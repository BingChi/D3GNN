import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import scipy.io as sio
import copy
import utils

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def SAM(X):
    # 对于两个特征，它们的余弦相似度就是两个特征在经过L2归一化之后的矩阵内积
    feature1 = X
    feature2 = X
    feature1 = F.normalize(feature1)  # F.normalize只能处理两维的数据，L2归一化
    feature2 = F.normalize(feature2)
    distance = feature1.mm(feature2.t())  # 计算余弦相似度
    # 将定义域限制在（-1，1）
    distance = torch.clamp(distance, -1, 1)
    SAM_value = torch.acos(distance)
    SAM_value = SAM_value.cpu().detach().numpy()
    SAM_value[np.isnan(SAM_value)] = 0
    SAM_value = torch.from_numpy(SAM_value.astype(np.float32)).to(device)
    return SAM_value

def EuclideanDistances(x):
    y = x
    m, n = x.size(0), y.size(0)
    # xx经过pow()方法对每单个数据进行二次方操作后，在axis=1 方向（横向，就是第一列向最后一列的方向）加和，此时xx的shape为(m, 1)，经过expand()方法，扩展n-1次，此时xx的shape为(m, n)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    # yy会在最后进行转置的操作
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    # torch.addmm(beta=1, input, alpha=1, mat1, mat2, out=None)，这行表示的意思是dist - 2 * x * yT
    dist.addmm_(1, -2, x, y.t())
    # clamp()函数可以限定dist内元素的最大最小范围，dist最后开方，得到样本之间的距离矩阵
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist

def DropEdge(A):
    percent = 0.8
    A = A.detach().cpu().numpy()
    A_triu = np.triu(A, k=0)  # 取A的上三角
    B = np.zeros((A.shape[0], A.shape[0]))  # 设定一个空矩阵
    k = np.argwhere(A_triu)  # 寻找上三角矩阵中不为0的边
    nnz = k.shape[0]  # 一共有多少条边
    perm = np.random.permutation(k)  # 随机排列
    preserve_nnz = int(nnz * percent)  # 取出比例为多少条
    perm = perm[:preserve_nnz]  # 对随机排列的边取出前preserve_nnz条
    for i in range(perm.shape[0]):
        B[perm[i, 0], perm[i, 1]] = A[perm[i, 0], perm[i, 1]]
    B = B + B.T - np.diag(np.diag(B))  # 将上三角转为对称阵
    B = torch.from_numpy(B.astype(np.float32)).to(device)
    return B

class GCNLayer_HSI(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, A: torch.Tensor):
        super(GCNLayer_HSI, self).__init__()
        self.A = A
        self.count = A.shape[0]
        self.BN = nn.BatchNorm1d(input_dim)
        self.Activation = nn.LeakyReLU()
        self.sigma1 = torch.nn.Parameter(torch.tensor([0.1], requires_grad=True))
        # 第一层GCN
        self.GCN_liner_theta_1 = nn.Sequential(nn.Linear(input_dim, 256))   # 用来更新邻接矩阵用，源代码用的是H_l
        self.GCN_liner_Attention_A = nn.Sequential(nn.Linear(self.count, self.count))
        self.GCN_liner_out_1 = nn.Sequential(nn.Linear(input_dim, output_dim))  # 图卷积层的权重更新
        nodes_count = self.A.shape[0]
        self.I = torch.eye(nodes_count, nodes_count, requires_grad=False).to(device)    # 单位矩阵
        self.mask = torch.ceil(self.A * 0.00001)    # ceil为向上取整，先乘一个极小值，再向上取整，即不为0的点的值为1

    def A_to_D_inv(self, A: torch.Tensor):
        D = A.sum(1)
        D_hat = torch.diag(torch.pow(D, -0.5))
        return D_hat

    def forward(self, H, model='normal'):
        # 方案一：minmax归一化，即正常的GCN操作
        tic = time.perf_counter()
        # sio.savemat(str(epoch) + "&" + str(tic) + "0" + "_H", {'H': H.detach().cpu().numpy()})
        H = self.BN(H)
        # sio.savemat(str(epoch) + "&" + str(tic) + "1" + "_H_BN", {'H_BN': H.detach().cpu().numpy()})
        H_xx1 = self.GCN_liner_theta_1(H)
        A = SAM(H_xx1)
        # sio.savemat(str(epoch) + "&" + str(tic) + "2" + "_A", {'A': A.detach().cpu().numpy()})
        # A = self.GCN_liner_Attention_A(A)
        A = torch.exp(-0.2 * A)
        A = torch.clamp(torch.sigmoid(A), min=0.1)
        A = DropEdge(A)
        A = A * self.mask + self.I
        A = A.detach().cpu().numpy()
        A = np.where(A > 0.7, A, 0)      # 给A卡一个阈值
        A = torch.from_numpy(A).to(device)
        # if model != 'normal':
        #     A = torch.clamp(A, 0.1)  # This is a trick.
        D_hat = self.A_to_D_inv(A)
        # sio.savemat(str(epoch) + "&" + str(tic) + "3" + "_D_hat", {'D_hat': D_hat.detach().cpu().numpy()})
        A_hat = torch.matmul(D_hat, torch.matmul(A, D_hat))     # matmul:矩阵乘法，有广播机制
        # sio.savemat(str(epoch) + "&" + str(tic) + "4" + "_A_hat", {'A_hat': A_hat.detach().cpu().numpy()})
        output = torch.mm(A_hat, self.GCN_liner_out_1(H))       # mm:矩阵乘法，无广播机制
        # sio.savemat(str(epoch) + "&" + str(tic) + "5" + "_output", {'output': output.detach().cpu().numpy()})
        output = self.Activation(output)
        # sio.savemat(str(epoch) + "&" + str(tic) + "6" + "_output_A", {'output_A': output.detach().cpu().numpy()})

        # 别人的方法
        # H = self.BN(H)
        # H_xx1 = self.GCN_liner_theta_1(H)
        # A = torch.clamp(torch.sigmoid(torch.matmul(H_xx1, H_xx1.t())), min=0.1) * self.mask + self.I
        # if model != 'normal': A = torch.clamp(A, 0.1) #This is a trick.
        # D_hat = self.A_to_D_inv(A)
        # A_hat = torch.matmul(D_hat, torch.matmul(A, D_hat))
        # output = torch.mm(A_hat, self.GCN_liner_out_1(H))
        # output = self.Activation(output)     # clamp限制数的下限

        # 方案二：softmax归一化 (加速运算)
        # H = self.BN(H)
        # H_xx1 = self.GCN_liner_theta_1(H)
        # e = torch.sigmoid(torch.matmul(H_xx1, H_xx1.t()))
        # zero_vec = -9e15 * torch.ones_like(e)
        # A = torch.where(self.mask > 0, e, zero_vec) + self.I   # 邻接矩阵
        # if model != 'normal':
        #     A = torch.clamp(A, 0.1) #This is a trick for the Indian Pines.
        # A = F.softmax(A, dim=1)
        # output = self.Activation(torch.mm(A, self.GCN_liner_out_1(H)))

        return output, A

class GCNLayer_LiDAR(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, A: torch.Tensor):
        super(GCNLayer_LiDAR, self).__init__()
        self.A = A
        self.count = A.shape[0]
        self.BN = nn.BatchNorm1d(input_dim)
        self.Activation = nn.LeakyReLU()
        self.sigma1 = torch.nn.Parameter(torch.tensor([0.1], requires_grad=True))
        # 第一层GCN
        self.GCN_liner_theta_1 = nn.Sequential(nn.Linear(input_dim, 256))   # 用来更新邻接矩阵用，源代码用的是H_l
        self.GCN_liner_Attention_A = nn.Sequential(nn.Linear(self.count, self.count))
        self.GCN_liner_out_1 = nn.Sequential(nn.Linear(input_dim, output_dim))  # 图卷积层的权重更新
        nodes_count = self.A.shape[0]
        self.I = torch.eye(nodes_count, nodes_count, requires_grad=False).to(device)    # 单位矩阵
        self.mask = torch.ceil(self.A * 0.00001)    # ceil为向上取整，先乘一个极小值，再向上取整，即不为0的点的值为1

    def A_to_D_inv(self, A: torch.Tensor):
        D = A.sum(1)
        D_hat = torch.diag(torch.pow(D, -0.5))
        return D_hat

    def forward(self, H, model='normal'):
        # 方案一：minmax归一化，即正常的GCN操作
        H = self.BN(H)
        H_xx1 = self.GCN_liner_theta_1(H)
        A = EuclideanDistances(H_xx1)
        # A = self.GCN_liner_Attention_A(A)
        A = torch.exp(-0.2 * A)
        A = torch.clamp(torch.sigmoid(A), min=0.1)
        A = DropEdge(A)
        A = A * self.mask + self.I

        # A = DropEdge(A)
        A = A.detach().cpu().numpy()
        A = np.where(A > 0.6, A, 0)      # 给A卡一个阈值
        A = torch.from_numpy(A).to(device)
        # if model != 'normal':
        #     A = torch.clamp(A, 0.1)  # This is a trick.
        D_hat = self.A_to_D_inv(A)
        A_hat = torch.matmul(D_hat, torch.matmul(A, D_hat))     # matmul:矩阵乘法，有广播机制
        output = torch.mm(A_hat, self.GCN_liner_out_1(H))       # mm:矩阵乘法，无广播机制
        output = self.Activation(output)

        # A = torch.zeros([H.shape[0], H.shape[0]]).cuda()
        # for i in range(H.shape[0]):
        #     for j in range(i, H.shape[0]):
        #         A[i, j] = A[j, i] = torch.dist(H[i, :], H[j, :])
        # print('已计算一次基于欧式距离的LiDAR邻接矩阵')

        # A = self.GCN_liner_Attention_A(A)

        # 别人的方法
        # H = self.BN(H)
        # H_xx1 = self.GCN_liner_theta_1(H)
        # A = torch.clamp(torch.sigmoid(torch.matmul(H_xx1, H_xx1.t())), min=0.1) * self.mask + self.I
        # if model != 'normal': A = torch.clamp(A, 0.1) #This is a trick.
        # D_hat = self.A_to_D_inv(A)
        # A_hat = torch.matmul(D_hat, torch.matmul(A, D_hat))
        # output = torch.mm(A_hat, self.GCN_liner_out_1(H))
        # output = self.Activation(output)     # clamp限制数的下限      # clamp限制数的下限


        # 方案二：softmax归一化 (加速运算)
        # H = self.BN(H)
        # H_xx1= self.GCN_liner_theta_1(H)
        # e = torch.sigmoid(torch.matmul(H_xx1, H_xx1.t()))
        # zero_vec = -9e15 * torch.ones_like(e)
        # A = torch.where(self.mask > 0, e, zero_vec) + self.I   # 邻接矩阵
        # if model != 'normal':
        #     A = torch.clamp(A, 0.1) #This is a trick for the Indian Pines.
        # A = F.softmax(A, dim=1)
        # output = self.Activation(torch.mm(A, self.GCN_liner_out_1(H)))

        return output, A

class GCNLayer_HSI_spectral(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, A: torch.Tensor):
        super(GCNLayer_HSI_spectral, self).__init__()
        self.A = A
        self.count = A.shape[0]
        self.BN = nn.BatchNorm1d(input_dim)
        self.Activation = nn.LeakyReLU()
        self.sigma1 = torch.nn.Parameter(torch.tensor([0.1], requires_grad=True))
        # 第一层GCN
        self.GCN_liner_theta_1 = nn.Sequential(nn.Linear(input_dim, 256))   # 用来更新邻接矩阵用，源代码用的是H_l
        self.GCN_liner_Attention_A = nn.Sequential(nn.Linear(self.count, self.count))
        self.GCN_liner_out_1 = nn.Sequential(nn.Linear(input_dim, output_dim))  # 图卷积层的权重更新
        nodes_count = self.A.shape[0]
        self.I = torch.eye(nodes_count, nodes_count, requires_grad=False).to(device)    # 单位矩阵
        self.mask = torch.ceil(self.A * 0.00001)    # ceil为向上取整，先乘一个极小值，再向上取整，即不为0的点的值为1

    def A_to_D_inv(self, A: torch.Tensor):
        D = A.sum(1)
        D_hat = torch.diag(torch.pow(D, -0.5))
        return D_hat

    def forward(self, H, model='normal'):
        # 方案一：minmax归一化，即正常的GCN操作
        # sio.savemat(str(epoch) + "&" + str(tic) + "0" + "_H", {'H': H.detach().cpu().numpy()})
        H = self.BN(H)
        # sio.savemat(str(epoch) + "&" + str(tic) + "1" + "_H_BN", {'H_BN': H.detach().cpu().numpy()})
        H_xx1 = self.GCN_liner_theta_1(H)
        A = SAM(H_xx1)
        # sio.savemat(str(epoch) + "&" + str(tic) + "2" + "_A", {'A': A.detach().cpu().numpy()})
        # A = self.GCN_liner_Attention_A(A)
        A = torch.exp(-0.2 * A)
        A = torch.clamp(torch.sigmoid(A), min=0.1)
        for (type) in [(2)]:
            if type == 1:
                A = DropEdge(A)
                A = A * self.mask + self.I
                A = A.detach().cpu().numpy()
                A = np.where(A > 0.7, A, 0)  # 给A卡一个阈值
                A = torch.from_numpy(A).to(device)
            if type == 2:
                k = 5
                n = A.shape[0]
                A_k_HSI = torch.zeros(n, n).to(device)
                index = torch.argsort(-A)  # argsort将元素从小到大排列，提取其index，此处应提取最大值
                for i in range(n):
                    for j in range(k):
                        A_k_HSI[i, index[i, j]] = 1
                A = torch.mul(A, A_k_HSI)
        # if model != 'normal':
        #     A = torch.clamp(A, 0.1)  # This is a trick.
        D_hat = self.A_to_D_inv(A)
        # sio.savemat(str(epoch) + "&" + str(tic) + "3" + "_D_hat", {'D_hat': D_hat.detach().cpu().numpy()})
        A_hat = torch.matmul(D_hat, torch.matmul(A, D_hat))     # matmul:矩阵乘法，有广播机制
        # sio.savemat(str(epoch) + "&" + str(tic) + "4" + "_A_hat", {'A_hat': A_hat.detach().cpu().numpy()})
        output = torch.mm(A_hat, self.GCN_liner_out_1(H))       # mm:矩阵乘法，无广播机制
        # sio.savemat(str(epoch) + "&" + str(tic) + "5" + "_output", {'output': output.detach().cpu().numpy()})
        output = self.Activation(output)
        # sio.savemat(str(epoch) + "&" + str(tic) + "6" + "_output_A", {'output_A': output.detach().cpu().numpy()})

        # 别人的方法
        # H = self.BN(H)
        # H_xx1 = self.GCN_liner_theta_1(H)
        # A = torch.clamp(torch.sigmoid(torch.matmul(H_xx1, H_xx1.t())), min=0.1) * self.mask + self.I
        # if model != 'normal': A = torch.clamp(A, 0.1) #This is a trick.
        # D_hat = self.A_to_D_inv(A)
        # A_hat = torch.matmul(D_hat, torch.matmul(A, D_hat))
        # output = torch.mm(A_hat, self.GCN_liner_out_1(H))
        # output = self.Activation(output)     # clamp限制数的下限

        # 方案二：softmax归一化 (加速运算)
        # H = self.BN(H)
        # H_xx1 = self.GCN_liner_theta_1(H)
        # e = torch.sigmoid(torch.matmul(H_xx1, H_xx1.t()))
        # zero_vec = -9e15 * torch.ones_like(e)
        # A = torch.where(self.mask > 0, e, zero_vec) + self.I   # 邻接矩阵
        # if model != 'normal':
        #     A = torch.clamp(A, 0.1) #This is a trick for the Indian Pines.
        # A = F.softmax(A, dim=1)
        # output = self.Activation(torch.mm(A, self.GCN_liner_out_1(H)))

        return output, A

class GCNLayer_LiDAR_elevation(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, A: torch.Tensor):
        super(GCNLayer_LiDAR_elevation, self).__init__()
        self.A = A
        self.count = A.shape[0]
        self.BN = nn.BatchNorm1d(input_dim)
        self.Activation = nn.LeakyReLU()
        self.sigma1 = torch.nn.Parameter(torch.tensor([0.1], requires_grad=True))
        # 第一层GCN
        self.GCN_liner_theta_1 = nn.Sequential(nn.Linear(input_dim, 256))   # 用来更新邻接矩阵用，源代码用的是H_l
        self.GCN_liner_Attention_A = nn.Sequential(nn.Linear(self.count, self.count))
        self.GCN_liner_out_1 = nn.Sequential(nn.Linear(input_dim, output_dim))  # 图卷积层的权重更新
        nodes_count = self.A.shape[0]
        self.I = torch.eye(nodes_count, nodes_count, requires_grad=False).to(device)    # 单位矩阵
        self.mask = torch.ceil(self.A * 0.00001)    # ceil为向上取整，先乘一个极小值，再向上取整，即不为0的点的值为1

    def A_to_D_inv(self, A: torch.Tensor):
        D = A.sum(1)
        D_hat = torch.diag(torch.pow(D, -0.5))
        return D_hat

    def forward(self, H, model='normal'):
        # 方案一：minmax归一化，即正常的GCN操作
        H = self.BN(H)
        H_xx1 = self.GCN_liner_theta_1(H)
        A = EuclideanDistances(H_xx1)
        # A = self.GCN_liner_Attention_A(A)
        A = torch.exp(-0.2 * A)
        A = torch.clamp(torch.sigmoid(A), min=0.1)
        for (type) in [(2)]:
            if type == 1:
                A = DropEdge(A)
                A = A * self.mask + self.I
                A = A.detach().cpu().numpy()
                A = np.where(A > 0.7, A, 0)  # 给A卡一个阈值
                A = torch.from_numpy(A).to(device)
            if type == 2:
                k = 5
                n = A.shape[0]
                A_k_LiDAR = torch.zeros(n, n).to(device)
                index = torch.argsort(-A)  # argsort将元素从小到大排列，提取其index，此处应提取最大值
                for i in range(n):
                    for j in range(k):
                        A_k_LiDAR[i, index[i, j]] = 1
                A = torch.mul(A, A_k_LiDAR)
        # if model != 'normal':
        #     A = torch.clamp(A, 0.1)  # This is a trick.
        D_hat = self.A_to_D_inv(A)
        A_hat = torch.matmul(D_hat, torch.matmul(A, D_hat))     # matmul:矩阵乘法，有广播机制
        output = torch.mm(A_hat, self.GCN_liner_out_1(H))       # mm:矩阵乘法，无广播机制
        output = self.Activation(output)

        # A = torch.zeros([H.shape[0], H.shape[0]]).cuda()
        # for i in range(H.shape[0]):
        #     for j in range(i, H.shape[0]):
        #         A[i, j] = A[j, i] = torch.dist(H[i, :], H[j, :])
        # print('已计算一次基于欧式距离的LiDAR邻接矩阵')

        # A = self.GCN_liner_Attention_A(A)

        # 别人的方法
        # H = self.BN(H)
        # H_xx1 = self.GCN_liner_theta_1(H)
        # A = torch.clamp(torch.sigmoid(torch.matmul(H_xx1, H_xx1.t())), min=0.1) * self.mask + self.I
        # if model != 'normal': A = torch.clamp(A, 0.1) #This is a trick.
        # D_hat = self.A_to_D_inv(A)
        # A_hat = torch.matmul(D_hat, torch.matmul(A, D_hat))
        # output = torch.mm(A_hat, self.GCN_liner_out_1(H))
        # output = self.Activation(output)     # clamp限制数的下限      # clamp限制数的下限


        # 方案二：softmax归一化 (加速运算)
        # H = self.BN(H)
        # H_xx1= self.GCN_liner_theta_1(H)
        # e = torch.sigmoid(torch.matmul(H_xx1, H_xx1.t()))
        # zero_vec = -9e15 * torch.ones_like(e)
        # A = torch.where(self.mask > 0, e, zero_vec) + self.I   # 邻接矩阵
        # if model != 'normal':
        #     A = torch.clamp(A, 0.1) #This is a trick for the Indian Pines.
        # A = F.softmax(A, dim=1)
        # output = self.Activation(torch.mm(A, self.GCN_liner_out_1(H)))

        return output, A

class GCNLayer_HSI_pixel(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super(GCNLayer_HSI_pixel, self).__init__()
        self.Activation = nn.LeakyReLU()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.sigma1 = torch.nn.Parameter(torch.tensor([0.1], requires_grad=True))
        # 第一层GCN
        self.GCN_liner_out_1 = nn.Sequential(nn.Linear(input_dim, output_dim))  # 图卷积层的权重更新

    def A_to_D_inv(self, A: torch.Tensor):
        D = A.sum(1)
        D_hat = torch.diag(torch.pow(D, -0.5))
        return D_hat

    def forward(self, H, A):
        H = torch.from_numpy(H.astype(np.float32)).to(device)
        A = torch.from_numpy(A.astype(np.float32)).to(device)
        A = torch.clamp(torch.sigmoid(A), min=0.1)
        D_hat = self.A_to_D_inv(A)
        A_hat = torch.matmul(D_hat, torch.matmul(A, D_hat))     # matmul:矩阵乘法，有广播机制
        output = torch.mm(A_hat, self.GCN_liner_out_1(H))       # mm:矩阵乘法，无广播机制
        output = self.Activation(output)
        output = output.cpu().detach().numpy()
        A = A.cpu().detach().numpy()
        return output, A

class GCNLayer_LiDAR_pixel(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super(GCNLayer_LiDAR_pixel, self).__init__()
        self.Activation = nn.LeakyReLU()
        self.sigma1 = torch.nn.Parameter(torch.tensor([0.1], requires_grad=True))
        # 第一层GCN
        self.GCN_liner_out_1 = nn.Sequential(nn.Linear(input_dim, output_dim))  # 图卷积层的权重更新

    def A_to_D_inv(self, A: torch.Tensor):
        D = A.sum(1)
        D_hat = torch.diag(torch.pow(D, -0.5))
        return D_hat

    def forward(self, H, A):
        H = torch.from_numpy(H.astype(np.float32)).to(device)
        A = torch.from_numpy(A.astype(np.float32)).to(device)
        A = torch.clamp(torch.sigmoid(A), min=0.1)
        D_hat = self.A_to_D_inv(A)
        A_hat = torch.matmul(D_hat, torch.matmul(A, D_hat))     # matmul:矩阵乘法，有广播机制
        output = torch.mm(A_hat, self.GCN_liner_out_1(H))       # mm:矩阵乘法，无广播机制
        output = self.Activation(output)
        output = output.cpu().detach().numpy()
        A = A.cpu().detach().numpy()
        return output, A

class P_HSI_update(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super(P_HSI_update, self).__init__()
        self.Activation = nn.LeakyReLU()
        self.sigma1 = torch.nn.Parameter(torch.tensor([0.1], requires_grad=True))
        self.P_update = nn.Sequential(nn.Linear(input_dim, output_dim))  # 图卷积层的权重更新

    def forward(self, P_HSI):
        mask = torch.ceil(P_HSI * 0.00001)
        P_HSI = self.P_update(P_HSI)
        P_HSI = P_HSI * mask
        return P_HSI

class P_LiDAR_update(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super(P_LiDAR_update, self).__init__()
        self.Activation = nn.LeakyReLU()
        self.sigma1 = torch.nn.Parameter(torch.tensor([0.1], requires_grad=True))
        self.P_update = nn.Sequential(nn.Linear(input_dim, output_dim))  # 图卷积层的权重更新

    def forward(self, P_LiDAR):
        mask = torch.ceil(P_LiDAR * 0.00001)
        P_LiDAR = self.P_update(P_LiDAR)
        P_LiDAR = P_LiDAR * mask
        return P_LiDAR

class GraphOnly(nn.Module):
    def __init__(self, height: int, width: int, changel: int, class_count: int, Q: torch.Tensor, A: torch.Tensor,
                 A_k_HSI: torch.Tensor, A_k_LiDAR: torch.Tensor, model='normal'):
        super(GraphOnly, self).__init__()
        self.class_count = class_count  # 类别数
        # 网络输入数据大小
        self.channel = changel
        self.height = height
        self.width = width
        self.Q = Q
        self.A = A
        self.A_k_HSI = A_k_HSI
        self.A_k_LiDAR = A_k_LiDAR
        self.model = model
        self.norm_col_Q = Q / (torch.sum(Q, 0, keepdim=True))  # 列归一化Q
        self.superpixel_count = Q.shape[1]

        layers_count = 3

        # ######空间双支路######
        # GCN for HSI
        self.GCN_Branch_HSI_spatial = nn.Sequential()
        self.GCN_Branch_HSI_spatial.add_module('GCN_Branch_HSI_spatial_1', GCNLayer_HSI(self.channel, 700, self.A))
        self.GCN_Branch_HSI_spatial.add_module('GCN_Branch_HSI_spatial_2', GCNLayer_HSI(700, 128, self.A))
        # self.GCN_Branch_HSI_spatial.add_module('GCN_Branch_HSI_spatial_3', GCNLayer_HSI(128, 128, self.A))
        # print(self.GCN_Branch_HSI_spatial)

        # GCN for LiDAR
        self.GCN_Branch_LiDAR_spatial = nn.Sequential()
        self.GCN_Branch_LiDAR_spatial.add_module('GCN_Branch_LiDAR_spatial_1', GCNLayer_LiDAR(1, 700, self.A))
        self.GCN_Branch_LiDAR_spatial.add_module('GCN_Branch_LiDAR_spatial_2', GCNLayer_LiDAR(700, 128, self.A))
        # self.GCN_Branch_LiDAR_spatial.add_module('GCN_Branch_LiDAR_spatial_3', GCNLayer_LiDAR(128, 128, self.A))
        # print(self.GCN_Branch_LiDAR_spatial)

        # ######光谱高程双支路######
        # GCN for HSI
        self.GCN_Branch_HSI_spectral = nn.Sequential()
        self.GCN_Branch_HSI_spectral.add_module('GCN_Branch_HSI_spectral_1', GCNLayer_HSI_spectral(self.channel, 700, self.A_k_HSI))
        self.GCN_Branch_HSI_spectral.add_module('GCN_Branch_HSI_spectral_2', GCNLayer_HSI_spectral(700, 128, self.A_k_HSI))
        # self.GCN_Branch_HSI_spatial.add_module('GCN_Branch_HSI_spectral_3', GCNLayer_HSI(128, 128, self.A_k_HSI))
        # print(self.GCN_Branch_HSI_spatial)

        # GCN for LiDAR
        self.GCN_Branch_LiDAR_elevation = nn.Sequential()
        self.GCN_Branch_LiDAR_elevation.add_module('GCN_Branch_LiDAR_elevation_1', GCNLayer_LiDAR_elevation(1, 700, self.A_k_LiDAR))
        self.GCN_Branch_LiDAR_elevation.add_module('GCN_Branch_LiDAR_elevation_2', GCNLayer_LiDAR_elevation(700, 128, self.A_k_LiDAR))
        # self.GCN_Branch_LiDAR_spatial.add_module('GCN_Branch_LiDAR_elevation_3', GCNLayer_LiDAR(128, 128, self.A_k_LiDAR))
        # print(self.GCN_Branch_LiDAR_spatial)

        # #####Softmax layer#####
        self.Fuse_linear = nn.Sequential(nn.Linear(512, 128))
        self.Softmax_linear = nn.Sequential(nn.Linear(512, self.class_count))
        self.Softmax_linear_MUUFL = nn.Sequential(nn.Linear(768, self.class_count))

    def forward(self, x_HSI: torch.Tensor, x_LiDAR: torch.Tensor):
        """
        :param x_HSI: H*W*C
        :param x_LiDAR: H*W*1
        :return: probability_map
        """
        # tic = time.perf_counter()
        dataset = x_LiDAR.shape[1]    # 1-其他数据集，2-MUUFL数据集
        if dataset == 1:
            H_HSI_spatial = H_HSI_spectral = x_HSI
            H_LiDAR_spatial = H_LiDAR_elevation = x_LiDAR

            # #####同构双支路#####
            for i in range(len(self.GCN_Branch_HSI_spatial)):
                H_HSI_spatial, _ = self.GCN_Branch_HSI_spatial[i](H_HSI_spatial)

            for i in range(len(self.GCN_Branch_LiDAR_spatial)):
                H_LiDAR_spatial, _ = self.GCN_Branch_LiDAR_spatial[i](H_LiDAR_spatial)

            # #####异构双支路#####
            for i in range(len(self.GCN_Branch_HSI_spectral)):
                H_HSI_spectral, _ = self.GCN_Branch_HSI_spectral[i](H_HSI_spectral)

            for i in range(len(self.GCN_Branch_LiDAR_elevation)):
                H_LiDAR_elevation, _ = self.GCN_Branch_LiDAR_elevation[i](H_LiDAR_elevation)

            GCN_result_HSI_spatial = torch.matmul(self.Q, H_HSI_spatial)  # 这里self.norm_row_Q == self.Q
            GCN_result_LiDAR_spatial = torch.matmul(self.Q, H_LiDAR_spatial)
            GCN_result_HSI_spectral = torch.matmul(self.Q, H_HSI_spectral)
            GCN_result_LiDAR_elevation = torch.matmul(self.Q, H_LiDAR_elevation)

            # Y = GCN_result_HSI_spatial
            # Y = GCN_result_LiDAR_spatial
            # Y = GCN_result_HSI_spectral
            # Y = GCN_result_LiDAR_elevation

            # Y = torch.cat([GCN_result_HSI_spatial, GCN_result_LiDAR_spatial], dim=1)
            # Y = torch.cat([GCN_result_HSI_spectral, GCN_result_LiDAR_elevation], dim=1)
            Y = torch.cat([GCN_result_HSI_spatial, GCN_result_LiDAR_spatial,
                           GCN_result_HSI_spectral, GCN_result_LiDAR_elevation], dim=1)
            # Y = GCN_result_HSI + GCN_result_HSI

            Y_feature = Y
            # Y = self.Fuse_linear(Y)
            Y = self.Softmax_linear(Y)
            Y = F.softmax(Y, -1)
            H_HSI = H_HSI_spectral
            H_LiDAR = H_LiDAR_elevation
            pass
        if dataset == 2:
            H_HSI_spatial = H_HSI_spectral = x_HSI
            H_LiDAR_spatial_1 = H_LiDAR_elevation_1 = x_LiDAR[:, 0].unsqueeze(dim=1)
            H_LiDAR_spatial_2 = H_LiDAR_elevation_2 = x_LiDAR[:, 1].unsqueeze(dim=1)



            # #####同构双支路#####
            for i in range(len(self.GCN_Branch_HSI_spatial)):
                H_HSI_spatial, _ = self.GCN_Branch_HSI_spatial[i](H_HSI_spatial)

            for i in range(len(self.GCN_Branch_LiDAR_spatial)):
                H_LiDAR_spatial_1, _ = self.GCN_Branch_LiDAR_spatial[i](H_LiDAR_spatial_1)

            for i in range(len(self.GCN_Branch_LiDAR_spatial)):
                H_LiDAR_spatial_2, _ = self.GCN_Branch_LiDAR_spatial[i](H_LiDAR_spatial_2)

            # #####异构双支路#####
            for i in range(len(self.GCN_Branch_HSI_spectral)):
                H_HSI_spectral, _ = self.GCN_Branch_HSI_spectral[i](H_HSI_spectral)

            for i in range(len(self.GCN_Branch_LiDAR_elevation)):
                H_LiDAR_elevation_1, _ = self.GCN_Branch_LiDAR_elevation[i](H_LiDAR_elevation_1)

            for i in range(len(self.GCN_Branch_LiDAR_elevation)):
                H_LiDAR_elevation_2, _ = self.GCN_Branch_LiDAR_elevation[i](H_LiDAR_elevation_2)

            GCN_result_HSI_spatial = torch.matmul(self.Q, H_HSI_spatial)  # 这里self.norm_row_Q == self.Q
            GCN_result_LiDAR_spatial_1 = torch.matmul(self.Q, H_LiDAR_spatial_1)
            GCN_result_LiDAR_spatial_2 = torch.matmul(self.Q, H_LiDAR_spatial_2)
            GCN_result_HSI_spectral = torch.matmul(self.Q, H_HSI_spectral)
            GCN_result_LiDAR_elevation_1 = torch.matmul(self.Q, H_LiDAR_elevation_1)
            GCN_result_LiDAR_elevation_2 = torch.matmul(self.Q, H_LiDAR_elevation_2)


            Y = torch.cat([GCN_result_HSI_spatial, GCN_result_LiDAR_spatial_1, GCN_result_LiDAR_spatial_2,
                           GCN_result_HSI_spectral, GCN_result_LiDAR_elevation_1, GCN_result_LiDAR_elevation_2], dim=1)
            # Y = GCN_result_HSI + GCN_result_HSI

            Y_feature = Y
            Y = self.Softmax_linear_MUUFL(Y)
            Y = F.softmax(Y, -1)
            H_HSI = H_HSI_spectral
            H_LiDAR = H_LiDAR_elevation_1



        return Y, H_HSI, H_LiDAR, Y_feature

        # 两组特征融合(两种融合方式)
        # Y = torch.cat([GCN_result, CNN_result], dim=-1)
        # Y = self.Softmax_linear(Y)
        # Y = F.softmax(Y, -1)
        # return Y