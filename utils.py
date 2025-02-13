import heapq
import math
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import random
from matplotlib import cm
import spectral as spy
from sklearn import metrics
import time
from sklearn import preprocessing
import torch
import torch.nn.functional as F
import torch.nn as nn
import copy

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



def Draw_Classification_Map(label, name: str, scale: float = 4.0, dpi: int = 400):
    '''
    get classification map , then save to given path
    :param label: classification label, 2D
    :param name: saving path and file's name
    :param scale: scale of image. If equals to 1, then saving-size is just the label-size
    :param dpi: default is OK
    :return: null
    '''
    fig, ax = plt.subplots()
    numlabel = np.array(label)
    v = spy.imshow(classes=numlabel.astype(np.int16), fignum=fig.number)
    ax.set_axis_off()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    fig.set_size_inches(label.shape[1] * scale / dpi, label.shape[0] * scale / dpi)
    foo_fig = plt.gcf()  # 'get current figure'
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    foo_fig.savefig(name + '.png', format='png', transparent=True, dpi=dpi, pad_inches=0)
    pass


def GT_To_One_Hot(gt, class_count):
    '''
    Convet Gt to one-hot labels
    :param gt:
    :param class_count:
    :return:
    '''
    GT_One_Hot = []  # 转化为one-hot形式的标签
    height = gt.shape[0]
    width = gt.shape[1]
    for i in range(gt.shape[0]):
        for j in range(gt.shape[1]):
            temp = np.zeros(class_count, dtype=np.float32)
            if gt[i, j] != 0:
                temp[int(gt[i, j]) - 1] = 1
            GT_One_Hot.append(temp)
    GT_One_Hot = np.reshape(GT_One_Hot, [height, width, class_count])
    return GT_One_Hot


def SAM_vector(H_i, H_j):
    SAM_value = math.sqrt(torch.dot(H_i, H_i)) * math.sqrt(torch.dot(H_j, H_j))
    SAM_value = torch.tensor(SAM_value)
    SAM_value = torch.dot(H_i, H_j) / SAM_value
    if SAM_value > 1 or SAM_value < -1:
        # print('ferergdfgrgewfwf')
        # print("%.10f"%SAM_value)
        SAM_value = 1
    SAM_value = math.acos(SAM_value)
    SAM_value = torch.tensor(SAM_value)
    return SAM_value


def SAM(X, Y):
    # 对于两个特征，它们的余弦相似度就是两个特征在经过L2归一化之后的矩阵内积
    feature1 = X
    feature2 = Y
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

def SAM_distance(X, Y):
    # 对于两个特征，它们的余弦相似度就是两个特征在经过L2归一化之后的矩阵内积
    feature1 = X
    feature2 = Y
    feature1 = F.normalize(feature1)  # F.normalize只能处理两维的数据，L2归一化
    feature2 = F.normalize(feature2)
    distance = feature1.mm(feature2.t())  # 计算余弦相似度
    # 将定义域限制在（-1，1）
    distance = torch.clamp(distance, -1, 1)
    # SAM_value = torch.acos(distance)
    # SAM_value = SAM_value.cpu().detach().numpy()
    # SAM_value[np.isnan(SAM_value)] = 0
    # SAM_value = torch.from_numpy(SAM_value.astype(np.float32)).to(device)
    return distance

def EuclideanDistances(x, y):
    m, n = x.size(0), y.size(0)
    # xx经过pow()方法对每单个数据进行二次方操作后，在axis=1 方向（横向，就是第一列向最后一列的方向）加和，此时xx的shape为(m, 1)，经过expand()方法，扩展n-1次，此时xx的shape为(m, n)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    # yy会在最后进行转置的操作
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    # torch.addmm(beta=1, input, alpha=1, mat1, mat2, out=None)，这行表示的意思是dist - 2 * x * yT
    dist = dist - 2 * torch.mm(x, y.t())
    # clamp()函数可以限定dist内元素的最大最小范围，dist最后开方，得到样本之间的距离矩阵
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist

def get_A_k(data_HSI, data_LiDAR, k):
    n = data_HSI.shape[0]

    # 代表与每个节点最相似的k个节点，相似则为1
    A_k_HSI = torch.zeros(n, n).to(device)
    A_k_HSI_1 = torch.zeros(n, n).to(device)
    A_k_LiDAR = torch.zeros(n, n).to(device)

    # 计算每个pixel与其他pixel的关系，此处为superpixel
    HSI_rel = SAM(data_HSI, data_HSI)
    LiDAR_rel = EuclideanDistances(data_LiDAR, data_LiDAR)


    # 求HSI中关联性最大的pixel索引
    index = torch.argsort(HSI_rel)    #argsort将元素从小到大排列，提取其index
    for i in range(n):
        for j in range(k):
            A_k_HSI[i, index[i, j]] = 1


    # 求LiDAR中关联性最大的pixel索引
    index = torch.argsort(LiDAR_rel)    #argsort将元素从小到大排列，提取其index
    for i in range(n):
        for j in range(k):
            A_k_LiDAR[i, index[i, j]] = 1

    return A_k_HSI, A_k_LiDAR

def get_A_r(A, r):
    adj = A
    if r == 1:
        adj = adj
    elif r == 2:
        adj = adj @ adj
    elif r == 3:
        adj = adj @ adj @ adj
    elif r == 4:
        adj = adj @ adj @ adj @ adj
    return adj

def entropy_loss(predict: torch.Tensor, reallabel_onehot: torch.Tensor, reallabel_mask: torch.Tensor):
    real_labels = reallabel_onehot
    we = -torch.mul(real_labels, torch.log(predict))
    we = torch.mul(we, reallabel_mask)
    pool_cross_entropy = torch.sum(we)
    return pool_cross_entropy / 180

def graph_loss(Q, HSI_result, LiDAR_result, A_r_HSI, A_r_LiDAR):
    n = Q.shape[1]

    # 计算分母
    distance_HSI = SAM_distance(HSI_result, HSI_result)
    distance_LiDAR = EuclideanDistances(LiDAR_result, LiDAR_result)

    mother_HSI = torch.sum(distance_HSI, dim=1)
    mother_LiDAR = torch.sum(distance_LiDAR, dim=1)
    # 计算时不包括当前特征和自己的关系，需要减1
    mother_HSI = mother_HSI - 1
    mother_LiDAR = mother_LiDAR - 1

    r_rela_HSI = torch.mul(A_r_LiDAR, distance_HSI)  # 哈达玛积，计算每个超像素与其r-hop节点的关系
    r_rela_HSI = torch.sum(r_rela_HSI, 1)
    son_HSI = r_rela_HSI

    r_rela_LIDAR = torch.mul(A_r_HSI, distance_LiDAR)  # 哈达玛积，计算每个超像素与其r-hop节点的关系
    r_rela_LIDAR = torch.sum(r_rela_LIDAR, 1)
    son_LiDAR = r_rela_LIDAR

    # 对定义域进行限制
    mother_HSI = torch.clamp(mother_HSI, min=1e-12)
    result_HSI = son_HSI / mother_HSI
    result_HSI = torch.clamp(result_HSI, min=1e-12)

    mother_LiDAR = torch.clamp(mother_LiDAR, min=1e-12)
    result_LiDAR = son_LiDAR / mother_LiDAR
    result_LiDAR = torch.clamp(result_LiDAR, min=1e-12)

    loss_HSI = -torch.log(result_HSI)
    loss_HSI = torch.sum(loss_HSI, 0) / n

    loss_LiDAR = -torch.log(result_LiDAR)
    loss_LiDAR = torch.sum(loss_LiDAR, 0) / n

    loss = (loss_HSI + loss_LiDAR) / 2
    return loss