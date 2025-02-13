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
import LDA_SLIC
import GraphOnly
import utils
from torch import autograd
from thop import profile

print('\n')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
path = '/data/Experimental data'

# FLAG =1, Houston 350
# FLAG =2, MUUFL
# FLAG =3, trento 30
samples_type = ['ratio', 'fixed', 'same_num'][2]     # 比例，固定训练集还是固定的数


for (FLAG, curr_train_ratio, Scale) in [(1, 30, 350)]:
    # 取值为1，0.1和100 FLAG用来选取数据集，curr_train_ratio为训练集比例，Scale为超像素数目
    # for (FLAG, curr_train_ratio,Scale) in [(2,0.01,100),(3,0.01,100)]:
    torch.cuda.empty_cache()
    OA_ALL = []
    AA_ALL = []
    KPP_ALL = []
    AVG_ALL = []
    Train_Time_ALL = []
    Test_Time_ALL = []
    
    Seed_List = [0]   # 随机种子点
    
    if FLAG == 1:
        data_HSI_mat = sio.loadmat(path + '/HSI and LiDAR/2012houston/HSI_data.mat')
        data_HSI = data_HSI_mat['HSI_data']
        data_LiDAR_mat = sio.loadmat(path + '/HSI and LiDAR/2012houston/LiDAR_data.mat')
        data_LiDAR = data_LiDAR_mat['LiDAR_data']

        gt_mat = sio.loadmat(path + '/HSI and LiDAR/2012houston/All_Label.mat')
        gt = gt_mat['All_Label']
        train_gt_mat = sio.loadmat(path + '/HSI and LiDAR/2012houston/Train_Label.mat')
        train_gt = train_gt_mat['Train_Label']

        # 参数预设
        # train_ratio = 0.05  # 训练集比例。注意，训练集为按照‘每类’随机选取
        val_ratio = 1  # 测试集比例.注意，验证集选取为从测试集整体随机选取，非按照每类
        class_count = 15  # 样本类别数
        learning_rate = 2e-4  # 学习率  默认2e-4
        max_epoch = 1000  # 迭代次数
        dataset_name = "Huston"  # 数据集名称
        trainloss_result = np.zeros([max_epoch + 1, 1])
        LiDAR_bands = 1
        # superpixel_scale=100
        pass    # pass 不做任何事情，一般用做占位语句
    if FLAG == 2:
        # data_HSI_mat = sio.loadmat('D:\\program\\python\\HSI and LiDAR\\data\\MUUFL\\HSI_data.mat')
        # data_HSI = data_HSI_mat['HSI_data']
        # # data_LiDAR_mat = sio.loadmat('D:\\program\\python\\HSI and LiDAR\\data\\MUUFL\\LiDAR_data.mat')
        # # data_LiDAR = data_LiDAR_mat['LiDAR_data']
        # # data_LiDAR_mat = sio.loadmat('D:\\program\\python\\HSI and LiDAR\\data\\MUUFL\\LiDAR_data_1.mat')
        # # data_LiDAR = data_LiDAR_mat['LiDAR_data_1']
        # data_LiDAR_mat = sio.loadmat('D:\\program\\python\\HSI and LiDAR\\data\\MUUFL\\LiDAR_data_2.mat')
        # data_LiDAR = data_LiDAR_mat['LiDAR_data_2']
        #
        # gt_mat = sio.loadmat('D:\\program\\python\\HSI and LiDAR\\data\\MUUFL\\All_Label.mat')
        # gt = gt_mat['All_Label']

        data_HSI_mat = sio.loadmat(path + '/HSI and LiDAR/MUUFL/HSI_data.mat')
        data_HSI = data_HSI_mat['HSI_data']
        # data_LiDAR_mat = sio.loadmat(path + '/HSI and LiDAR/MUUFL/LiDAR_data.mat')
        # data_LiDAR = data_LiDAR_mat['LiDAR_data']
        # data_LiDAR_mat = sio.loadmat(path + '/HSI and LiDAR/MUUFL/LiDAR_data_1.mat')
        # data_LiDAR = data_LiDAR_mat['LiDAR_data_1']
        data_LiDAR_mat = sio.loadmat(path + '/HSI and LiDAR/MUUFL/LiDAR_data_2.mat')
        data_LiDAR = data_LiDAR_mat['LiDAR_data_2']

        gt_mat = sio.loadmat(path + '/HSI and LiDAR/MUUFL/All_Label.mat')
        gt = gt_mat['All_Label']

        
        # 参数预设
        # train_ratio = 0.01  # 训练集比例。注意，训练集为按照‘每类’随机选取
        val_ratio = 0.01  # 测试集比例.注意，验证集选取为从测试集整体随机选取，非按照每类
        class_count = 11  # 样本类别数
        learning_rate = 2e-5  # 学习率
        max_epoch = 1000  # 迭代次数
        dataset_name = "MUUFL"  # 数据集名称
        trainloss_result = np.zeros([max_epoch + 1, 1])
        LiDAR_bands = 1
        # superpixel_scale = 100
        pass
    if FLAG == 3:
        # data_HSI_mat = sio.loadmat('D:\\program\\python\\HSI and LiDAR\\data\\Trento\\HSI_data.mat')
        # data_HSI = data_HSI_mat['HSI_data']
        # data_LiDAR_mat = sio.loadmat('D:\\program\\python\\HSI and LiDAR\\data\\Trento\\LiDAR_data.mat')
        # data_LiDAR = data_LiDAR_mat['LiDAR_data']
        #
        # gt_mat = sio.loadmat('D:\\program\\python\\HSI and LiDAR\\data\\Trento\\All_Label.mat')
        # gt = gt_mat['All_Label']
        # train_gt_mat = sio.loadmat('D:\\program\\python\\HSI and LiDAR\\data\\Trento\\Train_Label.mat')
        # train_gt = train_gt_mat['Train_Label']

        data_HSI_mat = sio.loadmat(path + '/HSI and LiDAR/Trento/HSI_data.mat')
        data_HSI = data_HSI_mat['HSI_data']
        data_LiDAR_mat = sio.loadmat(path + '/HSI and LiDAR/Trento/LiDAR_data.mat')
        data_LiDAR = data_LiDAR_mat['LiDAR_data']

        gt_mat = sio.loadmat(path + '/HSI and LiDAR/Trento/All_Label.mat')
        gt = gt_mat['All_Label']
        train_gt_mat = sio.loadmat(path + '/HSI and LiDAR/Trento/Train_Label.mat')
        train_gt = train_gt_mat['Train_Label']
        
        # 参数预设
        # train_ratio = 0.01  # 训练集比例。注意，训练集为按照‘每类’随机选取
        val_ratio = 1  # 测试集比例.注意，验证集选取为从测试集整体随机选取，非按照每类
        class_count = 6  # 样本类别数
        learning_rate = 2e-5  # 学习率
        max_epoch = 600  # 迭代次数
        dataset_name = "Trento"  # 数据集名称
        trainloss_result = np.zeros([max_epoch + 1, 1])
        LiDAR_bands = 1
        # superpixel_scale = 100
        pass

    ###########
    superpixel_scale = Scale
    train_samples_per_class = curr_train_ratio  # 当定义为每类样本个数时,则该参数更改为训练样本数
    val_samples = class_count   # 样本类别数
    train_ratio = curr_train_ratio  # 训练比例
    cmap = cm.get_cmap('jet', class_count + 1)
    plt.set_cmap(cmap)
    m, n, d = data_HSI.shape  # 高光谱数据的三个维度

    # 数据standardization标准化,即提前全局BN
    height, width, bands = data_HSI.shape  # 原始高光谱数据的三个维度

    data_HSI = np.reshape(data_HSI, [height * width, bands])    # 将数据转为HW * B
    minMax = preprocessing.StandardScaler()
    data_HSI = minMax.fit_transform(data_HSI)   # 这两行用来归一化数据，归一化时需要进行数据转换
    data_HSI = np.reshape(data_HSI, [height, width, bands])     # 将数据转回去 H * W * B

    data_LiDAR = np.reshape(data_LiDAR, [height * width, LiDAR_bands])    # 将数据转为HW * B
    minMax = preprocessing.StandardScaler()
    data_LiDAR = minMax.fit_transform(data_LiDAR)   # 这两行用来归一化数据，归一化时需要进行数据转换
    data_LiDAR = np.reshape(data_LiDAR, [height, width, LiDAR_bands])     # 将数据转回去 H * W * B
    
     # 打印每类样本个数
    gt_reshape = np.reshape(gt, [-1])
    for i in range(class_count):
        idx = np.where(gt_reshape == i + 1)[-1]
        samplesCount = len(idx)
        print('第' + str(i + 1) + '类的个数为' + str(samplesCount))

    def nocross(Q):
        Q_temp = Q
        Q = Q_temp
        [h, w, c] = net_input_HSI.shape

        # print(self.Q.shape)
        norm_col_Q = torch.sum(Q, 0, keepdim=True)
        x_HSI_flatten = net_input_HSI.reshape([h * w, -1])
        superpixels_flatten_HSI = torch.mm(Q.t(), x_HSI_flatten)
        x_LiDAR_flatten = net_input_LiDAR.reshape([h * w, -1])
        superpixels_flatten_LiDAR = torch.mm(Q.t(), x_LiDAR_flatten)

        # GCN层 1 转化为超像素 x_flat 乘以 列归一化Q
        H_HSI = superpixels_flatten_HSI / norm_col_Q.t().to(device)
        # print(H_HSI.shape)
        H_LiDAR = superpixels_flatten_LiDAR / norm_col_Q.t().to(device)
        # print(H_LiDAR.shape)
        return H_HSI, H_LiDAR
   
    for curr_seed in Seed_List:     # Seed_List为随机种子点，curr_seed从1到5，这里的作用是每次生成一样的随机数
        # step2:随机百分比的数据作为训练样本。方式：给出训练数据与测试数据的GT
        random.seed(curr_seed)      # 当seed()没有参数时，每次生成的随机数是不一样的，而当seed()有参数时，每次生成的随机数是一样的
        gt_reshape = np.reshape(gt, [-1])
        # gt_train_reshape = np.reshape(train_gt, [-1])
        train_rand_idx = []
        val_rand_idx = []
        if samples_type == 'ratio':     # 取一定比例训练
            for i in range(class_count):    # i从0跑到 class_count-1
                idx = np.where(gt_reshape == i + 1)[-1]
                samplesCount = len(idx)
                rand_list = [i for i in range(samplesCount)]  # 用于随机的列表
                rand_idx = random.sample(rand_list,
                                         np.ceil(samplesCount * train_ratio).astype('int32'))  # 随机数数量 四舍五入(改为上取整)
                rand_real_idx_per_class = idx[rand_idx]
                train_rand_idx.append(rand_real_idx_per_class)
            train_rand_idx = np.array(train_rand_idx)
            train_data_index = []
            for c in range(train_rand_idx.shape[0]):
                a = train_rand_idx[c]
                for j in range(a.shape[0]):
                    train_data_index.append(a[j])
            train_data_index = np.array(train_data_index)
            
            ##将测试集（所有样本，包括训练样本）也转化为特定形式
            train_data_index = set(train_data_index)
            all_data_index = [i for i in range(len(gt_reshape))]
            all_data_index = set(all_data_index)
            
            # 背景像元的标签
            background_idx = np.where(gt_reshape == 0)[-1]
            background_idx = set(background_idx)
            test_data_index = all_data_index - train_data_index - background_idx
            
            # 从测试集中随机选取部分样本作为验证集
            val_data_count = int(val_ratio * (len(test_data_index) + len(train_data_index)))  # 验证集数量
            val_data_index = random.sample(test_data_index, val_data_count)
            val_data_index = set(val_data_index)
            test_data_index = test_data_index - val_data_index  # 由于验证集为从测试集分裂出，所以测试集应减去验证集
            
            # 将训练集 验证集 测试集 整理
            test_data_index = list(test_data_index)
            train_data_index = list(train_data_index)
            val_data_index = list(val_data_index)

        if samples_type == 'fixed':  # 固定的训练集
            gt_train_reshape = np.reshape(train_gt, [-1])
            for i in range(class_count):  # i从0跑到 class_count-1
                idx = np.where(gt_train_reshape == i + 1)[-1]
                train_rand_idx.append(idx)
            train_rand_idx = np.array(train_rand_idx)
            train_data_index = []
            for c in range(train_rand_idx.shape[0]):
                a = train_rand_idx[c]
                for j in range(a.shape[0]):
                    train_data_index.append(a[j])
            train_data_index = np.array(train_data_index)

            ##将测试集（所有样本，包括训练样本）也转化为特定形式
            train_data_index = set(train_data_index)
            all_data_index = [i for i in range(len(gt_reshape))]
            all_data_index = set(all_data_index)

            # 背景像元的标签
            background_idx = np.where(gt_reshape == 0)[-1]
            background_idx = set(background_idx)
            # test_data_index = all_data_index - train_data_index - background_idx
            test_data_index = all_data_index - background_idx

            # 从测试集中随机选取部分样本作为验证集
            val_data_count = int(val_ratio * (len(test_data_index) + len(train_data_index)))  # 验证集数量
            val_data_index = random.sample(test_data_index, val_data_count)
            val_data_index = set(val_data_index)
            test_data_index = test_data_index - val_data_index  # 由于验证集为从测试集分裂出，所以测试集应减去验证集

            # 将训练集 验证集 测试集 整理
            test_data_index = list(test_data_index)
            train_data_index = list(train_data_index)
            val_data_index = list(val_data_index)

        if samples_type == 'same_num':      # 取固定数量训练
            for i in range(class_count):
                idx = np.where(gt_reshape == i + 1)[-1]
                samplesCount = len(idx)
                real_train_samples_per_class = train_samples_per_class
                rand_list = [i for i in range(samplesCount)]  # 用于随机的列表
                if real_train_samples_per_class > samplesCount:
                    real_train_samples_per_class = samplesCount
                rand_idx = random.sample(rand_list,
                                         real_train_samples_per_class)  # 随机数数量 四舍五入(改为上取整)
                rand_real_idx_per_class_train = idx[rand_idx[0:real_train_samples_per_class]]
                train_rand_idx.append(rand_real_idx_per_class_train)
            train_rand_idx = np.array(train_rand_idx)
            val_rand_idx = np.array(val_rand_idx)
            train_data_index = []
            for c in range(train_rand_idx.shape[0]):
                a = train_rand_idx[c]
                for j in range(a.shape[0]):
                    train_data_index.append(a[j])
            train_data_index = np.array(train_data_index)
            
            train_data_index = set(train_data_index)
            all_data_index = [i for i in range(len(gt_reshape))]
            all_data_index = set(all_data_index)
            
            # 背景像元的标签
            background_idx = np.where(gt_reshape == 0)[-1]
            background_idx = set(background_idx)
            # test_data_index = all_data_index - train_data_index - background_idx
            test_data_index = all_data_index - background_idx
            
            # 从测试集中随机选取部分样本作为验证集
            val_data_count = int(val_samples)  # 验证集数量
            val_data_index = random.sample(test_data_index, val_data_count)
            val_data_index = set(val_data_index)
            
            test_data_index = test_data_index - val_data_index
            # 将训练集 验证集 测试集 整理
            test_data_index = list(test_data_index)
            train_data_index = list(train_data_index)
            val_data_index = list(val_data_index)
        
        # 获取训练样本的标签图
        train_samples_gt = np.zeros(gt_reshape.shape)
        for i in range(len(train_data_index)):
            train_samples_gt[train_data_index[i]] = gt_reshape[train_data_index[i]]
            pass
        trainlabel = np.reshape(train_samples_gt, [height, width])
        sio.savemat("show_image\\" + "trainlabel_" + samples_type, {'trainlabel_' + samples_type: trainlabel})

        # 获取测试样本的标签图
        test_samples_gt = np.zeros(gt_reshape.shape)
        for i in range(len(test_data_index)):
            test_samples_gt[test_data_index[i]] = gt_reshape[test_data_index[i]]
            pass
        
        Test_GT = np.reshape(test_samples_gt, [m, n])  # 测试样本图
        
        # 获取验证集样本的标签图
        val_samples_gt = np.zeros(gt_reshape.shape)
        for i in range(len(val_data_index)):
            val_samples_gt[val_data_index[i]] = gt_reshape[val_data_index[i]]
            pass

        train_samples_gt = np.reshape(train_samples_gt, [height, width])
        test_samples_gt = np.reshape(test_samples_gt, [height, width])
        val_samples_gt = np.reshape(val_samples_gt, [height, width])

        train_samples_gt_onehot = utils.GT_To_One_Hot(train_samples_gt, class_count)
        test_samples_gt_onehot = utils.GT_To_One_Hot(test_samples_gt, class_count)
        val_samples_gt_onehot = utils.GT_To_One_Hot(val_samples_gt, class_count)

        train_samples_gt_onehot = np.reshape(train_samples_gt_onehot, [-1, class_count]).astype(int)
        test_samples_gt_onehot = np.reshape(test_samples_gt_onehot, [-1, class_count]).astype(int)
        val_samples_gt_onehot = np.reshape(val_samples_gt_onehot, [-1, class_count]).astype(int)

        ############制作训练数据和测试数据的gt掩膜.根据GT将带有标签的像元设置为全1向量##############
        # 训练集
        train_label_mask = np.zeros([m * n, class_count])
        temp_ones = np.ones([class_count])
        train_samples_gt = np.reshape(train_samples_gt, [m * n])
        for i in range(m * n):
            if train_samples_gt[i] != 0:
                train_label_mask[i] = temp_ones
        train_label_mask = np.reshape(train_label_mask, [m* n, class_count])

        # 测试集
        test_label_mask = np.zeros([m * n, class_count])
        temp_ones = np.ones([class_count])
        test_samples_gt = np.reshape(test_samples_gt, [m * n])
        for i in range(m * n):
            if test_samples_gt[i] != 0:
                test_label_mask[i] = temp_ones
        test_label_mask = np.reshape(test_label_mask, [m * n, class_count])

        # 验证集
        val_label_mask = np.zeros([m * n, class_count])
        temp_ones = np.ones([class_count])
        val_samples_gt = np.reshape(val_samples_gt, [m * n])
        for i in range(m * n):
            if val_samples_gt[i] != 0:
                val_label_mask[i] = temp_ones
        val_label_mask = np.reshape(val_label_mask, [m * n, class_count])

        # 只用HSI做超像素
        # ls = LDA_SLIC.LDA_SLIC(data_HSI, np.reshape(train_samples_gt, [height, width]), class_count-1)
        # HSI+LiDAR进行超像素
        data_all = np.concatenate((data_HSI, data_LiDAR), axis=2)
        ls = LDA_SLIC.LDA_SLIC(data_all, np.reshape(train_samples_gt, [height, width]), class_count - 1)

        tic0 = time.time()
        Q, S, A, Seg = ls.simple_superpixel(scale=superpixel_scale)
        # 存一下超像素的图，MDGCN用
        sio.savemat("superpixel\\" + "useful_sp_lab", {'useful_sp_lab': Seg})
        toc0 = time.time()
        LDA_SLIC_Time = toc0-tic0
        # np.save(dataset_name+'Seg',Seg)
        print("LDA-SLIC costs time: {}".format(LDA_SLIC_Time))

        # index_pixel_dict = {}  # 字典元素，用来存各个超像素块中的像素index
        # pixel_number_dict = {}  # 字典元素，用来存各个超像素块中分别包含多少个像素
        # for i in range(Q.shape[1]):  # i 为超像素区域的个数
        #     index_pixel_dict[i] = np.argwhere(Q[:, i].reshape(1, Q.shape[0]))[:, 1]  # 第m个超像素块中有哪些像素
        # for i in range(Q.shape[1]):
        #     pixel_number_dict[i] = len(index_pixel_dict[i])

        Q = torch.from_numpy(Q).to(device)
        A = torch.from_numpy(A).to(device)

        # 转到GPU
        train_samples_gt = torch.from_numpy(train_samples_gt.astype(np.float32)).to(device)
        test_samples_gt = torch.from_numpy(test_samples_gt.astype(np.float32)).to(device)
        val_samples_gt = torch.from_numpy(val_samples_gt.astype(np.float32)).to(device)
        # 转到GPU
        train_samples_gt_onehot = torch.from_numpy(train_samples_gt_onehot.astype(np.float32)).to(device)
        test_samples_gt_onehot = torch.from_numpy(test_samples_gt_onehot.astype(np.float32)).to(device)
        val_samples_gt_onehot = torch.from_numpy(val_samples_gt_onehot.astype(np.float32)).to(device)
        # 转到GPU
        train_label_mask = torch.from_numpy(train_label_mask.astype(np.float32)).to(device)
        test_label_mask = torch.from_numpy(test_label_mask.astype(np.float32)).to(device)
        val_label_mask = torch.from_numpy(val_label_mask.astype(np.float32)).to(device)

        net_input_HSI = np.array(data_HSI, np.float32)
        net_input_HSI = torch.from_numpy(net_input_HSI.astype(np.float32)).to(device)
        net_input_LiDAR = np.array(data_LiDAR, np.float32)
        net_input_LiDAR = torch.from_numpy(net_input_LiDAR.astype(np.float32)).to(device)

        [h, w, c] = net_input_HSI.shape


        x_HSI_flatten = net_input_HSI.reshape([h * w, -1])
        x_LiDAR_flatten = net_input_LiDAR.reshape([h * w, -1])

        # 不做交叉，直接做
        H_HSI, H_LiDAR = nocross(Q)
        # 分别计算HSI和LIDAR的初始k个邻居的图结构（取与当前超像素节点相近的k个点认为其有连接）
        k = 5
        A_k_HSI, A_k_LiDAR = utils.get_A_k(H_HSI, H_LiDAR, k)

        # 计算HSI和LiDAR的r-hop邻接矩阵
        r = 2
        A_r_HSI = utils.get_A_r(A_k_HSI, r)
        A_r_LiDAR = utils.get_A_r(A_k_LiDAR, r)

        # sio.savemat("A_k_HSI", {'A_k_HSI': A_k_HSI.cpu().numpy()})
        # sio.savemat("A_k_LiDAR", {'A_k_LiDAR': A_k_LiDAR.cpu().numpy()})
        # sio.savemat("A_r_HSI", {'A_r_HSI': A_r_HSI.cpu().numpy()})
        # sio.savemat("A_r_LiDAR", {'A_r_LiDAR': A_r_LiDAR.cpu().numpy()})

        net = GraphOnly.GraphOnly(height, width, bands, class_count, Q, A, A_k_HSI, A_k_LiDAR, model='smoothed')

        print("parameters", net.parameters(), len(list(net.parameters())))

        net.to(device)

        zeros = torch.zeros([m * n]).to(device).float()
        def evaluate_performance(network_output, train_samples_gt, train_samples_gt_onehot, require_AA_KPP=False,printFlag=True):
            if False==require_AA_KPP:
                with torch.no_grad():
                    available_label_idx = (train_samples_gt != 0).float()   # 有效标签的坐标,用于排除背景
                    available_label_count = available_label_idx.sum()   # 有效标签的个数
                    correct_prediction = torch.where(torch.argmax(network_output, 1) == torch.argmax(train_samples_gt_onehot, 1), available_label_idx, zeros).sum()
                    OA = correct_prediction.cpu()/available_label_count
                    
                    return OA
            else:
                with torch.no_grad():
                    #计算OA
                    available_label_idx = (train_samples_gt != 0).float()   # 有效标签的坐标,用于排除背景
                    available_label_count = available_label_idx.sum()   # 有效标签的个数
                    correct_prediction = torch.where(torch.argmax(network_output, 1) ==torch.argmax(train_samples_gt_onehot, 1), available_label_idx, zeros).sum()
                    OA = correct_prediction.cpu()/available_label_count
                    OA = OA.cpu().numpy()
                    
                    # 计算AA
                    zero_vector = np.zeros([class_count])
                    output_data = network_output.cpu().numpy()
                    train_samples_gt =  train_samples_gt.cpu().numpy()
                    train_samples_gt_onehot = train_samples_gt_onehot.cpu().numpy()
                    
                    output_data = np.reshape(output_data, [m * n, class_count])
                    idx = np.argmax(output_data, axis=-1)
                    for z in range(output_data.shape[0]):
                        if ~(zero_vector == output_data[z]).all():
                            idx[z] += 1
                    # idx = idx + train_samples_gt
                    count_perclass = np.zeros([class_count])
                    correct_perclass = np.zeros([class_count])
                    for x in range(len(train_samples_gt)):
                        if train_samples_gt[x] != 0:
                            count_perclass[int(train_samples_gt[x] - 1)] += 1
                            if train_samples_gt[x] == idx[x]:
                                correct_perclass[int(train_samples_gt[x] - 1)] += 1
                    test_AC_list = correct_perclass / count_perclass
                    test_AA = np.average(test_AC_list)

                    # 计算KPP
                    test_pre_label_list = []
                    test_real_label_list = []
                    output_data = np.reshape(output_data, [m * n, class_count])
                    idx = np.argmax(output_data, axis=-1)
                    idx = np.reshape(idx, [m, n])
                    for ii in range(m):
                        for jj in range(n):
                            if Test_GT[ii][jj] != 0:
                                test_pre_label_list.append(idx[ii][jj] + 1)
                                test_real_label_list.append(Test_GT[ii][jj])
                    test_pre_label_list = np.array(test_pre_label_list)
                    test_real_label_list = np.array(test_real_label_list)
                    kappa = metrics.cohen_kappa_score(test_pre_label_list.astype(np.int16),
                                                      test_real_label_list.astype(np.int16))
                    test_kpp = kappa

                    # 输出
                    if printFlag:
                        print("test OA=", OA, "AA=", test_AA, 'kpp=', test_kpp)
                        print('acc per class:')
                        print(test_AC_list)

                    OA_ALL.append(OA)
                    AA_ALL.append(test_AA)
                    KPP_ALL.append(test_kpp)
                    AVG_ALL.append(test_AC_list)
                    
                    # 保存数据信息
                    f = open('./results/' + dataset_name + '_results.txt', 'a+')
                    str_results = '\n======================' \
                                  + " learning rate=" + str(learning_rate) \
                                  + " epochs=" + str(max_epoch) \
                                  + " train ratio=" + str(train_ratio) \
                                  + " val ratio=" + str(val_ratio) \
                                  + " ======================" \
                                  + "\nOA=" + str(OA) \
                                  + "\nAA=" + str(test_AA) \
                                  + '\nkpp=' + str(test_kpp) \
                                  + '\nacc per class:' + str(test_AC_list) + "\n"
                                  # + '\ntrain time:' + str(time_train_end - time_train_start) \
                                  # + '\ntest time:' + str(time_test_end - time_test_start) \
                    f.write(str_results)
                    f.close()
                    return OA
        
        # #训练
        # optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)    # weight_decay=0.0001
        # best_loss = 99999
        # net.train()
        # tic1 = time.perf_counter()
        # for i in range(max_epoch+1):
        #     optimizer.zero_grad()  # zero the gradient buffers
        #     output, HSI_result, LiDAR_result, final_feature = net(H_HSI, H_LiDAR)
        #     loss_entropy = utils.entropy_loss(output, train_samples_gt_onehot, train_label_mask)
        #     # loss_graph = utils.graph_loss(Q, HSI_result, LiDAR_result, A_r_HSI, A_r_LiDAR)
        #     # loss = 1 * loss_entropy + 0 * loss_graph
        #     loss = loss_entropy
        #     # 开启检测功能
        #     # with torch.autograd.detect_anomaly():
        #     #     loss.backward(retain_graph=False)
        #     loss.backward(retain_graph=False)
        #     optimizer.step()  # Does the update
        #     if i % 10 == 0:
        #         with torch.no_grad():
        #             net.eval()
        #             output, HSI_result, LIDAR_result, final_feature = net(H_HSI, H_LiDAR)
        #             # sio.savemat(str(i) + "output", {'output': output.reshape([height, width, -1]).cpu().numpy()})
        #             trainloss = utils.entropy_loss(output, train_samples_gt_onehot, train_label_mask)
        #             trainOA = evaluate_performance(output, train_samples_gt, train_samples_gt_onehot)
        #             valloss = utils.entropy_loss(output, val_samples_gt_onehot, val_label_mask)
        #             valOA = evaluate_performance(output, val_samples_gt, val_samples_gt_onehot)
        #             print("{}\ttrain loss={}\t train OA={} val loss={}\t val OA={}".format(str(i + 1), trainloss, trainOA, valloss, valOA))
        #             # print('交叉熵损失：', loss_entropy, '图结构损失:', loss_graph)
        #
        #             ## 正常运行
        #             if valloss < best_loss:
        #                 best_loss = valloss
        #                 # torch.save(net.state_dict(), "model/best_model.pt")
        #                 torch.save(net.state_dict(), "model/Trento/best_model_" + str(i) + ".pt")
        #                 print('save model...')
        #             ## 做超参数测试
        #             # torch.save(net.state_dict(), "model/best_model.pt")
        #             # print('save model...')
        #         torch.cuda.empty_cache()
        #         net.train()
        #
        #     #  %%% 做收敛性分析 %%%
        #     trainloss_result[i] = loss.detach().cpu().numpy()
        #
        # toc1 = time.perf_counter()
        # print("\n\n====================training done. starting evaluation...========================\n")
        # training_time = toc1 - tic1 + LDA_SLIC_Time # 分割耗时需要算进去
        # Train_Time_ALL.append(training_time)
        
        torch.cuda.empty_cache()
        with torch.no_grad():
            net.load_state_dict(torch.load("model/Houston/best_model_140.pt"))
            net.eval()

            # ########## 计算模型复杂度 ##########
            temp_HSI = torch.randn(2422, 144).to(device)
            temp_LiDAR = torch.randn(2422, 1).to(device)
            flops, params = profile(net, inputs=(temp_HSI, temp_LiDAR))
            print(f"FLOPs: {flops}, Parameters: {params}")

            # ########## 继续测试 ##########
            tic2 = time.perf_counter()
            output, HSI_result, LIDAR_result, final_feature = net(H_HSI, H_LiDAR)
            toc2 = time.perf_counter()
            testloss = utils.entropy_loss(output, test_samples_gt_onehot, test_label_mask)
            testOA = evaluate_performance(output, test_samples_gt, test_samples_gt_onehot, require_AA_KPP=True, printFlag=False)
            print("{}\ttest loss={}\t test OA={}".format(str(i + 1), testloss, testOA))
            #计算
            classification_map = torch.argmax(output, 1).reshape([height, width]).cpu()+1
            utils.Draw_Classification_Map(classification_map, "results\\"+dataset_name+str(testOA))
            testing_time = toc2 - tic2 + LDA_SLIC_Time #分割耗时需要算进去
            Test_Time_ALL.append(testing_time)
            ## Saving data
            # sio.savemat(dataset_name+"softmax",{'softmax':output.reshape([height,width,-1]).cpu().numpy()})
            # np.save(dataset_name+"A_1", A_1.cpu().numpy())
            # np.save(dataset_name+"A_2", A_2.cpu().numpy())
            # sio.savemat("show_image\\" + "final_feature", {'final_feature': final_feature.cpu().numpy()})
            # HSI_result = torch.matmul(Q, HSI_result)
            # sio.savemat("show_image\\" + "HSI_result", {'HSI_result': HSI_result.cpu().numpy()})
            sio.savemat("./show_image/" + "result.mat", {'result': classification_map.cpu().numpy()})

    torch.cuda.empty_cache()
    del net
        
    OA_ALL = np.array(OA_ALL)
    AA_ALL = np.array(AA_ALL)
    KPP_ALL = np.array(KPP_ALL)
    AVG_ALL = np.array(AVG_ALL)
    Train_Time_ALL = np.array(Train_Time_ALL)
    Test_Time_ALL = np.array(Test_Time_ALL)

    if samples_type != 'fixed':
        print("\ntrain_ratio={}".format(curr_train_ratio),
            "\n==============================================================================")
    else:
        print("\ntrain sample is fixed",
            "\n==============================================================================")
    print('OA=', np.mean(OA_ALL), '+-', np.std(OA_ALL))
    print('AA=', np.mean(AA_ALL), '+-', np.std(AA_ALL))
    print('Kpp=', np.mean(KPP_ALL), '+-', np.std(KPP_ALL))
    print('AVG=', np.mean(AVG_ALL, 0), '+-', np.std(AVG_ALL, 0))
    print("Average training time:{}".format(np.mean(Train_Time_ALL)))
    print("Average testing time:{}".format(np.mean(Test_Time_ALL)))
    
    # 保存数据信息
    f = open('./results/' + dataset_name + '_results.txt', 'a+')
    str_results = '\n\n************************************************' \
    + "\ntrain_ratio={}".format(curr_train_ratio) \
    + '\nOA=' + str(np.mean(OA_ALL))+ '+-'+ str(np.std(OA_ALL)) \
    + '\nAA=' + str(np.mean(AA_ALL))+ '+-'+ str(np.std(AA_ALL)) \
    + '\nKpp=' + str(np.mean(KPP_ALL))+ '+-'+ str(np.std(KPP_ALL)) \
    + '\nAVG=' + str(np.mean(AVG_ALL,0))+ '+-'+ str(np.std(AVG_ALL, 0)) \
    + "\nAverage training time:{}".format(np.mean(Train_Time_ALL)) \
    + "\nAverage testing time:{}".format(np.mean(Test_Time_ALL))
    f.write(str_results)
    f.close()

    sio.savemat("./show_image/" + "trainloss_result.mat", {'trainloss_result': trainloss_result})
