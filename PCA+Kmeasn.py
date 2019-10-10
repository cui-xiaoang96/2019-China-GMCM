# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 15:03:05 2019

@author: MSI
"""



# basic moudle
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.image as mpimg
import math as mt

# seaborn moudle
import seaborn as sns

# os moudle
import os

# sklearn moudle
from sklearn.metrics import r2_score
from sklearn.preprocessing import *
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.cluster import KMeans
from sklearn.cluster import spectral_clustering
from sklearn.cluster import DBSCAN
from sklearn import cluster, datasets, mixture
from sklearn.cluster import Birch
from sklearn.cluster.bicluster import SpectralCoclustering
from sklearn.cluster.bicluster import SpectralBiclustering
from sklearn import metrics
from sklearn.metrics import pairwise_distances
from sklearn.metrics import davies_bouldin_score



# 数据读取
a1_name = './data/raw_data/a1.csv'
a2_name = './data/raw_data/a2.csv'
a3_name = './data/raw_data/a3.csv'
a1_all = np.array(pd.read_csv(a1_name, encoding='gbk', header=None))
a2_all = np.array(pd.read_csv(a2_name, encoding='gbk', header=None))
a3_all = np.array(pd.read_csv(a3_name, encoding='gbk', header=None))
x1 = np.load('./data/sport_data/xuan_rank1.npy')
x2 = np.load('./data/sport_data/xuan_rank2.npy')
x3 = np.load('./data/sport_data/xuan_rank3.npy')

# 运动片段截取
def cut_sport(x1, a1_all):
    a1 = np.zeros([x1.shape[0], a1_all.shape[1]])
    for i in range(x1.shape[0]):
        a1[i, :] = a1_all[x1[i, 0], :]
    return a1
a11 = cut_sport(x1, a1_all)
a22 = cut_sport(x2, a2_all)
a33 = cut_sport(x3, a3_all)
a_all = np.vstack((a11, a22, a33))


# 网络输出属性
a_net = np.load('./data/net_data/a.npy')
a = np.hstack((a_net, a_all))


# PCA降维
pca = PCA()   #保留所有成分
pca.fit(a)
pca_cp = pca.components_ #返回模型的各个特征向量
pca_vr = pca.explained_variance_ratio_ #返回各个成分各自的方差百分比(也称贡献率）
print(pca_vr)
# =============================================================================
# pca = PCA(n_components=3)   #保留所有成分
# pca.fit(a)
# pca_a = pca.transform(a)
# tsne_a = pca_a
# =============================================================================

# T-SNE降维
tsne = TSNE(n_components=3, verbose=1, n_iter=2000)
tsne.fit_transform(a)
tsne_a = tsne.embedding_


# 数据预处理
tsne_a = scale(tsne_a)


# 数据聚类
ncluster = 6
# kmean比较合适
km = KMeans(n_clusters=ncluster, verbose=0, init='random', n_init=100)
labels_kmean = km.fit_predict(tsne_a)
center = km.cluster_centers_
# =============================================================================
# # SpectralCoclustering可行
# model = SpectralCoclustering(n_clusters=ncluster, random_state=0)
# model.fit(tsne_a)
# labels_scc = model.row_labels_
# =============================================================================


# 画图
fig = plt.figure(1, figsize=(15, 15))
ax = Axes3D(fig)
ax.scatter(tsne_a[:, 0], tsne_a[:, 1], tsne_a[:, 2], c=labels_kmean)
plt.show()
print(max(labels_kmean))
print(min(labels_kmean))

# =============================================================================
# fig = plt.figure(2, figsize=(15, 15))
# ax = Axes3D(fig)
# ax.scatter(tsne_a[:, 0], tsne_a[:, 1], tsne_a[:, 2], c=labels_scc)
# plt.show()
# print(max(labels_scc))
# print(min(labels_scc))
# =============================================================================


# 寻找代表
def cent(a, b, j1=0.005, j2=0.05):
    k = 0
    k_list = []
    print('################')
    for i in range(a.shape[0]):
        if (abs(a[i, 0] - b[0]) < j1) & (abs(a[i, 1] - b[1]) < j2):
            k = k + 1
            print(a[i, 0] - b[0], a[i, 1] - b[1])
            print('0--', i, k)
            k_list.append(i)
    return np.array(k_list), k

c = np.ones([ncluster, 200]) * 9999
c_num = np.zeros(ncluster)
jj = np.array([[0.05, 0.05], [0.05, 0.05], [0.08, 0.08], [0.08, 0.08], [0.08, 0.08], [0.08, 0.08]])
for i in range(ncluster):
    c1, k1 = cent(tsne_a, center[i, :], j1=jj[i, 0], j2=jj[i, 1])
    for j in range(len(c1)):
        c[i, j] = c1[j]
        c_num[i] = k1


# 运动片段拼接
data1 = np.load('./data/sport_data/data_batch1.npy')
data2 = np.load('./data/sport_data/data_batch2.npy')
data3 = np.load('./data/sport_data/data_batch3.npy')

def xsp(num, data1=data1, data2=data2, data3=data3):
    if (num <= (data1.shape[0]-1)):
        p1 = data1[num]
    if (num > (data1.shape[0]-1)) & (num <= (data1.shape[0]+data2.shape[0]-1)):
        p1 = data2[num - data1.shape[0]]
    if (num > (data1.shape[0]+data2.shape[0]-1)):
        p1 = data3[num - data1.shape[0] - data2.shape[0]]
    print(len(p1))
    return p1
# 片段选取
p = []
for i in range(ncluster):
    print('#########  第' + str(i+1) +'类簇, 有' + str(int(c_num[i])) +  '个代表点#########')
    for j in range(int(c_num[i])):
        p.append(xsp(int(c[i, j])))


# 评价函数
def valuate(xx):
    print('平均速度: ', np.mean(xx), '平均速度差: ', np.mean(xx)-24.8584)
    print('速度标准差: ', np.std(xx, ddof=1), '速度标准差: ', np.std(xx, ddof=1)-14.2750)
    k = 0
    for i in range(len(xx)):
        if xx[i] < 0.0001:
            k = k+1
    print('怠速比: ', k/len(xx), '怠速比差: ', k/len(xx)-0.2332)
    return [abs((np.mean(xx)-24.8584)/24.8584), abs((np.std(xx, ddof=1)-14.2750)/14.2750), abs((k/len(xx)-0.2332)/0.2332)]

# 片段组合
val_cha = []
val_cha_x = []
length = []
pc = []
x = []
for i1 in range(int(c_num[0])):
    num1 = 0
    
    for i2 in range(int(c_num[1])):
        num2 = int(c_num[0])
        
        for i3 in range(int(c_num[2])):
            num3 = num2 + int(c_num[1])
            
            for i4 in range(int(c_num[3])):
                num4 = num3 + int(c_num[2])
                
                for i5 in range(int(c_num[4])):
                    num5 = num4 + int(c_num[3])
                    
                    for i6 in range(int(c_num[5])):
                        num6 = num5 + int(c_num[4])
                        
                    xx = np.hstack((p[i1+num1], p[i2+num2], p[i3+num3], p[i4+num4], p[i5+num5], p[i6+num6]))
                    print('################')
                    cha = valuate(xx)
                    if (len(xx) >= 1200) & (len(xx) <= 1300):
                        x.append(xx)
                        val_cha_x.append(cha)
                    val_cha.append(cha)
                    print('组合片段长度为:', len(xx), '排列组合为: '+ str([i1, i2, i3, i4, i5, i6]))
                    length.append(len(xx))
                    pc.append([i1, i2, i3, i4, i5, i6])
                    print('################')

val_cha = np.array(val_cha)
val_cha_x = np.array(val_cha_x)
length = np.array(length)

plt.figure(2, figsize=(30, 5))
plt.plot(val_cha)
plt.figure(3, figsize=(30, 5))
plt.plot(length)


plt.figure(4, figsize=(30, 5))
plt.plot(val_cha)
plt.plot(1200/length)


# =============================================================================
# np.save('./data/result_data/x.npy', x)
# np.save('./data/result_data/kmean_labels.npy', labels_kmean)
# np.save('./data/result_data/kmean_centers.npy', center)
# np.save('./data/result_data/representation_centers.npy', p)
# np.save('./data/result_data/representation_centers_num.npy', c_num)
# 
# a_net = pd.DataFrame(a_net)
# x = pd.DataFrame(x)
# pca_vr = pd.DataFrame(pca_vr)
# a_net.to_csv('./data/result_data/a_net.csv', index=None, header=None)
# x.to_csv('./data/result_data/x.csv', index=None, header=None)
# pca_vr.to_csv('./data/result_data/pca_vr.csv', index=None, header=None)
# =============================================================================



k = 0
plt.figure(4, figsize=(30, 5))
plt.plot(data1[k])












