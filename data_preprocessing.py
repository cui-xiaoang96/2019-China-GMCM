# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 15:03:05 2019

@author: MSI
"""



# basic moudle
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math as mt
import random
import time

# seaborn moudle
import seaborn as sns

# os moudle
import os

# sklearn moudle
from sklearn.metrics import r2_score
from sklearn import preprocessing



# 数据读取
data1 = pd.read_csv('./data/raw_data/data2.csv', encoding='gbk', header=None)
se1 = pd.read_csv('./data/raw_data/se2.csv', encoding='gbk', header=None)
data1 = np.array(data1)
se1 = np.array(se1)

# 运动片段截取
sport = []
mlen1 = []
mlen2 = []
mlen3 = []
for i in range(se1.shape[0]):
    batch = data1[int(se1[i, 0]): int(se1[i, 1]), 1]
    mlen1.append([i, len(batch)])
    if len(batch) <= 601.0:
        sport.append(batch)
        mlen2.append([i, len(batch)])
    else:
        mlen3.append([i, len(batch)])
mlen1 = np.array(mlen1)
mlen2 = np.array(mlen2)
mlen3 = np.array(mlen3)
np.save('./data/sport_data/data_batch2.npy', sport)

# 运动片段补全
data11 = np.zeros([len(sport), 600])
for i in range(len(sport)):
    data11[i, 0: len(sport[i])] = sport[i]

# 保存文件
data11_name = './data/sport_data/data2.npy'
mlen1_name = './data/sport_data/all_rank2.npy'
mlen2_name = './data/sport_data/xuan_rank2.npy'
mlen3_name = './data/sport_data/luo_rank2.npy'
np.save(data11_name, data11)
np.save(mlen1_name, mlen1)
np.save(mlen2_name, mlen2)
np.save(mlen3_name, mlen3)
























































































