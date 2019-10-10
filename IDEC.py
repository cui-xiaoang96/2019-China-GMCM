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

# keras moudle
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.losses import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
from keras.layers import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras import backend as K


# IDEC网络模型
def IDEC_1D(input_size = (200, 1), verbose = 0, activation_funcation = 'relu', conv_size = 3) :
    inputs = Input(input_size)
    print(inputs.shape)
    
    # 第一层
    conv1 = Conv1D(16, conv_size, padding='same', kernel_initializer='he_normal', activation = activation_funcation)(inputs)
    print(conv1.shape)
    conv1 = Conv1D(16, conv_size, padding='same', kernel_initializer='he_normal', activation = activation_funcation)(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling1D(pool_size=(2))(conv1)
    print(pool1.shape)
    
    # 第二层
    conv2 = Conv1D(32, conv_size, padding='same', kernel_initializer='he_normal', activation = activation_funcation)(pool1)
    conv2 = Conv1D(32, conv_size, padding='same', kernel_initializer='he_normal', activation = activation_funcation)(conv2)
    drop2 = BatchNormalization()(conv2)
    pool2 = MaxPooling1D(pool_size=(2))(drop2)
    print(pool2.shape)
    
    # 第三层
    conv3 = Conv1D(16, conv_size, padding='same', kernel_initializer='he_normal', activation = activation_funcation)(UpSampling1D(size = 2)(pool2))
    conv3 = Conv1D(16, conv_size, padding='same', kernel_initializer='he_normal', activation = activation_funcation)(conv3)
    conv3 = Conv1D(16, conv_size, padding='same', kernel_initializer='he_normal', activation = activation_funcation)(conv3)
    conv3 = BatchNormalization()(conv3)
    print(conv3.shape)
    
    # 第四层
    dens4 = Flatten()(conv3)
    dens4 = Dense(units=100, activation='relu')(dens4)
    dens4 = Dropout(0.2)(dens4)
    print(dens4.shape)
    
    # 第一层输出
    outp1 = Dense(units=10, activation='relu', name='output')(dens4)
    print(outp1.shape)
    
    # 第五层
    dens5 = Dense(units=100, activation='relu')(outp1)
    print(dens5.shape)
    dens5 = Reshape((100, 1))(dens5)
    print(dens5.shape)
    
    # 第六层
    conv6 = Conv1D(16, conv_size, padding='same', kernel_initializer='he_normal', activation = activation_funcation)(dens5)
    conv6 = Conv1D(16, conv_size, padding='same', kernel_initializer='he_normal', activation = activation_funcation)(conv6)
    conv6 = BatchNormalization()(conv6)
    conv6 = MaxPooling1D(pool_size=(2))(conv6)
    print(conv6.shape)
    
    # 第七层
    conv7 = Conv1D(32, conv_size, padding='same', kernel_initializer='he_normal', activation = activation_funcation)(conv6)
    conv7 = Conv1D(32, conv_size, padding='same', kernel_initializer='he_normal', activation = activation_funcation)(conv7)
    conv7 = BatchNormalization()(conv7)
    conv7 = MaxPooling1D(pool_size=(2))(conv7)
    print(conv7.shape)
    
    # 第七层
    conv8 = Conv1D(16, conv_size, padding='same', kernel_initializer='he_normal', activation = activation_funcation)(UpSampling1D(size = 2)(conv7))
    conv8 = Conv1D(16, conv_size, padding='same', kernel_initializer='he_normal', activation = activation_funcation)(conv8)
    conv8 = Conv1D(16, conv_size, padding='same', kernel_initializer='he_normal', activation = activation_funcation)(conv8)
    conv8 = BatchNormalization()(conv8)
    print(conv8.shape)
    
    # 第二层输出
    outp2 = Flatten()(conv8)
    outp2 = Dense(units=128, activation='relu')(outp2)
    outp2 = Dropout(0.2)(outp2)
    outp2 = Dense(units=input_size[0], activation='relu')(outp2)
    print(outp2.shape)
    
    model = Model(input = inputs, output = outp2)
    model.compile(optimizer = Adam(lr = 1e-4), loss='mse')
    return model

# 建立模型
model = IDEC_1D(input_size=(600, 1), verbose=1, activation_funcation = 'relu', conv_size = 3)

# 加载数据
data1 = np.load('./data/sport_data/data1.npy')
data2 = np.load('./data/sport_data/data2.npy')
data3 = np.load('./data/sport_data/data3.npy')

# 整理数据
data11 = np.zeros(data1.shape)
data22 = np.zeros(data2.shape)
data33 = np.zeros(data3.shape)
for i in range(data1.shape[0]):
    data11[i, :] = data1[i, :][::-1]
for i in range(data2.shape[0]):
    data22[i, :] = data2[i, :][::-1]
for i in range(data3.shape[0]):
    data33[i, :] = data3[i, :][::-1]
xx = np.vstack((data1, data2, data3))
yy = np.vstack((data11, data22, data33))
x = np.zeros([xx.shape[0], xx.shape[1], 1])
for i in range(x.shape[0]):
    x[i, :, 0] = xx[i, :]
y = yy

# 模型训练
history = model.fit(x, y, epochs=1000, shuffle=True, verbose=1)

# 输出属性
prediction = model.predict(x)
output = Model(inputs=model.input, outputs=model.get_layer('output').output)
output_data = output.predict(x)

plt.figure(1)
plt.plot(y[0, :])
plt.plot(prediction[0, :])


















































