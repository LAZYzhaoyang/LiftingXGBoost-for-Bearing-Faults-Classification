import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152

import tensorflow as tf
import numpy as np
import math
import glob
import matplotlib

import matplotlib.pyplot as plt
from matplotlib.pyplot import plot,savefig
from matplotlib.colors import ListedColormap

import xgboost as xgb
from xgboost import plot_importance
from xgboost import XGBClassifier

import sklearn as sk
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

from dataset import *
import mylib as ml2
from mylib import Standard_LiftNet, create_Standard_LiftNet, create_Standard_LiftNet_CWRU

from plotly.graph_objs import Scatter,Layout
import plotly
import plotly.offline as py
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from plotly.subplots import make_subplots
import pandas as pd
import plotly.figure_factory as ff

from keras.callbacks import TensorBoard

import seaborn as sea

matplotlib.use('Agg')

#setting offilne
plotly.offline.init_notebook_mode(connected=True)

#========================================================================================#
# Set the parameters
#------------------------default parameters-----------------------#
test_rate = 0.1
#lr=0.015
lr=0.02
momentum=0.8
decay=0.01
epochs = 1000

validation_split=0.1
steps_per_epoch=1
validation_steps=1
cutsize = 256
circle_num =1

snapshot = 100
bunch_steps = 450

label_name1 = ['NORMAL', 'IR', 'OR', 'BALL', 'JOINT']
label_name2 = ['NORMAL', 'BALL', 'IR', 'OR_3', 'OR_6', 'OR_12']
result_txt_list = []

head_of_name = './snapshot/Standard_LiftingNet_'
if not os.path.exists('snapshot'):
    os.mkdir('snapshot')

result_save_path = './result'
if not os.path.exists('result'):
    os.mkdir('result')

#------------------------adjustable parameters-----------------------#
whether_use_cwru_data = False

whether_expansion_test_data = 0
whether_expansion_train_data = 0
expansion_data_number = 200

train_noise_scales = 0
test_noise_scales = 0

whether_add_noise = 0

whether_append_expansion_train_data = 0
whether_append_expansion_test_data = 0

#artificial_feature_method: 1 is 19 features, 2 is 9 features
artificial_feature_method = 2

pca_parameters = 0.95

    
#-------------------------set read name-----------------------------#
class_num = 5
channel = 3
data_path = './data/our/'
input_shape = (640*circle_num,3)

if whether_use_cwru_data:
    head_of_name = head_of_name + 'CWRU_'
    class_num = 6
    data_path = './data/cwru/'
    cutsize = 1024
    channel = 2
    dataset_file_name = data_path + 'CWRU_dataset_' + str(cutsize) + '.npy'
    label_file_name = data_path + 'CWRU_label_' + str(cutsize) + '.npy'
    input_shape = (cutsize,channel)
    label_name = label_name2
    result_files_last_name = 'CWRU_data_with_' + str(train_noise_scales) + '_noise_train_model_and_test_in_'+ str(test_noise_scales)+'_noise_CWRU_data_test.png'
    result_txt_last_name = 'CWRU_data_with_' + str(train_noise_scales) + '_noise_train_model_and_test_in_'+ str(test_noise_scales)+'_noise_CWRU_data_test.txt'
else:
    head_of_name = head_of_name + 'Our_'
    data_path = './data/our/'
    dataset_file_name = data_path + 'dataset_' + str(circle_num) + '_' + str(cutsize) + '.npy'
    label_file_name = data_path + 'label_' + str(circle_num) + '_' + str(cutsize) + '.npy'
    label_name = label_name1
    result_files_last_name = 'our_'+ str(circle_num)+'_circles_data_with_' + str(train_noise_scales) + '_noise_trains_model_and_test_in_'+ str(test_noise_scales)+'_noise_our_data_test.png'
    result_txt_last_name = 'our_'+ str(circle_num)+'_circles_data_with_' + str(train_noise_scales) + '_noise_trains_model_and_test_in_'+ str(test_noise_scales)+'_noise_our_data_test.txt'

save_model_name = head_of_name + str(circle_num) + '_' + str(epochs)+ '_' + str(steps_per_epoch) + '_with_' + str(train_noise_scales) + '_noise.h5'
loss_map_name = head_of_name + str(circle_num) + '_' + str(epochs)+ '_' + str(steps_per_epoch) +'_with_' + str(train_noise_scales) + '_noise_loss.jpg'
acc_map_name = head_of_name + str(circle_num) + '_' + str(epochs)+ '_' + str(steps_per_epoch) +'_with_' + str(train_noise_scales) + '_noise_acc.jpg'

#========================================================================================#
#------------------------data processing---------------------------------#
# load data
dataset = np.load(dataset_file_name)
label = np.load(label_file_name)
print('dataset shape : ',dataset.shape)
print('label shape : ', label.shape)

x_number = dataset.shape[0]
x = dataset
y = label
print('x.shape: ',x.shape)
print('y.shape: ', y.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_rate, random_state=615)

print('x_train.shape: ', x_train.shape)
print('x_test.shape: ',x_test.shape)
print('y_train.shape: ', y_train.shape)
print('y_test.shape: ', y_test.shape)
x_train_number = x_train.shape[0]
x_test_number = x_test.shape[0]
print('x_train_number: ',x_train_number)
print('x_test_number: ',x_test_number)

bunch_size = math.floor(x_train_number / bunch_steps)
print('bunch_size: ', bunch_size)

x_train_truth_length = bunch_size * bunch_steps
x_train = x_train[:x_train_truth_length, :, :]
y_train = y_train[:x_train_truth_length,:]
print('x_train.shape: ', x_train.shape)
#------------------------expansion data parameters-----------------------#

if whether_expansion_train_data == 1:
    x_train, y_train = expansion_and_add_noise(x_train,y_train,exnumber=expansion_data_number, noise_scales=train_noise_scales,whether_link_original_data=whether_append_expansion_train_data,is_label_sort=0)
    print('data expansion')

if whether_expansion_test_data == 1:
    x_test, y_test = expansion_and_add_noise(x_test, y_test,exnumber=expansion_data_number,noise_scales=test_noise_scales,whether_link_original_data=whether_append_expansion_test_data,is_label_sort=0)

if whether_add_noise == 1:
    x_train = add_data_noise(x_train, train_noise_scales)
    x_test = add_data_noise(x_test, test_noise_scales)

x_train_number = x_train.shape[0]
print('x_train_number: ',x_train_number)
x_test_number = x_test.shape[0]
print('x_test_number: ',x_test_number)

#------------------------------artificial features extraction-----------------------------#
if artificial_feature_method == 1:
    artificial_feature_of_train_data = feature_extractor(x_train)
    artificial_feature_of_test_data = feature_extractor(x_test)
else:
    artificial_feature_of_train_data = feature_extractor2(x_train)
    artificial_feature_of_test_data = feature_extractor2(x_test)
#artificial_feature_data = artificial_feature_data.reshape(x_number,-1)[:,:select_feature_numbers]
print('artificial_feature_of_train_data.shape: ', artificial_feature_of_train_data.shape)
print('artificial_feature_of_test_data.shape: ', artificial_feature_of_test_data.shape)

artificial_feature_of_train_data = artificial_feature_of_train_data.reshape(x_train_number,-1)
artificial_feature_of_test_data = artificial_feature_of_test_data.reshape(x_test_number,-1)
print('artificial_feature_of_train_data.shape: ', artificial_feature_of_train_data.shape)
print('artificial_feature_of_test_data.shape: ', artificial_feature_of_test_data.shape)
#========================================================================================#
#--------------------------------Create LiftingNet-------------------------------------------#
#liftnet = create_LiftNet(class_num = class_num, channel = channel, circle_num = circle_num, input_shape=input_shape,lr=lr, momentum=momentum, decay=decay)
if whether_use_cwru_data:
    liftnet = create_Standard_LiftNet_CWRU(class_num = class_num, 
                                           channel = channel, 
                                           cut_size=cutsize, 
                                           input_shape=input_shape,
                                           lr=lr, momentum=momentum, 
                                           decay=decay)
else:
    liftnet = create_Standard_LiftNet(class_num = class_num, 
                                      channel = channel, 
                                      circle_num = circle_num, 
                                      input_shape=input_shape,
                                      lr=lr, momentum=momentum, 
                                      decay=decay)

#liftnet.summary()
"""
tbCallBack = TensorBoard(log_dir='./logs',  
                 histogram_freq=0,  # 按照何等频率（epoch）来计算直方图，0为不计算
#                  batch_size=32,     # 用多大量的数据计算直方图
                 write_graph=True,  # 是否存储网络结构图
                 write_grads=True, # 是否可视化梯度直方图
                 write_images=True,# 是否可视化参数
                 embeddings_freq=0, 
                 embeddings_layer_names=None, 
                 embeddings_metadata=None)
"""
#--------------------------------Train LiftingNet-------------------------------------------#
#history = liftnet.fit(x_train,y_train,validation_split=validation_split,epochs=epochs,batch_size=bunch_size,steps_per_epoch=steps_per_epoch,validation_steps=validation_steps,callbacks=[tbCallBack])
#history = liftnet.fit(x_train,y_train,validation_split=validation_split,epochs=epochs,batch_size=bunch_steps,steps_per_epoch=steps_per_epoch,validation_steps=validation_steps)
"""
loss, val_loss, acc, val_acc = liftnet.train(x_train,y_train,
                                             validation_split=validation_split,
                                             epochs=epochs,batch_size=bunch_steps,
                                             steps_per_epoch=steps_per_epoch,
                                             validation_steps=validation_steps,
                                             snapshot = snapshot,
                                             head_of_name = head_of_name,
                                             circle_num = circle_num)
print('------------------------------------------------------------')
print('finished train and plot loss')

plt.figure(num=0)
newepochs = range(1,len(loss)+1)
plt.plot(newepochs,loss,'b',label='train loss')
plt.plot(newepochs,val_loss,'r',label='val_loss')
plt.title('train and val loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.savefig(loss_map_name)

plt.figure(num=1)
newepochs = range(1,len(acc)+1)
plt.plot(newepochs,acc,'b',label='train acc')
plt.plot(newepochs,val_acc,'r',label='val_acc')
plt.title('train and val acc')
plt.xlabel('epochs')
plt.ylabel('acc')
plt.legend()
plt.savefig(acc_map_name)

liftnet.save_weights(save_model_name)
"""
liftnet.load_weights(save_model_name)

print('save the model')

#========================================================================================#
#--------------------------------LiftingNet Predict-------------------------------------------#
print('------------------------------------------------------------')
print('test LiftingNet model')
result_txt_list.append('LiftingNet result: \n')
#test
LiftingNet_predict = liftnet.predict(x_test, steps=1)

if whether_use_cwru_data:
    LiftngNet_test_result = ml2.evaluate_model(LiftingNet_predict ,y_test, 
                                               whether_save_result=1,  
                                               whether_use_CWRU_data_label=1)
else:
    LiftngNet_test_result = ml2.evaluate_model(LiftingNet_predict ,y_test, 
                                               whether_save_result=1,  
                                               whether_use_CWRU_data_label=0)
result_txt_list.extend(LiftngNet_test_result)

LiftingNet_predict2 = ml2.pre_to_index(LiftingNet_predict)

ans = LiftingNet_predict2
cnt1 = 0
cnt2 = 0
for i in range(len(y_test)):
    if ans[i] == y_test[i]:
        cnt1 += 1
    else:
        cnt2 += 1
print("Accuracy: %.2f %% " % (100 * cnt1 / (cnt1 + cnt2)))
result_txt_list.append('Accuracy: '+str(100 * cnt1 / (cnt1 + cnt2))+'%\n')
result_txt_list.append('\n')

confusion_matrix_map_name1 = result_save_path + '/LiftingNet_result_of_'+result_files_last_name
plot_confusion_matrix(ans,y_test,label_name, save_name=confusion_matrix_map_name1)

print('------------------------------------------------------------')
#========================================================================================#
#--------------------------------SVM with 9AF Train and Predict-------------------------------------------#
print('------------------------------------------------------------')
print('test svm with 9AF model')
result_txt_list.append('svm 9AF result: \n')
sc = StandardScaler()
sc.fit(artificial_feature_of_train_data)

svm_x_train = sc.transform(artificial_feature_of_train_data)
svm_x_test = sc.transform(artificial_feature_of_test_data)

svm_y_train = y_train.reshape((y_train.shape[0],))
svm_y_test = y_test.reshape((y_test.shape[0],))

print('svm_y_train.shape: ', svm_y_train.shape)
print('svm_y_test.shape: ', svm_y_test.shape)

svm = SVC(kernel='linear',C=1.0,random_state= 0)

svm.fit(svm_x_train, svm_y_train)

svm_predict = svm.predict(svm_x_test)

svm_test_result = ml2.evaluate_model3(svm_predict, svm_y_test, 
                                      whether_save_result=1, 
                                      whether_use_CWRU_data_label=whether_use_cwru_data)
result_txt_list.extend(svm_test_result)

ans = svm_predict
cnt1 = 0
cnt2 = 0
for i in range(len(y_test)):
    if ans[i] == y_test[i]:
        cnt1 += 1
    else:
        cnt2 += 1
print("Accuracy: %.2f %% " % (100 * cnt1 / (cnt1 + cnt2)))
result_txt_list.append('Accuracy: '+str(100 * cnt1 / (cnt1 + cnt2))+'%\n')
result_txt_list.append('\n')

print('ans.shape: ', ans.shape)
print('y_test.shape: ', y_test.shape)

confusion_matrix_map_name2 = result_save_path + '/SVM_9AF_result_of_'+result_files_last_name
plot_confusion_matrix(ans,y_test,label_name, save_name=confusion_matrix_map_name2)
print('------------------------------------------------------------')


#========================================================================================#
#--------------------------------XGBoost with 9AF Train and Predict-------------------------------------------#
print('------------------------------------------------------------')
print('test xgboost with 9AF model')
result_txt_list.append('xgboost 9AF result: \n')
# create XGBoost
xgboost_model = XGBClassifier(learning_rate=0.1,
                                n_estimators=1000,         # 树的个数--1000棵树建立xgboost
                                max_depth=16,               # 树的深度
                                min_child_weight = 1,      # 叶子节点最小权重
                                gamma=0.1,                  # 惩罚项中叶子结点个数前的参数
                                subsample=0.8,             # 随机选择80%样本建立决策树
                                objective='multi:softmax', # 指定损失函数
                                )

xgboost_y_train = y_train.reshape((y_train.shape[0],))
xgboost_y_test = y_test.reshape((y_test.shape[0],))


# train xgboost
xgboost_model.fit(artificial_feature_of_train_data, xgboost_y_train)

xgboost_predict = xgboost_model.predict(artificial_feature_of_test_data)

xgboost_result = ml2.evaluate_model3(xgboost_predict, xgboost_y_test, 
                                     whether_save_result=1, 
                                     whether_use_CWRU_data_label=whether_use_cwru_data)
result_txt_list.extend(xgboost_result)

ans = xgboost_predict
cnt1 = 0
cnt2 = 0
for i in range(len(y_test)):
    if ans[i] == y_test[i]:
        cnt1 += 1
    else:
        cnt2 += 1
print("Accuracy: %.2f %% " % (100 * cnt1 / (cnt1 + cnt2)))
result_txt_list.append('Accuracy: '+str(100 * cnt1 / (cnt1 + cnt2))+'%\n')
result_txt_list.append('\n')

print('ans.shape: ', ans.shape)
print('y_test.shape: ', y_test.shape)

confusion_matrix_map_name3 = result_save_path + '/XGBoost_with_9AF_result_of_'+result_files_last_name
plot_confusion_matrix(ans,y_test,label_name, save_name=confusion_matrix_map_name3)
print('------------------------------------------------------------')

"""
#========================================================================================#
#--------------------------------XGBoost PCA AF Train and Predict-------------------------------------------#
print('------------------------------------------------------------')
print('test xgboost with PCA 9AF model')

pca_feature_extractor1 = PCA(n_components=pca_parameters)
pca_feature_extractor1.fit(artificial_feature_of_train_data)
pca_feature_for_train1 = pca_feature_extractor1.transform(artificial_feature_of_train_data)
pca_feature_for_test1 = pca_feature_extractor1.transform(artificial_feature_of_test_data)
print('pca_feature_for_train.shape: ', pca_feature_for_train1.shape)
print('pca_feature_for_test.shape: ', pca_feature_for_test1.shape)

#create xgboost train dataset
x_train_for_xgboost_PCA_AF = pca_feature_for_train1.reshape(x_train_number,-1)
x_test_for_xgboost_PCA_AF = pca_feature_for_test1.reshape(x_test_number,-1)

#x_train_for_xgboost_AF_and_LF  = np.concatenate((x_train_for_xgboost_AF_and_LF ,artificial_feature_of_train_data),axis=1)
#x_test_for_xgboost_AF_and_LF  = np.concatenate((x_test_for_xgboost_AF_and_LF ,artificial_feature_of_test_data),axis=1)

print('x_train_for_xgboost_AF_and_LF.shape: ', x_train_for_xgboost_PCA_AF.shape)
print('x_test_for_xgboost_AF_and_LF.shape: ', x_test_for_xgboost_PCA_AF.shape)

#create xgboost model
xgboost_model_for_PCA_AF = XGBClassifier(learning_rate=0.1,
                                            n_estimators=1000,         # 树的个数--1000棵树建立xgboost
                                            max_depth=16,               # 树的深度
                                            min_child_weight = 1,      # 叶子节点最小权重
                                            gamma=0.1,                  # 惩罚项中叶子结点个数前的参数
                                            subsample=0.8,             # 随机选择80%样本建立决策树
                                            objective='multi:softmax' # 指定损失函数
                                            )

print('create the model')
result_txt_list.append('xgboost PCA AF result: \n')

xgboost_PCAaf_y_train = y_train.reshape((y_train.shape[0],))
xgboost_PCAaf_y_test = y_test.reshape((y_test.shape[0],))

xgboost_model_for_PCA_AF.fit(x_train_for_xgboost_PCA_AF, xgboost_PCAaf_y_train)
print('fit the model')
xgboost_model_for_PCA_AF_predict = xgboost_model_for_PCA_AF.predict(x_test_for_xgboost_PCA_AF)
print('finish predict')
xgboost_with_PCA_AF_result = ml2.evaluate_model3(xgboost_model_for_PCA_AF_predict, 
                                                xgboost_PCAaf_y_test, whether_save_result=1, 
                                                whether_use_CWRU_data_label=whether_use_cwru_data)
result_txt_list.extend(xgboost_with_PCA_AF_result)
print('save the result')
ans = xgboost_model_for_PCA_AF_predict
cnt1 = 0
cnt2 = 0
for i in range(len(y_test)):
    if ans[i] == y_test[i]:
        cnt1 += 1
    else:
        cnt2 += 1
print("Accuracy: %.2f %% " % (100 * cnt1 / (cnt1 + cnt2)))
result_txt_list.append('Accuracy: '+str(100 * cnt1 / (cnt1 + cnt2))+'%\n')
result_txt_list.append('\n')

print('ans.shape: ', ans.shape)
print('y_test.shape: ', y_test.shape)

confusion_matrix_map_name4 = result_save_path + '/XGBoost_with_PCA_AF_result_of_'+result_files_last_name
plot_confusion_matrix(ans,y_test,label_name, save_name=confusion_matrix_map_name4)

"""
#========================================================================================#
#--------------------------------XGBoost PCA LF Train and Predict-------------------------------------------#
print('------------------------------------------------------------')
print('test xgboost with PCA LF model')
#LiftingNet features
print('start extract feature')
feature_data_for_train = liftnet.feature_extractor(x_train)
feature_data_for_test = liftnet.feature_extractor(x_test)
print('extracted feature')

feature_data_for_train = feature_data_for_train.numpy()
feature_data_for_test = feature_data_for_test.numpy()

print('feature_data_for_train.shape: ', feature_data_for_train.shape)
print('feature_data_for_test.shape: ', feature_data_for_test.shape)

print('finished feature extract')
pca_feature_extractor2 = PCA(n_components=pca_parameters)
pca_feature_extractor2.fit(feature_data_for_train)
pca_feature_for_train2 = pca_feature_extractor2.transform(feature_data_for_train)
pca_feature_for_test2 = pca_feature_extractor2.transform(feature_data_for_test)
print('pca_feature_for_train.shape: ', pca_feature_for_train2.shape)
print('pca_feature_for_test.shape: ', pca_feature_for_test2.shape)
#create xgboost train dataset
x_train_for_xgboost_PCA_LF = pca_feature_for_train2.reshape(x_train_number,-1)
x_test_for_xgboost_PCA_LF = pca_feature_for_test2.reshape(x_test_number,-1)

print('x_train_for_xgboost_AF_and_LF.shape: ', x_train_for_xgboost_PCA_LF.shape)
print('x_test_for_xgboost_AF_and_LF.shape: ', x_test_for_xgboost_PCA_LF.shape)

#create xgboost model
xgboost_model_for_PCA_LF = XGBClassifier(learning_rate=0.1,
                                            n_estimators=1000,         # 树的个数--1000棵树建立xgboost
                                            max_depth=16,               # 树的深度
                                            min_child_weight = 1,      # 叶子节点最小权重
                                            gamma=0.1,                  # 惩罚项中叶子结点个数前的参数
                                            subsample=0.8,             # 随机选择80%样本建立决策树
                                            objective='multi:softmax' # 指定损失函数
                                            )

print('create the model')
result_txt_list.append('xgboost PCA LF result: \n')

xgboost_PCAlf_y_train = y_train.reshape((y_train.shape[0],))
xgboost_PCAlf_y_test = y_test.reshape((y_test.shape[0],))

xgboost_model_for_PCA_LF.fit(x_train_for_xgboost_PCA_LF, xgboost_PCAlf_y_train)
print('fit the model')
xgboost_model_for_PCA_LF_predict = xgboost_model_for_PCA_LF.predict(x_test_for_xgboost_PCA_LF)
print('finish predict')
xgboost_with_PCA_LF_result = ml2.evaluate_model3(xgboost_model_for_PCA_LF_predict, 
                                                xgboost_PCAlf_y_test, whether_save_result=1, 
                                                whether_use_CWRU_data_label=whether_use_cwru_data)
result_txt_list.extend(xgboost_with_PCA_LF_result)
print('save the result')
ans = xgboost_model_for_PCA_LF_predict
cnt1 = 0
cnt2 = 0
for i in range(len(y_test)):
    if ans[i] == y_test[i]:
        cnt1 += 1
    else:
        cnt2 += 1
print("Accuracy: %.2f %% " % (100 * cnt1 / (cnt1 + cnt2)))
result_txt_list.append('Accuracy: '+str(100 * cnt1 / (cnt1 + cnt2))+'%\n')
result_txt_list.append('\n')

print('ans.shape: ', ans.shape)
print('y_test.shape: ', y_test.shape)

confusion_matrix_map_name5 = result_save_path + '/XGBoost_with_PCA_LF_result_of_'+result_files_last_name
plot_confusion_matrix(ans,y_test,label_name, save_name=confusion_matrix_map_name5)
"""
#========================================================================================#
#--------------------------------XGBoost PCA(9AF + LF) Train and Predict-------------------------------------------#
print('------------------------------------------------------------')
print('test xgboost with PCA (9AF+LF) model')

LF_AF_feature_for_train = np.concatenate((feature_data_for_train, artificial_feature_of_train_data),axis=1)
LF_AF_feature_for_test = np.concatenate((feature_data_for_test, artificial_feature_of_test_data),axis=1)

print('feature_data_for_train.shape: ', LF_AF_feature_for_train.shape)
print('feature_data_for_test.shape: ', LF_AF_feature_for_test.shape)

print('finished feature extract')
pca_feature_extractor3 = PCA(n_components=pca_parameters)
pca_feature_extractor3.fit(LF_AF_feature_for_train)
pca_feature_for_train3 = pca_feature_extractor3.transform(LF_AF_feature_for_train)
pca_feature_for_test3 = pca_feature_extractor3.transform(LF_AF_feature_for_test)
print('pca_feature_for_train.shape: ', pca_feature_for_train3.shape)
print('pca_feature_for_test.shape: ', pca_feature_for_test3.shape)


#create xgboost train dataset
x_train_for_xgboost_PCA_AF_and_LF = pca_feature_for_train3.reshape(x_train_number,-1)
x_test_for_xgboost_PCA_AF_and_LF = pca_feature_for_test3.reshape(x_test_number,-1)

#x_train_for_xgboost_AF_and_LF  = np.concatenate((x_train_for_xgboost_AF_and_LF ,artificial_feature_of_train_data),axis=1)
#x_test_for_xgboost_AF_and_LF  = np.concatenate((x_test_for_xgboost_AF_and_LF ,artificial_feature_of_test_data),axis=1)

print('x_train_for_xgboost_AF_and_LF.shape: ', x_train_for_xgboost_PCA_AF_and_LF.shape)
print('x_test_for_xgboost_AF_and_LF.shape: ', x_test_for_xgboost_PCA_AF_and_LF.shape)

#create xgboost model
xgboost_model_for_PCA_AF_and_LF = XGBClassifier(learning_rate=0.1,
                                            n_estimators=1000,         # 树的个数--1000棵树建立xgboost
                                            max_depth=16,               # 树的深度
                                            min_child_weight = 1,      # 叶子节点最小权重
                                            gamma=0.1,                  # 惩罚项中叶子结点个数前的参数
                                            subsample=0.8,             # 随机选择80%样本建立决策树
                                            objective='multi:softmax' # 指定损失函数
                                            )

print('create the model')
result_txt_list.append('xgboost PCA 9AF and LF result: \n')

xgboost_PCAaflf_y_train = y_train.reshape((y_train.shape[0],))
xgboost_PCAaflf_y_test = y_test.reshape((y_test.shape[0],))

xgboost_model_for_PCA_AF_and_LF.fit(x_train_for_xgboost_PCA_AF_and_LF, xgboost_PCAaflf_y_train)
print('fit the model')
xgboost_model_for_PCA_AF_and_LF_predict = xgboost_model_for_PCA_AF_and_LF.predict(x_test_for_xgboost_PCA_AF_and_LF)
print('finish predict')
xgboost_with_PCA_AF_LF_result = ml2.evaluate_model3(xgboost_model_for_PCA_AF_and_LF_predict, 
                                                xgboost_PCAaflf_y_test, whether_save_result=1, 
                                                whether_use_CWRU_data_label=whether_use_cwru_data)
result_txt_list.extend(xgboost_with_PCA_AF_LF_result)
print('save the result')
ans = xgboost_model_for_PCA_AF_and_LF_predict
cnt1 = 0
cnt2 = 0
for i in range(len(y_test)):
    if ans[i] == y_test[i]:
        cnt1 += 1
    else:
        cnt2 += 1
print("Accuracy: %.2f %% " % (100 * cnt1 / (cnt1 + cnt2)))
result_txt_list.append('Accuracy: '+str(100 * cnt1 / (cnt1 + cnt2))+'%\n')
result_txt_list.append('\n')

print('ans.shape: ', ans.shape)
print('y_test.shape: ', y_test.shape)

confusion_matrix_map_name6 = result_save_path + '/XGBoost_with_PCA_9AFandLF_result_of_'+result_files_last_name
plot_confusion_matrix(ans,y_test,label_name, save_name=confusion_matrix_map_name6)

"""
#========================================================================================#
#--------------------------------XGBoost 9AF + LF Train and Predict-------------------------------------------#
print('------------------------------------------------------------')
print('test xgboost with 9AF and PCA LF model (the Proposed Method）')
print('finished feature extract')
pca_feature_extractor = PCA(n_components=pca_parameters)
pca_feature_extractor.fit(feature_data_for_train)
pca_feature_for_train = pca_feature_extractor.transform(feature_data_for_train)
pca_feature_for_test = pca_feature_extractor.transform(feature_data_for_test)
print('pca_feature_for_train.shape: ', pca_feature_for_train.shape)
print('pca_feature_for_test.shape: ', pca_feature_for_test.shape)


#create xgboost train dataset
x_train_for_xgboost_AF_and_LF = pca_feature_for_train.reshape(x_train_number,-1)
x_test_for_xgboost_AF_and_LF = pca_feature_for_test.reshape(x_test_number,-1)

x_train_for_xgboost_AF_and_LF  = np.concatenate((x_train_for_xgboost_AF_and_LF ,artificial_feature_of_train_data),axis=1)
x_test_for_xgboost_AF_and_LF  = np.concatenate((x_test_for_xgboost_AF_and_LF ,artificial_feature_of_test_data),axis=1)

print('x_train_for_xgboost_AF_and_LF.shape: ', x_train_for_xgboost_AF_and_LF.shape)
print('x_test_for_xgboost_AF_and_LF.shape: ', x_test_for_xgboost_AF_and_LF.shape)

#create xgboost model
xgboost_model_for_AF_and_LF = XGBClassifier(learning_rate=0.1,
                                            n_estimators=1000,         # 树的个数--1000棵树建立xgboost
                                            max_depth=16,               # 树的深度
                                            min_child_weight = 1,      # 叶子节点最小权重
                                            gamma=0.1,                  # 惩罚项中叶子结点个数前的参数
                                            subsample=0.8,             # 随机选择80%样本建立决策树
                                            objective='multi:softmax' # 指定损失函数
                                            )

print('create the model')
result_txt_list.append('xgboost 9AF and LF result: \n')

xgboost_aflf_y_train = y_train.reshape((y_train.shape[0],))
xgboost_aflf_y_test = y_test.reshape((y_test.shape[0],))

xgboost_model_for_AF_and_LF.fit(x_train_for_xgboost_AF_and_LF, xgboost_aflf_y_train)
print('fit the model')
xgboost_model_for_AF_and_LF_predict = xgboost_model_for_AF_and_LF.predict(x_test_for_xgboost_AF_and_LF)
print('finish predict')
xgboost_with_AF_LF_result = ml2.evaluate_model3(xgboost_model_for_AF_and_LF_predict, 
                                                xgboost_aflf_y_test, whether_save_result=1, 
                                                whether_use_CWRU_data_label=whether_use_cwru_data)
result_txt_list.extend(xgboost_with_AF_LF_result)
print('save the result')
ans = xgboost_model_for_AF_and_LF_predict
cnt1 = 0
cnt2 = 0
for i in range(len(y_test)):
    if ans[i] == y_test[i]:
        cnt1 += 1
    else:
        cnt2 += 1
print("Accuracy: %.2f %% " % (100 * cnt1 / (cnt1 + cnt2)))
result_txt_list.append('Accuracy: '+str(100 * cnt1 / (cnt1 + cnt2))+'%\n')
result_txt_list.append('\n')

print('ans.shape: ', ans.shape)
print('y_test.shape: ', y_test.shape)

confusion_matrix_map_name7 = result_save_path + '/XGBoost_with_9AFandLF_result_of_'+result_files_last_name
plot_confusion_matrix(ans,y_test,label_name, save_name=confusion_matrix_map_name7)
#========================================================================================#



result_txt_name = result_save_path + '/result_of_' + result_txt_last_name
if os.path.exists(result_txt_name):
    os.remove(result_txt_name)

f = open(result_txt_name,'w')
for info in result_txt_list:
    f.writelines(info)

print('OK')

print('program end')









