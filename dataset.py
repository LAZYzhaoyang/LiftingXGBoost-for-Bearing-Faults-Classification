"""
# MIT Copyright 2019
# zhaoyang
# 2019 10 19
"""
from __future__ import  absolute_import
from __future__ import  division
#import tensorflow as tf
import numpy as np
import math
import glob
import matplotlib
from scipy.io import loadmat
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot,savefig
import random
from sklearn.metrics import confusion_matrix
import seaborn as sea

def read_and_savefig(loadpath,savepath,fig_size=(20,2)):
    label_name = ['/normal/', '/inner_ring/', '/outer_ring/', '/roller/', '/joint/']
    fig_num = 0
    for name in label_name:
        fnames = glob.glob(loadpath + name + '*.txt')
        for file_name in fnames:
            read_data = np.loadtxt(file_name,skiprows=16)
            f1 = plt.figure(num=fig_num,figsize=fig_size)
            plot(read_data[:,0],read_data[:,1])
            savefig_path = file_name.replace(loadpath,savepath)
            savefig_path = savefig_path.replace('.txt','.jpg')
            savefig(savefig_path)
            fig_num += 1

def read_and_savedataset(loadpath,savepath):
    label_name = ['/normal/', '/inner_ring/', '/outer_ring/', '/roller/', '/joint/']
    label = []
    dataset = []
    for i in range(len(label_name)):
        name = label_name[i]
        print(name,i)
        fnames = glob.glob(loadpath + name + '*.txt')
        for file_name in fnames:
            read_data = np.loadtxt(file_name,skiprows=16)
            read_data = read_data[:256000,1].T
            read_data = read_data[np.newaxis,:]
            if len(dataset)==0:
                dataset = read_data
                label.append(i)
                continue
            dataset = np.concatenate((dataset,read_data),axis = 0)
            label.append(i)
    label = np.array(label)
    label = label[:,np.newaxis]
    print(label.shape)
    print(dataset.shape)
    total_dataset = np.concatenate((label,dataset),axis=1)
    print(total_dataset.shape)
    np.savetxt(savepath + '/label.csv', label, delimiter = ',')
    np.savetxt(savepath + '/dataset.csv', dataset, delimiter = ',')
    np.savetxt(savepath + '/total_dataset.csv', total_dataset, delimiter = ',')
    print('dataset saved')

def diliver_test_and_train_data(dataset, label, test_rate = 0.4):
    num = dataset.shape[0]
    permutation = np.random.permutation(num)
    dataset = dataset[permutation,:]
    label = label[permutation,:]
    print(dataset.shape)
    print(label.shape)
    print(label.T)
    test_num = round(test_rate * num)
    train_num = num - test_num
    train_index = random.sample(range(0,num-1),train_num)
    print('train_num:',train_num)
    data_index = np.zeros([num,1],dtype=np.int)
    data_index[train_index] = 1
    test_index = np.where(data_index==0)
    test_index=test_index[0]

    train_data = dataset[train_index,:,np.newaxis]
    test_data = dataset[test_index,:,np.newaxis]
    train_label = label[train_index]
    test_label = label[test_index]
    dataset = dataset[:, :, np.newaxis]

    return train_data, train_label, test_data, test_label, dataset, label

def mix_data(dataset, label):
    num = dataset.shape[0]
    permutation = np.random.permutation(num)
    dataset = dataset[permutation,:,:]
    label = label[permutation,:]
    print(dataset.shape)
    print(label.shape)
    return dataset, label

def read_and_plot(filename,savepath,cutsize=1000,fig_size=(20,2)):
    read_data = np.loadtxt(filename,skiprows=16)[:256000,:]
    split_data = np.vsplit(read_data,cutsize)
    for i in range(len(split_data)):
        plt.figure(num=i,figsize=fig_size)
        plot(split_data[i][:,0],split_data[i][:,1])
        savefig_path = savepath + str(i) + '.jpg'
        savefig(savefig_path)


def split_data_by_max_min(data,cutsize=256,max_min=None,direction='v'):
    if direction != 'v':
        read_data = data.T
    else:
        read_data = data
    
    if len(read_data.shape) == 1:
        read_data = read_data[:,np.newaxis]

    
    if read_data.shape[1] != 1:
        print('error data.shape must be [n,1], you may need to change the direction')
        return EOFError
    

    split_data = np.vsplit(read_data,cutsize)
    
    if max_min == None:
        max_size = 0
        min_size = 0
        for i in range(len(split_data)):
            max_size = max_size + np.max(split_data[i])
            min_size = min_size + np.min(split_data[i])
        mean_max = max_size/len(split_data)
        mean_min = min_size/len(split_data)
        max_min = abs(mean_max)
        if abs(mean_min) > abs(mean_max):
            max_min = abs(mean_min)
    index = []
    for i in range(len(split_data)):
        if np.max(split_data[i]) >= max_min:
            index.append(1)
            continue
        if np.min(split_data[i]) <= (0 - max_min):
            index.append(1)
            continue
        index.append(0)
    #print(index)
    return split_data, index

def split_data_by_std(data,cutsize=256,std_level=None,direction='v'):
    if direction != 'v':
        read_data = data.T
    else:
        read_data = data
    
    if len(read_data.shape) == 1:
        read_data = read_data[:,np.newaxis]
    
    print(read_data.shape)
    if read_data.shape[1] != 1:
        print('error data.shape must be [n,1], you may need to change the direction')
        return EOFError
    

    split_data = np.vsplit(read_data,cutsize)
    
    if std_level == None:
        all_std = []
        std_add = 0
        for i in range(len(split_data)):
            std_add = std_add + np.std(split_data[i])
            all_std.append(np.std(split_data[i]))
        mean_std = std_add/len(split_data)
        std_level = mean_std

    index = []
    for i in range(len(split_data)):
        if np.std(split_data[i]) >= std_level:
            index.append(1)
            continue
        index.append(0)
    #print(index)
    return split_data, index


def index_first_final(index):
    start = 1
    end = 1

    first_one = 0
    final_one = len(index)-1
    setp1 = 1
    setp2 = 1
    while start:
        if index[first_one] == 1:
            break
        first_one += 1
        setp1 += 1
        if setp1 >= len(index):
            print('cannot find the one')
            break

    while end:
        if index[final_one] == 1:
            break
        final_one = final_one - 1
        setp2 += 1
        if setp1 >= len(index):
            print('cannot find the one')
            break
    print('first: ' + str(first_one) + ' final: ' + str(final_one))
    return first_one, final_one

def cut_data(split_data, ch1_index, ch2_index, ch3_index):
    ch1_first, ch1_final = index_first_final(ch1_index)
    ch2_first, ch2_final = index_first_final(ch2_index)
    ch3_first, ch3_final = index_first_final(ch3_index)

    first_index = ch1_first
    final_index = ch1_final

    if ch2_first < first_index:
        first_index = ch2_first
    if ch3_first < first_index:
        first_index = ch3_first
    if ch2_final > final_index:
        final_index = ch2_final
    if ch3_final > final_index:
        final_index = ch3_final
    stack_data = []

    for i in range(first_index,final_index):
        if len(stack_data)==0:
            stack_data = split_data[i]
            continue
        stack_data = np.vstack((stack_data,split_data[i]))
    print('end vstack')
    print(stack_data.shape)
    return stack_data


def split_dataset(dataset,num_circle = 1, cutsize=256, time_limit = 20,circle_per_second=20, split_model = 'std', max_min=None, direction='v', std_level=None, whether_trans_to_CWRU_data=0, CWRU_data_length=1024):
    if direction != 'v':
        dataset = dataset.T
    
    if split_model == 'std':
        _, ch1_index = split_data_by_std(dataset[:,0],cutsize=cutsize,std_level=std_level)
        _, ch2_index = split_data_by_std(dataset[:,1],cutsize=cutsize,std_level=std_level)
        _, ch3_index = split_data_by_std(dataset[:,2],cutsize=cutsize,std_level=std_level)
    else:
        _, ch1_index = split_data_by_max_min(dataset[:,0],cutsize=cutsize,max_min=max_min)
        _, ch2_index = split_data_by_max_min(dataset[:,1],cutsize=cutsize,max_min=max_min)
        _, ch3_index = split_data_by_max_min(dataset[:,2],cutsize=cutsize,max_min=max_min)

    split_data = np.vsplit(dataset,cutsize)
    stack_data = cut_data(split_data,ch1_index,ch2_index,ch3_index)
    
    stack_data_length = stack_data.shape[0]
    data_length = dataset.shape[0]
    data_per_circle = round(data_length/time_limit/circle_per_second)
    data_size = data_per_circle * num_circle
    if whether_trans_to_CWRU_data==1:
        data_size = CWRU_data_length

    how_much_data = math.floor(stack_data_length/data_size)
    true_data_length = how_much_data * data_size

    stack_data = stack_data[:true_data_length,:]
    split_stack_data = np.vsplit(stack_data,how_much_data)
    split_stack_data = np.array(split_stack_data)
    print('success split')
    print(split_stack_data.shape)

    return split_stack_data, how_much_data

def read_split_and_savedataset(loadpath,savepath,datalength = 256000,num_circle = 1, cutsize=256, time_limit = 20, circle_per_second=20, split_model = 'std', max_min=None, direction='v', std_level=None, whether_trans_to_CWRU_data=0, CWRU_data_length=1024):
    label_name = ['/normal/', '/inner_ring/', '/outer_ring/', '/roller/', '/joint/']
    if whether_trans_to_CWRU_data == 1:
        label_name = ['/normal/', '/roller/', '/inner_ring/', '/outer_ring/']
    label = []
    dataset = []
    for i in range(len(label_name)):
        name = label_name[i]
        print(name,i)
        fnames = glob.glob(loadpath + name + '*ch2.txt')

        for file_name1 in fnames:
            file_name2 = file_name1.replace('ch2','ch3')
            file_name3 = file_name1.replace('ch2','ch4')
            read_data1 = np.loadtxt(file_name1,skiprows=16)[:datalength,1]
            read_data2 = np.loadtxt(file_name2,skiprows=16)[:datalength,1]
            read_data3 = np.loadtxt(file_name3,skiprows=16)[:datalength,1]
            read_data = np.vstack((read_data1,read_data2))
            read_data = np.vstack((read_data,read_data3)).T
            split_data, num = split_dataset(read_data, num_circle=num_circle,cutsize=cutsize, time_limit = time_limit,circle_per_second=circle_per_second, split_model = split_model, max_min=max_min, direction=direction, std_level=std_level, whether_trans_to_CWRU_data=whether_trans_to_CWRU_data, CWRU_data_length=CWRU_data_length)

            for j in range(num):
                label.append(i)
            if len(dataset)==0:
                dataset = split_data
                continue
            dataset = np.concatenate((dataset,split_data),axis = 0)
    print('read finished')
    label = np.array(label)
    label = label[:,np.newaxis]
    if whether_trans_to_CWRU_data==1:
        dataset = dataset[:,:,:2]
    print('dataset shape : ',dataset.shape)
    print('label shape : ', label.shape)
    save_dataset_name = savepath + '/dataset_' + str(num_circle) + '_' + str(cutsize) + '.npy'
    save_label_name = savepath + '/label_' + str(num_circle) + '_' + str(cutsize) + '.npy'
    if whether_trans_to_CWRU_data==1:
        save_dataset_name = savepath + '/dataset_trans_to_CWRU_data_' + str(CWRU_data_length) + '.npy'
        save_label_name = savepath + '/label_trans_to_CWRU_data_' + str(CWRU_data_length) + '.npy'
    np.save(save_label_name, label)
    np.save(save_dataset_name, dataset)
    #np.savetxt(savepath + '/total_dataset.csv', total_dataset, delimiter = ',')

    print('dataset saved')

def feature_extractor(dataset):
    #---------------------Peak amplitude-----------------------#
    XP = np.max(dataset,axis=1)
    #-----------------Peak to Peak amplitude-------------------#
    XPP = np.max(dataset,axis=1) - np.min(dataset,axis=1)
    #------------------------RMS-------------------------------#
    square_data = dataset * dataset
    N = dataset.shape[1]
    print('N:',N)
    sum_square = np.sum(square_data,axis=1)/N
    print('sum_square.shape(): ',sum_square.shape)
    RMS = np.sqrt(sum_square)
    print('RMS.shape(): ',RMS.shape)

    #----------------------Variance-----------------------------#
    data_mean = np.mean(dataset,axis=1)
    print('data_mean.shape(): ',data_mean.shape)
    data_std = np.std(dataset,axis=1)
    print('data_std.shape(): ', data_std.shape)

    new_data_mean = data_mean[:,np.newaxis,:]
    """
    padding_data_mean = []
    for i in range(N):
        if len(padding_data_mean)==0:
            padding_data_mean = new_data_mean
            continue
        padding_data_mean = np.concatenate([padding_data_mean,new_data_mean],axis=1)
    print('padding_data_mean.shape: ', padding_data_mean.shape)
    """

    #d_data_mean = dataset - padding_data_mean
    d_data_mean = dataset - new_data_mean
    print('d_data_mean.shape: ',d_data_mean.shape)

    Variance = np.sum(np.power(d_data_mean, 2), axis=1)/((N-1) * np.power(data_std, 2))
    print('Variance.shape: ',Variance.shape)

    #--------------------Skewness----------------------------#
    Skewness = np.sum(np.power(d_data_mean, 3), axis=1)/((N-1) * np.power(data_std, 3))
    print('Skewness.shape: ',Skewness.shape)

    #--------------------Kurtosis----------------------------#
    Kurtosis = np.sum(np.power(d_data_mean, 4), axis=1)/((N-1) * np.power(data_std, 4))
    print('Kurtosis.shape: ', Kurtosis.shape)

    #--------------------Shape factor------------------------#
    SF = np.sqrt(np.sum(np.power(dataset, 2),axis=1)/N)/(np.sum(np.abs(dataset),axis=1)/N)
    print('SF.shape: ',SF.shape)

    #---------------------Crest factor----------------------#
    CF = np.max(np.abs(dataset),axis=1)/RMS
    print('CF.shape: ',CF.shape)

    #---------------------impulse factor--------------------#
    IF = np.max(np.abs(dataset),axis=1)/(np.sum(np.abs(dataset),axis=1)/N)
    print('IF.shape: ',IF.shape)

    #---------------------Margin factor---------------------#
    MF = np.max(np.abs(dataset),axis=1)/np.power((np.sum(np.abs(dataset),axis=1)/N),2)
    print('MF.shape: ',MF.shape)

    #---------------------mobility-------------------------#
    diff1_data = dataset[:,1:,:] - dataset[:,:-1,:]
    print('diff1_data.shape: ',diff1_data.shape)
    diff2_data = diff1_data[:,1:,:] - diff1_data[:,:-1,:]
    print('diff2_data.shape: ', diff2_data.shape)

    std_diff1 = np.std(diff1_data,axis=1)
    std_diff2 = np.std(diff2_data,axis=1)

    print('std_diff1.shape: ', std_diff1.shape)
    print('std_diff2.shape: ', std_diff2.shape)

    mobility = std_diff1 / data_std
    print('mobility.shape: ', mobility.shape)

    #----------------------complexity----------------------#
    complexity = (std_diff2/std_diff1)/mobility
    print('complexity.shape: ',complexity.shape)

    #----------------------frequency centre----------------#
    pi = math.pi
    FC = np.sum(diff1_data * dataset[:,1:,:],axis=1)/(2*pi*np.sum(square_data,axis=1))
    print('FC.shape: ',FC.shape)

    #--------------------mean square frequency-------------#
    MSF = np.sum(np.power(diff1_data,2),axis=1)/(4*pi*pi*np.sum(square_data,axis=1))
    print('MSF.shape: ', MSF.shape)

    #---------------root mean square frequency--------------#
    RMSF = np.sqrt(MSF)
    print('RMSF.shape: ', RMSF.shape)

    #---------------root variance frequency----------------#
    RVF = np.sqrt((MSF - np.power(FC,2)))
    print('RVF.shape: ', RVF.shape)

    #------------------organize feature--------------------#
    XP = XP[:, np.newaxis, :]
    XPP = XPP[:, np.newaxis, :]
    sum_square = sum_square[:, np.newaxis, :]
    RMS = RMS[:, np.newaxis, :]
    data_mean = data_mean[:, np.newaxis, :]
    data_std = data_std[:, np.newaxis, :]
    Variance = Variance[:, np.newaxis, :]
    Skewness = Skewness[:, np.newaxis, :]
    Kurtosis = Kurtosis[:, np.newaxis, :]
    SF = SF[:, np.newaxis, :]
    CF = CF[:, np.newaxis, :]
    IF = IF[:, np.newaxis, :]
    MF = MF[:, np.newaxis, :]
    mobility = mobility[:, np.newaxis, :]
    complexity = complexity[:, np.newaxis, :]
    FC = FC[:, np.newaxis, :]
    MSF = MSF[:, np.newaxis, :]
    RMSF = RMSF[:, np.newaxis, :]
    RVF = RVF[:, np.newaxis, :]

    feature_data = np.concatenate([XP,XPP,sum_square,RMS,data_mean,data_std,Variance,Skewness,Kurtosis,SF,CF,IF,MF,mobility,complexity,FC,MSF,RMSF,RVF],axis=1)
    return feature_data


def feature_extractor2(dataset):
    N = dataset.shape[1]
    #---------------------Square mean root---------------------#
    p1 = np.square(np.sum(np.sqrt(np.abs(dataset)),axis=1)/N)
    print('p1.shape: ', p1.shape)
    #---------------------Mean absolute------------------------#
    p2 = np.sum(np.abs(dataset),axis=1)/N
    print('p2.shape: ', p2.shape)
    #---------------------Root mean square---------------------#
    p3 = np.sqrt(np.sum(np.square(dataset),axis=1)/N)
    print('p3.shape: ', p3.shape)
    #---------------------Kurtosis-----------------------------#
    p4 = np.sum(np.power(dataset,4),axis=1)/(N*np.power(p3,4))
    print('p4.shape: ', p4.shape)
    #---------------------Skewness-----------------------------#
    p5 = np.sum(np.power(dataset,3),axis=1)/(N*np.power(p3,3))
    print('p5.shape: ', p5.shape)
    #---------------------Crest factor-------------------------#
    p6 = np.max(np.abs(dataset),axis=1)/p3
    print('p6.shape: ', p6.shape)
    #---------------------Shape factor-------------------------#
    p7 = p3/p2
    print('p7.shape: ', p7.shape)
    #---------------------Clearance factor---------------------#
    p8 = np.max(np.abs(dataset),axis=1)/p1
    print('p8.shape: ', p8.shape)
    #---------------------Impulse indicator--------------------#
    p9 = np.max(np.abs(dataset),axis=1)/p2
    print('p9.shape: ', p9.shape)
    #------------------organize feature--------------------#
    p1 = p1[:, np.newaxis, :]
    p2 = p2[:, np.newaxis, :]
    p3 = p3[:, np.newaxis, :]
    p4 = p4[:, np.newaxis, :]
    p5 = p5[:, np.newaxis, :]
    p6 = p6[:, np.newaxis, :]
    p7 = p7[:, np.newaxis, :]
    p8 = p8[:, np.newaxis, :]
    p9 = p9[:, np.newaxis, :]

    feature_data = np.concatenate([p1, p2, p3, p4, p5, p6, p7, p8, p9],axis=1)
    return feature_data

def select_svm_xgboost_common_feature(svm_feature_index, xgb_feature_index, common_feature_number= 20, select_method= 1):
    #if select_method == 1 then choose each class svm feature compare with the xgboost feature
    #if select_method == 2 then choose the svm common feature compare with the xgboost feature
    #return the common feature index (list)
    class_number = svm_feature_index.shape[0]
    if select_method == 1:
        common_feature = []
        for i in range(class_number):
            common_feature.append(list(set(svm_feature_index[i,:]).intersection(set(xgb_feature_index))))
        return common_feature
    else:
        svm_common_feature = []
        #common_feature = []
        for i in range(class_number-1):
            now_feature_index = svm_feature_index[i,:]
            next_feature_index = svm_feature_index[i+1,:]
            common_feature = list(set(now_feature_index).intersection(set(next_feature_index)))
            print('the '+str(i)+'th:')
            if len(svm_common_feature)==0:
                svm_common_feature = common_feature
                print(svm_common_feature)
                continue
            svm_common_feature = list(set(svm_common_feature).union(set(common_feature)))
            print(svm_common_feature)
        print('the final svm feature')
        print(svm_common_feature)
        print(len(svm_common_feature))
        
        select_feature =  list(set(svm_common_feature).intersection(set(xgb_feature_index)))
        print(select_feature)
        print(len(select_feature))
        return select_feature

def data_expension(dataset, label, exnumber=50):
    label_num = []
    k=0
    for i in range(label.shape[0]):
        if label[i,0]==k:
            label_num.append(i)
            k=k+1
    label_num.append(label.shape[0])
    classification_data = []
    for i in range(k):
        new_class_data = dataset[label_num[i]:label_num[i+1]-1,:,:]
        classification_data.append(new_class_data)
    expansion_dataset = []
    expansion_label = []
    for i in range(len(classification_data)):
        expansion_data = classification_data[i]
        sample_num = expansion_data.shape[0]
        weight = np.random.rand(exnumber,2)
        weight = weight/np.sum(weight,axis=1)[:,np.newaxis]
        index = np.random.randint(0,sample_num,[exnumber,2])
        merge_data = np.zeros([exnumber,expansion_data.shape[1],expansion_data.shape[2]])
        merge_label = np.ones([exnumber,1])*i
        for j in range(exnumber):
            merge_data[j,:,:] = expansion_data[index[j,0]]*weight[j,0] + expansion_data[index[j,1]]*weight[j,1]

        if len(expansion_dataset) == 0:
            expansion_dataset = merge_data
            expansion_label = merge_label
            continue
        expansion_dataset = np.concatenate((expansion_dataset,merge_data),axis=0)
        expansion_label = np.concatenate((expansion_label,merge_label),axis=0)
    print('expansion_dataset.shape: ', expansion_dataset.shape)
    print('expansion_label.shape: ', expansion_label.shape)

    return expansion_dataset, expansion_label

def data_expension2(dataset, label, exnumber=50):
    label_num = [0]
    index_num = label[0,0]
    label_class = []
    label_class.append(index_num)
    for i in range(label.shape[0]):
        if label[i,0]!=index_num:
            label_num.append(i)
            index_num = label[i,0]
            label_class.append(index_num)
    label_num.append(label.shape[0])
    classification_data = []
    for i in range(len(label_num)-1):
        new_class_data = dataset[label_num[i]:label_num[i+1],:,:]
        classification_data.append(new_class_data)
        print('new_class_data.shape: ', new_class_data.shape)
    expansion_dataset = []
    expansion_label = []
    for i in range(len(classification_data)):
        expansion_data = classification_data[i]
        sample_num = expansion_data.shape[0]
        weight = np.random.rand(exnumber,2)
        weight = weight/np.sum(weight,axis=1)[:,np.newaxis]
        index = np.random.randint(0,sample_num,[exnumber,2])
        merge_data = np.zeros([exnumber,expansion_data.shape[1],expansion_data.shape[2]])
        merge_label = np.ones([exnumber,1])*label_class[i]
        for j in range(exnumber):
            merge_data[j,:,:] = expansion_data[index[j,0]]*weight[j,0] + expansion_data[index[j,1]]*weight[j,1]
        print('merge_data.shape: ', merge_data.shape)
        print('merge_label.shape: ', merge_label.shape)
        if expansion_dataset == []:
            expansion_dataset = merge_data
            expansion_label = merge_label
            continue
        expansion_dataset = np.concatenate((expansion_dataset,merge_data),axis=0)
        expansion_label = np.concatenate((expansion_label,merge_label),axis=0)
    print('expansion_dataset.shape: ', expansion_dataset.shape)
    print('expansion_label.shape: ', expansion_label.shape)
    return expansion_dataset, expansion_label

def add_data_noise(dataset, noise_scale = 0.01):
    noise = np.random.randn(dataset.shape[0],dataset.shape[1],dataset.shape[2])
    data_max_abs_x = np.max(np.abs(dataset),axis=1)[:,np.newaxis,:]

    print('data_max_abs_x.shape: ',data_max_abs_x.shape)
    noise_data = noise_scale*data_max_abs_x*noise + dataset
    print('noise_data.shape: ', noise_data.shape)
    return noise_data

def expansion_and_add_noise(dataset, label, exnumber=50, noise_scales=0.01, is_label_sort=0, whether_link_original_data=1):

    if is_label_sort == 0:
        index = np.argsort(label,axis = 0)
        dataset1 = dataset[index[:,0],:,:]
        label1 = label[index[:,0],:]

    expansion_data, expansion_label = data_expension(dataset1,label1,exnumber)

    if noise_scales!=0:
        expansion_data = add_data_noise(expansion_data, noise_scale=noise_scales)
    if whether_link_original_data==1:
        dataset = np.concatenate((dataset,expansion_data),axis=0)
        label = np.concatenate((label,expansion_label),axis=0)
        print('save with original data')
        print('after expansion: ')
        print('dataset shape : ',dataset.shape)
        print('label shape : ', label.shape)
        return dataset, label
    else:
        print('save without original data')
        print('after expansion: ')
        print('expansion_data.shape: ', expansion_data.shape)
        print('expansion_label.shape: ', expansion_label.shape)
        return expansion_data, expansion_label

def load_dataset(data_path, circle_num=1, cutsize=256):
    dataset_file_name = data_path + 'dataset_' + str(circle_num) + '_' + str(cutsize) + '.npy'
    label_file_name = data_path + 'label_' + str(circle_num) + '_' + str(cutsize) + '.npy'
    dataset = np.load(dataset_file_name)
    label = np.load(label_file_name)
    print('dataset shape : ',dataset.shape)
    print('label shape : ', label.shape)
    return dataset, label

def load_CWRU_data(data_path,cut_size=1024):
    dataset_file_name = data_path + '/CWRU_dataset_' + str(cut_size) + '.npy'
    label_file_name = data_path + '/CWRU_label_' + str(cut_size) + '.npy'
    dataset = np.load(dataset_file_name)
    label = np.load(label_file_name)
    print('dataset shape : ',dataset.shape)
    print('label shape : ', label.shape)
    return dataset, label
    

def read_CWRU_data(data_file_name, channel=2):
    origin_data = loadmat(data_file_name)

    #data_header = ['DE_time', 'FE_time', 'BA_time']
    if channel==2:
        data_header = ['DE_time', 'FE_time']
    else:
        data_header = ['DE_time', 'FE_time', 'BA_time']
    items_list = list(origin_data.keys())
    #print(items_list)

    read_data = []

    for i in range(len(data_header)):
        current_item = find_item_name(items_list, data_header[i])
        #print(current_item)
        current_data = origin_data[current_item][np.newaxis, :, :]
        if read_data==[]:
            read_data = current_data
            continue
        read_data = np.concatenate([read_data,current_data],axis=2)
    #print(read_data.shape)
    return read_data

def find_item_name(items_list, item_key):
    for i in range(len(items_list)):
        if items_list[i].find(item_key)>=0:
            return items_list[i]
    print('cannot find the item')
    return -1

def cut_CWRU_data(data, label_index, cut_size=1024):
    data_long = data.shape[1]
    cut_num = math.floor(data_long/cut_size)
    #print('cut_num: ', cut_num)
    cut_long = cut_num*cut_size
    split_data =  data[:,:cut_long,:].reshape((-1, cut_size, data.shape[2]))
    #print(split_data.shape)
    label = [label_index for _ in range(cut_num)]
    return split_data, label

def read_and_save_CWRU_data(data_path, save_path=None, cut_size=1024, origin_index=0, first_index=0, second_index=0, whether_trans_to_our_data=0, trans_length=1, whether_use_normal=0, trans_label = None):
    origin_level_document = ['/12K_Drive_End', '/48K_Drive_End', '/Fan_End']
    first_level_document = ['/1797', '/1772', '/1750', '/1730']
    second_level_document = ['/7', '/14', '/21', '/28']
    third_level_document = ['/normal/', '/ball/', '/inner_ring/', '/outer_ring_3/', '/outer_ring_6/', '/outer_ring_12/']
    if whether_trans_to_our_data==1:
        cut_size = trans_length*640
        if whether_use_normal==0:
            third_level_document = ['/ball/', '/inner_ring/', '/outer_ring_3/', '/outer_ring_6/', '/outer_ring_12/']
    read_category = data_path + origin_level_document[origin_index] + first_level_document[first_index] + second_level_document[second_index]
    dataset = []
    label = []
    for i in range(len(third_level_document)):
        read_path = read_category + third_level_document[i]
        fnames = glob.glob(read_path + '*.mat')
        for file_name in fnames:
            if whether_use_normal==0:
                read_data = read_CWRU_data(file_name,channel=3)
            else:
                read_data = read_CWRU_data(file_name, channel=2)
            if whether_trans_to_our_data==1:
                read_data, read_label = cut_CWRU_data(read_data, trans_label[i], cut_size=cut_size)
            else:
                read_data, read_label = cut_CWRU_data(read_data, i, cut_size=cut_size)
            if dataset == []:
                dataset = read_data
                label.extend(read_label)
                continue
            dataset = np.concatenate((dataset, read_data), axis=0)
            label.extend(read_label)
    label = np.array(label)
    label = label[:,np.newaxis]
    dataset_save_name = save_path+'/CWRU_dataset_'+str(cut_size)+'.npy'
    label_save_name = save_path+'/CWRU_label_'+str(cut_size)+'.npy'
    if whether_trans_to_our_data==1:
        dataset_save_name = save_path+'/CWRU_dataset_trans_to_'+str(trans_length)+'_circles_data.npy'
        label_save_name = save_path+'/CWRU_label_trans_to_'+str(trans_length)+'_circles_data.npy'
    np.save(label_save_name, label)
    np.save(dataset_save_name, dataset)
    print('saved data')
    return dataset, label

def plot_confusion_matrix(predict_y, true_y, label_name, save_name='Confusion Matrix.png', format='png'):
    cm = confusion_matrix(true_y, predict_y)
    cm = cm.astype('int32')
    cmap = sea.cubehelix_palette(start = 1.5, rot = 3, gamma=0.8, as_cmap = True)
    f= plt.figure(figsize = (16,9))
    ax1 = f.add_subplot(1,1,1)

    #sea.heatmap(cm,annot=True,ax= ax1, linewidths=0.05, cmap='rainbow',fmt='d')
    sea.heatmap(cm,annot=True,ax= ax1, linewidths=0.05, cmap=cmap,fmt='d')
    ax1.set_title("Confusion Matrix")
    ax1.set_xlabel('Predict')
    ax1.set_ylabel('True')
    ax1.set_xticklabels(label_name)
    ax1.set_yticklabels(label_name,rotation=0)

    #plot_confusion_matrix(cm, label_name, "Confusion Matrix")
    print(cm)
    plt.savefig(save_name, format=format)
    plt.cla()
    plt.close("all")

def calculation_the_accuracy(pred_y, true_y):
    cnt1 = 0
    cnt2 = 0
    for i in range(len(true_y)):
        if pred_y[i]==true_y[i]:
            cnt1 += 1
        else:
            cnt2 += 1
    print("Accuracy: %.2f %% " % (100 * cnt1 / (cnt1 + cnt2)))
    return 100 * cnt1 / (cnt1 + cnt2)