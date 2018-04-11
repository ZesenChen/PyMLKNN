# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 09:31:16 2017

@author: 14094
"""

import numpy as np
import scipy.io as sio
import evaluate as ev
#import sampleProcess as sp
from sklearn.model_selection import KFold
from knn import *
 
    
class ML_KNN(object):
    s = 1
    k = 10
    labels_num = 0
    train_data_num = 0
    train_data = np.array([])
    train_target = np.array([])
    #test_data = np.array([])
    #test_target = np.array([])
    rtl = np.array([])    
    Ph1 = np.array([])#P(H1)
    Ph0 = np.array([])
    Peh1 = np.array([])
    Peh0 = np.array([])
    predict_labels = np.array([])
    def __init__(self, _train_data, _train_target, _k):
        self.train_data = _train_data
        self.train_target = _train_target
        self.k = _k
        self.labels_num = _train_target.shape[1]
        self.train_data_num = self.train_data.shape[0]
        self.Ph1 = np.zeros((self.labels_num,))
        self.Ph0 = np.zeros((self.labels_num,))
        self.Peh1 = np.zeros((self.labels_num, self.k + 1))
        self.Peh0 = np.zeros((self.labels_num, self.k + 1))
    
    def fit(self):
        for i in range(self.labels_num):
            y = 0
            for j in range(self.train_data_num):
                if self.train_target[j][i] == 1:
                    y = y + 1
            self.Ph1[i] = (self.s + y)/(self.s*2 + self.train_data_num)
        self.Ph0 = 1 - self.Ph1
                   
        for i in range(self.labels_num):
            c1 = np.zeros((self.k + 1,))
            c0 = np.zeros((self.k + 1,))
            for j in range(self.train_data_num):
                temp = 0
                KNN = knn(self.train_data, j, self.k)
                for k in range(self.k):
                    if self.train_target[int(KNN[k])][i] == 1:
                        temp = temp + 1
                if self.train_target[j][i] == 1:
                    c1[temp] = c1[temp] + 1
                else:
                    c0[temp] = c0[temp] + 1
            
            for l in range(self.k + 1):
                self.Peh1[i][l] = (self.s + c1[l])/(self.s*(self.k + 1) + c1.sum())
                self.Peh0[i][l] = (self.s + c0[l])/(self.s*(self.k + 1) + c0.sum())
                
            
    def predict(self, _test_data):
        self.rtl = np.zeros((_test_data.shape[0], self.labels_num))
        test_data_num = _test_data.shape[0]
        self.predict_labels = np.zeros((test_data_num, self.labels_num))
        for i in range(test_data_num):
            KNN = knn1(self.train_data, _test_data[i], self.k)
            for j in range(self.labels_num):
                temp = 0
                y1 = 0
                y0 = 0
                for k in range(self.k):
                    if self.train_target[int(KNN[k])][j] == 1:
                        temp = temp + 1
                y1 = self.Ph1[j]*self.Peh1[j][temp]
                y0 = self.Ph0[j]*self.Peh0[j][temp]
                self.rtl[i][j] = self.Ph1[j]*self.Peh1[j][temp]/(self.Ph1[j]*self.Peh1[j][temp] + self.Ph0[j]*self.Peh0[j][temp])
                if y1 > y0:
                    self.predict_labels[i][j] = 1 
                else:
                    self.predict_labels[i][j] = 0
        #print(self.predict_labels)
        return self.predict_labels
                    
                
    def evaluate(self, _test_target):#the function must be invoked after the predict function has been invoked
        print("Hamming loss:")
        print(ev.HammingLoss(self.predict_labels, _test_target))
        print("Average precision:")
        print(ev.avgprec(self.rtl, _test_target))
        print("Coverage:")
        print(ev.Coverage(self.rtl, _test_target))
        print("One error:")
        print(ev.OneError(self.rtl, _test_target))
        print("Ranking loss:")
        print(ev.rloss(self.rtl, _test_target))
        print("Subset accury:")
        print(ev.SubsetAccuracy(self.predict_labels, _test_target))
        print("AUC:")
        print(ev.MacroAveragingAUC(self.rtl, _test_target))

if __name__ == "__main__":
    kf = KFold(n_splits=10, shuffle=True, random_state=2017)
    dataset = sio.loadmat('sample.mat')
    data,target = dataset['data'],dataset['target'].T
    for tr_index,val_index in kf.split(data):
        tr_X,val_X = data[tr_index],data[val_index]
        tr_Y,val_Y = target[tr_index],target[val_index]
        mlKnn = ML_KNN(tr_X, tr_Y, 10) 
        mlKnn.fit()
        labels1 = mlKnn.predict(val_X)
        mlKnn.evaluate(val_Y)
