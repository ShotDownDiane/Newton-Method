# -*- coding: utf-8 -*-
"""
Created on Sun Dec  3 17:29:45 2017

@author: Faaiz
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

wq=pd.read_csv("winequality-red.csv",sep=';')

# normalizing the data 
normalized=(wq-wq.min())/(wq.max()-wq.min())

split=np.random.rand(len(normalized))<=0.8
wq_train = normalized[split]
wq_test = normalized[~split]

X_train=wq_train.iloc[:,0:11]
bias= np.ones((len(X_train),1))
X_train.insert(0,'bias',bias)
print(X_train)
Y_train=wq_train['quality']
print(Y_train)
X_test=wq_test.iloc[:,0:11]
bias= np.ones((len(X_test),1))
X_test.insert(0,'bias',bias)
print(X_test)
Y_test=wq_test['quality']
print(Y_test)

def sigmoid(XB):
    return 1 / (1 + np.exp(-XB)) 

def RMSE(X,Y,Beta):
      XB = np.dot(X, Beta)
      error = Y-XB
      rmse = np.sqrt(np.mean((error)**2))
      return rmse

def gradient(X_train,Y_train,Beta):
    XB = np.dot(X_train,Beta) 
    XB = np.dot(sigmoid(XB),XB)                                                      
    PY = np.dot((Y_train-XB),X_train)                                       
    return PY                           

def newtons_method(X_train, Y_train,L): 
    Beta = np.zeros(X_train.shape[1])      
    A=[] #stores the RMSE of trained data set
    B=[] #stores the RMSE of test data set                                                                                                                                                                                                                                                                                                   
    for i in range(iterations):
        g = gradient(X_train, Y_train,Beta)
        
       # Beta = Beta - alpha * np.dot(H_inv,g) + 2*L*Beta   
        Beta = Beta - alpha * g + 2*L*Beta  
        print(Beta)
        RMSE_train = RMSE(X_train,Y_train,Beta)
        RMSE_test = RMSE(X_test,Y_test,Beta)      
        A = np.append(A,RMSE_train)
        B = np.append(B,RMSE_test)
        
    plt.plot(A,range(i+1),c="r")  
    plt.title("RMSE vs iterations")
    plt.plot(-B,range(i+1))  
    plt.legend()
    plt.figure() 
    

L=0.0001      
alpha=0.00001      
iterations = 5
newtons_method(X_train,Y_train,L)

L=0.001      
alpha=0.0001   
newtons_method(X_train,Y_train,L)

L=0.01      
alpha=0.001  
newtons_method(X_train,Y_train,L)
