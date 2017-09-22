#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 17:16:06 2017

@author: James
"""

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from numpy import linalg as LA

#covariance
def covariance(a,b):
    c=sum(a-np.mean(a))*sum(b-np.mean(b))/(len(a)-1)
    return c

def PCA(Data):
    Data=Data-Data.mean()
    cov=np.cov(Data,rowvar=False)
    evals, evecs=LA.eigh(cov,UPLO='L')
    num = np.argsort(evals)[::-1]
    evecs = evecs[:,num]
    evals = evals[num]
    output=np.matmul(Data,evecs)
    plt.scatter(Data[:,1],Data[:,2])
    plt.title("Original Data")
    plt.show()
    plt.scatter(output[:,1],output[:,2])
    plt.title("Transformed Data")
    plt.show()
    return output,evals,evecs
    
Data=np.loadtxt(open("dataset_1.csv", "rb"), delimiter=",", skiprows=1)

PCA(Data)
