# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from numpy import linalg as LA

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

#Question 1

Data=np.loadtxt(open("dataset_1.csv", "rb"), delimiter=",", skiprows=1)

print("x variance is "+str(np.var(Data[:,0])))
print("y variance is "+str(np.var(Data[:,1])))
print("z variance is "+str(np.var(Data[:,2])))

print("Covariance of x and y is "+str(covariance(Data[:,0],Data[:,1])))
print("Covariance of y and z is "+str(covariance(Data[:,1],Data[:,2])))

PCA(Data)

#Question 3
print(np.array([[0, -1], [2, 3]], int))
eigenvalues, eigenvectors =LA.eigh(np.array([[0, -1], [2, 3]]))
print(eigenvalues)
print(eigenvectors)
