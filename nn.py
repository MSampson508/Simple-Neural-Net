import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import json

def forwardPropTest(w1, w2, b1, b2, Xi):
    Z1 = w1.dot(Xi) + b1
    #print(Z1)
    A1 = ReLU(Z1)
    #print(A1)
    Z2 = w2.dot(A1) + b2
    #print(Z2)
    A2 = softmax(Z2)
    #print(A2)
    pred = A2[:, 0].argmax()

    return pred, A2[:, 0], Z2[:, 0] 
    

def ReLU(Z):
    return np.maximum(0,Z)

def softmax(Z, axis=0):              # columns are samples: (10, 1000)
    Z = Z - Z.max(axis=axis, keepdims=True)   # stability
    eZ = np.exp(Z)
    return eZ / eZ.sum(axis=axis, keepdims=True)



def oneHot(Y):
    oneHotY = np.zeros((Y.size, Y.max() + 1), dtype=np.float32)
    oneHotY[np.arange(Y.size), Y] = 1
    oneHotY = oneHotY.T
    return oneHotY

def get_Predictions(A2):
    return np.argmax(A2, 0)
def checkAccuracy(predictions, Y):
    
    return np.sum(predictions == Y) / Y.size



