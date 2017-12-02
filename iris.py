import numpy as np
from NeuralNet import *
import sys

# OUTPUTS
SETOSA_N = "Iris-setosa"
SETOSA_V = np.array([1,0,0])

VERSICOLOR_N = "Iris-versicolor"
VERSICOLOR_V = np.array([0,1,0])

VIRGINICA_N = "Iris-virginica"
VIRGINICA_V = np.array([0,0,1])

name_to_vec = {SETOSA_N:SETOSA_V,
               VERSICOLOR_N:VERSICOLOR_V,
                VIRGINICA_N:VIRGINICA_V}

def line_to_vec(line):
    x = np.zeros(4)
    line = line.replace("\n", "").split(",")
    try:
        for i in range(len(line) - 1):
            x[i] = float(line[i])
    except:
        return None, None
    return x, name_to_vec[line[-1]]

def make_samples(filename):
    with open(filename , "r") as f:
        lines = f.readlines()
        X, Y = line_to_vec(lines[0])
        for i in range(1, len(lines)):
            x,y = line_to_vec(lines[i])
            if x is not None:
                X = np.vstack((X,x))
                Y = np.vstack((Y,y))
    return X,Y

# make samples and train the net
X, y = make_samples("Iris/training_set.txt")
N = NeuralNet([4,10,7,3], 3, 1e-4)
N.train(X,y, 100000)
# test NN
Xt,yt = make_samples("Iris/data.txt")
y_hat = N.forward(Xt)
y_hat[y_hat <= 0.90] = 0; y_hat[y_hat>0.90] = 1
result = np.abs(np.sum(y_hat-yt, axis=1))
result[np.abs(result)>1] = 1 # count every error ones
result = 1 - (np.sum(result)/Xt.shape[0])
print("Corrct:", "%.2f" % result)
