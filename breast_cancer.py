import numpy as np
from NeuralNet import *

name_to_vec = {"2":np.array([1,0]), "4":np.array([0,1])}

def line_to_vec(line):
    x = np.zeros(9)
    line = line.replace("\n", "").split(",")
    try:
        for i in range(1,len(line)-1):
            x[i-1] = float(line[i])
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
X, y = make_samples("BreastCancer/training_set.txt")
N = NeuralNet([9,16,16,2], 3, 1e-4)
N.train(X,y, 100000)
# test NN
Xt,yt = make_samples("BreastCancer/data.txt")
y_hat = N.forward(Xt)
y_hat[y_hat <= 0.90] = 0; y_hat[y_hat>0.90] = 1
result = np.abs(np.sum(y_hat-yt, axis=1))
result[np.abs(result)>1] = 1 # count every error ones
result = 1 - (np.sum(result)/Xt.shape[0])
print("Corrct:", "%.2f" % result)
