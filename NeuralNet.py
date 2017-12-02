import numpy as np
import sys

class NeuralNet(object):
    """
    A simple neural net implementation.
    No bias but with penalty on overfitting
    """

    def __init__(self, sizes, alpha, lamda):
        """
        Gets the sizes of the layer includin input and output
        """
        self.sizes = sizes
        self.init_w()
        self.alpha = alpha
        self.lamda = lamda

    def init_w(self):
        """
        Initialize random weights in the range [0,1]
        """
        self.W = [np.ndarray] * len(self.sizes)
        for i in range(1, len(self.sizes)):
            self.W[i] = np.random.rand(self.sizes[i-1], self.sizes[i])

    def sigmoid(self, z):
        return 1/(1+np.exp(-z))

    def sigmoid_p(self, z):
        return np.exp(-z)/((1+np.exp(-z))**2)

    def forward(self, X):
        """
        Calculate f(X)
        """
        # initialize variables
        self.z = [np.ndarray] * len(self.sizes)
        self.a = [np.ndarray] * len(self.sizes)
        self.a[0] = X
        # strat the chain
        for i in range(1,len(self.sizes)):
            self.z[i] = self.a[i-1].dot(self.W[i])
            self.a[i] = self.sigmoid(self.z[i])
        return self.a[-1]

    def cost(self, X, y):
        y_hat = self.forward(X)
        sum_w = 0
        for i in range(1, len(self.W)):
            sum_w += np.sum(self.W[i]**2)
        penalty = 0.5 * self.lamda * sum_w
        return 0.5  * np.sum((y-y_hat)**2) / X.shape[0] + penalty

    def cost_prime(self, X, y):
        """
        Return the derivative
        """
        djdw = [np.ndarray] * len(self.sizes)
        y_hat = self.forward(X)
        # calculate deriv for output layer
        delta = (-(y-y_hat)) * self.sigmoid_p(self.z[-1])
        djdw[-1] = self.a[-2].T.dot(delta)/X.shape[0]+self.lamda*self.W[-1]
        # calculate for hidden layers
        for i in range(len(self.W)-2,0,-1):
            delta = delta.dot(self.W[i+1].T) * self.sigmoid_p(self.z[i])
            djdw[i] = self.a[i-1].T.dot(delta)/X.shape[0]+self.lamda*self.W[i]
        # return the derivatives
        return djdw

    def update_w(self, X, y):
        djdw = self.cost_prime(X,y)
        for i in range(1, len(self.W)):
            self.W[i] -= self.alpha * djdw[i]
        return djdw

    def get_params(self):
        params = self.W[1].ravel()
        for i in range(2, len(self.W)):
            params = np.concatenate((params, self.W[i].ravel()))

        return params

    def set_params(self, params):
        start = 0
        for i in range(1, len(self.sizes)):
            end = start + self.sizes[i-1] * self.sizes[i]
            self.W[i] = params[start:end].reshape(self.sizes[i-1],self.sizes[i])
            start = end

    def norm(self, x):
        return np.sqrt(np.sum(x**2))

    def compute_grad(self, X, y, new_djdw=None):
        djdw = self.cost_prime(X,y) if new_djdw is None else new_djdw
        grad = djdw[1].ravel()
        for i in range(2, len(djdw)):
            grad = np.concatenate((grad, djdw[i].ravel()))
        return grad

    def test_deriv(self, X, y):
        params_initial = self.get_params()
        numgrad = np.zeros(params_initial.shape)
        perturb = np.zeros(params_initial.shape)
        e = 1e-4

        for p in range(len(params_initial)):
            #Set perturbation vector
            perturb[p] = e
            self.set_params(params_initial + perturb)
            loss2 = self.cost(X, y)

            self.set_params(params_initial - perturb)
            loss1 = self.cost(X, y)

            #Compute Numerical Gradient
            numgrad[p] = (loss2 - loss1) / (2*e)

            #Return the value we changed to zero:
            perturb[p] = 0

        #Return Params to original value:
        self.set_params(params_initial)
        grad = self.compute_grad(X,y)
        return self.norm(grad - numgrad)/self.norm(grad + numgrad)

    def train(self, X, y, num, print_e=False):
        print("Started trainig on input of size:", X.shape[0], "for:", num, "iterations"); sys.stdout.flush()
        for i in range(num):
            djdw = self.update_w(X,y)
            if print_e and i%int(num/5)==0:
                print("Current cost:", self.cost(X,y))
                print("Deriv norm:", self.norm(self.compute_grad(X,y,djdw)))
                sys.stdout.flush()
