import numpy as np
class LogisticRegression:

    def __init__(self, alpha = 0.01, regLambda=0.01, epsilon=0.0001, maxNumIters = 10000):

        self.alpha = alpha
        self.regLambda = regLambda
        self.epsilon = epsilon
        self.maxNumIters = maxNumIters


    def computeCost(self, theta, X, y, regLambda):
        '''
        Computes the objective function
        Arguments:
            theta is d-dimensional numpy vector
            X is a n-by-d numpy matrix
            y is an n-dimensional numpy vector
            regLambda is the scalar regularization constant
        Returns:
            a scalar value of the cost  ** make certain you're not returning a 1 x 1 matrix! **
        '''
        m = len(y)  # number of training examples
        h_theta = self.sigmoid(X @ theta)  # model predictions using sigmoid function
        cost = -(1 / m) * (y @ np.log(h_theta) + (1 - y) @ np.log(1 - h_theta))

        # Add regularization term (excluding theta[0] for bias term)
        reg_cost = (regLambda / (2 * m)) * np.sum(theta[1:] ** 2)

        return cost + reg_cost

    
    
    def computeGradient(self, theta, X, y, regLambda):
        '''
        Computes the gradient of the objective function
        Arguments:
            theta is d-dimensional numpy vector
            X is a n-by-d numpy matrix
            y is an n-dimensional numpy vector
            regLambda is the scalar regularization constant
        Returns:
            the gradient, an d-dimensional vector
        '''
        m = len(y)
        h_theta = self.sigmoid(X @ theta)
        grad = (1 / m) * (X.T @ (h_theta - y))
        grad[1:] += (regLambda/m) * theta[1:]

        return grad


    def fit(self, X, y):

        '''
        Trains the model
        Arguments:
            X is a n-by-d numpy matrix
            y is an n-dimensional numpy vector
        ** the d here is different from above! (due to augmentation) **
        '''
        X = np.c_[np.ones((X.shape[0], 1)), X]  # intercept term
        theta = np.zeros(X.shape[1])  #theta

        for i in range(self.maxNumIters):
            gradient = self.computeGradient(theta, X, y, self.regLambda)
            new_theta = theta - self.alpha * gradient  # Gradient descent

            # Check for convergence
            if np.linalg.norm(new_theta - theta) < self.epsilon:
                break
            theta = new_theta

        self.theta = theta



    def predict(self, X):
        '''
        Used the model to predict values for each instance in X
        Arguments:
            X is a n-by-d numpy matrix
        Returns:
            an n-dimensional numpy vector of the predictions, the output should be binary (use h_theta > .5)
        '''
        X = np.c_[np.ones((X.shape[0], 1)), X] #
        h_theta = self.sigmoid(X @ self.theta)
        return (h_theta >= 0.5).astype(int)

    def sigmoid(self, Z):
        '''
                Applies the sigmoid function on every element of Z
                Arguments:
                    Z can be a (n,) vector or (n , m) matrix
                Returns:
                    A vector/matrix, same shape with Z, that has the sigmoid function applied elementwise
                '''
        return 1/ (1+ np.exp(-Z))


