## Implementation of Regularized Logistic Regression
<p>This assignment involves implementing a classification model using the regularized logistic regression algorithm in Python. The LogisticRegression class includes methods to train the model, make predictions, compute the cost function, calculate gradients, and apply the sigmoid function. The project utilizes L2 regularization to improve model accuracy and prevent overfitting.</p>

## Methods and Explanations:

**Sigmoid Function (sigmoid)**
The sigmoid function is the core activation function that enables the model to make classification decisions. It maps the input z to a range between 0 and 1, providing probability predictions using the formula h(z) = 1 / (1 + exp(-z)).

**Cost Function (computeCost)**
computeCost calculates the logistic regression cost function, which evaluates the accuracy of the model’s predictions. A regularization term is added to prevent overfitting, thus improving the model’s generalization ability. The computeCost function returns the cost J(θ) of the model.

**Gradient Calculation (computeGradient)**
computeGradient calculates the gradient of the cost function and uses it for parameter updates in the gradient descent algorithm. The regularization term is applied to all parameters except the bias term (theta[0]).

**Model Training (fit)**
The fit function trains the model using the gradient descent algorithm on the training data. During training, computeGradient is used to update the theta parameters in each iteration. The training process stops when the gradient falls below a certain threshold (epsilon) or when the maximum number of iterations is reached.

**Prediction (predict)**
The predict function makes classification predictions for new data points after training. If the sigmoid function output is above 0.5, it classifies the observation as 1; otherwise, it classifies it as 0.

## Conclusion:
The regularized logistic regression algorithm implemented in this assignment demonstrates good performance and generalization ability on the training data. With regularization, the model avoids overfitting, achieving a more balanced learning process. Each function in the code is designed to cover all necessary steps for training and testing the model.

