# Machine Learning Online Class - Exercise 2: Logistic Regression
#
#  Instructions
#  ------------
#
#  This file contains code that helps you get started on the logistic
#  regression exercise. You will need to complete the following functions
#  in this exericse:
#
#     sigmoid.m
#     costFunction.m
#     predict.m
#     costFunctionReg.m
#
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.
#

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.optimize import minimize
from scipy import optimize
import numpy as np


def sigmoid(z):
    """ calculate sigmoid function on x (being it scalar, vector or matrix)"""
    return (1/(1+np.exp(-z)))

def cost(theta, X, y):
    import math
    m = X.shape[0]
    j = y.dot(np.log(sigmoid(np.dot(X, theta))))  + (1 - y).dot(np.log(1 - sigmoid(np.dot(X, theta))))
    print("cost:", -j/m)
    if math.isnan(j):
        return 1e8
    else:
        return (-j/m)


def grad(theta, X, y):
    m = X.shape[0]
    print("grad:", ((sigmoid(np.dot(X, theta)) - y).dot(X))/m )
    return ((sigmoid(np.dot(X, theta)) - y).dot(X))/m


## Load Data
#  The first two columns contains the exam scores and the third column
#  contains the label.
print('Plotting Data ...\n')
x1, x2, y1, mycolor, y = [], [], [], [], []
with open('ex2data1.txt','r') as f:
    for l in f:
        row = l.split(',')
        x1.append(float(row[0])/1)
        x2.append(float(row[1])/1)
        y.append(float(row[2]))
        if row[2].startswith('0'):
            y1.append('o')
            mycolor.append('red')
        else:
            y1.append('x')
            mycolor.append('blue')

y = np.asarray(y)

for xx,yy,marker,col in zip(x1, x2, y1, mycolor):
    plt.scatter(xx, yy, marker=marker, color=col)

# Put some labels
plt.xlabel('Exam 1 score')
plt.ylabel('Exam 2 score')

red_patch = mpatches.Patch(color='red', label='Not admitted')
blue_patch = mpatches.Patch(color='blue', label='Admitted')
plt.legend(handles=[red_patch, blue_patch])
#plt.show()



## ============ Part 2: Compute Cost and Gradient ============
#  In this part of the exercise, you will implement the cost and gradient
#  for logistic regression. You neeed to complete the code in
#  costFunction.m

#  Setup the data matrix appropriately, and add ones for the intercept term
#[m, n] = size(X);

# Add intercept term to x and X_test
rows = len(x1)
myones = np.ones(rows)
#print(myones.transpose)
#print(x.transpose)
#data = np.concatenate([x1], [x2], axis=0)
data = np.vstack((x1, x2)).transpose()
print(data.shape)
m, features = data.shape
features += 1
myones = np.ones((m, 1))
print(myones.shape)
X = np.concatenate((myones, data), 1)
print (X.shape)

initial_theta = np.zeros((features))

# Compute and display initial cost and gradient
print (sigmoid(initial_theta))
print ("cost:", cost(initial_theta, X, y))
print ("gradient:", grad(initial_theta, X, y))


## ============= Part 3: Optimizing using fminunc  =============
#  In this exercise, you will use a built-in function (fminunc) to find the
#  optimal parameters theta.

theta = initial_theta
print("theta;", theta.shape, X.shape, y.shape)
result = minimize(fun=cost, x0=theta, args=(X,y), jac=grad )
print(result.x)

#print("CALLING")
#res = optimize.fmin_cg(f=costFunction(theta, x, y), x0=theta, args=(x,y), fprime=gradient(theta, x, y))

# Print theta to screen
print('Cost at theta found by fminunc: ', cost(result.x,X,y))
print('Expected cost (approx): 0.203')
print('theta: ', result.x)
print('Expected theta (approx):')
print(' -25.161 0.206  0.201')

# Plot Boundary

#plotDecisionBoundary(theta, X, y);

# Put some labels
#hold on;
# Labels and Legend
#xlabel('Exam 1 score')
#ylabel('Exam 2 score')

# Specified in plot order
#legend('Admitted', 'Not admitted')
#hold off;

#fprintf('\nProgram paused. Press enter to continue.\n');
#pause;

## ============== Part 4: Predict and Accuracies ==============
#  After learning the parameters, you'll like to use it to predict the outcomes
#  on unseen data. In this part, you will use the logistic regression model
#  to predict the probability that a student with score 45 on exam 1 and
#  score 85 on exam 2 will be admitted.
#
#  Furthermore, you will compute the training and test set accuracies of
#  our model.
#
#  Your task is to complete the code in predict.m

#  Predict probability for a student with score 45 on exam 1
#  and score 85 on exam 2
theta=result.x
print("arg: ", np.array([1, 45, 85]).dot(theta))
prob = sigmoid(np.array([1, 45, 85]).dot(theta))
print('For a student with scores 45 and 85, we predict an admission probability of ', prob)
#fprintf('Expected value: 0.775 +/- 0.002\n\n');

# Compute accuracy on our training set
#p = predict(theta, X);

#fprintf('Train Accuracy: #f\n', mean(double(p == y)) * 100);
#fprintf('Expected accuracy (approx): 89.0\n');
#fprintf('\n');


