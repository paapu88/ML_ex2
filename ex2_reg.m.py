"""classify data to two sets using logistic regression

   polynomial features are added to get shape for the decision boundary

   but features are scaled down (penalized) to avoid overfitting

   Ng, machine learning, exercise 2

"""



import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.optimize import minimize

def sigmoid(z):
    """ calculate sigmoid function on x (being it scalar, vector or matrix)"""
    return (1/(1+np.exp(-z)))

def cost(theta, X, y, mylambda):
    import math
    m = X.shape[0]
    print("SHAPES", X.shape,theta.shape)
    print(sigmoid(np.dot(X, theta)))
    j = y.dot(np.log(sigmoid(np.dot(X, theta))))  + (1 - y).dot(np.log(1 - sigmoid(np.dot(X, theta)))) \
    - mylambda/2*theta.dot(theta)
    print("cost:", -j/m)
    if math.isnan(j):
        return 1e8
    else:
        return (-j/m)

def grad(theta, X, y, mylambda):
    m = X.shape[0]
    mygrad = (((sigmoid(np.dot(X, theta)) - y).dot(X)) + mylambda*theta) / m
    mygrad[0] = mygrad[0] - mylambda*theta[0]/m
    # print("grad:", ((sigmoid(np.dot(X, theta)) - y).dot(X))/m )
    return mygrad

def mapFeature(X1, X2):
    """
    # MAPFEATURE Feature mapping function to polynomial features
    #  MAPFEATURE(X1, X2) maps the two input features
    #   to quadratic features used in the regularization exercise.
    #   Returns a new feature array with more features, comprising of
    #   X1, X2, X1.^2, X2.^2, X1*X2, X1*X2.^2, etc..
    #   Inputs X1, X2 must be the same size
    """
    degree = 6
    X=[]
    for i in range(degree+1):
        for j in range(i+1):
            X.append(np.power(X1,(i-j))*np.power(X2,j))

    return np.column_stack((X))



## Load Data
#  The first two columns contains the exam scores and the third column
#  contains the label.
print('Plotting Data ...\n')
x1, x2, y1, mycolor, y = [], [], [], [], []
with open('ex2data2.txt','r') as f:
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
plt.xlabel('Microchip Test 1')
plt.ylabel('Microchip Test 2')

red_patch = mpatches.Patch(color='red', label='y=0')
blue_patch = mpatches.Patch(color='blue', label='y=1')
plt.legend(handles=[red_patch, blue_patch])
#plt.show()

rows = len(x1)
myones = np.ones(rows)
data = np.vstack((x1, x2)).transpose()
print(data.shape)
m, features = data.shape
features += 1
myones = np.ones((m, 1))
print(myones.shape)
X = np.concatenate((myones, data), 1)
print (X.shape)
X = mapFeature(X[:,1], X[:,2])
print(X.shape)

# Initialize fitting parameters
initial_theta = np.zeros(X.shape[1])

# Set regularization parameter lambda to 1
mylambda = 2

#Compute and display initial cost and gradient for regularized logistic regression
mycost = cost(initial_theta, X, y, mylambda)
print(mycost)
mygrad = grad(initial_theta, X, y, mylambda)
print("mygrad,", mygrad)

result = minimize(fun=cost, x0=initial_theta, args=(X,y,mylambda), jac=grad )
print(result)
theta = result.x

# show the decision boundary
from numpy.random import uniform, seed
from matplotlib.mlab import griddata
import matplotlib.pyplot as plt
import numpy as np
# make up data.
#npts = int(raw_input('enter # of random points to plot:'))
#seed(0)
npts = 100
xx1, xx2 = [], []
for ix in np.linspace(-1,1,npts):
    for iy in np.linspace(-1,1,npts):
        xx1.append(ix)
        xx2.append(iy)
xx1 = uniform(-1, 1, npts)
xx2 = uniform(-1, 1, npts)

#z = x*np.exp(-x**2 - y**2)
# define grid.


rows = len(xx1)
print("rows", rows)
myones = np.ones(rows)
data = np.vstack((xx1, xx2)).transpose()
print(data.shape)
m, features = data.shape
features += 1
myones = np.ones((m, 1))
print(myones.shape)
X = np.concatenate((myones, data), 1)
print (X.shape)
X = mapFeature(X[:,1], X[:,2])
print(X.shape)
Z = sigmoid(np.dot(X, theta))
print("Z:",Z.shape)

# grid the data.
xi = np.linspace(-1.1, 1.1, 100)
yi = np.linspace(-1.1, 1.1, 200)
zi = griddata(xx1, xx2, Z, xi, yi, interp='linear')
# contour the gridded data, plotting dots at the nonuniform data points.
CS = plt.contour(xi, yi, zi, levels = [0.5])
plt.clabel(CS, inline=1, fontsize=10)
#CS = plt.contour(xi, yi, zi, 15, linewidths=0.5, colors='k')
#CS = plt.contourf(xi, yi, zi, 15,
#                  vmax=abs(zi).max(), vmin=-abs(zi).max())
#plt.colorbar()  # draw colorbar
# plot data points.
plt.scatter(xx1, xx2, marker='o', s=5, zorder=10)
plt.xlim(-1, 1)
plt.ylim(-1, 1)
plt.title('griddata test (%d points)' % npts)
plt.show()