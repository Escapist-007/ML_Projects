#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from scipy.io import loadmat
from scipy.optimize import minimize
from sklearn import svm
import matplotlib.pyplot as plt


# In[2]:


def preprocess():
    """ 
     Input:
     Although this function doesn't have any input, you are required to load
     the MNIST data set from file 'mnist_all.mat'.

     Output:
     train_data: matrix of training set. Each row of train_data contains 
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     validation_data: matrix of training set. Each row of validation_data 
       contains feature vector of a image
     validation_label: vector of label corresponding to each image in the 
       training set
     test_data: matrix of training set. Each row of test_data contains 
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set
    """

    mat = loadmat('mnist_all.mat')  # loads the MAT object as a Dictionary

    n_feature = mat.get("train1").shape[1]
    n_sample = 0
    for i in range(10):
        n_sample = n_sample + mat.get("train" + str(i)).shape[0]
    n_validation = 1000
    n_train = n_sample - 10 * n_validation

    # Construct validation data
    validation_data = np.zeros((10 * n_validation, n_feature))
    for i in range(10):
        validation_data[i * n_validation:(i + 1) * n_validation, :] = mat.get("train" + str(i))[0:n_validation, :]

    # Construct validation label
    validation_label = np.ones((10 * n_validation, 1))
    for i in range(10):
        validation_label[i * n_validation:(i + 1) * n_validation, :] = i * np.ones((n_validation, 1))

    # Construct training data and label
    train_data = np.zeros((n_train, n_feature))
    train_label = np.zeros((n_train, 1))
    temp = 0
    for i in range(10):
        size_i = mat.get("train" + str(i)).shape[0]
        train_data[temp:temp + size_i - n_validation, :] = mat.get("train" + str(i))[n_validation:size_i, :]
        train_label[temp:temp + size_i - n_validation, :] = i * np.ones((size_i - n_validation, 1))
        temp = temp + size_i - n_validation

    # Construct test data and label
    n_test = 0
    for i in range(10):
        n_test = n_test + mat.get("test" + str(i)).shape[0]
    test_data = np.zeros((n_test, n_feature))
    test_label = np.zeros((n_test, 1))
    temp = 0
    for i in range(10):
        size_i = mat.get("test" + str(i)).shape[0]
        test_data[temp:temp + size_i, :] = mat.get("test" + str(i))
        test_label[temp:temp + size_i, :] = i * np.ones((size_i, 1))
        temp = temp + size_i

    # Delete features which don't provide any useful information for classifiers
    sigma = np.std(train_data, axis=0)
    index = np.array([])
    for i in range(n_feature):
        if (sigma[i] > 0.001):
            index = np.append(index, [i])
    train_data = train_data[:, index.astype(int)]
    validation_data = validation_data[:, index.astype(int)]
    test_data = test_data[:, index.astype(int)]

    # Scale data to 0 and 1
    train_data /= 255.0
    validation_data /= 255.0
    test_data /= 255.0

    return train_data, train_label, validation_data, validation_label, test_data, test_label


# In[3]:


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


# In[4]:


def blrObjFunction(initialWeights, *args):
    """
    blrObjFunction computes 2-class Logistic Regression error function and
    its gradient.

    Input:
        initialWeights: the weight vector (w_k) of size (D + 1) x 1 
        train_data: the data matrix of size N x D
        labeli: the label vector (y_k) of size N x 1 where each entry can be either 0 or 1 representing the label of corresponding feature vector

    Output: 
        error: the scalar value of error function of 2-class logistic regression
        error_grad: the vector of size (D+1) x 1 representing the gradient of
                    error function
    """
    train_data, labeli = args

    n_data = train_data.shape[0]
    n_features = train_data.shape[1]
    error = 0
    error_grad = np.zeros((n_features + 1, 1))

    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data
    
    # Adding biases
    biases = np.full((n_data,1),1)
    X = np.concatenate((biases,train_data),axis=1)
    
    newRow = n_features + 1
    col = 1
    w = initialWeights.reshape(newRow,col)
    
    theta_value = sigmoid(np.dot(X,w))
    
    part1 = labeli * np.log(theta_value)
    part2 = (1.0 - labeli) * np.log(1.0 - theta_value)
    
    error_func = part1 + part2
    
    error = (- 1.0 * np.sum(error_func) ) / n_data
    
    error_grad = np.sum((theta_value - labeli)*X, axis=0) / n_data
    

    return error, error_grad


# In[5]:


def blrPredict(W, data):
    """
     blrObjFunction predicts the label of data given the data and parameter W 
     of Logistic Regression
     
     Input:
         W: the matrix of weight of size (D + 1) x 10. Each column is the weight 
         vector of a Logistic Regression classifier.
         X: the data matrix of size N x D
         
     Output: 
         label: vector of size N x 1 representing the predicted label of 
         corresponding feature vector given in data matrix

    """
    label = np.zeros((data.shape[0], 1))

    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data
    
    N = data.shape[0]
    D = data.shape[1]
    
    # Adding biases
    biases = np.full((N,1),1)
    X = np.concatenate((biases,data),axis=1)
    
    prob = sigmoid(np.dot(X,W))
    
    l = np.argmax(prob,axis=1)
    
    label = l.reshape((N,1)) 

    return label


# In[6]:


def mlrObjFunction(params, *args):
    """
    mlrObjFunction computes multi-class Logistic Regression error function and
    its gradient.

    Input:
        initialWeights_b: the weight vector of size (D + 1) x 10
        train_data: the data matrix of size N x D
        labeli: the label vector of size N x 1 where each entry can be either 0 or 1
                representing the label of corresponding feature vector

    Output:
        error: the scalar value of error function of multi-class logistic regression
        error_grad: the vector of size (D+1) x 10 representing the gradient of
                    error function
    """
    n_class = 10
    train_data, labeli = args
    n_data    = train_data.shape[0]
    n_feature = train_data.shape[1]
    error = 0
    error_grad = np.zeros((n_feature + 1, n_class))

    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data
    
    # Adding biases
    biases = np.full((n_data,1),1)
    X      = np.concatenate((biases,train_data), axis=1)
    
    W       = params.reshape((n_feature + 1, n_class))
    
    
    upper = np.exp(np.dot(X,W))
        
    below = np.sum(upper,axis=1)
    below = below.reshape(below.shape[0],1)
    
    theta_value = upper/below
   
    innerSum = np.sum(Y*np.log(theta_value))

    error = - (np.sum(innerSum))
    
    error_grad = np.dot(X.T,(theta_value - labeli))
    error_grad = error_grad.ravel()
    
    return error, error_grad


# In[7]:


def mlrPredict(W, data):
    """
     mlrObjFunction predicts the label of data given the data and parameter W
     of Logistic Regression

     Input:
         W: the matrix of weight of size (D + 1) x 10. Each column is the weight
         vector of a Logistic Regression classifier.
         X: the data matrix of size N x D

     Output:
         label: vector of size N x 1 representing the predicted label of
         corresponding feature vector given in data matrix

    """
    label = np.zeros((data.shape[0], 1))
    row   = data.shape[0]
    
    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data
    
    # Adding biases
    biases = np.full((row,1),1)
    X      = np.concatenate((biases,data), axis=1)

    t = np.sum(np.exp(np.dot(X,W)),axis=1)    
    t = t.reshape(t.shape[0],1)
    
    theta_value = np.exp(np.dot(X,W))/t
    
    label = np.argmax(theta_value,axis=1)
    label = label.reshape(row,1)
    
    return label


# In[9]:


"""
Script for Logistic Regression
"""
train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()

# number of classes
n_class = 10

# number of training samples
n_train = train_data.shape[0]

# number of features
n_feature = train_data.shape[1]

Y = np.zeros((n_train, n_class))
for i in range(n_class):
    Y[:, i] = (train_label == i).astype(int).ravel()

# Logistic Regression with Gradient Descent
W = np.zeros((n_feature + 1, n_class))
initialWeights = np.zeros((n_feature + 1, 1))
opts = {'maxiter': 100}
for i in range(n_class):
    labeli = Y[:, i].reshape(n_train, 1)
    args = (train_data, labeli)
    nn_params = minimize(blrObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)
    W[:, i] = nn_params.x.reshape((n_feature + 1,))

# Find the accuracy on Training Dataset
predicted_label = blrPredict(W, train_data)
print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label == train_label).astype(float))) + '%')

# Find the accuracy on Validation Dataset
predicted_label = blrPredict(W, validation_data)
print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label == validation_label).astype(float))) + '%')

# Find the accuracy on Testing Dataset
predicted_label = blrPredict(W, test_data)
print('\n Testing set Accuracy:' + str(100 * np.mean((predicted_label == test_label).astype(float))) + '%')

"""
Script for Support Vector Machine
"""

print('\n\n--------------SVM-------------------\n\n')
##################
# YOUR CODE HERE #
##################

# Random Selection of Samples
index = np.random.randint(50000, size = 10000)
trsvm_data = train_data[index,:]
trsvm_label = train_label[index,:]

# Linear
linear_mod = svm.SVC(kernel='linear')
linear_mod.fit(trsvm_data, trsvm_label)

print('\n---------Linear Kernel---------\n')
print('\n Training Accuracy -->' + str(100 * linear_mod.score(train_data, train_label)) + '%')
print('\n Validation Accuracy -->' + str(100 * linear_mod.score(validation_data, validation_label)) + '%')
print('\n Testing Accuracy -->' + str(100 * linear_mod.score(test_data, test_label)) + '%')

# RBF with Gamma = 1

rbf_mod = svm.SVC(kernel='rbf', gamma = 1.0)
rbf_mod.fit(trsvm_data, trsvm_label)

print('\n---------RBF Kernel with Gamma = 1---------\n')
print('\n Training Accuracy -->' + str(100 * rbf_mod.score(trsvm_data, trsvm_label)) + '%')
print('\n Validation Accuracy -->' + str(100 * rbf_mod.score(validation_data, validation_label)) + '%')
print('\n Testing Accuracy -->' + str(100 * rbf_mod.score(test_data, test_label)) + '%')

# RBF with default gamma(0.0)

rbf_mod1 = svm.SVC(kernel='rbf', gamma = 'auto')
rbf_mod1.fit(trsvm_data, trsvm_label)

print('\n---------RBF Kernel with Gamma = default---------\n')
print('\n Training Accuracy -->' + str(100 * rbf_mod1.score(train_data, train_label)) + '%')
print('\n Validation Accuracy -->' + str(100 * rbf_mod1.score(validation_data, validation_label)) + '%')
print('\n Testing Accuracy -->' + str(100 * rbf_mod1.score(test_data, test_label)) + '%')

# RBF with default gamma and changing C 

accuracy = np.zeros((11,3), float)
C_values = np.array([1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
inpt = 0

# iterating C values
for c in C_values:
    print("C Value: \n", c)
    rbf_mod2 = svm.SVC(kernel = 'rbf', C = c)
    rbf_mod2.fit(trsvm_data, trsvm_label.ravel())
    if inpt <= 10: 
        accuracy[inpt][0] = 100 * rbf_mod2.score(train_data, train_label)
        accuracy[inpt][1] = 100 * rbf_mod2.score(validation_data, validation_label)
        accuracy[inpt][2] = 100 * rbf_mod2.score(test_data, test_label)
        
        print('\n---------RBF Kernel with Gamma = default and C = '+ str(c) +'---------\n')
        print('\n Training Accuracy -->' + str(accuracy[inpt][0]) + '%')
        print('\n Validation Accuracy -->' + str(accuracy[inpt][1]) + '%')
        print('\n Testing Accuracy -->' + str(accuracy[inpt][2]) + '%')
        
    inpt = inpt + 1
 
'''
Figure and Title   
''' 
plt.figure(figsize=(16,12))
plt.title('Accuracy vs C',pad=10,fontsize=20,fontweight = 'bold')

plt.xlabel('Value of C', labelpad=20, weight='bold', size=15)
plt.ylabel('Accuracy',   labelpad=20, weight='bold', size=15)

plt.xticks( np.array([1,  10,  20,  30,  40,  50,  60,  70,  80,  90, 100]), fontsize=15)
plt.yticks( np.arange(85,100, step=0.5),  fontsize=15)


plt.plot(C_values, accuracy[:,0], color='g')
plt.plot(C_values, accuracy[:,1], color='b')
plt.plot(C_values, accuracy[:,2], color='r')

plt.legend(['Training_Data','Validation_Data','Test_Data'])

## running on complete set

rbf_model_full = svm.SVC(kernel = 'rbf', gamma = 'auto', C = 70)
rbf_model_full.fit(train_data, train_label.ravel())

print('----------\n RBF with FULL training set with best C : \n------------')
print('\n Training Accuracy:' + str(100 * rbf_model_full.score(train_data, train_label)) + '%')
print('\n Validation Accuracy:' + str(100 * rbf_model_full.score(validation_data, validation_label)) + '%')
print('\n Testing Accuracy:' + str(100 * rbf_model_full.score(test_data, test_label)) + '%')
 

"""
Script for Extra Credit Part
"""
# FOR EXTRA CREDIT ONLY
W_b = np.zeros((n_feature + 1, n_class))
initialWeights_b = np.zeros((n_feature + 1, n_class))
opts_b = {'maxiter': 100}

args_b = (train_data, Y)
nn_params = minimize(mlrObjFunction, initialWeights_b, jac=True, args=args_b, method='CG', options=opts_b)
W_b = nn_params.x.reshape((n_feature + 1, n_class))

# Find the accuracy on Training Dataset
predicted_label_b = mlrPredict(W_b, train_data)
print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label_b == train_label).astype(float))) + '%')

# Find the accuracy on Validation Dataset
predicted_label_b = mlrPredict(W_b, validation_data)
print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label_b == validation_label).astype(float))) + '%')

# Find the accuracy on Testing Dataset
predicted_label_b = mlrPredict(W_b, test_data)
print('\n Testing set Accuracy:' + str(100 * np.mean((predicted_label_b == test_label).astype(float))) + '%')


# In[ ]:




