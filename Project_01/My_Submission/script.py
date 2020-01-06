import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from numpy.linalg import det, inv
from math import sqrt, pi
import scipy.io
import matplotlib.pyplot as plt
import pickle
import sys

# Done

def ldaLearn(X,y):
    '''
      Inputs
         X - a N x d matrix with each row corresponding to a training example
         y - a N x 1 column vector indicating the labels for each training example
    
      Outputs
        means - A d x k matrix containing learnt means for each of the k classes
        covmat - A single d x d learnt covariance matrix 
    '''
   
    labels = np.unique(y)
  
    total_label   = labels.shape[0]
    total_feature = X.shape[1]
    
    means  = np.zeros([total_label,total_feature])

    r = 0
    for i in labels:
        data = X[np.where(y == i)[0],]
        m = np.mean(data,axis=0)
        means[r,] = m
        r +=1
    
    X_transpose = np.transpose(X)
    
    covmat = np.cov(X_transpose)
    return means,covmat


# Done

def qdaLearn(X,y):
    '''
     Inputs
         X - a N x d matrix with each row corresponding to a training example
         y - a N x 1 column vector indicating the labels for each training example
    
     Outputs
         means - A d x k matrix containing learnt means for each of the k classes
         covmats - A list of k d x d learnt covariance matrices for each of the k classes
    '''
   
    
    # IMPLEMENT THIS METHOD
    covmats = []
    labels = np.unique(y)
    
    total_label   = labels.shape[0]
    total_feature = X.shape[1]
    
    means  = np.zeros([total_label,total_feature])
     
    r = 0
    for i in labels:
        data = X[np.where(y == i)[0],]
        m = np.mean(data,axis=0)
        means[r,] = m
        r +=1
        data_transpose = np.transpose(data)
        covmats.append(np.cov(data_transpose))
        
    return means,covmats



# Done

def ldaTest(means,covmat,Xtest,ytest):
    r = Xtest.shape[0]
    c = means.shape[0]
    res = np.zeros((r,c))
    f = 1/np.sqrt((2*pi)**means.shape[1]*det(covmat))
    for j in range(means.shape[0]):
        res[:,j] = f * np.exp(-0.5*np.array([np.dot(np.dot((Xtest[i,:] - means[j,:]),inv(covmat)),np.transpose(Xtest[i,:] - means[j,:])) for i in range(Xtest.shape[0])]))

    ypred = np.argmax(res,axis=1) + 1
    res = (ypred == ytest.ravel())
    acc_data = np.where(res)[0]
    acc = len(acc_data)
    return float(acc)/len(ytest),ypred


# Done

def qdaTest(means,covmats,Xtest,ytest):
    
    res = np.zeros((Xtest.shape[0],means.shape[0]))
    for j in range(means.shape[0]):
        f = 1/np.sqrt((2*pi)**means.shape[1]*det(covmats[j]))
        res[:,j] = f * np.exp(-0.5*np.array([np.dot(np.dot((Xtest[i,:] - means[j,:]),inv(covmats[j])),np.transpose(Xtest[i,:] - means[j,:])) for i in range(Xtest.shape[0])]))
    ypred = np.argmax(res,axis=1) + 1
    res = (ypred == ytest.ravel())
    acc = len(np.where(res)[0])
    return float(acc)/len(ytest),ypred



# Done

def learnOLERegression(X,y):
    '''
    Inputs:                                                         
          X = N x d  (Input data matrix for training)
          y = N x 1  (Target vector for training)                                                            
    Output: 
          w = d x 1   (Learned weight vector)
    '''
    
    # The formula for learning w in OLE : w = Inverse((Xtranspose * X)) * Xtranspose * y

    X_transpose   = np.transpose(X)
    X_X_transpose = np.dot(X_transpose,X)
    Inverse_X_X_transpose = np.linalg.inv(X_X_transpose) 
    
    w = np.dot(np.dot(Inverse_X_X_transpose,X_transpose),y)
    
    return w


# Done

def learnRidgeRegression(X,y,lambd):
    '''
    Inputs:                                                         
          X = N x d  (Input data matrix for training)
          y = N x 1  (Target vector for training)
          lambd = ridge parameter (scalar)
    Output: 
          w = d x 1   (Learned weight vector)
    '''
    # The formula for learning w in Ridge Regression : 
    #    w = Inverse(( Lamda* Identity(d)) + Xtranspose * X) * Xtranspose * y
    
    I = np.identity(X.shape[1])
    lambd_I = np.dot(lambd,I)
    
    X_transpose   = np.transpose(X)
    X_X_transpose = np.dot(X_transpose,X)
    
    Inverse_part = np.linalg.inv(lambd_I + X_X_transpose) 
    
    w = np.dot(np.dot(Inverse_part,X_transpose),y)                                         
    return w


# Done

def testOLERegression(w,Xtest,ytest):
    '''
    Inputs:
        w = d x 1
        Xtest = N x d
        ytest = X x 1
    Output:
        mse
    '''
    y_predict = np.dot(Xtest,w)
    
    diff = (ytest - y_predict)    
    diff_transpose = np.transpose(diff)
    N = 1 /len(Xtest)
    
    mse = np.dot( np.dot(N,diff_transpose), diff )
    
    return mse


# Done

def regressionObjVal(w, X, y, lambd):
    '''
      compute squared error (scalar) and gradient of squared error with respect 
      to w (vector) for the given data X and y and the regularization parameter lambda
      
    '''   
    # The formula for learning w in Ridge Regression using Gradient Descent :
    #     XTranspose * ( y - Xw)
    
    w_tranpose    = np.asmatrix(w).transpose()
    X_w_tranpose  = np.dot(X,w_tranpose)
    
    diff = (y - X_w_tranpose)
    
    diff_transpose = np.transpose(diff)
    
    diff_diff    = (np.dot(diff_transpose,diff))  
    w_w_tranpose = np.dot(np.asmatrix(w),w_tranpose)
    
   
    error = 0.5*(diff_diff + lambd*w_w_tranpose)
    
    error_grad = -(np.dot(np.transpose(X),diff)) + lambd*w_tranpose
    error_grad = np.squeeze(np.array(error_grad))

    return error, error_grad


# Done
def mapNonLinear(x,p):
    '''
    Inputs:                                                                  
        x - a single column vector (N x 1)                                       
        p - integer (>= 0)                                                       
    Outputs:                                                                 
        Xp - (N x (p+1)) 
    
    '''
    Xp = np.zeros((x.shape[0],p+1))
    for i in range(p+1):
        Xp[:,i] = pow(x,i)
    return Xp


        # Main script

    
# Problem 1

if sys.version_info.major == 2:
    X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'))
else:
    X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'),encoding = 'latin1')

# LDA
means,covmat = ldaLearn(X,y)
ldaacc,ldares = ldaTest(means,covmat,Xtest,ytest)
print('LDA Accuracy = '+str(ldaacc))

# QDA
means,covmats = qdaLearn(X,y)
qdaacc,qdares = qdaTest(means,covmats,Xtest,ytest)
print('QDA Accuracy = '+str(qdaacc))

# plotting boundaries
x1 = np.linspace(-5,20,100)
x2 = np.linspace(-5,20,100)
xx1,xx2 = np.meshgrid(x1,x2)
xx = np.zeros((x1.shape[0]*x2.shape[0],2))
xx[:,0] = xx1.ravel()
xx[:,1] = xx2.ravel()

fig = plt.figure(figsize=[18,9])
plt.subplot(1, 2, 1)

zacc,zldares = ldaTest(means,covmat,xx,np.zeros((xx.shape[0],1)))
plt.contourf(x1,x2,zldares.reshape((x1.shape[0],x2.shape[0])),alpha=0.3)
plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest[:,0])
plt.title('LDA')
plt.subplot(1, 2, 2)

zacc,zqdares = qdaTest(means,covmats,xx,np.zeros((xx.shape[0],1)))
plt.contourf(x1,x2,zqdares.reshape((x1.shape[0],x2.shape[0])),alpha=0.3)
plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest[:,0])
plt.title('QDA')



# Problem 2

if sys.version_info.major == 2:
    X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'))
else:
    X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'),encoding = 'latin1')

# add intercept
X_i = np.concatenate((np.ones((X.shape[0],1)), X), axis=1)
Xtest_i = np.concatenate((np.ones((Xtest.shape[0],1)), Xtest), axis=1)

w = learnOLERegression(X,y)
mle = testOLERegression(w,Xtest,ytest)

w_i = learnOLERegression(X_i,y)
mle_i = testOLERegression(w_i,Xtest_i,ytest)

print('MSE without intercept '+str(mle))
print('MSE with intercept '+str(mle_i))




# Problem 3

k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
mses3_train = np.zeros((k,1))
mses3 = np.zeros((k,1))
for lambd in lambdas:
    w_l = learnRidgeRegression(X_i,y,lambd)
    mses3_train[i] = testOLERegression(w_l,X_i,y)
    mses3[i] = testOLERegression(w_l,Xtest_i,ytest)
    i = i + 1
fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(lambdas,mses3_train)
plt.title('MSE for Train Data')
plt.subplot(1, 2, 2)
plt.plot(lambdas,mses3)
plt.title('MSE for Test Data')

plt.show()


# Problem 4

k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
mses4_train = np.zeros((k,1))
mses4 = np.zeros((k,1))
opts = {'maxiter' : 20}    # Preferred value.                                                
w_init = np.ones((X_i.shape[1],1))
for lambd in lambdas:
    args = (X_i, y, lambd)
    w_l = minimize(regressionObjVal, w_init, jac=True, args=args,method='CG', options=opts)
    w_l = np.transpose(np.array(w_l.x))
    w_l = np.reshape(w_l,[len(w_l),1])
    mses4_train[i] = testOLERegression(w_l,X_i,y)
    mses4[i] = testOLERegression(w_l,Xtest_i,ytest)
    i = i + 1
fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(lambdas,mses4_train)
plt.plot(lambdas,mses3_train)
plt.title('MSE for Train Data')
plt.legend(['Using scipy.minimize','Direct minimization'])

plt.subplot(1, 2, 2)
plt.plot(lambdas,mses4)
plt.plot(lambdas,mses3)
plt.title('MSE for Test Data')
plt.legend(['Using scipy.minimize','Direct minimization'])
plt.show()



# Problem 5

pmax = 7
lambda_opt = 0.06 # REPLACE THIS WITH lambda_opt estimated from Problem 3
mses5_train = np.zeros((pmax,2))
mses5 = np.zeros((pmax,2))
for p in range(pmax):
    Xd = mapNonLinear(X[:,2],p)
    Xdtest = mapNonLinear(Xtest[:,2],p)
    w_d1 = learnRidgeRegression(Xd,y,0)
    mses5_train[p,0] = testOLERegression(w_d1,Xd,y)
    mses5[p,0] = testOLERegression(w_d1,Xdtest,ytest)
    w_d2 = learnRidgeRegression(Xd,y,lambda_opt)
    mses5_train[p,1] = testOLERegression(w_d2,Xd,y)
    mses5[p,1] = testOLERegression(w_d2,Xdtest,ytest)

fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(range(pmax),mses5_train)
plt.title('MSE for Train Data')
plt.legend(('No Regularization','Regularization'))
plt.subplot(1, 2, 2)
plt.plot(range(pmax),mses5)
plt.title('MSE for Test Data')
plt.legend(('No Regularization','Regularization'))
plt.show()
