import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt
import time
import pickle
import matplotlib.pyplot as plt
import pandas as pd




def initializeWeights(n_in, n_out):
    epsilon = sqrt(6) / sqrt(n_in + n_out + 1)
    W = (np.random.rand(n_out, n_in + 1) * 2 * epsilon) - epsilon
    return W




def sigmoid(z):
    return (1.0 / (1.0 + np.exp(-z)))



def sigmoid_derivative(z):
    sigm = 1.0 / (1.0 + np.exp(-z))
    return sigm * (1.0 - sigm)
    



def feature_indices(boolean_value):
    
    featureCount = 0
    global featureIndices
    
    for i in range(len(boolean_value)):
        if boolean_value[i]==False:
            featureCount += 1
            featureIndices.append(i)
            print(i,end =" ")
    print(" ")
    print("Total number of selected features : ", featureCount)




def preprocess():
    
    # loads the MAT object as a Dictionary
    mnist = loadmat('mnist_all.mat') 

    # Split the training sets into two sets of 50000 randomly sampled training examples & 10000 validation examples. 
    
                    ############## TRAIN DATA ############
    tmp = []
    for i in range(10):
        idx = 'train'+ str(i)
        train_mat = mnist[idx]
        labels = np.full((train_mat.shape[0],1),i)
        labeled_train_mat = np.concatenate((train_mat,labels),axis=1)
        tmp.append(labeled_train_mat)

    all_labeled_train = np.concatenate((tmp[0],tmp[1],tmp[2],tmp[3],tmp[4],tmp[5],tmp[6],tmp[7],tmp[8],tmp[9]), axis=0)
    
    np.random.shuffle(all_labeled_train)
    
    labeled_train = all_labeled_train[0:50000,:]
    train_data    = labeled_train[:,0:784]
    train_label   = labeled_train[:,784]

    train_data = train_data / 255.0

    labeled_validation = all_labeled_train[50000:60000,:]
    validation_data    = labeled_validation[:,0:784] 
    validation_label   = labeled_validation[:,784]

    validation_data = validation_data / 255.0  
    
                ############## TEST DATA ############
    tmp1 = []
    for i in range(10):
        idx = 'test'+ str(i)
        test_mat = mnist[idx]
        labels = np.full((test_mat.shape[0],1),i)
        labeled_test_mat = np.concatenate((test_mat,labels),axis=1)
        tmp1.append(labeled_test_mat)

    all_labeled_test = np.concatenate((tmp1[0],tmp1[1],tmp1[2],tmp1[3],tmp1[4],tmp1[5],tmp1[6],tmp1[7],tmp1[8],tmp1[9]), axis=0)

    np.random.shuffle(all_labeled_test)
    
    test_data    = all_labeled_test[:,0:784]
    test_label   = all_labeled_test[:,784]

    test_data = test_data / 255.0

    # Feature selection
    
    combined  = np.concatenate((train_data, validation_data),axis=0)
    reference = combined[0,:]
    boolean_value_columns = np.all(combined == reference, axis = 0)
    
    # Print the selected features
    feature_indices(boolean_value_columns)
    
    final = combined[:,~boolean_value_columns]
    
    tr_R = train_data.shape[0]
    vl_R = validation_data.shape[0]

    
    train_data      = final[0:tr_R,:]
    validation_data = final[tr_R:,:]
    test_data = test_data[:,~boolean_value_columns]
    
    return train_data, train_label, validation_data, validation_label, test_data, test_label
    


def nnObjFunction(params, *args):
    
    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args

    w1 = params[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    
    obj_val = 0
    n = training_data.shape[0]
    ''' 
                                Step 01: Feedforward Propagation 
    '''
    
    '''Input Layer --> Hidden Layer
    '''
    # Adding bias node to every training data. Here, the bias value is 1 for every training data
    # A training data is a feature vector X. 
    # We have 717 features for every training data

    biases1 = np.full((n,1), 1)
    training_data_bias = np.concatenate((biases1, training_data),axis=1)
    
    # aj is the linear combination of input data and weight (w1) at jth hidden node. 
    # Here, 1 <= j <= no_of_hidden_units
    aj = np.dot( training_data_bias, np.transpose(w1))
    
    # zj is the output from the hidden unit j after applying sigmoid as an activation function
    zj = sigmoid(aj)
    
    '''Hidden Layer --> Output Layer
    '''
    
    # Adding bias node to every zj. 
    
    m = zj.shape[0]
    
    biases2 = np.full((m,1), 1)
    zj_bias = np.concatenate((biases2, zj), axis=1)
    
    # bl is the linear combination of hidden units output and weight(w2) at lth output node. 
    # Here, l = 10 as we are classifying 10 digits
    bl = np.dot(zj_bias, np.transpose(w2))
    ol = sigmoid(bl)
    
    ''' 
                            Step 2:  Error Calculation by error function
    '''
    # yl --> Ground truth for every training dataset
    yl = np.full((n, n_class), 0)

    for i in range(n):
        trueLabel = training_label[i]
        yl[i][trueLabel] = 1
    
    yl_prime = (1.0-yl)
    ol_prime = (1.0-ol)
    
    lol = np.log(ol)
    lol_prime = np.log(ol_prime)
    
    # Our Error function is "negative log-likelihood"
    # We need elementwise multiplication between the matrices
    
    error = np.sum( np.multiply(yl,lol) + np.multiply(yl_prime,lol_prime) )/((-1)*n)

#     error = -np.sum( np.sum(yl*lol + yl_prime*lol_prime, 1))/ n
    
    ''' 
                         Step 03: Gradient Calculation for Backpropagation of error
    '''
    
    delta = ol- yl
    gradient_w2 = np.dot(delta.T, zj_bias)
   
    temp = np.dot(delta,w2) * ( zj_bias * (1.0-zj_bias))
    
    gradient_w1 = np.dot( np.transpose(temp), training_data_bias)
    gradient_w1 = gradient_w1[1:, :]
    
    ''' 
                                Step 04: Regularization 
    '''
    regularization =  lambdaval * (np.sum(w1**2) + np.sum(w2**2)) / (2*n)
    obj_val = error + regularization
    
    gradient_w1_reg = (gradient_w1 + lambdaval * w1)/n
    gradient_w2_reg = (gradient_w2 + lambdaval * w2)/n

    obj_grad = np.concatenate((gradient_w1_reg.flatten(), gradient_w2_reg.flatten()), 0)

    return (obj_val, obj_grad)




def nnPredict(w1, w2, training_data):

    n = training_data.shape[0]

    biases1 = np.full((n,1),1)
    training_data = np.concatenate((biases1, training_data), axis=1)

    aj = np.dot(training_data, w1.T)
    zj = sigmoid(aj)
    
    m = zj.shape[0]
    
    biases2 = np.full((m,1), 1)
    zj = np.concatenate((biases2, zj), axis=1)

    bl = np.dot(zj, w2.T)
    ol = sigmoid(bl)

    labels = np.argmax(ol, axis=1)

    return labels




featureIndices=[]

train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()

# Store values for all iterations
totalTime = []

train_accuracy=[]
validation_accuracy=[]
test_accuracy=[]

l = []
m = []


n_input = train_data.shape[1]
n_class = 10

# Hyper-parameters

lambdavalues    = np.arange(0,70,10)
n_hidden_values = np.arange(4,24,4)


for lambdavalue in lambdavalues:
    
    for n_hidden in n_hidden_values:

        trainingStart = time.time()

        initial_w1 = initializeWeights(n_input, n_hidden)
        initial_w2 = initializeWeights(n_hidden, n_class)

        initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()), 0)
        
        args = (n_input, n_hidden, n_class, train_data, train_label, lambdavalue)

        opts = {'maxiter': 50}  # Preferred value.

        nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)

        # Reshape nnParams from 1D vector into w1 and w2 matrices
        w1 = nn_params.x[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
        w2 = nn_params.x[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))


        # Accuracy on Training Data
        predicted_label = nnPredict(w1, w2, train_data)
        print('Training set Accuracy:' + str(100 * np.mean((predicted_label == train_label).astype(float))) + '%', end=" ")
        
        trc = str(100 * np.mean((predicted_label == train_label).astype(float)))
        train_accuracy.append(float(trc))
       

        # Accuracy on Validation Data
        predicted_label = nnPredict(w1, w2, validation_data)
        print('|| Validation set Accuracy:' + str(100 * np.mean((predicted_label == validation_label).astype(float))) + '%', end=" ")
        
        vc = str(100 * np.mean((predicted_label == validation_label).astype(float)))
        validation_accuracy.append(float(vc))
        
        # Accuracy on Test Data
        predicted_label = nnPredict(w1, w2, test_data)
        print('|| Test set Accuracy:' + str(100 * np.mean((predicted_label == test_label).astype(float))) + '%', end=" ")
        
        tec = str(100 * np.mean((predicted_label == test_label).astype(float)))
        test_accuracy.append(float(tec))
        
        trainingEnd = time.time()

        totalTime.append(trainingEnd-trainingStart)
        m.append(n_hidden)
        l.append(lambdavalue)
        
        print('|| λ= ', lambdavalue)




results = pd.DataFrame(np.column_stack([l, m, train_accuracy, validation_accuracy, test_accuracy, totalTime]), 
                      columns=['λ', 'm','Train_Accuracy', 'Validation_Accuracy', 'Test_Accuracy', 'Training_Time'])
results = results.sort_values(by=['Test_Accuracy'], ascending=False)


# In[11]:


results.head(10)


# In[12]:


optimal_lambda = results.iloc[0,0]
optimal_m = results.iloc[0,1]

print("Optimal Lambda :",optimal_lambda)
print("Optimal hidden units :", optimal_m)


# In[13]:


rows_with_optimal_lambda = results[results.λ == optimal_lambda]
rows_with_optimal_m      = results[results.m == optimal_m]




rows_with_optimal_m


rows_with_optimal_m = rows_with_optimal_m.sort_values(by=['λ'])
rows_with_optimal_m


rows_with_optimal_lambda


rows_with_optimal_lambda = rows_with_optimal_lambda.sort_values(by=['m'])
rows_with_optimal_lambda



# Figure & Title
plt.figure(figsize=(16,12))
plt.title('Accuracy vs Number of Hidden Units (m)', pad=10, fontsize = 20, fontweight = 'bold')

# Axis Labeling
plt.xlabel('Number of Hidden Input (m)',labelpad=20, weight='bold', size=15)
plt.ylabel('Accuracy', labelpad=20, weight='bold', size=15)

# Axis ticks
plt.xticks( np.arange( 4,24, step=4), fontsize = 15)
plt.yticks( np.arange(70,95, step=2), fontsize = 15)

plt.plot(rows_with_optimal_lambda.m, rows_with_optimal_lambda.Train_Accuracy,  color='g')
plt.plot(rows_with_optimal_lambda.m, rows_with_optimal_lambda.Validation_Accuracy, color='b')
plt.plot(rows_with_optimal_lambda.m, rows_with_optimal_lambda.Test_Accuracy,  color='r')

ss = 'λ = ' + str(optimal_lambda) + ''
plt.text(16,86, s=ss, fontsize=25)
plt.legend(('Training Accuracy','Validation Accuracy','Testing Accuracy'),fontsize = 15)
plt.show()


# In[19]:


# Figure & Title
plt.figure(figsize=(16,12))
plt.title('Accuracy vs Number of Hidden Units (m)', pad=10, fontsize = 20, fontweight = 'bold')

# Axis Labeling
plt.xlabel('Number of Hidden Input (m)',labelpad=20, weight='bold', size=15)
plt.ylabel('Accuracy', labelpad=20, weight='bold', size=15)

# Axis ticks
plt.xticks( np.arange( 4,24, step=4), fontsize = 15)
plt.yticks( np.arange(70,95, step=2), fontsize = 15)

plt.scatter(rows_with_optimal_lambda.m, rows_with_optimal_lambda.Train_Accuracy,  color='g')
plt.scatter(rows_with_optimal_lambda.m, rows_with_optimal_lambda.Validation_Accuracy, color='b')
plt.scatter(rows_with_optimal_lambda.m, rows_with_optimal_lambda.Test_Accuracy,  color='r')

ss = 'λ = ' + str(optimal_lambda) + ''
plt.text(16,86, s=ss, fontsize=25)
plt.legend(('Training Accuracy','Validation Accuracy','Testing Accuracy'),fontsize = 15)
plt.show()


# ## <font color=blue> Training Time vs Number of Hidden Units

# In[28]:


# Figure & Title
plt.figure(figsize=(16,12))
plt.title('Training_Time vs Number of Hidden Units(m)', pad=10, fontsize = 20, fontweight = 'bold')

# Axis Labeling
plt.xlabel('Number of Hidden Input',labelpad=20, weight='bold', size=15)
plt.ylabel('Training_Time', labelpad=20, weight='bold', size=15)

# Axis ticks
plt.xticks( np.arange( 4,24, step=4), fontsize = 15)
plt.yticks( fontsize = 15)

ss = 'λ = ' + str(optimal_lambda) + ''
plt.text(8,24.25, s=ss, fontsize=25)
plt.plot(rows_with_optimal_lambda.m, rows_with_optimal_lambda.Training_Time,  color='c')

plt.show()


# ## <font color=blue> Accuracy vs Lamda

# In[26]:


# Figure & Title
plt.figure(figsize=(16,12))
plt.title('Accuracy vs λ', pad=10, fontsize = 20, fontweight = 'bold')

# Axis Labeling
plt.xlabel('λ'        ,labelpad=20, weight='bold', size=15)
plt.ylabel('Accuracy', labelpad=20, weight='bold', size=15)

# Axis ticks
plt.xticks( np.arange( 0,65, step=5), fontsize = 15)
plt.yticks( fontsize = 15)

plt.plot(rows_with_optimal_m.λ, rows_with_optimal_m.Train_Accuracy,  color='g')
plt.plot(rows_with_optimal_m.λ, rows_with_optimal_m.Validation_Accuracy, color='b')
plt.plot(rows_with_optimal_m.λ, rows_with_optimal_m.Test_Accuracy,  color='r')

ss = 'm = ' + str(optimal_m) + ''
plt.text(10,93.5, s=ss, fontsize=25)
plt.legend(('Training Accuracy','Validation Accuracy','Testing Accuracy'),fontsize = 15)
plt.show()


# In[22]:


len(featureIndices)



# # <font color = green> Pickle object Creation with Optimal parameters

# In[29]:


# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1]

# set the number of nodes in hidden unit (not including bias unit)
n_hidden = 20

# set the number of nodes in output unit
n_class = 10

# initialize the weights into some random matrices
initial_w1 = initializeWeights(n_input, n_hidden)
initial_w2 = initializeWeights(n_hidden, n_class)

# unroll 2 weight matrices into single column vector
initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()), 0)

# set the regularization hyper-parameter
lambdaval = 30

args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

# Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example

opts = {'maxiter': 50}  # Preferred value.

nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)

# In Case you want to use fmin_cg, you may have to split the nnObjectFunction to two functions nnObjFunctionVal
# and nnObjGradient. Check documentation for this function before you proceed.
# nn_params, cost = fmin_cg(nnObjFunctionVal, initialWeights, nnObjGradient,args = args, maxiter = 50)


# Reshape nnParams from 1D vector into w1 and w2 matrices
w1 = nn_params.x[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
w2 = nn_params.x[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

# Test the computed parameters

predicted_label = nnPredict(w1, w2, train_data)

# find the accuracy on Training Dataset

print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label == train_label).astype(float))) + '%')

predicted_label = nnPredict(w1, w2, validation_data)

# find the accuracy on Validation Dataset

print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label == validation_label).astype(float))) + '%')

predicted_label = nnPredict(w1, w2, test_data)

# find the accuracy on Validation Dataset

print('\n Test set Accuracy:' + str(100 * np.mean((predicted_label == test_label).astype(float))) + '%')


parameters = [featureIndices, int(optimal_m), w1, w2, int(optimal_lambda)]
pickle.dump(parameters, open('params.pickle', 'wb'))


