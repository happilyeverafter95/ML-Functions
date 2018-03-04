import numpy as np 

# Logistic Regression 
# Takes input of training set (seperated by labels and features) and features of testing set 
# Outputs fitted testing set 
def log_reg(train_y, train_X, test_X): 
    converge = False 
    
    # Initial values for weights 
    beta = np.random.standard_normal(train_X.shape[1])*0.001
    
    count = 0 
    
    while(converge == False):
        # Fischer Scoring 
        proj = np.dot(train_X, beta) 
        pred = 1/(1 + np.exp(-proj))
        score = train_X.transpose().dot(train_y - pred)
        information = train_X.transpose().dot(pred.dot(1-pred)).dot(train_X)
        # update weights
        old_beta = beta
        beta = beta + np.dot(np.linalg.inv(information), score)
        count = count + 1 
        if count % 1000 == 0: 
            print("RSS over Training Set: " + sum(abs(train_y - np.round(pred))) + "; iteration: " + count)
        if sum(abs(old_beta - beta))/train_X.shape[1] < 0.0001: # threshold for convergence
            converge = True
        
    # Using log reg to fit testing set     
    y_pred = 1/(1+np.exp(-np.dot(test_X, beta)))
    return y_pred