import numpy as np 

# Single Layer Neural Network 
# Takes input of training set (seperated by labels and features) and features of testing set, number of hidden units (h) 
# Outputs fitted testing set 
# Note: instead of iterating through epoches, we iterate until weights converge
def neural_net(train_X,train_y,h, test_X, test_y):
    converge = False 
    
    # initialize initial weights
    hidden_layer = np.random.standard_normal((train_X.shape[1],h))
    output_layer = np.random.standard_normal((h,1))*0.00001
    epoch = 0 
    while (converge == False):
        # back propagation 
        layer_input = np.dot(train_X, hidden_layer) 
        activated_layer = 1/(1+np.exp(-layer_input)) 
        output_input = np.dot(activated_layer,output_layer)      
        output = 1/(1 + np.exp(-output_input))
        error = -2*(train_y - output) 
        slope_output = output*(1 - output) 
        slope_layer = activated_layer*(1 - activated_layer) 
        delta_out = error * slope_output 
        layer_error = np.dot(delta_out, output_layer.transpose())
        delta_layer = layer_error * slope_layer 
        # update weights
        old_hidden_layer = hidden_layer 
        old_output_layer = output_layer 
        output_layer = output_layer - (np.dot(activated_layer.transpose(), delta_out) + 0.01 * output_layer)*0.0015  # 0.0015 is the learning rate; 0.01 is the weight decay parameter
        hidden_layer = hidden_layer - (np.dot(train_X.transpose(), delta_layer) + 0.01 * hidden_layer)*0.0015
        epoch = epoch + 1
        if epoch % 1000 == 0: 
            print("RSS over Training Set: " +  sum(abs(train_y - np.round(output)))/176 + "; Epoch: " + epoch)
        if sum(sum(abs(hidden_layer - old_hidden_layer))) < 0.0006 and sum(abs(output_layer - old_output_layer)) < 0.0006: 
            converge = True
        
    # using NN to fit testing set
    layer_input_test = np.dot(test_X, hidden_layer) 
    activated_layer_test = 1/(1+np.exp(-layer_input_test)) 
    y_pred = np.dot(activated_layer_test,output_layer)
    
    return y_pred
