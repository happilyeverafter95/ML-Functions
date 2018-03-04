# Support Vector Machines 

from scipy.optimize import minimize 

# Hard margin; assuming everything is linearly separable
# Intakes training set (split by labels and features) and outputs [slope, intercept] of the seperator hyperplane 
def HardMarg(train_X,train_y):
    n = train_y.shape[0]
    S = np.multiply(train_y.T, train_X).T.dot(np.multiply(train_y.T,train_X))
    fun = lambda a: (-a.dot(np.ones((n,1))) + 0.5*a.dot(S).dot(a.T)) # lagrangian formulation; would like to maximize margins 
    bnds = ((0,float('inf')),)*n 
    cons = {'type': 'eq', 'fun': lambda a: float(a.dot(train_y))}
    result = minimize(fun, np.random.standard_normal((1,n)), method = 'SLSQP', bounds = bnds, constraints = cons)
    B = np.multiply(np.matrix(result.x), train_y.T).dot(train_X.T)
    z = list(result.x)
    index_0 = z.index(sorted(z)[-1])
    B0 = np.asscalar(1/train_y[index_0] - B.dot(train_X[:,index_0]))
    return [B, B0]

# Soft margin; no linearly separable assumption 
# Intakes training set (split by labels and features) and outputs [slope, intercept] of the seperator hyperplane 
# gamma is the free parameter 
# Note: did not include C, the variable which penalizes slackness 
def SoftMarg(train_X,train_y, gamma): 
    n = train_y.shape[0]
    S = np.multiply(train_y.T, train_X).T.dot(np.multiply(train_y.T,train_X))
    fun = lambda a: (-a.dot(np.ones((n,1))) + 0.5*a.dot(S).dot(a.T)) # lagrangian formulation; would like to maximize margins 
    bnds = ((0,gamma),)*n 
    cons = {'type': 'eq', 'fun': lambda a: float(a.dot(train_y))}
    result = minimize(fun, np.random.standard_normal((1,n)), method = 'SLSQP', bounds = bnds, constraints = cons)
    B = np.multiply(np.matrix(result.x), train_y.T).dot(train_X.T)
    index_0 = list(result.x).index(sorted(result.x[(result.x < gamma)])[-1])
    B0 = np.asscalar(1/train_y[index_0] - B.dot(train_X[:,index_0]))
    return [B, B0]

# Classifies test set after receiving [slope, intercept] of the seperator hyperplane 
def classify(Xtest, b, b0): 
    yhat = [] 
    for i in range(0,Xtest.shape[1]):
        temp1 = b.dot(Xtest[:,i]) + b0
        if np.asscalar(temp1) < 0: 
            temp = -1
        else: 
            temp = 1
        yhat.append(temp) 
    return yhat