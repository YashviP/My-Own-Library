import numpy as np

class LogisticRegression:
	def __init__(self, alpha=0.01, num_iter=100000, fit_intercept=True):
		self.alpha = alpha
		self.num_iter = num_iter
		self.fit_intercept = fit_intercept
	
	def add_intercept(self, X):
		intercept = np.ones((X.shape[0], 1))
		return np.concatenate((intercept, X), axis=1)
    
	def sigmoid(self, z):
		return 1 / (1 + np.exp(-z))

	def loss(self, h, y):
		return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()
    
	def fit(self, X, y):
		m=X.shape[0]
		if self.fit_intercept:
			X = self.add_intercept(X)
        
		# weights initialization
		self.theta = np.zeros(X.shape[1])
        
		for i in range(self.num_iter):
			z = np.dot(X, self.theta)
			h = self.sigmoid(z)
			gradient = np.dot(X.T, (h - y)) /m 
			self.theta -= self.alpha * gradient
               
		z = np.dot(X, self.theta)
		h = self.sigmoid(z)       
		loss = self.loss(h, y)   
		print('loss:{}'.format(loss))

	def predict_prob(self, X):
		if self.fit_intercept:
			X = self.add_intercept(X)
    
		return self.sigmoid(np.dot(X, self.theta))
    
	def predict(self, X):
		return self.predict_prob(X).round()
