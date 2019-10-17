import numpy as np

class SVM:
	def __init__(self,n_iter=100000):
		self.n_iter=n_iter
 

	def fit(self,X, Y):
		for i in range(0,Y.shape[0]):
			if Y[i]==0:
				Y[i]=-1
		self.w = np.zeros(len(X[0]))
		eta = 1
		epochs = self.n_iter


		for epoch in range(1,epochs):
			for i, x in enumerate(X):
				if (Y[i]*np.dot(X[i], self.w)) < 1:
					self.w = self.w + eta * ( (X[i] * Y[i]) + (-2  *(1/epoch)* self.w) )
					self.cost=1-(Y[i]*np.dot(X[i], self.w))
				else:
					self.w = self.w + eta * (-2  *(1/epoch)* self.w)
					self.cost=0
			

		return self
	
	def predict(self,X):
		return np.sign(np.dot(X,self.w))
	
