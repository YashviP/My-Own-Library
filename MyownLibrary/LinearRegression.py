import numpy as np

# Gradient descent technique is used 
# alpha: learning rate
# n_iter : no. of passes 
# w: weights 
# cost=total error after each iteration 
# m=size of input vector
# x=input vector
# y_pred=predicted vector
# y=actual target vector


class LinearRegression:
	def __init__(self,alpha=0.05,n_iter=1000):
		self.alpha=alpha
		self.n_iter=n_iter

	def fit(self,x,y):
		self.cost=[]
		self.w=np.zeros((x.shape[1],1))
		m=x.shape[0]
		
		for _ in range(self.n_iter):
			y_pred=np.dot(x,self.w)
			resd=y_pred-y
			gradient_vector=np.dot(x.T,resd)
			self.w -= (self.alpha/m)*gradient_vector
			cost=np.sum((resd**2))/(2*m)
			self.cost.append(cost)
		return self

	def predict(self,x):
		return np.dot(x,self.w)


	def score(self,y_pred,y_actual):
		sum_squared_err=np.sum((y_pred-y_actual)**2)
		total_sum_square=np.sum((y_actual-np.mean(y_actual))**2)
		r2_score=1-(sum_squared_err/total_sum_square)
		return r2_score


				


			

