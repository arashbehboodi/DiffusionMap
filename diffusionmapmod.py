__author__="Arash Behboodi"
# This is written for Python 3.5
# This program is written to test diffusion maps. The implementation is based on:
# Alfonso S. Bandeira, Ten Lectures and Forty-Two Open Problems in the Mathematics of Data Science.
############################################################################
# General Libraries to be included.
import numpy as np
from numpy import linalg as la

############################################################################
# diffusionmap(X,n,eps,t,k)
# The function gets the data matrix where samples are put in columns of the matrix.
# t indicates the number of iterations. 
# k indicates the target dimension for dimensionality reduction
# eps is the parameter of Guassian kernel that is used.
 ############################################################################
def diffusionmap(X,n,eps,t,k):
	##################################
	# Running diffusion maps
	##################################
	##################################
	# print('-------------------------------')
	# print('-------------------------------')
	# print('Diffusion Maps')
	# print('-------------------------------')
	# print('-------------------------------')
	#### Finding the distance matrix
	x1temp=-2*X@X.T
	x2temp=np.outer(np.diag(X@X.T),np.ones((n,1)))+np.outer(np.ones((n,1)),np.diag(X@X.T).T)
	DX=x1temp+x2temp
	#### Kernel function for weight matrx
	## Gaussian Kernel
	## Constructing the weight matrix
	W=np.exp(-DX/eps)
	Deg=W@np.ones((n,1))
	D=np.diag(Deg.reshape(n,))
	## Transition matrix
	M=la.inv(D)@W
	#######################
	## Constructing the matrix S
	S=D**(1/2)@M@la.inv(D)**(1/2)
	## Spectral decomposition
	eigvalCov, eigvecCov = la.eig(S)
	idx = eigvalCov.argsort()[::-1]   
	eigvalCov = eigvalCov[idx]
	eigvecCov = eigvecCov[:,idx]
	## Diffusion Map
	phiD=la.inv(D)**(1/2)@eigvecCov
	lambdaD=eigvalCov**t
	##########################
	## Final Matrix with columns as the vectors
	DiffM=np.diag(lambdaD)@phiD.T
	Difftruncated=DiffM[1:k+1,:]
	return Difftruncated
