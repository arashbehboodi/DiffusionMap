# Diffusion Maps
------

This program is written to test diffusion maps. The implementation is based on:

* Alfonso S. Bandeira, Ten Lectures and Forty-Two Open Problems in the Mathematics of Data Science.

The function is given by *diffusionmap(X,n,eps,t,k)*. The function gets the data matrix where samples are put in columns of the matrix.
*t* indicates the number of iterations.  *k* indicates the target dimension for dimensionality reduction. *eps* is the parameter of Guassian kernel that is used.