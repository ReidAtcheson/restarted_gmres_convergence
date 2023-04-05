import numpy as np
import scipy.linalg as la
import arnoldi
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


seed=1234
rng=np.random.default_rng(seed)

m=100
separation=4.0
xs=rng.normal(loc=0.0,scale=0.3,size=(2,m))
L=np.array(
        [[1.0,0.0],
         [1.0,1.0]])
xs=L@xs+np.array([1.0,separation])[:,None]
ys=rng.normal(loc=0.0,scale=0.3,size=(2,m))
L=np.array(
        [[1.0,0.0],
        [-1.0,1.0]])
ys=L@ys+np.array([1.0,-separation])[:,None]


plt.scatter(xs[0,:],xs[1,:])
plt.scatter(ys[0,:],ys[1,:])
plt.savefig('ex01/ex01.png')
