import numpy as np
import scipy.linalg as la
from sympy import symbols

#Factorizes A as
#A * V[:,0:k] = V * H
#Where V spans a Krylov space of A
#and V is orthogonal in the standard l2 inner product
def arnoldi(A,v,k):
    dtype=A(v).dtype
    norm=np.linalg.norm
    dot=np.vdot
    eta=1.0/np.sqrt(2.0)

    m=len(v)
    #Make V a matrix of size m x k+1 with same datatype as A
    V=np.zeros((m,k+1),dtype=dtype)
    H=np.zeros((k+1,k),dtype=dtype)
    #V[:,0]=v/norm(v)
    V[:,0]=v/norm(v)
    for j in range(0,k):
        w=A(V[:,j])

        h=V[:,0:j+1].conj().T @ w
        f=w-V[:,0:j+1] @ h

        s = V[:,0:j+1].conj().T @ f

        f = f - V[:,0:j+1] @ s

        h = h + s
        beta=norm(f)
        #H[j+1,j]=beta
        H[0:j+1,j]=h
        H[j+1,j]=beta
        #V[:,j+1]=f/beta
        V[:,j+1]=f.flatten()/beta
    return V,H



def gmres_step(A,b,z,k):
    m=len(b)
    x=np.zeros_like(b)
    r=b-A(x)
    V,H=arnoldi(A,r,k)
    P=arnoldi_basis(H,r,z)

    beta=np.linalg.norm(r)
    e=np.zeros(k+1,dtype=x.dtype)
    e[0]=beta
    y,_,_,_=la.lstsq(H,e)
    x=x+V[:,0:k]@y
    return x,1-z*(P[:,0:k]@y)
