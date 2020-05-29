
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import multiprocessing
from sklearn.cluster import KMeans


def sub_ESRL(X_dat,rmin,rmax):
    n=X_dat.shape[0]
    K0=np.random.choice(range(2,int(n/3)+1), size=1, replace=True, p=None)
    r01=np.unique(np.random.choice(range(X_dat.shape[1]), size=X_dat.shape[1], replace=True))
    X_dat1=X_dat.iloc[:,r01]
    U, s, VT =np.linalg.svd(X_dat1)
    D=np.diag(s)
    cu=s.cumsum()/sum(s)
    bb= np.random.uniform(rmin,rmax,1)
    m0=np.min(np.where(cu>bb))
    if (m0==0):  m0=1
    U=U[:,0:(np.shape(X_dat1)[1]-1)]
    rec0= U[:,:m0].dot(D[:m0,:m0].dot(VT[:m0,:m0]))
    kmeans = KMeans(n_clusters=K0[0]).fit(rec0).labels_
    return(kmeans)

#def hammingD(dat):
#    ss=np.shape(dat)[1]
#    dismat=np.array([])
#    for i in range(ss-1):
#        for j in range(i+1,ss):
#        dismat[i,j]=np.mean(dat[:,i]==dat[:,j])
#    return(1-dismat)

def hammingD(dat):
    ss=np.shape(dat)[1]
    dismat=np.array([])
    for i in range(ss-1):
        for j in range(i+1,ss):
            dismat=np.append(dismat,np.mean(dat[:,i]==dat[:,j]))
    return(1-dismat)

def ESRL(x_dat,rmin=.50,rmax=.85,B=1000):
    num_cores = multiprocessing.cpu_count()
    par0=Parallel(n_jobs=num_cores-1)(
    delayed(sub_ESRL)(x_dat,rmin,rmax)
    for i in range(B))
    h=hammingD(np.transpose(par0).T)
    return(h)    
