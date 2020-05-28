# ESRL: Ensemble SVD-regularized 
Clustering is an unsupervised technique to find underlying structure in a dataset by grouping
data points into subsets that are as homogeneous as possible, clustering is a widely used unsupervised technique for identifying natural classes within a set of data. [Amiri et al. (2020)]() proposed a clustering technique for general clustering problems which provide a kind of dissimilarity. The proposed is fully nonparametric and it generates clusters for a given desired number of clusters K. They also discussed estimating the size of cluster.

## Contents
1. [ESRL](#esrl)
2. [Data set](#datasets)
 - [GARBER data](#garber-data)
 - [Scale the data](#scale-the-data)
3. [How to rung](#how-to-run)
5. [References](#references)

# ESRL
We implemented the methods discussed Amiri et al. (2020) in R and uploaded in Github. To load the codes in R, run the following script.
```
library("RCurl")
script <- getURL("https://raw.githubusercontent.com/saeidamiri1/ESLR/master/RCode/ESRL.R", ssl.verifypeer = FALSE)
eval(parse(text = script))
 ```
The ```ESRL()``` provides the dissimiliary matrix
```
ESRL(x,B=2000,rmin=0.5,rmax=0.85)
```

The arguments are: ```x``` is the observation, use the R's matrix format. ```B``` is number of run to get a stabilized clusters, we used B=2000 in our computations. Concerning ```B```, run the code with different Bs and if you see huge different in result, increase the number of iterations. rmin and rmax are the minimum and maximum size of the proportion of total variance in SVD to get the stabilized dissimilarity matrix. We used knmin=.5 and knmax=.85.


# Data sets
###  GARBER data
To study the performance of the SHC’s with high dimensional data, we used the microarray data from Garber et al. (2001). The data are the 916-dimensional gene expression profiles for lung tissue from n = 72 subjects. Of these, five subjects were normal and 67 had lung tumors. The classification of the tumors into 6 classes (plus normal) was done by a pathologist giving seven classes total.

```
library("pvclust")
data(lung)
attach(lung)

garber<-t(lung)
garber<-garber[-c(1,20),]

for(i in 1:(dim(garber)[2])){
   garber[is.na(garber[,i]),i]<-mean(garber[,i],na.rm=T)
}
garber<-as.matrix(garber)

# extract the tru label
row1<-grep("Adeno",row.names(garber), perl=TRUE, value=FALSE)
row2<-grep("normal",row.names(garber), perl=TRUE, value=FALSE)
row3<-grep("SCLC",row.names(garber), perl=TRUE, value=FALSE)
row4<-grep("SCC",row.names(garber), perl=TRUE, value=FALSE)
row5<-grep("node",row.names(garber), perl=TRUE, value=FALSE)
row6<-grep("LCLC",row.names(garber), perl=TRUE, value=FALSE)

cagarber<-NULL
cagarber[row1]<-1
cagarber[row2]<-2
cagarber[row3]<-3
cagarber[row4]<-4
cagarber[row5]<-5
cagarber[row6]<-6
```

### Scale the data
We generated many simulated convex data sets with  very different scales in one dimension (vertical) but similar scales on another (horizontal). One of them is Github and accessible via the following scripts   

```
X_g<-Scale(garber)
```

# How to run
Load the following libraries which run the computations in parallel,

```
library("foreach")
library("doParallel")
```
Once the data and the codes are loaded in R, the clustering can be obtained using the following script

```
CLUS<-ESRL(X_g,rmin=0.5,rmax=0.85,B=2000)
CLUS<-as.dist(CLUS)
```

The dendrogram can be also plotted,
```
plot(hclust(CLUS, method="average"),h=-1)
```

# References
Amiri, S., Saunier, N. (2020). ESRL: Ensemble SVD-regularized learning to achieve clustering . ([pdf](), [journal]())

### License
Copyright (c) 2020 Saeid Amiri
**[⬆ back to top](#contents)**