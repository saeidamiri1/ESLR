ESRL<-function(x,rmin=.50,rmax=.85,B=1000){
sub_ESRL<-function(X0,rmin,rmax){
  n<-nrow(X0)
  K0 <- sample(2:(n/3 - 2), 1)
  r01 <- unique(sample(ncol(X0), replace = TRUE))
  X01 <- X0[, r01]
  s <- svd(X01)
  D <- diag(s$d)
  cu<-cumsum(s$d)/sum(s$d)
  bb<-runif(1,min=rmin, max=rmax)
  m0<-min(which(cu>bb))
  if (m0==1) {
    rec0<-s$u[,1:m0] * D[1:m0,1:m0] * (s$v[,1:m0])
  } else {
    rec0<-s$u[,1:m0] %*% D[1:m0,1:m0] %*% t(s$v[,1:m0])
  }
 return(kmeans(rec0,K0)$cluster)
}

hammingD<-DistM<-function(dat){
  Len<-dim(dat)
  ss<-Len[1]
  dismat<- matrix(1,ncol=ss,ss)
  for(i in 1:ss){
    if (i==ss) break
    for(j in (i:ss)){
      dismat[i,j]<-mean(dat[i,]==dat[j,],na.rm=T)
    }
  }
  return(1-t(dismat))
}


require("doParallel", character.only = T)
require("foreach", character.only = T)

cl <- makeCluster(detectCores()-1) 
registerDoParallel(cl) 
ens = foreach(i = 1:B,
              .combine = "rbind") %dopar% {
                fit1 <- sub_ESRL(x,knmin,knmax)
                fit1
              }
stopCluster(cl)



esrl<-hammingD(t(ens))
return(esrl)
}

