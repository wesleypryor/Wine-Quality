#Libraries
library(kknn)
library(class)
library(party)
library(rpart)
library(rattle)
library(exact2x2)
library(e1071)
library(rpart.plot)
library(mlbench)
library(nnet)
library(kernlab)
library(microbenchmark)
library(flexclust)
library(pROC)
library(ggplot2)
library(MASS)

#Finding Beta Hat Calculations
beta.hat <- function(X,Y){
  beta.hat <- (solve(t(X)%*%X)%*%t(X)%*%Y)
  return(betahat = beta.hat)
}

#Residuals Function and Calculation
residuals <- function(X,Y){
  e <- Y-X%*%OptLeastSquares(X,Y)
  return(res=e)
}

#Estimation of Sigma function
sigma.hat <- function(X,Y){
  s <- sqrt(1/(nrow(X)-ncol(X))*sum(residuals(X,Y)^2))
  return(s)
}

#Covariance calculations
cov.beta.hat <- function(X,Y){
  cvbtht <- sigma.hat(X,Y)^2*solve(t(X)%*%X)
  return(cvbtht)
}

#F-test function
f.test=function(model0,model){
  e02=sum((model0$residuals)^2)   
  e2=sum((model$residuals)^2)     
  df0=model0$df.residual          
  df=model$df.residual            
  
  f.stat=((e02-e2)/(df0-df))/(e2/df)
  
  p.value=pf(f.stat,
             df1=df0-df,
             df2=df,
             lower.tail=FALSE)
  
  return(list(f.stat=f.stat,p.value=p.value))
}

#Box-Cox
box.cox <- function(Y,X,lambdarange){
  k.2 <- (prod(Y^(1/length(Y))))
  res.sum.squares <- NULL
  for(i in 1:length(lambdarange)){
    if(lambdarange[i]!=0){
      k.1 <- 1/(lambdarange[i]*k.2^(lambdarange[i]-1))
      W <- k.1*(Y^lambdarange[i]-1)
    } else{
      W <- k.2*(log(Y))
    }
    W <- as.vector(W)
    temp.model <- lm(W~X)
    res.sum.squares[i] <- sum(temp.model$residuals^2)
  }
  optimal.rss.location <- which.min(res.sum.squares)
  optimal.lambda <- lambdarange[optimal.rss.location]
  return(lamba=optimal.lambda)
}

#Importing Dataset
WhiteData <- read.csv("~/Data Mining 1 Fall 2016/FinalProject/winequality-white.csv", sep=";")
attach(WhiteData)
quality <- as.factor(WhiteData$quality)


#Histograms of the Data
par(mfrow=c(2,2), oma = c(1,1,0,0) + 0.1, mar = c(3,3,1,1) + 0.1)
barplot((table(quality)), col=c("slateblue4", "slategray", "slategray1", "slategray2", "slategray3", "skyblue4"))
mtext("Quality", side=1, outer=F, line=2, cex=0.8)
truehist(fixed.acidity, h = 0.5, col="slategray3")
mtext("Fixed Acidity", side=1, outer=F, line=2, cex=0.8)
truehist(volatile.acidity, h = 0.05, col="slategray3")
mtext("Volatile Acidity", side=1, outer=F, line=2, cex=0.8)
truehist(citric.acid, h = 0.1, col="slategray3")
mtext("Citric Acid", side=1, outer=F, line=2, cex=0.8)

truehist(residual.sugar, h = 0.5, col="slategray3")
mtext("Residual Sugar", side=1, outer=F, line=2, cex=0.8)
truehist(chlorides, h = 0.05, col="slategray3")
mtext("Chlorides", side=1, outer=F, line=2, cex=0.8)
truehist(free.sulfur.dioxide, h = 0.1, col="slategray3")
mtext("Free Sulfur Dioxide", side=1, outer=F, line=2, cex=0.8)
truehist(total.sulfur.dioxide, h = 0.5, col="slategray3")
mtext("Total Sulfur Dioxide", side=1, outer=F, line=2, cex=0.8)

truehist(density, h = 0.05, col="slategray3")
mtext("Density", side=1, outer=F, line=2, cex=0.8)
truehist(pH, h = 0.1, col="slategray3")
mtext("pH", side=1, outer=F, line=2, cex=0.8)
truehist(sulphates, h = 0.5, col="slategray3")
mtext("Sulphates", side=1, outer=F, line=2, cex=0.8)
truehist(alcohol, h = 0.5, col="slategray3")
mtext("Alcohol", side=1, outer=F, line=2, cex=0.8)

par(mfrow=c(1,1))

#Preparing the Data for outliers
limout <- rep(0,11)
for (i in 1:11){
  t1 <- quantile(WhiteData[,i], 0.75)
  t2 <- IQR(WhiteData[,i], 0.75)
  limout[i] <- t1 + 1.5*t2
}
WhiteWineIndex <- matrix(0, 4898, 11)
for (i in 1:4898)
  for (j in 1:11){
    if (WhiteData[i,j] > limout[j]) WhiteWineIndex[i,j] <- 1
  }
WWInd <- apply(WhiteWineIndex, 1, sum)
WhiteWineTemp <- cbind(WWInd, WhiteData)
Indexes <- rep(0, 208)
j <- 1
for (i in 1:4898){
  if (WWInd[i] > 0) {Indexes[j]<- i
  j <- j + 1}
  else j <- j
}
WhiteWineLib <-WhiteData[-Indexes,]

X.1 <- WhiteWineLib[,1]
X.2 <- WhiteWineLib[,2]
X.3 <- WhiteWineLib[,3]
X.4 <- WhiteWineLib[,4]
X.5 <- WhiteWineLib[,5]
X.6 <- WhiteWineLib[,6]
X.7 <- WhiteWineLib[,7]
X.8 <- WhiteWineLib[,8]
X.9 <- WhiteWineLib[,9]
X.10 <- WhiteWineLib[,10]
X.11 <- WhiteWineLib[,11]

X <- cbind(X.1,X.2,X.3,X.4,X.5,X.6,X.7,X.8,X.9,X.10,X.11)
Y <- WhiteWineLib[,12]

#Split of Data into training and testing data
set.seed(1506)
index <- sample(1:nrow(WhiteWineLib), size = round(0.7*nrow(WhiteWineLib),0))
wine.train <- WhiteWineLib[index,]
wine.test <- WhiteWineLib[-index,]

#Linear Model Initial Application
linmodel <- lm(quality~., data=wine.train)
summary(linmodel)

Y.hat <- predict(linmodel, newdata = wine.test)
e <- wine.test$quality-Y.hat
plot(Y.hat,e)
plot(Y.hat, wine.test$quality)
RMSE <- sqrt(mean(e^2))

#Step application to determine the best predictors for quality
stepmod <- step(lm(quality ~ 1, wine.train), scope=list(lower=~1,  upper = ~fixed.acidity+volatile.acidity+citric.acid+residual.sugar+chlorides+free.sulfur.dioxide+total.sulfur.dioxide+pH+sulphates+alcohol), direction="forward", trace = FALSE)
step.model <- lm(quality~ alcohol+volatile.acidity+residual.sugar+free.sulfur.dioxide+sulphates+chlorides+pH, data = wine.train)

#F.test for normal and step model
f.test(step.model,linmodel)

#Box-Cox Transformation
lambdarang <- seq(-3,3,0.1)
box.cox(Y,X,lambdarang)

lambdarange <- seq(0.5,1.5,0.01)
box.cox(Y,X,lambdarange)

Y.tilde <- Y^0.68
as.data.frame(X)
adj.model <- lm(Y.tilde~X.1+X.2+X.3+X.4+X.5+X.6+X.7+X.8+X.9+X.10+X.11)
Y.tilde.hat <- predict(adj.model)
e.tilde <- Y.tilde-Y.tilde.hat
plot(Y.tilde.hat , Y.tilde)
plot(Y.tilde,e.tilde)

adj.RMSE <- sqrt(mean(e.tilde^2))

#Decision Trees
wine.tree <- rpart(quality~., data = wine.train)
fancyRpartPlot(wine.tree)
wine.pred <- predict(wine.tree, newdata=wine.test)
tree.RMSE <- sqrt(mean(wine.test$quality-wine.pred)^2)

#Cross Validation
createfolds=function(n,K){
  reps=ceiling(n/K)
  non_rand_folds = rep(1:K,reps) 
  non_rand_folds = non_rand_folds[1:n] 
  folds=sample(non_rand_folds) 
  return(folds[1:n])
}
kflval <- function(k,data){
  folds = createfolds(nrow(data),k)
  accvector = 1:k
  for(k in 1:k){
    temptrain = data[folds!=k,]
    temptest = data[folds==k,]
    temptree = rpart(quality~.,data=temptrain)
    temppred = predict(temptree, newdata=temptest)
    analysis = sqrt(mean(temptest$quality-temppred)^2)
    accvector[k] = analysis
  }
  return(mean(accvector))
}

kflval(10,WhiteData)

#K nearest neighbors
x = WhiteData[,1:11]
xbar = apply(x,2,mean)
xbarMat = cbind(rep(1,nrow(WhiteData)))%*%xbar
s = apply(x,2,sd)
sMat = cbind(rep(1,nrow(WhiteData)))%*%s
z = (x-xbarMat)/sMat
Y = WhiteData$quality
z = cbind(z,Y)

z.split <- sample(nrow(z),round(nrow(z)*0.7,0))
z.train <- z[z.split,]
z.test <- z[-z.split,]

wine.knn <- knn(train = z.train, test = z.test, k=3, cl = Y[z.split])
wine.knn <- as.integer(wine.knn)
residual.knn <- Y[-z.split]-wine.knn
knn.RMSE <- sqrt(mean(residual.knn^2))

#Neural Networks
wine.nnet <- nnet(quality~.,data=wine.train,size=10,linout=FALSE, maxit = 500)
pred.nnet <- predict(wine.nnet, newdata = wine.test)
nnet.residuals <- wine.test$quality-pred.nnet
nnet.RMSE <- sqrt(mean(nnet.residuals^2))
