#rm(list = ls())
library(ggplot2)
library(lattice)
library(dplyr)
library(tidyr)
library(lubridate)
library(AppliedPredictiveModeling)
library(caret)
library(ElemStatLearn)
library(pgmm)
library(rpart)
library(rpart.plot)
library(lubridate)
library(e1071)
library(gbm)
library(forecast)

setwd("/Users/Nael/Desktop/Coursera/8. Machine Learning/Project")
testing<-read.csv("pml-testing.csv")
training<-read.csv("pml-training.csv")
#unique(training$classe)


#=====================================================================================
# Preprocessing
#=====================================================================================

set.seed(5-11-2017)
inTrain = createDataPartition(training$classe, p = 3/4)[[1]]
train = training[inTrain,]
validation = training[-inTrain,]
dim(train); dim(validation)

# strategy - we delete columns which have more than 95% NAs in them
train_1<-train[colSums(!is.na(train)) > (dim(train)[1])*0.05] # delete "_1" in front of train
dim(train_1)

train_2<-train_1[colSums(train_1!="" & train_1!="#DIV/0!") > (dim(train)[1])*0.05]
dim(train_2)

train_3<-train_2[,-7:-1]
dim(train_3)

testing_1<-testing[colSums(!is.na(testing)) > (dim(testing)[1])*0.05] # same columns for test
dim(testing_1)
testing_1
testing_2<-testing_1[colSums(testing_1!="" & testing_1!="#DIV/0!") > (dim(testing_1)[1])*0.05]
dim(testing_2)
testing_3<-testing_2[,-7:-1]
dim(testing_3)



val_1<-validation[colSums(!is.na(validation)) > (dim(validation)[1])*0.05] # same columns for test
dim(val_1)
val_2<-val_1[colSums(val_1!="" & val_1!="#DIV/0!") > (dim(val_1)[1])*0.05]
dim(val_2)
val_3<-val_2[,-7:-1]
dim(val_3)
# Some interesing pattern has been revealed - we remove only those columns which have 
# 299 non-NAs values. This means that NAs have a systematic pattern! -> we could check 
# it in the appendix of our analysis

# train_2<-train[colSums(!is.na(train)) > 298]
# dim(train_2)


#=====================================================================================
# Random forest with parallel calculation
#=====================================================================================

# Step 1: Configure parallel processing

# Parallel processing in caret can be accomplished with the parallel and doParallel 
# packages. The following code loads the required libraries (note, these libraries 
# also depend on the iterators and foreach libraries).

library(parallel)
library(doParallel)
cluster <- makeCluster(detectCores() - 1) # convention to leave 1 core for OS
registerDoParallel(cluster)

# Step 2: Configure trainControl object

fitControl <- trainControl(method = "cv",
                           number = 10,
                           allowParallel = TRUE)

# Step 3: Develop training model
# devtools::install_github('topepo/caret/pkg/caret')


start.time <- Sys.time()
#RF_fit <- train(classe~., method="rf", data=train_3, trControl = fitControl, preProcess="pca", tuneGrid = data.frame(mtry = 3))
RF_fit <- train(classe~., method="rf", data=train_3, trControl = fitControl, tuneGrid = data.frame(mtry = 3))

end.time <- Sys.time()
print(time.taken <- end.time - start.time)

# Step 4: De-register parallel processing cluster
# After processing the data, we explicitly shut down the cluster by calling 
# the stopCluster() and registerDoSEQ() functions. registerDoSEQ() function 
# is required to force R to return to single threaded processing.
stopCluster(cluster)
registerDoSEQ()
# At this point we have a trained model in the fit object, and can take a 
# number of steps to evaluate the suitability of this model, including 
# accuracy and a confusion matrix that is based on comparing the modeled data 
# to the held out folds.

RF_fit
RF_fit$resample
confusionMatrix.train(RF_fit)

pred<-predict(RF_fit, newdata=val_3)
length(pred)
confusionMatrix(val_3$classe, pred)


pred2<-predict(RF_fit, newdata=testing_3)
pred2
#  Accuracy (average) : 0.9653
#  Accuracy (average) : 0.9757
#  Accuracy (average) : 0.9944  10 fold cv, mtry=10    5.64 min




# model using PCA with principal components explaining 80% of the variance in the predictors.
# RF_with_PCA_fit <- train(classe~., data=train, 
#                preProcess="pca", method="rf", na.remove=T,
#                trControl=trainControl(preProcOptions = list(thresh = 0.8)) )











