# Machine Learning: Final Project - formated
Nail Allayarov  
5 11 2017  


## Background

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: <http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har> (see the section on the Weight Lifting Exercise Dataset).

## Weight Lifting Exercises Dataset
This human activity recognition research has traditionally focused on discriminating between different activities, i.e. to predict "which" activity was performed at a specific point in time (like with the Daily Living Activities dataset above). The approach we propose for the Weight Lifting Exercises dataset is to investigate "how (well)" an activity was performed by the wearer. The "how (well)" investigation has only received little attention so far, even though it potentially provides useful information for a large variety of applications,such as sports training.

In this work (see the paper) we first define quality of execution and investigate three aspects that pertain to qualitative activity recognition: the problem of specifying correct execution, the automatic and robust detection of execution mistakes, and how to provide feedback on the quality of execution to the user. We tried out an on-body sensing approach (dataset here), but also an "ambient sensing approach" (by using Microsoft Kinect - dataset still unavailable)

Six young health participants were asked to perform one set of 10 repetitions of the Unilateral Dumbbell Biceps Curl in five different fashions: exactly according to the specification (Class A), throwing the elbows to the front (Class B), lifting the dumbbell only halfway (Class C), lowering the dumbbell only halfway (Class D) and throwing the hips to the front (Class E).

Class A corresponds to the specified execution of the exercise, while the other 4 classes correspond to common mistakes. Participants were supervised by an experienced weight lifter to make sure the execution complied to the manner they were supposed to simulate. The exercises were performed by six male participants aged between 20-28 years, with little weight lifting experience. We made sure that all participants could easily simulate the mistakes in a safe and controlled manner by using a relatively light dumbbell (1.25kg).



## Load the libraries:


```r
library(ggplot2); library(lattice)
library(dplyr); library(tidyr)
library(lubridate); library(caret)
library(AppliedPredictiveModeling)
library(ElemStatLearn)
library(pgmm); library(rpart)
library(rpart.plot); library(e1071)
library(gbm); library(forecast)
```

## Set the working directory and load the data:


```r
setwd("/Users/Nael/Desktop/Coursera/8. Machine Learning/Project")
testing<-read.csv("pml-testing.csv")
training<-read.csv("pml-training.csv")
```

## Let us first explore the training set:


```r
dim(training)
```

```
## [1] 19622   160
```

```r
levels(training$classe)
```

```
## [1] "A" "B" "C" "D" "E"
```
Since we have pretty large data set, let us divide it into training and validation sets:

```r
set.seed(5-11-2017)
inTrain = createDataPartition(training$classe, p = 3/4)[[1]]
train = training[inTrain,]
validation = training[-inTrain,]
```
Let us check the sizes of our new data sets:

```r
dim(train)
```

```
## [1] 14718   160
```

```r
dim(validation)
```

```
## [1] 4904  160
```

## Data cleaning

We can delete columns which have more than 95 percent NAs:

```r
train<-train[colSums(!is.na(train)) > (dim(train)[1])*0.05]
validation<-validation[colSums(!is.na(validation)) > (dim(validation)[1])*0.05] 
```

In the next step we delete columns which have more than 95 percent of empty symbols or #DIV/0! errors:

```r
train<-train[colSums(train!="" & train!="#DIV/0!") > (dim(train)[1])*0.05]
validation<-validation[colSums(validation!="" & validation!="#DIV/0!") > (dim(validation)[1])*0.05]
```

Finally, we omit some less important columns:

```r
train<-train[,-7:-1]
validation<-validation[,-7:-1]
```

## Random forest with parallel calculation

For our analysis we will use the random forest algorithm with parallel calculation for faster performing. See:
<https://github.com/lgreski/datasciencectacontent/blob/master/markdown/pml-randomForestPerformance.md>

### Step 1: Configure parallel processing

Parallel processing in caret can be accomplished with the `parallel` and `doParallel` packages:


```r
library(parallel)
library(doParallel)
```

```
## Loading required package: foreach
```

```
## Loading required package: iterators
```

```r
cluster <- makeCluster(detectCores() - 1) # convention to leave 1 core for OS
registerDoParallel(cluster)
```

### Step 2: Configure trainControl object

The most critical arguments for the `trainControl` function are the resampling method `method`, the `number` that specifies the quantity of folds for k-fold cross-validation, and `allowParallel` which tells `caret` to use the cluster that we've registered in the previous step. For our analysis we used `cv` method instead of `bootsrapping` in order to increase or calculation speed. k=10 is the pretty balanced choise between the accuracy and speed/variance:


```r
fitControl <- trainControl(method = "cv",
                           number = 10,
                           allowParallel = TRUE)
```

### Step 3: Develop training model

`devtools::install_github('topepo/caret/pkg/caret')` command in order to make sure the caret library is up to date. 

Let us run our machine learning algorithm:


```r
RF_fit <- train(classe~., method="rf", data=train, trControl = fitControl, tuneGrid = data.frame(mtry = 3))
```

### Step 4: De-register parallel processing cluster

After processing the data, we explicitly shut down the cluster by calling the `stopCluster()` and `registerDoSEQ()` functions. `registerDoSEQ()` function is required to force R to return to single threaded processing.


```r
stopCluster(cluster)
registerDoSEQ()
```

At this point we have a trained model in the fit object, and can take a number of steps to evaluate the suitability of this model, including accuracy and a confusion matrix that is based on comparing the modeled data to the held out folds.


```r
RF_fit
```

```
## Random Forest 
## 
## 14718 samples
##    52 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (10 fold) 
## Summary of sample sizes: 13248, 13247, 13246, 13247, 13245, 13245, ... 
## Resampling results:
## 
##   Accuracy   Kappa    
##   0.9927981  0.9908889
## 
## Tuning parameter 'mtry' was held constant at a value of 3
```

```r
RF_fit$resample
```

```
##     Accuracy     Kappa Resample
## 1  0.9918367 0.9896744   Fold01
## 2  0.9918423 0.9896796   Fold02
## 3  0.9898098 0.9871062   Fold03
## 4  0.9925221 0.9905398   Fold04
## 5  0.9918534 0.9896929   Fold05
## 6  0.9932111 0.9914106   Fold06
## 7  0.9925272 0.9905464   Fold07
## 8  0.9986404 0.9982804   Fold08
## 9  0.9938900 0.9922730   Fold09
## 10 0.9918478 0.9896861   Fold10
```

```r
confusionMatrix.train(RF_fit)
```

```
## Cross-Validated (10 fold) Confusion Matrix 
## 
## (entries are percentual average cell counts across resamples)
##  
##           Reference
## Prediction    A    B    C    D    E
##          A 28.4  0.1  0.0  0.0  0.0
##          B  0.0 19.2  0.2  0.0  0.0
##          C  0.0  0.0 17.3  0.3  0.0
##          D  0.0  0.0  0.0 16.1  0.0
##          E  0.0  0.0  0.0  0.0 18.3
##                             
##  Accuracy (average) : 0.9928
```

As we can see we get a very accurate model with 99% accuracy. Lets take a look at the prediction power for the validation data set:


```r
confusionMatrix(validation$classe, predict(RF_fit, newdata=validation))
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1393    0    2    0    0
##          B    3  943    3    0    0
##          C    0    6  848    1    0
##          D    0    0   13  791    0
##          E    0    0    0    4  897
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9935          
##                  95% CI : (0.9908, 0.9955)
##     No Information Rate : 0.2847          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9917          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9979   0.9937   0.9792   0.9937   1.0000
## Specificity            0.9994   0.9985   0.9983   0.9968   0.9990
## Pos Pred Value         0.9986   0.9937   0.9918   0.9838   0.9956
## Neg Pred Value         0.9991   0.9985   0.9956   0.9988   1.0000
## Prevalence             0.2847   0.1935   0.1766   0.1623   0.1829
## Detection Rate         0.2841   0.1923   0.1729   0.1613   0.1829
## Detection Prevalence   0.2845   0.1935   0.1743   0.1639   0.1837
## Balanced Accuracy      0.9986   0.9961   0.9887   0.9953   0.9995
```

As we can see the result on vlidation data set is even better. We used it on testing data set (20 observations) and predicted all correctly. 
