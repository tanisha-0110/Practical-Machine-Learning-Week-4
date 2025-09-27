# Practical-Machine-Learning-Week-4
Introduction
Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset).

The training data for this project are available here:

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv

The test data are available here:

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv

library(caret)
## Warning: package 'caret' was built under R version 4.1.2
## Loading required package: ggplot2
## Warning: package 'ggplot2' was built under R version 4.1.2
## Loading required package: lattice
library(rpart)
## Warning: package 'rpart' was built under R version 4.1.2
library(rpart.plot)
## Warning: package 'rpart.plot' was built under R version 4.1.2
library(rattle)
## Warning: package 'rattle' was built under R version 4.1.2
## Loading required package: tibble
## Warning: package 'tibble' was built under R version 4.1.2
## Loading required package: bitops
## Rattle: A free graphical interface for data science with R.
## Version 5.4.0 Copyright (c) 2006-2020 Togaware Pty Ltd.
## Type 'rattle()' to shake, rattle, and roll your data.
library(randomForest)
## Warning: package 'randomForest' was built under R version 4.1.2
## randomForest 4.7-1
## Type rfNews() to see new features/changes/bug fixes.
## 
## Attaching package: 'randomForest'
## The following object is masked from 'package:rattle':
## 
##     importance
## The following object is masked from 'package:ggplot2':
## 
##     margin
library(gbm)
## Warning: package 'gbm' was built under R version 4.1.2
## Loaded gbm 2.1.8
Reading CSVs
training_set<- read.csv("pml-training.csv")
testing_set<- read.csv("pml-testing.csv")
Cleaning the Data
nearZeroVar <- nearZeroVar(training_set)
train_data <- training_set[,-nearZeroVar]
test_data <- testing_set[,-nearZeroVar]
str(test_data)
NaCols <- sapply(train_data, function(x) mean(is.na(x))) > 0.95
train_data <- train_data[,NaCols == FALSE]
test_data <- test_data[,NaCols == FALSE]
str(test_data)
Removing the first 6 non-numeric variables
train_data <- train_data[,7:59]
test_data <- test_data[,7:59]
str(test_data)
Creating testing and a validation set 60/40 split
inTrain<- createDataPartition(train_data$classe, p=0.6, list=FALSE)
training<- train_data[inTrain,]
validating<- train_data[-inTrain,]
dim(training)
## [1] 11776    53
dim(validating)
## [1] 7846   53
Runing a Decision Tree model
DT_modelfit<- train(classe ~. , data=training, method= "rpart")
fancyRpartPlot(DT_modelfit$finalModel)


DT_prediction <- predict(DT_modelfit, validating)
confusionMatrix(as.factor(DT_prediction), as.factor(validating$classe))
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2037  641  624  577  203
##          B   37  517   53  231  192
##          C  153  360  691  478  402
##          D    0    0    0    0    0
##          E    5    0    0    0  645
## 
## Overall Statistics
##                                           
##                Accuracy : 0.4958          
##                  95% CI : (0.4847, 0.5069)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.341           
##                                           
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9126  0.34058  0.50512   0.0000  0.44730
## Specificity            0.6357  0.91893  0.78496   1.0000  0.99922
## Pos Pred Value         0.4990  0.50194  0.33157      NaN  0.99231
## Neg Pred Value         0.9482  0.85314  0.88251   0.8361  0.88924
## Prevalence             0.2845  0.19347  0.17436   0.1639  0.18379
## Detection Rate         0.2596  0.06589  0.08807   0.0000  0.08221
## Detection Prevalence   0.5203  0.13128  0.26561   0.0000  0.08284
## Balanced Accuracy      0.7742  0.62976  0.64504   0.5000  0.72326
The Decision Tree Model has a low accuracy level

Running a Random Forest Model
RF_modelfit <- train(classe ~ ., data = training, method = "rf", ntree = 100)
RF_prediction<- predict(RF_modelfit, validating)
qplot(RF_prediction,validating$classe, colour=validating$classe)


RF_confusionMatrix<-confusionMatrix(as.factor(RF_prediction), as.factor(validating$classe))
RF_confusionMatrix
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2227   14    0    0    0
##          B    3 1489   13    1    1
##          C    1   14 1351   14    6
##          D    0    1    4 1268    0
##          E    1    0    0    3 1435
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9903          
##                  95% CI : (0.9879, 0.9924)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9877          
##                                           
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9978   0.9809   0.9876   0.9860   0.9951
## Specificity            0.9975   0.9972   0.9946   0.9992   0.9994
## Pos Pred Value         0.9938   0.9881   0.9747   0.9961   0.9972
## Neg Pred Value         0.9991   0.9954   0.9974   0.9973   0.9989
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2838   0.1898   0.1722   0.1616   0.1829
## Detection Prevalence   0.2856   0.1921   0.1767   0.1622   0.1834
## Balanced Accuracy      0.9976   0.9890   0.9911   0.9926   0.9973
The Random Forest Model accuracy is 99%

Running a Gradient Boosting Model
gbm_modelfit<- train(classe~., data=training, method="gbm", verbose= FALSE)
gbm_prediction<- predict(gbm_modelfit, validating)
qplot(gbm_prediction,validating$classe, colour=validating$classe)


gbm_confusionMatrix<-confusionMatrix(as.factor(gbm_prediction), as.factor(validating$classe))
gbm_confusionMatrix
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2194   58    0    2    2
##          B   23 1419   50    5   19
##          C   12   41 1305   37   21
##          D    2    0   13 1237   11
##          E    1    0    0    5 1389
## 
## Overall Statistics
##                                          
##                Accuracy : 0.9615         
##                  95% CI : (0.957, 0.9657)
##     No Information Rate : 0.2845         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.9513         
##                                          
##  Mcnemar's Test P-Value : 1.994e-14      
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9830   0.9348   0.9539   0.9619   0.9632
## Specificity            0.9890   0.9847   0.9829   0.9960   0.9991
## Pos Pred Value         0.9725   0.9360   0.9216   0.9794   0.9957
## Neg Pred Value         0.9932   0.9844   0.9902   0.9926   0.9918
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2796   0.1809   0.1663   0.1577   0.1770
## Detection Prevalence   0.2875   0.1932   0.1805   0.1610   0.1778
## Balanced Accuracy      0.9860   0.9597   0.9684   0.9790   0.9812
The Gradient Boosting Model accuracy is 96%

Conclusion
The Random Forest model is more accurate than Gradient Boosting Model at ~ 99% accuracy. Expected out-of-sample error = 1 - accuracy of cross-validation testing = 0.01

#Running The Random Forest model on the test data

test_prediction<- predict(RF_modelfit, test_data)
test_prediction
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
