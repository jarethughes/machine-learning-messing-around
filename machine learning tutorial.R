#Setting up
library(caret)
setwd("C:/Users/Jaret/Documents/GitHub/machine-learning-messing-around")

#Load dataset
rdat = read.csv("iris.csv", header=FALSE)

#
#Prep data
#

#Name the columns
colnames(rdat) = c("Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width", "Species")

#Create a validation index of 80% of the dataset to training
val_index = createDataPartition(rdat$Species, p=0.80, list=FALSE)

#Section off the data for validation
val = rdat[-val_index,]
rdat = rdat[val_index,]
val$Species = factor(val$Species)
#
#Data Summary
#

#Gives the dimensions of the dataset
dim(rdat)

#Lists out the types of each attribute in the dataset
sapply(rdat, class)

#Render a preview of the dataset
head(rdat)

#Look at the class distribution in the data
percent = prop.table(table(rdat$Species))*100
cbind(freq=table(rdat$Species), percentage=percent)

#Summary Statistics
summary(rdat)

#
#Visualize the dataset
#

#Split inputs and output
x = rdat[,1:4]
y = factor(rdat[,5])

#Create boxplots for each attribute, all in one image
par(mfrow=c(1,4))
  for(i in 1:4){
    boxplot(x[,i], main=names(iris)[i])
  }

#Scatterplot matrix
featurePlot(x=x, y=y, plot="ellipse")

#Box and whisker plot for each attribute
featurePlot(x=x, y=y, plot="box")

#Density plots for each attribute by class
scales = list(x=list(relation="free"), y=list(relation="free"))
featurePlot(x=x, y=y, plot="density", scales=scales)

#
#Evaluate some algorithms
#

#Set up 10-fold cross validation
control = trainControl(method="cv", number=10)
metric = "Accuracy"

#A.) Linear algorithms
set.seed(7)
fit.lda = train(Species~., data=rdat, method="lda", metric=metric, trControl=control)

#B.) Nonlinear algorithms
#CART
set.seed(7)
fit.cart = train(Species~., data=rdat, method="rpart", metric=metric, trControl=control)
#kNN
set.seed(7)
fit.knn = train(Species~., data=rdat, method="knn", metric=metric, trControl=control)

#C.) Advanced Algorithms
#SVM
set.seed(7)
fit.svm = train(Species~., data=rdat, method="svmRadial", metric=metric, trControl=control)
#Random Forest
set.seed(7)
fit.rf = train(Species~., data=rdat, method="rf", metric=metric, trControl=control)

#Summarize accuracy of the models
results = resamples(list(lda=fit.lda, cart=fit.cart, knn=fit.knn, svm=fit.svm, rf=fit.rf))
summary(results)

#Compare accuracy of models
dotplot(results)
#The best model is shown to be LDA...

#Summarize best model
print(fit.lda)

#
#Make predictions
#

#Estimate skill of LDA using validation set
predicts = predict(fit.lda, val)
confusionMatrix(predicts, val$Species)
