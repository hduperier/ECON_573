## Establishing Libraries
library(ggplot2)
library(leaps)
library(ISLR2)
data("Boston")
attach(Boston)
library(glmnet)
library(pls)

## Question 10a, Chapter 6 Exercises

set.seed(1)
p<-20
n<-1000
x<-matrix(rnorm(n*p),n,p)
beta<-rnorm(p)
beta[4]<-0
beta[7]<-0
beta[9]<-0
beta[13]<-0
beta[19]<-0
epsilon<-rnorm(p)
y<-x%*%beta + epsilon


## Question 10b
data10<- data.frame(y,x)
n.dat<-dim(data10)[1]
set.seed(5)
rowsTest<- sample(1:n.dat, n.dat/10)
testing<-data10[rowsTest,]
dim(testing)

training<-data10[-rowsTest,]
dim(training)

## Question 10c
regressFitMod= regsubsets(y ~ ., training, nvmax=p)
valuErrors<-rep(NA, p)
colNames<- colnames(data10)
for (i in 1:p) {
  coefficient<- coef(regressFitMod, id=i)
  predic<- as.matrix(training[, colNames %in% names(coefficient)]) %*% coefficient[names(coefficient) %in% colNames]
  valuErrors[i]<- mean((training$y - predic)^2)
}
dt.val.errs<- data10.table(valuErrors, seq(1:20))
ggplot(dt.val.errs, aes(x<-V2, y<-valuErrors)) + geom_line()

## Question 11a
### Best Subset Selection
predict.regsubsets = function(object, newdata, id, ...) {
  formTest = as.formula(object$call[[2]])
  mat = model.matrix(formTest, newdata)
  coefi = coef(object, id = id)
  mat[, names(coefi)] %*% coefi
}

k = 10
p = ncol(Boston) - 1
folds = sample(rep(1:k, length = nrow(Boston)))
cv.errors = matrix(NA, k, p)
for (i in 1:k) {
  bestFit = regsubsets(crim ~ ., data = Boston[folds != i, ], nvmax = p)
  for (j in 1:p) {
    pred = predict(bestFit, Boston[folds == i, ], id = j)
    cv.errors[i, j] = mean((Boston$crim[folds == i] - pred)^2)
  }
}
rmse.cv = sqrt(apply(cv.errors, 2, mean))
plot(rmse.cv, pch = 19, type = "b")
summary(bestFit)
which.min(rmse.cv)
bostonBSMErr=(rmse.cv[which.min(rmse.cv)])^2
bostonBSMErr

print("Cross-validation selects a 9-variable model based on the Test MSE. At 9-variables, the CV estimate for the test MSE is 43.47287–the lowest MSE reported.")

### The Lasso
bostonX=model.matrix(crim~., data=Boston)[,-1]
bostonY=Boston$crim
bostonLasso=cv.glmnet(bostonX, bostonY, alpha=1, type.measure = "mse")
plot(bostonLasso)

print("To predict the training model on the test model, I need to find the lambda that reduces error the most.")
coef(bostonLasso)
bostonLassoErr<-(bostonLasso$cvm[bostonLasso$lambda==bostonLasso$lambda.1se])
bostonLassoErr
print("Lasso is inly a variable reduction method and because of this, the lasso model that reduces the MSE only includes 1 variable (rad) and has an MSE of 54.83663.")

### Ridge Regression
bostonRidgeReg=cv.glmnet(bostonX, bostonY, type.measure = "mse", alpha=0)
plot(bostonRidgeReg)
print("Ridge Regression attempts to keep all variables unlike the Lasso method.")
coef(bostonRidgeReg)
bostonRidgeErr<-bostonRidgeReg$cvm[bostonRidgeReg$lambda==bostonRidgeReg$lambda.lse]
bostonRidgeErr
print("Compared to the Best Subset Selection and the Lasso, the Ridge Regression does not perform well.")

### PCR
bostonPCR = pcr(crim~., data=Boston, scale=TRUE, validation="CV")
summary(bostonPCR)
print("The most appropiate PCR model would include 8 components and that would explain 93.45% of the predictors by the model. At 8 components, MSE is 44.38224. Overall, this model works pretty well.")

## Question 11b
print("Since the model that had the lowest cross-val error is the best subset selection model, I would propose this model as I computed above. This model also has an MSE of 43.47287.")

##Question 11c
print("The Best Subset Selection Model only includes 9 variable as I explained above. More variation of the response would be included if the model were to include the left out features. Since we are aiming to have low variance and low MSE in the model prediction accuracy, this is the best since the variables it includes are 'zn, indus, nox, dis, rad, ptratio, black, lstat, and medv'")
