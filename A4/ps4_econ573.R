library(ISLR2)
library(gbm)
library(glmnet)
library(randomForest)
library(MASS)
library(tidyverse)
library(ggplot2)
library(ggthemes)
library(broom)
library(knitr)
library(caret)
library(splines)

# Chapter 5
## 3a
print("The k-fold cross-validation is implemented by dividing the set of observations into k folds of approximately equal size. This first fold is treated as a validation set, and the method is fit on the remaining k-1 folds. The mean squared error is then computed on the observations in the held-out gold. This procedure is repeated k times; each time, a different group of observations is treated as a validation set. This process results in k estimates of the test error, MSE1, MSE2, …, MSEk. The k-fold CV estimate is computer by averaging these values. ")
## 5a
data("Default")
set.seed(1)
glm.fit <- glm(default ~ income + balance, data = Default, family = binomial)
summary(glm.fit)

## 5b
### 5bi
trainingSet<- sample(dim(Default)[1], dim(Default)[1]/2)
### 5bii
glmFit<- glm(default ~ income + balance, data=Default, family=binomial, subset=traingSet)
### 5biii
glm
### 5biv

## 5c

# Chapter 7
## 9a
data("Boston")
set.seed(1)
theme_set(theme_tufte(base_size = 20) + theme(legend.position = 'top'))
data('Boston')

model <- lm(nox ~ poly(dis, 3), data = Boston)
tidy(model) %>%
  kable(digits = 3)
Boston %>%
  mutate(pred = predict(model, Boston)) %>%
  ggplot() +
  geom_point(aes(dis, nox, col = '1')) +
  geom_line(aes(dis, pred, col = '2'), size = 1) +
  scale_color_manual(name = 'Value Type/Color:',
                     labels = c('Observed', 'Predicted'),
                     values = c('#7BAFD4', '#8B0000'))
print("Each power of the dis coefficient is found to be statistically significant according to the model. The plot also seems to describe the data without overfitting.")

## 9b
errors <- list()
models <- list()
pred_df <- data_frame(V1 = 1:506)
for (i in 1:9) {
  models[[i]] <- lm(nox ~ poly(dis, i), data = Boston)
  preds <- predict(models[[i]])
  pred_df[[i]] <- preds
  errors[[i]] <- sqrt(mean((Boston$nox - preds)^2))
}
errors <- unlist(errors)
names(pred_df) <- paste('Level', 1:9)
data_frame(RMSE = errors) %>%
  mutate(Poly = row_number()) %>%
  ggplot(aes(Poly, RMSE, fill = Poly == which.min(errors))) +
  geom_col() + 
  guides(fill = "none") +
  scale_x_continuous(breaks = 1:9) +
  coord_cartesian(ylim = c(min(errors), max(errors))) +
  labs(x = 'Degree Rate')
print("From this we see that the model with the highest polynomial degree has the lowest RSS when fitted and tested on the same data.")
Boston %>%
  cbind(pred_df) %>%
  gather(Polynomial, prediction, -(1:14)) %>%
  mutate(Polynomial = factor(Polynomial, 
                             levels = unique(as.character(Polynomial)))) %>%
  ggplot() + 
  ggtitle('Predicted Values for Each Level of Polynomial') +
  geom_point(aes(dis, nox, col = '1')) + 
  geom_line(aes(dis, prediction, col = '2'), size = 1) +
  scale_color_manual(name = 'Value Type',
                     labels = c('Observed', 'Predicted'),
                     values = c('#7BAFD4', '#8B0000')) +
  facet_wrap(~ Polynomial, nrow = 2)

## 9c
errors <- list()
folds <- sample(1:10, 506, replace = TRUE)
errors <- matrix(NA, 10, 9)
for (k in 1:10) {
  for (i in 1:9) {
    model <- lm(nox ~ poly(dis, i), data = Boston[folds != k,])
    pred <- predict(model, Boston[folds == k,])
    errors[k, i] <- sqrt(mean((Boston$nox[folds == k] - pred)^2))
  }
}
errors <- apply(errors, 2, mean)
data_frame(RMSE = errors) %>%
  mutate(Poly = row_number()) %>%
  ggplot(aes(Poly, RMSE, fill = Poly == which.min(errors))) +
  geom_col() + theme_tufte() + guides(fill = FALSE) +
  scale_x_continuous(breaks = 1:9) +
  coord_cartesian(ylim = range(errors))
print("The model with a polynomial degree of 4 is chosen when tested on out-of-sample data. This is the highest polynomial degree that does not show signs of overfitting like 5 through 9 do.")

## 9d
model <- lm(nox ~ bs(dis, df = 4), data = Boston)
kable(tidy(model), digits = 3)
Boston %>%
  mutate(pred = predict(model)) %>%
  ggplot() +
  geom_point(aes(dis, nox, col = '1')) + 
  geom_line(aes(dis, pred, col = '2'), size = 1.5) +
  scale_color_manual(name = 'Value Type',
                     labels = c('Observed', 'Predicted'),
                     values = c('#7BAFD4', '#8B0000')) +
  theme_tufte(base_size = 13)
print("This model finds all the different bases to be statistically significant and the pred line seems to fit data well without overfitting.")

## 9e
errors <- list()
models <- list()
pred_df <- data_frame(V1 = 1:506)
for (i in 1:9) {
  models[[i]] <- lm(nox ~ bs(dis, df = i), data = Boston)
  preds <- predict(models[[i]])
  pred_df[[i]] <- preds
  errors[[i]] <- sqrt(mean((Boston$nox - preds)^2))
}

names(pred_df) <- paste(1:9, 'Degrees of Freedom')
data_frame(RMSE = unlist(errors)) %>%
  mutate(df = row_number()) %>%
  ggplot(aes(df, RMSE, fill = df == which.min(errors))) +
  geom_col() + guides(fill = FALSE) + theme_tufte() +
  scale_x_continuous(breaks = 1:9) +
  coord_cartesian(ylim = range(errors))
Boston %>%
  cbind(pred_df) %>%
  gather(df, prediction, -(1:14)) %>%
  mutate(df = factor(df, levels = unique(as.character(df)))) %>%
  ggplot() + ggtitle('Predicted Values for Each Level of Polynomial') +
  geom_point(aes(dis, nox, col = '1')) + 
  geom_line(aes(dis, prediction, col = '2'), size = 1.5) +
  scale_color_manual(name = 'Value Type',
                     labels = c('Observed', 'Predicted'),
                     values = c('#56B4E9', '#E69F00')) +
  facet_wrap(~ df, nrow = 3)
print("When trained and tested on the same data, the higher complexity models are deemed the best as shown on the plots above.")

## 9f
folds <- sample(1:10, size = 506, replace = TRUE)
errors <- matrix(NA, 10, 9)
models <- list()
for (k in 1:10) {
  for (i in 1:9) {
    models[[i]] <- lm(nox ~ bs(nox, df = i), data = Boston[folds != k,])
    pred <- predict(models[[i]], Boston[folds == k,])
    errors[k, i] <- sqrt(mean((Boston$nox[folds == k] - pred)^2))
  }
}
errors <- apply(errors, 2, mean)
data_frame(RMSE = errors) %>%
  mutate(df = row_number()) %>%
  ggplot(aes(df, RMSE, fill = df == which.min(errors))) +
  geom_col() + theme_tufte() + guides(fill = FALSE) +
  scale_x_continuous(breaks = 1:9) +
  coord_cartesian(ylim = range(errors))

print("These were validated on out-of-sample data. Due to this, a simpler model is chosen, with df of 4. This is similar to polynomial validation, in that this is the most complex model that does not begin to show signs of overfitting like 5 through 9 do.")

# Chapter 8
## 10a
data("Hitters")
Hitters = na.omit(Hitters)
Hitters$Salary = log(Hitters$Salary)

## 10b
train = 1:200
hitters.train = Hitters[train,]
hitters.test = Hitters[-train,]

## 10c
set.seed(1)
pows = seq(-10, -0.2, by = 0.1)
lambdas = 10^pows
train.err = rep(NA, length(lambdas))
for (i in 1:length(lambdas)) {
  boost.hitters = gbm(Salary ~ ., data = hitters.train, distribution = "gaussian", n.trees = 1000, shrinkage = lambdas[i])
  pred.train = predict(boost.hitters, hitters.train, n.trees = 1000)
  train.err[i] = mean((pred.train - hitters.train$Salary)^2)
}
plot(lambdas, train.err, type = "b", xlab = "Shrinkage-values", ylab = "Training-MSE")

## 10d
set.seed(1)
test.err <- rep(NA, length(lambdas))
for (i in 1:length(lambdas)) {
  boost.hitters = gbm(Salary ~ ., data = hitters.train, distribution = "gaussian", n.trees = 1000, shrinkage = lambdas[i])
  yhat = predict(boost.hitters, hitters.test, n.trees = 1000)
  test.err[i] = mean((yhat - hitters.test$Salary)^2)
}
plot(lambdas, test.err, type = "b", xlab = "Shrinkage-values", ylab = "Test-MSE")
min(test.err)
lambdas[which.min(test.err)]

## 10e
fitFirst = lm(Salary ~ ., data = hitters.train)
predFirst = predict(fitFirst, hitters.test)
mean((predFirst - hitters.test$Salary)^2)
x = model.matrix(Salary ~ ., data = hitters.train)
x.test = model.matrix(Salary ~ ., data = hitters.test)
y = hitters.train$Salary
fitSnd = glmnet(x, y, alpha = 0)
predSnd = predict(fitSnd, s = 0.01, newx = x.test)
mean((predSnd - hitters.test$Salary)^2)

print("The test MSE for boosting is lower than for linear and ridge regression shown by the data above.")

## 10f
boost.hitters <- gbm(Salary ~ ., data = hitters.train, distribution = "gaussian", n.trees = 1000, shrinkage = lambdas[which.min(test.err)])
summary(boost.hitters)
print("The variable 'CAtBat' is the most important predictor in the boosted model.")

## 10g
set.seed(1)
bag.hitters <- randomForest(Salary ~ ., data = hitters.train, mtry = 19, ntree = 500)
yhat.bag <- predict(bag.hitters, newdata = hitters.test)
mean((yhat.bag - hitters.test$Salary)^2)
print("The test set MSE for this approach is 0.2299324 which is slightly lower than the test MSE for boosting.")
