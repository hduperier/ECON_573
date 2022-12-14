install.packages("ISLR2")
library(ISLR2)
data("Auto")

#Question 1
print("Conceptual Exercises")
print("1. The null hypothesis for radio is that in the presence of TV and newspaper ads, radio ads have no effect on sales. The null hypothesis for TV is that in the presence of radio and newspaper ads, TV ads have no effect on sales. The null hypothesis for newpaper is that in the presense of radio and TV ads, newspaper ads have no effect on sales. The low p-values of TV and Radio suggest that their null hypotheses are false and the high p-value of newspaper suggests that the null hypothesis is true.")

#Question 3
print("")
print("3b: 137.1")
print("3c: This is false. To determine if the interaction term is statistically significant or not, we must look at the p-value of the regression coefficient.")

#Question 4
print("4a: We would expect the cubic regression to have a lower training RSS than the linear regression because it could potentially allow for a tighter fit.")
print("4b: We would expect the cubic regression to have a higher test RSS as the overfit would have more error than the linear regression.")
print("4c: We would expect the cubic regression to have a lower training RSS than the linear regression because of the cubic regression's higher flexibility. Since it has higher flexibility, it will follow the point more closely and reduce training error when compared to a linear regression.")
print("4d: Since we don't know how far from linear the relationship is between X and Y, there is not enough information to determine which regression would give a lower test RSS. If closer to linear than cubic, the linear regression would give a lower test RR, but if the inverse was true, it would potentially have a higher test RSS.")

#Question 8a
simple.fit<-lm(mpg~horsepower, data=Auto)
summary(simple.fit)
print("8ai. Yes, there is a relationship between the predictor and the response since the p-value<2.2e-16.")
print("8aii. The R-squared value indicates that about 61% of the variation in mpg (response variable) is due to horsepower (predictor variable).")
print("8aiii. The relationship between the predictor (horsepower) and the response (mpg) is negative.")
print("8aiv. The predicted mpg associated with a horsepower of 98 is 24.47.")
q8PredMPG<-predict(simple.fit,data.frame(horsepower=c(98)), interval="prediction")
q8ConfMPG<-predict(simple.fit,data.frame(horsepower=c(98)), interval="confidence")
print(q8PredMPG)
print(q8ConfMPG)


#Question 8b
attach(Auto)
plot(horsepower, mpg)
abline(simple.fit,lwd=5, col="red")

#Question 8c
which.max(hatvalues(simple.fit))
par(mfrow=c(2,2))
plot(simple.fit)

#Question 9a
pairs(Auto)

#Question 9b
Auto$name<-NULL
cor(Auto,method= c("pearson"))

#Question 9c
q9sim.fit<-lm(mpg~.,data=Auto)
summary(q9sim.fit)
print("9ci. Yes, there is a relationship between the predictors and the response because the p-value<2.2e-16")
print("9cii. Displacement, weight, year, and origin appear to have a statistically significant relationship to the response.")
print("9ciii. The coefficient for the year variable suggests when every other variable held constant, the mpg value increases about 0.75 with each year that passes, meaning as cars get newer, the mpg goes up.")

#Question 9d
which.max(hatvalues(q9sim.fit))
par(mfrow = c(2,2))
plot(q9sim.fit)
print("9d:The Residuals Vs Fitted graph shows that there is a non-linear relationship between the response and predictors. The next graph shows that the residuals are normally distributed and slightly right skewed. The third graph shows that the constant variance of error assumption is not true for this model and the last graph shows that there are no leverage points, but there is an observation that stands out as a potential leverage point.")

#Question 9e
q9sim.fit<-lm(mpg ~.-name+displacement:weight, data = Auto)
summary(q9sim.fit)
print("9e: Yes, some interactions do appear to be statistically significant.")

#Question 9f
q9sim.fit = lm(mpg ~.-name+I((displacement)^2)+log(displacement)+displacement:weight, data = Auto)
summary(q9sim.fit)

#Question 13a
set.seed(1)
x<-rnorm(100)

#Question 13b
eps<-rnorm(100,sd=sqrt(0.25))

#Question 13c
y<- -1+0.5*x+eps
print(length(y))
print("B0=-1   B1=0.5")

#Question 13d
plot(x, y)
print("13d: The relationship between x and y looks to be mostly linear with slight noise introduced by the eps variable.")

#Question 13e
q13sim.fit<-lm(y~x)
summary(q13sim.fit)
print("13e: The hat values are pretty similar to the original values.")

#Question 13f
plot(x,y)
abline(q13sim.fit, col="red")
abline(-1, 0.5, col="green")
legend("topleft", c("least-square","regress"), col=c("red", "green"), lty= c(1,1))

#Question 13g
q13gsim.fit<-lm(y~x+ I(x^2))
summary(q13gsim.fit)
print("13g: There is not sufficient evidence that the quadratic term improves the model fit as its p-value is higher than 0.05.")

#Question 13h
set.seed(1)
eps <- rnorm(100, sd = 0.125)
x <- rnorm(100)
y <- -1 + 0.5 * x + eps
plot(x, y)
q13hsim.fit <- lm(y~x)
summary(q13hsim.fit)
abline(q13hsim.fit, col = "red")
abline(-1, 0.5, col = "green")
legend("topleft", c("Least square", "Regression"), col = c("red", "green"), lty = c(1, 1))
print("13h: Reduced the noise by decreasing the variance of the normal distribution used to generate the error term. The relationship is now mostly linear and has a much highger R^2")


#Question 13i
set.seed(1)
eps<- rnorm(100, sd = 0.5)
x<- rnorm(100)
y<- -1 + 0.5 * x + eps
plot(x, y)
q13isim.fit<- lm(y~x)
summary(q13isim.fit)
abline(q13isim.fit, col = "red")
abline(-1, 0.5, col = "green")
legend("topleft", c("Least square", "Regression"), col = c("red", "green"), lty = c(1, 1))
print("13i: We increased the noise by increasing the variance of the normal distribution used to generate the error term. We have a much lower R^2 and RSE.")

#Question 13j
confint(q13sim.fit)
confint(q13hsim.fit)
confint(q13isim.fit)
print("13j: As noise increases, so does the interval. The opposite also seems to be true.")

#Question 15a
data("Boston")
q15znsim.fit<-lm(crim~zn, data=Boston)
summary(q15znsim.fit)
q15indus.fit<-lm(crim~indus, data=Boston)
summary(q15indus.fit)
q15chas.fit<-lm(crim~chas, data=Boston)
summary(q15chas.fit)
q15nox.fit<-lm(crim~nox, data=Boston)
summary(q15nox.fit)
q15rm.fit<-lm(crim~rm, data=Boston)
summary(q15rm.fit)
q15age.fit<-lm(crim~age, data=Boston)
summary(q15age.fit)
q15dis.fit<-lm(crim~dis, data=Boston)
summary(q15dis.fit)
q15rad.fit<-lm(crim~rad, data=Boston)
summary(q15rad.fit)
q15tax.fit<-lm(crim~tax, data=Boston)
summary(q15tax.fit)
q15ptratio.fit<-lm(crim~ptratio, data=Boston)
summary(q15ptratio.fit)
q15lstat.fit<-lm(crim~lstat, data=Boston)
summary(q15lstat.fit)
q15medv.fit<-lm(crim~medv, data=Boston)
summary(q15medv.fit)
print("15a :Each predictor is statistically significant except for chas as it is the only predictor to have a p-value > 0.05.")

#Question 15b
q15bsim.fit<-lm(crim~.,data=Boston)
summary(q15bsim.fit)
print("15b: From this, we can reject the null hypothesis for the following predictors: zn, dis, rad, and medv.")

#Question 15c
simple.reg<- vector("numeric",0)
simple.reg<- c(simple.reg, fit.zn$coefficient[2])
simple.reg<- c(simple.reg, fit.indus$coefficient[2])
simple.reg<- c(simple.reg, fit.chas$coefficient[2])
simple.reg<- c(simple.reg, fit.nox$coefficient[2])
simple.reg<- c(simple.reg, fit.rm$coefficient[2])
simple.reg<- c(simple.reg, fit.age$coefficient[2])
simple.reg<- c(simple.reg, fit.dis$coefficient[2])
simple.reg<- c(simple.reg, fit.rad$coefficient[2])
simple.reg<- c(simple.reg, fit.tax$coefficient[2])
simple.reg<- c(simple.reg, fit.ptratio$coefficient[2])
simple.reg<- c(simple.reg, fit.black$coefficient[2])
simple.reg<- c(simple.reg, fit.lstat$coefficient[2])
simple.reg<- c(simple.reg, fit.medv$coefficient[2])
mult.reg<- vector("numeric", 0)
mult.reg<- c(mult.reg, fit.all$coefficients)
mult.reg<- mult.reg[-1]
plot(simple.reg, mult.reg, col = "blue")
print("15c: There is difference between the simple and multi regression coefficients due to that in the simple regresssion, the slope term represents the average effect of an increase in the predictor, while ignoring other predictors, but in the multi regress, the slope term represents the average effect of an increase in the predictor, while holding other predictors fixed. This difference allows for the multi regression to suggest no relationship between the response and some of the predictors while the simple regression implies the opposite since the correlations between the predictors show some strong relationships.")

#Question 15d
zn2.fit<- lm(crim ~ poly(zn, 3), data=Boston)
summary(zn2.fit)
indus2.fit <- lm(crim ~ poly(indus, 3), data=Boston)
summary(indus2.fit)
nox2.fit <- lm(crim ~ poly(nox, 3), data=Boston)
summary(nox2.fit)
rm2.fit <- lm(crim ~ poly(rm, 3), data=Boston)
summary(rm2.fit)
age2.fit <- lm(crim ~ poly(age, 3), data=Boston)
summary(age2.fit)
dis2.fit <- lm(crim ~ poly(dis, 3), data=Boston)
summary(dis2.fit)
rad2.fit <- lm(crim ~ poly(rad, 3), data=Boston)
summary(rad2.fit)
tax2.fit <- lm(crim ~ poly(tax, 3), data=Boston)
summary(tax2.fit)
ptratio2.fit <- lm(crim ~ poly(ptratio, 3), data=Boston)
summary(ptratio2.fit)
fit.lstat2 <- lm(crim ~ poly(lstat, 3), data=Boston)
summary(fit.lstat2)
fit.medv2 <- lm(crim ~ poly(medv, 3), data=Boston)
summary(fit.medv2)
print("15d: For zn, rm, rad, tax, and lstat as predictors, the p-values suggest that the cubic coefficient is not statistically significant. For indus, nox, age, dis, ptratio, and medv as predictor, the p-values suggest the cubic coefficient is ok.")
