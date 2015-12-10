##Test if data is downloaded and download if it doesn't exist
if(!file.exists("./train.csv")){download.file(url = "https://www.kaggle.com/c/titanic/download/train.csv", destfile = "./train.csv")}
if(!file.exists("./test.csv")){download.file(url = "https://www.kaggle.com/c/titanic/download/test.csv", destfile = "./test.csv")}

##Load dependencies and set random seed
library(caret, quietly=TRUE)
library(randomForest)
library(pROC)
set.seed(2233)

##Load the training and test data
training <- read.csv("./train.csv")
testing <- read.csv("./test.csv")

##Data Prep
training$Survived <- as.factor(training$Survived)
training$Age[is.na(training$Age)] <- mean(training$Age)
training$Fare[is.na(training$Fare)] <- median(training$Fare, na.rm=T)
training$Embarked[training$Embarked==""]<- "S"
training$Embarked<- as.factor(training$Embarked)
training$Sex <- as.factor(training$Sex)
training$Pclass <- as.factor(training$Pclass)

testing$Age[is.na(testing$Age)] <- mean(testing$Age)
testing$Fare[is.na(testing$Fare)] <- median(testing$Fare, na.rm=T)
testing$Embarked[testing$Embarked==""]<- "S"
testing$Embarked<- as.factor(testing$Embarked)
testing$Sex <- as.factor(testing$Sex)
testing$Pclass <- as.factor(testing$Pclass)

##Create new features
training$FamSize <- training$SibSp + training$Parch + 1
training$CabinIdent <- as.factor(ifelse(training$Cabin=="", 0, 1))

testing$FamSize <- testing$SibSp + testing$Parch + 1
testing$CabinIdent <- as.factor(ifelse(testing$Cabin=="", 0, 1))

##Create test and validation data sets
inTrain <- createDataPartition(y=training$Survived, p=0.8, list=FALSE)
data_train <- training[inTrain, ]
data_val <- training[-inTrain, ]
dim(data_train); dim(data_val)

##Exploring the Data
featurePlot(x=data_train[,c("Pclass","Age", "SibSp", "Parch", "Fare")],
            y= data_train$Survived, plot = "pairs")
plot(data_train$Fare, data_train$Cabin)
plot(data_train$Fare, data_train$Ticket)
plot(data_train$Fare, data_train$Embarked)
plot(data_train$Fare, data_train$Parch)
hist(data_train$Parch)

##Build and test logistic regression models
log.mod <- glm(Survived ~ Pclass+Sex+Age+FamSize+CabinIdent+Fare+Embarked, data = data_train, family = binomial(link = logit))
summary(log.mod)
data_train$SurvivedYhat <- predict(log.mod, type = "response")
data_train$SurvivedYhat <- ifelse(data_train$SurvivedYhat > 0.5, 1.0, 0.0)
confusionMatrix(data_train$Survived, data_train$SurvivedYhat)
auc(roc(data_train$Survived, data_train$SurvivedYhat))

##Export rpart prediction for submission
submission <- data.frame(PassengerId = testing$PassengerId)
submission$Survived <- predict(log.mod, testing, type = "response")
submission$Survived <- ifelse(submission$Survived > 0.5, 1.0, 0.0)
write.csv(submission, file = "log_reg_submission.csv", row.names=FALSE)

##Build and test random forest models
rpart.mod <- train(data_train$Survived ~ Pclass+Sex+Age+SibSp, data = data_train, method = "rpart")
confusionMatrix(data_val$Survived, predict(rpart.mod, data_val))

rpart.mod1 <- train(data_train$Survived ~ Pclass+Sex+Age+SibSp+Fare+Embarked, data = data_train, method = "rpart")
confusionMatrix(data_val$Survived, predict(rpart.mod1, data_val))

rpart.mod2 <- train(data_train$Survived ~ Pclass+Sex+Age+FamSize+CabinIdent+Fare, data = data_train, method = "rpart", preProcess = "pca")
confusionMatrix(data_val$Survived, predict(rpart.mod2, data_val))

rpart.mod.all <- train(training$Survived ~ Pclass+Sex+Age+SibSp, data = training, method = "rpart")
confusionMatrix(data_val$Survived, predict(rpart.mod.all, data_val))

##Export rpart prediction for submission
submission <- data.frame(PassengerId = testing$PassengerId)
submission$Survived <- predict(rpart.mod, testing)
write.csv(submission, file = "rpart_submission.csv", row.names=FALSE)

##Build and test neural net model
nnet.mod <- train(data_train$Survived ~ Pclass+Sex+Age+SibSp+Fare+Embarked, data = data_train, method = "nnet")
confusionMatrix(data_val$Survived, predict(nnet.mod, data_val))

nnet.mod1 <- train(data_train$Survived ~ Pclass+Sex+Age+Fare+Embarked+FamSize+CabinIdent, data = data_train, method = "nnet")
confusionMatrix(data_val$Survived, predict(nnet.mod1, data_val))

#nnet.mod2 <- train(Survived ~ Pclass+Sex+Age+Fare+Embarked+FamSize+CabinIdent, data = training, method = "nnet")

##Export neural net prediction for submission
submission <- data.frame(PassengerId = testing$PassengerId)
submission$Survived <- predict(nnet.mod1, testing)
write.csv(submission, file = "nnet1_submission.csv", row.names=FALSE)

##Other areas of exploration: imputing Age with linear model and pulling out title data from the Name field.
