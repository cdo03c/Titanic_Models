setwd("~/Documents/Github/Titanic_Models")

##Test if data is downloaded and download if it doesn't exist
if(!file.exists("./train.csv")){download.file(url = "https://www.kaggle.com/c/titanic/download/train.csv", destfile = "./train.csv")}
if(!file.exists("./test.csv")){download.file(url = "https://www.kaggle.com/c/titanic/download/test.csv", destfile = "./test.csv")}

##Load dependencies and set random seed
library(caret, quietly=TRUE)
library(xgboost)
library(Ckmeans.1d.dp)
library(readr)
library(stringr)
library(caret)
library(car)
library(randomForest)
library(pROC)
set.seed(2233)

##Load the training and test data
training <- read.csv("./train.csv")
testing <- read.csv("./test.csv")

##Combine training and test data for prep
training$Source <- "Train"
testing$Source <- "Test"
testing$Survived <- NA
df <- rbind(training, testing)

##Data Prep
df$Survived <- as.factor(df$Survived)
df$Fare[is.na(df$Fare)] <- median(df$Fare, na.rm=T)
df$Embarked[df$Embarked==""]<- "S"
df$Embarked<- as.factor(df$Embarked)
df$Sex <- as.factor(df$Sex)
df$Pclass <- as.factor(df$Pclass)

##Create new features
df$FamSize <- df$SibSp + df$Parch + 1
df$CabinIdent <- as.factor(ifelse(df$Cabin=="", 0, 1))
#Extract the title from the Name field
for(i in 1:nrow(df)){
    t <- strsplit(as.character(df$Name), ",")[[i]]
    y <- strsplit(t[2], " ")[[1]]
    df$title[i] <- y[2]
}
#Turn title into a factor
df$title <- as.factor(df$title)
#Recode the levels for title into three levels: 1 is formal title,
#2 is yound person's title, and 3 is normal title
levels(df$title) <- c(1,1,1,1,1,1,1,1,2,2,2,2,3,3,2,1,1,1)

#Builds a linear model for age
agemod <- lm(Age ~ title+FamSize+Sex+Pclass+Embarked-1, df)
#Imputes missing Age values from linear model
df$Age[is.na(df$Age)] <- round(predict(agemod, df))

#Removed the Passenger ID and Embarked based evaluation of feature importance
df <- df[,!(names(df) %in% "Embarked")]

#One Hot Encode variables and remove non-numeric columns
ohe_feats = 'Sex'
dummies <- dummyVars(~ Sex, data = df)
df_all_ohe <- as.data.frame(predict(dummies, newdata = df))
df_all_combined <- cbind(df[,-c(which(colnames(df) %in% ohe_feats))],df_all_ohe)

drops <- c("Name", "Ticket", "Cabin")
df <- df_all_combined[ , !(names(df_all_combined) %in% drops)]


rm(df_all_ohe, df_all_combined)

##Create test and validation data sets
testing <- df[df$Source == "Test",]
training <- df[df$Source=="Train",]
#Removed the source field and the survived field for testing only
training <- training[,!(names(training) %in% c("Source", "PassengerId""))]
testing <- testing[,!(names(testing) %in% c("Source", "Survived"))]
inTrain <- createDataPartition(y=training$Survived, p=0.8, list=FALSE)
data_train <- training[inTrain, ]
data_val <- training[-inTrain, ]

labels <- data_train$Survived
labels <- ifelse(labels == 1 , 0, 1)
data_train <- data_train[,!(names(data_train) %in% "Survived")]

#dim(data_train); dim(data_val)

##Build XGBoost model
xgb <- xgboost(data = data.matrix(data_train), 
               label = labels, 
               verbose = 2,
               eta = 0.1,
               nround=25, 
               eval_metric = "error",
               #eval_metric = "merror",
               objective = "binary:logistic",
               nthread = 2
)

#Importance Matrix
names <- dimnames(data.matrix(data_train))[[2]]
importance_matrix <- xgb.importance(names, model = xgb)
xgb.plot.importance(importance_matrix)

# predict values in test set
y_pred <- predict(xgb, data.matrix(data_val[,-1]))
y_pred <- as.numeric(y_pred > 0.5)
confusionMatrix(data_val$Survived,y_pred)

##Build Final XGBoost model
labels <- as.numeric(training$Survived)
labels <- ifelse(labels == 1 , 0, 1)
training <- training[,!(names(training) %in% "Survived")]
xgb <- xgboost(data = data.matrix(training), 
               label = labels, 
               verbose = 2,
               eta = 0.1,
               nround=25, 
               eval_metric = "error",
               objective = "binary:logistic",
               nthread = 2
)

##Export neural net prediction for submission
submission <- data.frame(PassengerId = testing$PassengerId)
submission$Survived 
pred <- predict(xgb, data.matrix(testing[,-1]))
write.csv(submission, file = "nnet2_submission.csv", row.names=FALSE)
