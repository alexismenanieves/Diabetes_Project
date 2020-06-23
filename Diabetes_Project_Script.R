# Header ------------------------------------------------------------------
# Diabetes Project Script
# Author: Manuel Alexis Mena Nieves
# Date: June 22,2020
# Repo: https://github.com/alexismenanieves/Diabetes_Project

# Step 1. Getting the data ------------------------------------------------

# Load the libraries
repo <- "http://cran.us.r-project.org"
if(!require(tidyverse)) install.packages("tidyverse", repos = repo)
if(!require(caret)) install.packages("caret", repos = repo)
if(!require(skimr)) install.packages("skimr", repos = repo)
if(!require(rpart)) install.packages("rpart", repos = repo)
if(!require(randomForest)) install.packages("randomForest", repos = repo)
if(!require(gbm)) install.packages("gbm", repos = repo)
if(!require(kernlab) install.packages("kernlab", repos = repo)

# Read the file
url <- paste0("https://raw.githubusercontent.com/alexismenanieves/",
              "Diabetes_Project/master/dataset.txt")
dataset <- read.csv(url)

# A first view of the data, dimensions and variables
dim(dataset)
as_tibble(dataset)
summary(dataset)
dataset %>% group_by(Outcome) %>% 
  summarise(count = n()) %>% mutate(freq = count/sum(count))

# Let's apply some changes on the outcome name and encoding
dataset$Outcome <- as.factor(ifelse(dataset$Outcome == 1,"Yes","No"))
names(dataset)[9]<- "Diabetes"

# Let's divide the dataset into a train and test set
set.seed(1979)
tt_index <- createDataPartition(dataset$Age, times = 1, p = 0.9, list = FALSE)
train_set <- dataset[tt_index,]
test_set <- dataset[-tt_index,]

# Step 2. Exploratory Analysis --------------------------------------------

# See how many observations and variables are available
str(train_set)
# Glimpse of mean, median and NA's
summary(train_set)
# Count how many zeros are in each variable
colSums(train_set[,-9] == 0, na.rm = TRUE)
# Convert zeros to NA
train_set[,c(2:6)] <- apply(train_set[,c(2:6)], 2, function(x) {ifelse(x==0, NA, x)} )
# See the descriptive stats from the data
skim(train_set)
# Plot all variables and see the effects on outcome
train_set %>% gather(key = "Variable", value = "Measure", -Diabetes) %>% 
  ggplot(aes(Measure, fill = Diabetes)) + geom_density(alpha = 0.3) + 
  facet_wrap(~Variable,ncol = 4, scales = "free")
train_set %>% gather(key = "Variable", value = "Measure", -Diabetes) %>% 
  ggplot(aes(Diabetes, Measure, fill = Diabetes)) + geom_boxplot() + 
  facet_wrap(~Variable,ncol = 4, scales = "free")

# Step 3. Preprocess the data ---------------------------------------------

# Set seed
set.seed(1979)
# Divide the train set into a new train set and a validation set
tv_index <- createDataPartition(train_set$Age, times = 1, p = 0.8, list = FALSE)
validation_set <- train_set[-tv_index,]
train_set <- train_set[tv_index,]

# Proceed with preprocessing, imputing median values into NAs and applying range
preProcess_data <- preProcess(train_set, method = c("medianImpute","range"))
train_set <- predict(preProcess_data, newdata = train_set)

# Step 4. Create a simple ML model and obtain metrics ---------------------

# Train knn on train set for a first glimpse of accuracy
model_KNN <- train(Diabetes ~., 
                   data = train_set, 
                   trControl = trainControl(sampling = "down"),
                   tuneGrid = data.frame(k = seq(3,50,1)),
                   method = "knn")

# Plot accuracy 
plot(model_KNN, main = "Accuracy of KNN model")
# Plot variable importance
plot(varImp(model_KNN), main = "Variable importance for KNN model")
model_KNN$bestTune
max(model_KNN$results$Accuracy)

# Predict the results using knn model and evaluate accuracy
validation_set <- predict(preProcess_data, newdata = validation_set)
final_model <- predict(model_KNN, newdata = validation_set)
confusionMatrix(final_model, validation_set$Diabetes)$overall[["Accuracy"]]

# Step 5. Create an ensemble model and select the best model --------------
models <- c("knn","rpart","rf","gbm","svmLinear")
knn_tg = data.frame(k = seq(3,50,1))
rpart_tg = data.frame(cp = seq(0,0.1,len = 50))
rf_tg = data.frame(mtry = seq(1,8,1))
gbm_tg = expand.grid(n.trees = c(100, 1000, 10000),
                     interaction.depth = c(1,3,5),
                     shrinkage = 0.001,
                     n.minobsinnode = 1)
svmLinear_tg = data.frame(C = c(0, 0.01, 0.05, 0.1, seq(0.25, 2, 0.25)))

suppressMessages(fits <- lapply(models, function(model){ 
  print(model)
  train(Diabetes ~ ., 
        data = train_set, 
        trControl = trainControl(sampling = "up"),
        tuneGrid = switch(model,
                          "knn" = knn_tg,
                          "rpart" = rpart_tg,
                          "rf" = rf_tg,
                          "gbm" = gbm_tg,
                          "svmLinear" = svmLinear_tg),
        method = model)
}))

names(fits) <- models

pred <- sapply(fits, function(object) 
  predict(object, newdata = validation_set))
dim(pred)

acc <- colMeans(pred == validation_set$Diabetes)
acc
mean(acc)
votes <- rowMeans(pred == "Yes")
y_hat <- ifelse(votes > 0.5, "Yes", "No")
mean(y_hat == validation_set$Diabetes)

ind <- acc >= mean(acc)
best_votes <- rowMeans(pred[,ind]=="Yes")
best_y_hat <- ifelse(best_votes > 0.5, "Yes", "No")
mean(best_y_hat == validation_set$Diabetes)
