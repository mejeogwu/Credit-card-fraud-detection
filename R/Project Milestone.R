############################################################
# TOTAL MILESTONE PROJECT: FRAUD DETECTION ANALYSIS
# Models: Logistic Regression and Decision Tree
############################################################

# -----------------------------
# 0. Install and load packages
# -----------------------------

packages <- c("ggplot2", "pROC", "rpart")

for (pkg in packages) {
  if (!require(pkg, character.only = TRUE)) {
    install.packages(pkg)
    library(pkg, character.only = TRUE)
  }
}

# -----------------------------
# I. Import datasets
# -----------------------------

ccd <- read.csv("fraudTrain.csv")
ccdt <- read.csv("fraudTest.csv")

# Keep original datasets unchanged for reference
ccd_original <- ccd
ccdt_original <- ccdt

# -----------------------------
# II. Dataset exloration
# -----------------------------

# Dimension of the datasets
dim(ccd)
dim(ccdt)

# First and last five transactions in train dataset
head(ccd, 5)
tail(ccd, 5)

# First and last five transactions in test dataset
head(ccdt, 5)
tail(ccdt, 5)

# Check missing values
sum(is.na(ccd))
sum(is.na(ccdt))

# Feature names
names(ccd)
names(ccdt)

# Summary statistics for transaction amount
summary(ccd$amt)
summary(ccdt$amt)

# Variance and standard deviation for transaction amount
var(ccd$amt)
sd(ccd$amt)
var(ccdt$amt)
sd(ccdt$amt)

# Count transactions where amount is greater than the mean
sum(ccd$amt > mean(ccd$amt, na.rm = TRUE))
sum(ccdt$amt > mean(ccdt$amt, na.rm = TRUE))

# -----------------------------
# 3. Prepare fraud labels for exploration
# -----------------------------

# Create readable labels without overwriting the original numeric target
ccd$fraud_label <- factor(
  ccd$is_fraud,
  levels = c(0, 1),
  labels = c("Normal", "Fraud")
)

ccdt$fraud_label <- factor(
  ccdt$is_fraud,
  levels = c(0, 1),
  labels = c("Normal", "Fraud")
)

# -----------------------------
# 4. Violin plots with boxplots
# -----------------------------

# Train violin plot with boxplot inside
ggplot(ccd, aes(x = fraud_label, y = amt, fill = fraud_label)) +
  geom_violin(trim = FALSE) +
  geom_boxplot(width = 0.15, outlier.color = "blue", fill = "gray") +
  scale_y_log10() +
  labs(
    title = "Violin Plot of Transaction Amount by Class - Train Dataset",
    x = "Transaction Class",
    y = "Amount of Money Transferred"
  ) +
  theme_minimal()

# Test violin plot with boxplot inside
ggplot(ccdt, aes(x = fraud_label, y = amt, fill = fraud_label)) +
  geom_violin(trim = FALSE) +
  geom_boxplot(width = 0.15, outlier.color = "red", fill = "green") +
  scale_y_log10() +
  labs(
    title = "Violin Plot of Transaction Amount by Class - Test Dataset",
    x = "Transaction Class",
    y = "Amount of Money Transferred"
  ) +
  theme_minimal()

############################################################
# MILESTONE 3: SCALING AND TRAIN/TEST SPLIT
############################################################

# -----------------------------
# 1. Show first five transactions before scaling
# -----------------------------

head(ccd, 5)
head(ccdt, 5)

# -----------------------------
# 3.Save the new dataset in another data frame.
# -----------------------------

ccd_scaled <- ccd
ccdt_scaled <- ccdt

# -----------------------------
# 2. Apply scale function to amount feature
# -----------------------------

ccd_scaled$amt_scaled <- as.numeric(scale(ccd_scaled$amt))
ccdt_scaled$amt_scaled <- as.numeric(scale(ccdt_scaled$amt))

# -----------------------------
# 4. Could you show the first 5 transactions after the changes?
# -----------------------------

head(ccd_scaled, 5)
head(ccdt_scaled, 5)

# -----------------------------
# II. Modeling The Data
# -----------------------------


# This avoids mixing row indexes from fraudTrain and fraudTest.
# The model is trained and tested from the same scaled train dataset.
ccd_model <- ccd_scaled[, c("amt", "amt_scaled", "is_fraud", "fraud_label")]

# Make sure target variable is clean numeric 0/1
ccd_model$is_fraud <- ifelse(
  ccd_model$is_fraud %in% c(1, "1", "Fraud", "fraud", TRUE),
  1,
  0
)

ccd_model$is_fraud <- as.numeric(ccd_model$is_fraud)
ccd_model <- na.omit(ccd_model)

# Confirm target variable is valid for logistic regression
table(ccd_model$is_fraud, useNA = "ifany")
range(ccd_model$is_fraud)

# -----------------------------
# 1. Select a seed value for each split operation.
# -----------------------------
set.seed(123)


# -----------------------------
# 2. Set the split ratio (75%/25%).
# -----------------------------

split_ratio <- 0.75

train_index <- sample(
  1:nrow(ccd_model),
  size = split_ratio * nrow(ccd_model)
)

# -----------------------------
# 3. Divide the dataset and save the new splits into train and test sets.
# -----------------------------

train_set <- ccd_model[train_index, ]
test_set <- ccd_model[-train_index, ]

# -----------------------------
# 4. Print out the train set and test set dimensions.
# -----------------------------
# Print train and test dimensions
dim(train_set)
dim(test_set)

# -----------------------------
# 5. Give a summary of train and test sets.
# -----------------------------
# Summary of train and test sets
summary(train_set)
summary(test_set)

# -----------------------------
# 6. Create the histogram plot based on the Amount feature for the two sets.
# -----------------------------

ggplot(train_set, aes(x = amt_scaled)) +
  geom_histogram(bins = 50, fill = "blue") +
  labs(
    title = "Histogram of Standardized Transaction Amounts - Training Set",
    x = "Standardized Amount",
    y = "Frequency"
  ) +
  theme_minimal()

ggplot(test_set, aes(x = amt_scaled)) +
  geom_histogram(bins = 50, fill = "red") +
  labs(
    title = "Histogram of Standardized Transaction Amounts - Testing Set",
    x = "Standardized Amount",
    y = "Frequency"
  ) +
  theme_minimal()

############################################################
# MILESTONE 4: Correlation Plot and Setting & Fitting the Model
############################################################


# -----------------------------
# 1. Correlation Plot
# -----------------------------

#  Create Class feature based on is_fraud

train_set$Class <- factor(
  train_set$is_fraud,
  levels = c(0, 1),
  labels = c("Not Fraud", "Fraud")
)

test_set$Class <- factor(
  test_set$is_fraud,
  levels = c(0, 1),
  labels = c("Not Fraud", "Fraud")
)

# Scatter plot: amount and Class feature

plot(
  train_set$amt,
  train_set$is_fraud,
  main = "Relationship Between Amount and Fraud Class",
  xlab = "Transaction Amount",
  ylab = "Class: 0 = Not Fraud, 1 = Fraud",
  pch = 19,
  col = ifelse(train_set$is_fraud == 1, "red", "blue")
)

legend(
  "topright",
  legend = c("Not Fraud", "Fraud"),
  col = c("blue", "red"),
  pch = 19
)

# -----------------------------
# II. Set Your Model
# -----------------------------

# Set and train the logistic regression model.


logistic_model <- glm(
  is_fraud ~ amt_scaled,
  data = train_set,
  family = binomial
)



# -----------------------------
# 1. Apply logistic regression model to test data
# -----------------------------

test_set$logistic_predicted_probability <- predict(
  logistic_model,
  newdata = test_set,
  type = "response"
)

test_set$logistic_predicted_class <- ifelse(
  test_set$logistic_predicted_probability >= 0.5,
  1,
  0
)

# -----------------------------
# 2. Model summary
# -----------------------------

summary(logistic_model)

# -----------------------------
# 3. Plot logistic regression model
# -----------------------------

plot(
  train_set$amt_scaled,
  train_set$is_fraud,
  main = "Train Logistic Regression Model",
  xlab = "Scaled Amount",
  ylab = "Probability of Fraud",
  pch = 19,
  col = ifelse(train_set$is_fraud == 1, "red", "blue")
)

curve(
  predict(
    logistic_model,
    newdata = data.frame(amt_scaled = x),
    type = "response"
  ),
  add = TRUE,
  col = "yellow",
  lwd = 2
)

plot(
  test_set$amt_scaled,
  test_set$is_fraud,
  main = "Test Logistic Regression Model",
  xlab = "Scaled Amount",
  ylab = "Probability of Fraud",
  pch = 19,
  col = ifelse(test_set$is_fraud == 1, "red", "blue")
)

curve(
  predict(
    logistic_model,
    newdata = data.frame(amt_scaled = x),
    type = "response"
  ),
  add = TRUE,
  col = "green",
  lwd = 2
)

# -----------------------------
# 4. Print out the confusion matrix.
# -----------------------------

logistic_confusion_matrix <- table(
  Actual = test_set$is_fraud,
  Predicted = test_set$logistic_predicted_class
)

logistic_confusion_matrix

logistic_accuracy <- sum(diag(logistic_confusion_matrix)) / sum(logistic_confusion_matrix)
logistic_accuracy

# -----------------------------
# 5. ROC curve for your model.
# -----------------------------

roc_logistic <- roc(
  response = test_set$is_fraud,
  predictor = test_set$logistic_predicted_probability,
  levels = c(0, 1),
  direction = "<"
)

plot(
  roc_logistic,
  main = "ROC Curve for Logistic Regression",
  col = "blue",
  lwd = 2
)

# -----------------------------
# 6. Calculate the AUC for your model’s curve.
# -----------------------------

auc_logistic <- auc(roc_logistic)
auc_logistic


# -----------------------------
# IV. More Training and Test Sets
# -----------------------------
# -----------------------------
# 19. Run logistic regression 10 times with different splits
# -----------------------------

seeds <- c(101, 202, 303, 404, 505, 606, 707, 808, 909, 1001)

auc_results <- data.frame(
  Run = integer(),
  Seed = integer(),
  AUC = numeric()
)

for (i in 1:10) {
  set.seed(seeds[i])
  
  train_index_loop <- sample(
    1:nrow(ccd_model),
    size = split_ratio * nrow(ccd_model)
  )
  
  train_loop <- ccd_model[train_index_loop, ]
  test_loop <- ccd_model[-train_index_loop, ]
  
  logistic_model_loop <- glm(
    is_fraud ~ amt_scaled,
    data = train_loop,
    family = binomial
  )
  
  test_loop$predicted_probability <- predict(
    logistic_model_loop,
    newdata = test_loop,
    type = "response"
  )
  # -----------------------------
  # 2. Save all the AUC results in a table.
  # -----------------------------  
  roc_result_loop <- roc(
    response = test_loop$is_fraud,
    predictor = test_loop$predicted_probability,
    levels = c(0, 1),
    direction = "<"
  )
  
  auc_results <- rbind(
    auc_results,
    data.frame(
      Run = i,
      Seed = seeds[i],
      AUC = as.numeric(auc(roc_result_loop))
    )
  )
}

auc_results

# -----------------------------
# 3. Calculate the median of the results (median of the AUC for ten runs) as the final result.
# -----------------------------
median_auc <- median(auc_results$AUC)
median_auc

############################################################
# MILESTONE 5: Applying and Analyzing Another Algorithm
############################################################


# -----------------------------
# I. Apply Another Algorithm
# -----------------------------

# Apply the Decision Tree model on the same train and test sets from 


decision_tree_model <- rpart(
  factor(is_fraud) ~ amt_scaled,
  data = train_set,
  method = "class"
)

# Plot decision tree using base R
plot(decision_tree_model, uniform = TRUE, main = "Decision Tree Model")
text(decision_tree_model, use.n = TRUE, cex = 0.8)

# -----------------------------
# Predict decision tree probabilities on SAME test set
# -----------------------------

dt_probabilities <- predict(
  decision_tree_model,
  newdata = test_set,
  type = "prob"
)

# Probability for class 1 means fraud probability
test_set$dt_predicted_probability <- dt_probabilities[, "1"]

test_set$dt_predicted_class <- ifelse(
  test_set$dt_predicted_probability >= 0.5,
  1,
  0
)

# -----------------------------
# Decision tree confusion matrix and accuracy
# -----------------------------

decision_tree_confusion_matrix <- table(
  Actual = test_set$is_fraud,
  Predicted = test_set$dt_predicted_class
)

decision_tree_confusion_matrix

decision_tree_accuracy <- sum(diag(decision_tree_confusion_matrix)) / sum(decision_tree_confusion_matrix)
decision_tree_accuracy

# -----------------------------
# Decision tree ROC curve and AUC
# -----------------------------

roc_decision_tree <- roc(
  response = test_set$is_fraud,
  predictor = test_set$dt_predicted_probability,
  levels = c(0, 1),
  direction = "<"
)

auc_decision_tree <- auc(roc_decision_tree)
auc_decision_tree

############################################################
# MODEL COMPARISON AND HYPOTHESIS TESTING
############################################################

# -----------------------------
# Plot both ROC curves on the same graph
# -----------------------------

plot(
  roc_logistic,
  col = "blue",
  lwd = 2,
  main = "ROC Curve Comparison: Logistic Regression vs Decision Tree"
)

lines(
  roc_decision_tree,
  col = "red",
  lwd = 2
)

legend(
  "bottomright",
  legend = c(
    paste("Logistic Regression AUC =", round(as.numeric(auc_logistic), 4)),
    paste("Decision Tree AUC =", round(as.numeric(auc_decision_tree), 4))
  ),
  col = c("blue", "red"),
  lwd = 2
)


#  AUC and accuracy comparison table


auc_comparison <- data.frame(
  Model = c("Logistic Regression", "Decision Tree"),
  AUC = c(as.numeric(auc_logistic), as.numeric(auc_decision_tree)),
  Accuracy = c(logistic_accuracy, decision_tree_accuracy)
)

auc_comparison





# -----------------------------
# 1. Hypothesis testing: DeLong paired ROC test
# -----------------------------
# Test choice:
# DeLong's paired ROC test because both models are evaluated on the SAME test_set.
# H0: There is no significant difference between the two AUC values.
# H1: There is a significant difference between the two AUC values.
# Decision rule: If p-value < 0.05, reject H0.

roc_test_result <- roc.test(
  roc_logistic,
  roc_decision_tree,
  method = "delong",
  paired = TRUE
)

roc_test_result
# -----------------------------
# 2. Calculate the P value. 
# -----------------------------
p_value <- roc_test_result$p.value
p_value


if (p_value < 0.05) {
  print("The p-value is less than 0.05. Reject H0. The AUC values are significantly different.")
} else {
  print("The p-value is greater than or equal to 0.05. Fail to reject H0. The AUC values are not significantly different.")
}

# -----------------------------
# 3. Analyze your findings and explain them. 
# -----------------------------

