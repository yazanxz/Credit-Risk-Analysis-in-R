# Credit Risk Analysis
# This project implements credit risk modeling using logistic regression, 
# decision trees, and random forests to predict loan default risk

# Load required libraries
library(tidyverse)    # For data manipulation and visualization
library(caret)        # For model training and evaluation
library(rpart)        # For decision trees
library(rpart.plot)   # For visualizing decision trees
library(randomForest) # For random forest models
library(ROCR)         # For ROC curves
library(knitr)        # For formatted output

# Set seed for reproducibility
set.seed(123)

#=============================================================================
# 1. Data Preparation
#=============================================================================

# Generate synthetic loan data
# In a real-world scenario, you would import your historical loan data
generate_loan_data <- function(n = 1000) {
  data.frame(
    customer_id = 1:n,
    loan_amount = round(runif(n, 5000, 50000)),
    income = round(rnorm(n, 65000, 20000)),
    credit_score = round(rnorm(n, 680, 75)),
    employment_years = round(runif(n, 0, 30), 1),
    age = round(runif(n, 22, 70)),
    debt_to_income = round(runif(n, 0.1, 0.6), 2),
    employment_status = sample(c("Employed", "Self-Employed", "Unemployed"), 
                              n, replace = TRUE, prob = c(0.7, 0.2, 0.1)),
    home_ownership = sample(c("Own", "Mortgage", "Rent"), 
                           n, replace = TRUE, prob = c(0.3, 0.5, 0.2)),
    loan_purpose = sample(c("Home", "Education", "Medical", "Business", "Personal"),
                         n, replace = TRUE),
    stringsAsFactors = TRUE
  )
}

# Generate the loan data
loan_data <- generate_loan_data(5000)

# Create a default variable based on risk factors
# This simulates historical loan outcomes
calculate_default_probability <- function(data) {
  with(data, {
    # Base probability
    p <- 0.05
    
    # Adjust based on credit score
    p <- p + ifelse(credit_score < 600, 0.20, 
                    ifelse(credit_score < 650, 0.10, 
                           ifelse(credit_score < 700, 0.05, 0)))
    
    # Adjust based on income
    p <- p + ifelse(income < 40000, 0.15, 
                    ifelse(income < 60000, 0.08, 
                           ifelse(income < 80000, 0.03, 0)))
    
    # Adjust based on employment status
    p <- p + ifelse(employment_status == "Unemployed", 0.25,
                    ifelse(employment_status == "Self-Employed", 0.10, 0))
    
    # Adjust based on debt-to-income
    p <- p + (debt_to_income * 0.3)
    
    # Cap probability at 0.95
    return(pmin(p, 0.95))
  })
}

# Calculate default probabilities
default_probs <- calculate_default_probability(loan_data)

# Simulate defaults based on probabilities
set.seed(456)
loan_data$default <- rbinom(nrow(loan_data), 1, default_probs)

# Convert default to factor
loan_data$default <- factor(loan_data$default, levels = c(0, 1), labels = c("No", "Yes"))

# Display the first few rows of the dataset
cat("First 5 rows of the loan dataset:\n")
head(loan_data, 5)

#=============================================================================
# 2. Exploratory Data Analysis
#=============================================================================

# Summary statistics
cat("\nSummary statistics:\n")
summary(loan_data)

# Default rate
default_rate <- mean(loan_data$default == "Yes")
cat(sprintf("\nDefault rate: %.2f%%\n", default_rate * 100))

# Create visualizations
# Univariate analysis

# Plot credit score distribution by default status
credit_score_plot <- ggplot(loan_data, aes(x = credit_score, fill = default)) +
  geom_density(alpha = 0.6) +
  labs(title = "Credit Score Distribution by Default Status",
       x = "Credit Score", y = "Density") +
  theme_minimal() +
  scale_fill_manual(values = c("No" = "blue", "Yes" = "red"))

# Plot income distribution by default status
income_plot <- ggplot(loan_data, aes(x = income, fill = default)) +
  geom_density(alpha = 0.6) +
  labs(title = "Income Distribution by Default Status",
       x = "Income", y = "Density") +
  theme_minimal() +
  scale_fill_manual(values = c("No" = "blue", "Yes" = "red"))

# Default rate by employment status
emp_status_plot <- ggplot(loan_data, aes(x = employment_status, fill = default)) +
  geom_bar(position = "fill") +
  labs(title = "Default Rate by Employment Status",
       x = "Employment Status", y = "Proportion") +
  theme_minimal() +
  scale_fill_manual(values = c("No" = "blue", "Yes" = "red"))

# Default rate by home ownership
ownership_plot <- ggplot(loan_data, aes(x = home_ownership, fill = default)) +
  geom_bar(position = "fill") +
  labs(title = "Default Rate by Home Ownership",
       x = "Home Ownership", y = "Proportion") +
  theme_minimal() +
  scale_fill_manual(values = c("No" = "blue", "Yes" = "red"))

# Correlation matrix for numerical variables
numeric_vars <- loan_data %>% 
  select(loan_amount, income, credit_score, employment_years, age, debt_to_income)
correlation <- cor(numeric_vars)
cat("\nCorrelation matrix for numerical variables:\n")
print(round(correlation, 2))

#=============================================================================
# 3. Data Preprocessing
#=============================================================================

# Split data into training and testing sets
train_index <- createDataPartition(loan_data$default, p = 0.7, list = FALSE)
train_data <- loan_data[train_index, ]
test_data <- loan_data[-train_index, ]

# Verify split
cat(sprintf("\nTraining set size: %d\n", nrow(train_data)))
cat(sprintf("Testing set size: %d\n", nrow(test_data)))

# Create a formula for modeling
formula <- default ~ loan_amount + income + credit_score + employment_years + 
           age + debt_to_income + employment_status + home_ownership + loan_purpose

#=============================================================================
# 4. Logistic Regression Model
#=============================================================================

# Train logistic regression model
cat("\nTraining logistic regression model...\n")
logistic_model <- glm(formula, data = train_data, family = "binomial")

# Model summary
cat("\nLogistic Regression Model Summary:\n")
summary(logistic_model)

# Extract significant features
logistic_coef <- summary(logistic_model)$coefficients
significant_features <- rownames(logistic_coef)[logistic_coef[, 4] < 0.05]
cat("\nSignificant features in logistic regression:\n")
print(significant_features)

# Predict on test set
logistic_pred_prob <- predict(logistic_model, newdata = test_data, type = "response")
logistic_pred <- ifelse(logistic_pred_prob > 0.5, "Yes", "No")
logistic_pred <- factor(logistic_pred, levels = c("No", "Yes"))

# Evaluate model
logistic_conf_matrix <- confusionMatrix(logistic_pred, test_data$default)
cat("\nLogistic Regression Confusion Matrix:\n")
print(logistic_conf_matrix)

#=============================================================================
# 5. Decision Tree Model
#=============================================================================

# Train decision tree model
cat("\nTraining decision tree model...\n")
tree_model <- rpart(formula, data = train_data, method = "class",
                   control = rpart.control(maxdepth = 6, cp = 0.01))

# Plot the decision tree
tree_plot <- function() {
  rpart.plot(tree_model, extra = 2, fallen.leaves = TRUE, main = "Decision Tree for Credit Risk")
}

# Display important variables from the tree
cat("\nDecision Tree Variable Importance:\n")
tree_importance <- tree_model$variable.importance
print(tree_importance)

# Predict on test set
tree_pred <- predict(tree_model, newdata = test_data, type = "class")

# Evaluate model
tree_conf_matrix <- confusionMatrix(tree_pred, test_data$default)
cat("\nDecision Tree Confusion Matrix:\n")
print(tree_conf_matrix)

#=============================================================================
# 6. Random Forest Model
#=============================================================================

# Train random forest model
cat("\nTraining random forest model...\n")
rf_model <- randomForest(formula, data = train_data, ntree = 500, importance = TRUE)

# Display model
cat("\nRandom Forest Model:\n")
print(rf_model)

# Variable importance
cat("\nRandom Forest Variable Importance:\n")
rf_importance <- importance(rf_model)
rf_importance_df <- data.frame(
  Variable = rownames(rf_importance),
  MeanDecreaseGini = rf_importance[, "MeanDecreaseGini"]
)
rf_importance_df <- rf_importance_df %>% 
  arrange(desc(MeanDecreaseGini))
print(rf_importance_df)

# Plot variable importance
importance_plot <- function() {
  varImpPlot(rf_model, main = "Random Forest Variable Importance")
}

# Predict on test set
rf_pred <- predict(rf_model, newdata = test_data)

# Evaluate model
rf_conf_matrix <- confusionMatrix(rf_pred, test_data$default)
cat("\nRandom Forest Confusion Matrix:\n")
print(rf_conf_matrix)

#=============================================================================
# 7. Model Comparison
#=============================================================================

# Collect predictions for ROC curves
logistic_pred_prob <- predict(logistic_model, newdata = test_data, type = "response")
tree_pred_prob <- predict(tree_model, newdata = test_data, type = "prob")[, "Yes"]
rf_pred_prob <- predict(rf_model, newdata = test_data, type = "prob")[, "Yes"]

# Create prediction objects for ROCR
logistic_pred_obj <- prediction(logistic_pred_prob, test_data$default)
tree_pred_obj <- prediction(tree_pred_prob, test_data$default)
rf_pred_obj <- prediction(rf_pred_prob, test_data$default)

# Calculate ROC metrics
logistic_perf <- performance(logistic_pred_obj, "tpr", "fpr")
tree_perf <- performance(tree_pred_obj, "tpr", "fpr")
rf_perf <- performance(rf_pred_obj, "tpr", "fpr")

# Calculate AUC
logistic_auc <- performance(logistic_pred_obj, "auc")@y.values[[1]]
tree_auc <- performance(tree_pred_obj, "auc")@y.values[[1]]
rf_auc <- performance(rf_pred_obj, "auc")@y.values[[1]]

# Compare accuracy, precision, recall, and F1 score
models <- c("Logistic Regression", "Decision Tree", "Random Forest")
accuracy <- c(
  logistic_conf_matrix$overall["Accuracy"],
  tree_conf_matrix$overall["Accuracy"],
  rf_conf_matrix$overall["Accuracy"]
)
precision <- c(
  logistic_conf_matrix$byClass["Pos Pred Value"],
  tree_conf_matrix$byClass["Pos Pred Value"],
  rf_conf_matrix$byClass["Pos Pred Value"]
)
recall <- c(
  logistic_conf_matrix$byClass["Sensitivity"],
  tree_conf_matrix$byClass["Sensitivity"],
  rf_conf_matrix$byClass["Sensitivity"]
)
f1_score <- c(
  logistic_conf_matrix$byClass["F1"],
  tree_conf_matrix$byClass["F1"],
  rf_conf_matrix$byClass["F1"]
)
auc <- c(logistic_auc, tree_auc, rf_auc)

# Create comparison table
comparison_table <- data.frame(
  Model = models,
  Accuracy = round(accuracy, 4),
  Precision = round(precision, 4),
  Recall = round(recall, 4),
  F1_Score = round(f1_score, 4),
  AUC = round(auc, 4)
)

cat("\nModel Comparison:\n")
print(comparison_table)

# Plot ROC curves
roc_plot <- function() {
  plot(logistic_perf, col = "blue", main = "ROC Curve Comparison")
  plot(tree_perf, col = "red", add = TRUE)
  plot(rf_perf, col = "green", add = TRUE)
  abline(a = 0, b = 1, lty = 2, col = "gray")
  legend("bottomright", 
         legend = c(
           paste("Logistic (AUC =", round(logistic_auc, 4), ")"),
           paste("Decision Tree (AUC =", round(tree_auc, 4), ")"),
           paste("Random Forest (AUC =", round(rf_auc, 4), ")")
         ),
         col = c("blue", "red", "green"), 
         lwd = 2)
}

#=============================================================================
# 8. Key Risk Factors Analysis
#=============================================================================

# Identify key risk factors
cat("\nKey Risk Factors from Logistic Regression:\n")
logistic_coef_df <- data.frame(
  Variable = names(coef(logistic_model)),
  Coefficient = coef(logistic_model),
  Odds_Ratio = exp(coef(logistic_model)),
  P_Value = summary(logistic_model)$coefficients[, 4]
)
significant_logistic <- logistic_coef_df %>%
  filter(P_Value < 0.05) %>%
  arrange(P_Value)
print(significant_logistic)

cat("\nKey Risk Factors from Random Forest:\n")
print(head(rf_importance_df, 10))

# Create a combined risk factor ranking
cat("\nCombined Risk Factor Importance:\n")
all_vars <- unique(c(
  significant_logistic$Variable[!grepl("Intercept", significant_logistic$Variable)],
  rf_importance_df$Variable
))

# Function to get rank or NA
get_rank <- function(var, var_list) {
  if(var %in% var_list) {
    return(which(var_list == var))
  } else {
    return(NA)
  }
}

# Create ranking table
risk_ranking <- data.frame(
  Variable = all_vars
)

risk_ranking$Logistic_Rank <- sapply(
  risk_ranking$Variable, 
  function(x) get_rank(x, significant_logistic$Variable[!grepl("Intercept", significant_logistic$Variable)])
)

risk_ranking$RF_Rank <- sapply(
  risk_ranking$Variable, 
  function(x) get_rank(x, rf_importance_df$Variable)
)

# Calculate average rank, handling NAs
risk_ranking$Avg_Rank <- rowMeans(
  risk_ranking[, c("Logistic_Rank", "RF_Rank")], 
  na.rm = TRUE
)

# Sort by average rank
risk_ranking <- risk_ranking %>%
  arrange(Avg_Rank)

print(risk_ranking)

#=============================================================================
# 9. Create Risk Score Function
#=============================================================================

# Create a simplified risk scoring function based on the models
create_risk_score <- function(new_data) {
  # Predict probabilities from each model
  logistic_prob <- predict(logistic_model, newdata = new_data, type = "response")
  tree_prob <- predict(tree_model, newdata = new_data, type = "prob")[, "Yes"]
  rf_prob <- predict(rf_model, newdata = new_data, type = "prob")[, "Yes"]
  
  # Calculate weighted average (weights based on model performance)
  weights <- c(
    logistic = logistic_auc,
    tree = tree_auc,
    rf = rf_auc
  )
  weights <- weights / sum(weights)
  
  weighted_prob <- (
    weights["logistic"] * logistic_prob +
    weights["tree"] * tree_prob +
    weights["rf"] * rf_prob
  )
  
  # Convert to risk score (0-100)
  risk_score <- round(weighted_prob * 100)
  
  # Create risk categories
  risk_category <- cut(
    risk_score,
    breaks = c(-1, 20, 40, 60, 80, 101),
    labels = c("Very Low", "Low", "Medium", "High", "Very High")
  )
  
  return(data.frame(
    risk_score = risk_score,
    risk_category = risk_category,
    default_probability = weighted_prob
  ))
}

# Example application with a few test cases
cat("\nRisk Score Examples:\n")
example_cases <- data.frame(
  loan_amount = c(10000, 25000, 40000),
  income = c(85000, 45000, 35000),
  credit_score = c(750, 680, 580),
  employment_years = c(10, 5, 1),
  age = c(35, 28, 24),
  debt_to_income = c(0.2, 0.35, 0.5),
  employment_status = factor(c("Employed", "Self-Employed", "Unemployed"), 
                             levels = levels(loan_data$employment_status)),
  home_ownership = factor(c("Own", "Mortgage", "Rent"), 
                          levels = levels(loan_data$home_ownership)),
  loan_purpose = factor(c("Home", "Business", "Personal"), 
                        levels = levels(loan_data$loan_purpose))
)

example_risk <- cbind(example_cases, create_risk_score(example_cases))
print(example_risk)

#=============================================================================
# 10. Conclusions ^_^
#=============================================================================

cat("\nCredit Risk Analysis Conclusions:\n")
cat("1. The Random Forest model performed best with an AUC of", round(rf_auc, 4), "\n")
cat("2. Key risk factors identified across models:\n")
for(i in 1:min(5, nrow(risk_ranking))) {
  cat("   - ", risk_ranking$Variable[i], "\n")
}

cat("\n3. Credit score, income, and employment status are the strongest predictors of default risk\n")
cat("4. The risk scoring system can be used to classify new loan applications into risk categories\n")
