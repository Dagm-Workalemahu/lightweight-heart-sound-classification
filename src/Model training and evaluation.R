# --- Libraries ---
library(tidyverse)
library(caret)
library(randomForest)
library(xgboost)
library(tuneR)
library(seewave)
library(e1071)
library(reshape2)
library(ggplot2)
library(cowplot)

set.seed(123)

# ===============================
# 1) Load and Prepare Data
# ===============================
df <- read_csv("features_labeled.csv")  # updated features file

df <- df %>%
  dplyr::filter(!is.na(label) & label != "unknown") %>%
  dplyr::mutate(label = factor(label))

df <- df %>% dplyr::select(-any_of("file_path"))

# ===============================
# 2) Preprocess Data
# ===============================
pre_proc <- preProcess(df %>% select(-label), method = c("center", "scale"))
df_scaled <- predict(pre_proc, df %>% select(-label))
df_final <- bind_cols(df_scaled, label = df$label)

# ===============================
# 3) Train Models with CV Probabilities
# ===============================
train_ctrl <- trainControl(
  method = "repeatedcv",
  number = 10,
  repeats = 3,
  sampling = "up",
  classProbs = TRUE,
  savePredictions = "final"
)

formula <- as.formula("label ~ .")

knn_model <- train(formula, data = df_final, method = "knn", trControl = train_ctrl)
svm_model <- train(formula, data = df_final, method = "svmRadial", trControl = train_ctrl)
rf_model  <- train(formula, data = df_final, method = "rf", trControl = train_ctrl)
xgb_model <- train(formula, data = df_final, method = "xgbTree", trControl = train_ctrl)

saveRDS(knn_model, "knn_model.rds")
saveRDS(svm_model, "svm_model.rds")
saveRDS(rf_model,  "rf_model.rds")
saveRDS(xgb_model, "xgb_model.rds")
saveRDS(pre_proc,  "preproc_scaler.rds")

# ===============================
# 4) Metrics Function
# ===============================
calculate_metrics <- function(true_labels, predictions) {
  cm <- table(True = true_labels, Pred = predictions)
  
  precision <- diag(cm) / colSums(cm)
  recall <- diag(cm) / rowSums(cm)
  f1 <- 2 * precision * recall / (precision + recall)
  
  precision[is.na(precision)] <- 0
  recall[is.na(recall)] <- 0
  f1[is.na(f1)] <- 0
  
  balanced_acc <- mean(recall)
  
  list(
    confusion_matrix = cm,
    balanced_accuracy = balanced_acc,
    macro_precision = mean(precision),
    macro_recall = mean(recall),
    macro_f1 = mean(f1)
  )
}

# ===============================
# 5) Compute Metrics and Ensemble Using CV Probabilities
# ===============================
models <- list(KNN = knn_model, SVM = svm_model, RF = rf_model, XGB = xgb_model)
metrics_list <- list()
cv_probs_list <- list()

for (m in names(models)) {
  pred_df <- models[[m]]$pred %>%
    dplyr::group_by(rowIndex) %>%
    dplyr::filter(dplyr::row_number() == 1) %>%
    dplyr::ungroup()
  
  metrics_list[[m]] <- calculate_metrics(pred_df$obs, pred_df$pred)
  
  cv_probs <- pred_df %>% dplyr::select(all_of(levels(df_final$label))) %>% as.matrix()
  cv_probs_list[[m]] <- cv_probs
}

# Ensemble with CV probabilities weighted by macro F1
model_weights <- sapply(metrics_list, function(x) x$macro_f1)
model_weights[is.na(model_weights)] <- 0
if (sum(model_weights) == 0) model_weights <- rep(1, length(models))
model_weights <- model_weights / sum(model_weights)
saveRDS(model_weights, "model_weights.rds")

ensemble_prob <- Reduce(`+`, Map(`*`, cv_probs_list, model_weights))
ensemble_pred <- factor(colnames(ensemble_prob)[max.col(ensemble_prob)], levels = levels(df_final$label))
ensemble_metrics <- calculate_metrics(df_final$label, ensemble_pred)

# ===============================
# 6) Performance Visualization
# ===============================
metrics_df <- data.frame(
  Model = c(names(metrics_list), "Ensemble"),
  Macro_F1 = c(sapply(metrics_list, function(x) x$macro_f1), ensemble_metrics$macro_f1),
  Balanced_Accuracy = c(sapply(metrics_list, function(x) x$balanced_accuracy), ensemble_metrics$balanced_accuracy)
)

metrics_long <- metrics_df %>%
  pivot_longer(cols = c(Macro_F1, Balanced_Accuracy),
               names_to = "Metric",
               values_to = "Value")

ggplot(metrics_long, aes(x = Model, y = Value, fill = Metric)) +
  geom_bar(stat = "identity", position = position_dodge(width = 0.8)) +
  ylim(0, 1) +
  geom_text(aes(label = round(Value, 2)),
            position = position_dodge(width = 0.8),
            vjust = -0.3,
            size = 2) +
  theme_minimal() +
  labs(title = "Macro F1 and Balanced Accuracy per Model",
       y = "Score",
       x = "Model") +
  scale_fill_manual(values = c("Macro_F1" = "steelblue", "Balanced_Accuracy" = "darkorange")) +
  theme(axis.text.x = element_text(angle = 15, hjust = 1))

# Confusion matrices
plot_confusion_matrix <- function(conf_matrix, title) {
  as.data.frame(conf_matrix) %>%
    rename(Predicted = Pred, Actual = True, Count = Freq) %>%
    ggplot(aes(x = Actual, y = Predicted, fill = Count)) +
    geom_tile(color = "white") +
    geom_text(aes(label = Count), color = "black", size = 4) +
    scale_fill_gradient(low = "white", high = "brown") +
    theme_minimal() +
    labs(title = title, x = "Actual Label", y = "Predicted Label")
}

p_knn_cm <- plot_confusion_matrix(metrics_list$KNN$confusion_matrix, "KNN Confusion Matrix")
p_svm_cm <- plot_confusion_matrix(metrics_list$SVM$confusion_matrix, "SVM Confusion Matrix")
p_rf_cm  <- plot_confusion_matrix(metrics_list$RF$confusion_matrix, "RF Confusion Matrix")
p_xgb_cm <- plot_confusion_matrix(metrics_list$XGB$confusion_matrix, "XGB Confusion Matrix")
p_ensemble_cm <- plot_confusion_matrix(ensemble_metrics$confusion_matrix, "Ensemble Confusion Matrix")

print(p_knn_cm)
print(p_svm_cm)
print(p_rf_cm)
print(p_xgb_cm)
print(p_ensemble_cm)
