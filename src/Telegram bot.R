library(telegram.bot)
library(tuneR)
library(seewave)
library(dplyr)
library(tibble)
library(caret)
library(randomForest)
library(xgboost)
library(e1071)
library(readr)
library(av)
library(tools)

# ===============================
# 1) Load models and preprocessing
# ===============================
pre_proc <- readRDS("preproc_scaler.rds")
model_weights <- readRDS("model_weights.rds")
knn_model <- readRDS("knn_model.rds")
svm_model <- readRDS("svm_model.rds")
rf_model  <- readRDS("rf_model.rds")
xgb_model <- readRDS("xgb_model.rds")

# ===============================
# 2) Feature Extraction Function
# ===============================
extract_features_file <- function(file_path, n_mfcc = 13) {
  wav <- readWave(file_path)
  wav <- mono(wav, "left")
  sr <- wav@samp.rate
  
  mfcc_mat <- melfcc(wav, sr = sr, numcep = n_mfcc)
  if (ncol(mfcc_mat) != n_mfcc && nrow(mfcc_mat) == n_mfcc) mfcc_mat <- t(mfcc_mat)
  
  mfcc_means <- apply(mfcc_mat, 2, mean, na.rm = TRUE)
  mfcc_sds   <- apply(mfcc_mat, 2, sd, na.rm = TRUE)
  
  delta_mat <- apply(mfcc_mat, 2, function(x) c(NA, diff(x)))
  deltadelta_mat <- apply(delta_mat, 2, function(x) c(NA, diff(x)))
  
  delta_means     <- apply(delta_mat, 2, mean, na.rm = TRUE)
  deltadelta_means <- apply(deltadelta_mat, 2, mean, na.rm = TRUE)
  
  feats <- c(mfcc_means, mfcc_sds, delta_means, deltadelta_means)
  names(feats) <- c(
    paste0("mfcc_", 1:n_mfcc),
    paste0("mfcc_sd_", 1:n_mfcc),
    paste0("delta_", 1:n_mfcc),
    paste0("deltadelta_", 1:n_mfcc)
  )
  
  as_tibble_row(as.list(feats))
}

# ===============================
# 3) Telegram MarkdownV2 Escape
# ===============================
escape_markdown <- function(text) {
  gsub("([_\\*\\[\\]\\(\\)~`>#+\\-=|{}.!])", "\\\\\\1", text, perl = TRUE)
}

# ===============================
# 4) Telegram Handlers
# ===============================
TOKEN <- "YOUR BOT TOKEN HERE"  # <-- Replace with your bot token or load from env

# /start command
start_handler <- function(bot, update) {
  msg <- "👋 Hello! Send me a heart sound recording (voice or audio file: .wav, .mp3, .m4a, .ogg, .mp4) and I will predict its class."
  bot$sendMessage(chat_id = update$message$chat_id, text = escape_markdown(msg), parse_mode = "MarkdownV2")
}

# Audio/voice handler
audio_handler <- function(bot, update) {
  chat_id <- update$message$chat_id
  file_info <- update$message$voice %||% update$message$audio %||% update$message$document
  
  if (is.null(file_info)) {
    bot$sendMessage(chat_id = chat_id,
                    text = escape_markdown("⚠️ Please send a voice recording or an audio file (.wav, .mp3, .m4a, .ogg, .mp4)."),
                    parse_mode = "MarkdownV2")
    return()
  }
  
  orig_file_name <- file_info$file_name %||% paste0("voice_", Sys.time(), ".ogg")
  file_meta <- bot$getFile(file_info$file_id)
  ext <- tools::file_ext(file_meta$file_path)
  if (ext == "") ext <- "ogg"
  
  temp_file <- tempfile(fileext = paste0(".", ext))
  download.file(
    paste0("https://api.telegram.org/file/bot", TOKEN, "/", file_meta$file_path),
    temp_file,
    mode = "wb"
  )
  
  bot$sendMessage(chat_id = chat_id, text = escape_markdown("🎧 Processing your audio..."), parse_mode = "MarkdownV2")
  
  result <- tryCatch({
    # Convert to WAV if necessary
    if (tolower(ext) != "wav") {
      temp_wav <- tempfile(fileext = ".wav")
      av::av_audio_convert(temp_file, temp_wav, channels = 1, sample_rate = 44100)
      temp_file <- temp_wav
    }
    
    feat <- extract_features_file(temp_file)
    feat_scaled <- predict(pre_proc, feat)
    
    # Per-model predictions
    knn_pred <- predict(knn_model, feat_scaled)
    svm_pred <- predict(svm_model, feat_scaled)
    rf_pred  <- predict(rf_model, feat_scaled)
    xgb_pred <- predict(xgb_model, as.matrix(feat_scaled))
    
    # Probabilities
    knn_prob <- predict(knn_model, feat_scaled, type = "prob")
    svm_prob <- predict(svm_model, feat_scaled, type = "prob")
    rf_prob  <- predict(rf_model, feat_scaled, type = "prob")
    xgb_prob <- predict(xgb_model, as.matrix(feat_scaled), type = "prob")
    
    # Weighted soft voting using RF & XGB (preserves your pipeline)
    avg_prob <- Reduce(`+`, Map(`*`, list(rf_prob, xgb_prob), model_weights))
    final_class <- names(avg_prob)[which.max(avg_prob)]
    confidence <- round(max(avg_prob) * 100, 1)
    
    list(
      File = orig_file_name,
      Final_Class = final_class,
      Confidence = confidence,
      Per_Model = c(KNN = as.character(knn_pred),
                    SVM = as.character(svm_pred),
                    RF  = as.character(rf_pred),
                    XGB = as.character(xgb_pred)),
      Probabilities = avg_prob
    )
    
  }, error = function(e) {
    bot$sendMessage(chat_id = chat_id, text = escape_markdown(paste("❌ Error:", e$message)), parse_mode = "MarkdownV2")
    return(NULL)
  })
  
  if (!is.null(result)) {
    prob_text <- paste0(names(result$Probabilities), ": ", round(result$Probabilities * 100, 1), "%", collapse = "\n")
    model_text <- paste0(names(result$Per_Model), ": ", result$Per_Model, collapse = "\n")
    
    # --- Reinstated notes & disclaimer ---
    notes_block <- paste0(
      "\n\n💡 Note:\n",
      "- 'normal' indicates no detected murmur or artifact.\n",
      "- 'murmur' indicates potential murmur sounds that may require follow-up.\n",
      "- 'extrahls' indicates extra heart sounds; 'extrastole' indicates premature/extra beats.\n\n",
      "⚠️ Disclaimer:\n",
      "This tool is provided for research and educational purposes only. It is NOT a medical diagnostic device. Predictions are probabilistic and should NOT be used as the sole basis for clinical decisions. Always consult a qualified healthcare professional for diagnosis and treatment."
    )
    
    msg <- paste0(
      "🎵 **Heart Sound Prediction**\n\n",
      "🔹 **File:** ", result$File, "\n\n",
      "🏆 **Final Predicted Class:** *", result$Final_Class, "*\n",
      "🔹 **Confidence:** ", result$Confidence, "%\n\n",
      "🧩 **Per-Model Predictions:**\n", model_text, "\n\n",
      "📊 **Class Probabilities (Weighted):**\n", prob_text,
      notes_block
    )
    
    safe_msg <- escape_markdown(msg)
    bot$sendMessage(chat_id = chat_id, text = safe_msg, parse_mode = "MarkdownV2")
  }
  
  unlink(temp_file)
}

# ===============================
# 5) Bot Setup
# ===============================
updater <- Updater(token = TOKEN)
updater <- updater + CommandHandler("start", start_handler)
updater <- updater + MessageHandler(audio_handler)

cat("🤖 Bot is starting...\n")
updater$start_polling()
cat("✅ Bot running and listening for messages.\n")
updater$idle()
