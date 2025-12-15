# ============================================
# Heartbeat Feature Extraction (R)
# ============================================

library(tidyverse)
library(tuneR)
library(seewave)
library(httr)
library(tools)

# -------------------------
# 1) Configuration
# -------------------------

# Use current working directory to save features file
base_dir <- getwd()

dataset_url <- "https://github.com/Wadahupy/Heartbeat-Sounds-Dataset/archive/refs/heads/master.zip"
save_dir <- file.path(base_dir, "heartbeat_sounds_dataset")
features_file <- file.path(base_dir, "features_labeled.csv")
n_mfcc <- 13

dir.create(save_dir, showWarnings = FALSE, recursive = TRUE)

# -------------------------
# 2) Download & unzip
# -------------------------
zip_path <- tempfile(fileext = ".zip")
cat("⬇️ Downloading dataset...\n")
GET(dataset_url, write_disk(zip_path, overwrite = TRUE))
unzip(zip_path, exdir = save_dir)
cat("✅ Extraction complete.\n")

# -------------------------
# 3) Locate all .wav files
# -------------------------
wav_files <- list.files(save_dir, pattern = "\\.wav$", recursive = TRUE, full.names = TRUE)
cat("🎧 Found", length(wav_files), "audio files.\n")

# -------------------------
# 4) Load label CSV (optional)
# -------------------------
label_csv_path <- list.files(save_dir, pattern = "\\.csv$", recursive = TRUE, full.names = TRUE)
labels_df <- NULL
if (length(label_csv_path) > 0) {
  labels_df <- read_csv(label_csv_path[1], show_col_types = FALSE)
  names(labels_df) <- tolower(trimws(names(labels_df)))
  cat("📄 Loaded label file:", label_csv_path[1], "\n")
} else {
  cat("⚠️ No CSV found — will infer labels from file paths.\n")
}

# -------------------------
# 5) Infer labels from path (extrahls and extrastole distinct)
# -------------------------
infer_label_from_path <- function(path) {
  path_lower <- tolower(path)
  if (grepl("normal", path_lower)) return("normal")
  if (grepl("murmur", path_lower)) return("murmur")
  if (grepl("extrahls", path_lower)) return("extrahls")
  if (grepl("extrastole", path_lower)) return("extrastole")
  if (grepl("artifact", path_lower)) return("artifact")
  if (grepl("unlabelledtest|unlabeled", path_lower)) return("unknown")
  return("unknown")
}

# -------------------------
# 6) MFCC + richer features extraction
# -------------------------
extract_features_file <- function(file_path, n_mfcc = 13) {
  wav <- readWave(file_path)
  wav <- mono(wav, "left")
  sr <- wav@samp.rate
  
  # MFCC matrix
  mfcc_mat <- melfcc(wav, sr = sr, numcep = n_mfcc)
  
  # Ensure correct orientation
  if (ncol(mfcc_mat) != n_mfcc && nrow(mfcc_mat) == n_mfcc) mfcc_mat <- t(mfcc_mat)
  
  # MFCC mean & SD
  mfcc_means <- apply(mfcc_mat, 2, mean, na.rm = TRUE)
  mfcc_sds   <- apply(mfcc_mat, 2, sd, na.rm = TRUE)
  
  # Delta & DeltaDelta
  delta_mat <- apply(mfcc_mat, 2, function(x) c(NA, diff(x)))
  deltadelta_mat <- apply(delta_mat, 2, function(x) c(NA, diff(x)))
  
  delta_means <- apply(delta_mat, 2, mean, na.rm = TRUE)
  deltadelta_means <- apply(deltadelta_mat, 2, mean, na.rm = TRUE)
  
  # Combine features
  feats <- c(mfcc_means, mfcc_sds, delta_means, deltadelta_means)
  names(feats) <- c(
    paste0("mfcc_", 1:n_mfcc),
    paste0("mfcc_sd_", 1:n_mfcc),
    paste0("delta_", 1:n_mfcc),
    paste0("deltadelta_", 1:n_mfcc)
  )
  as_tibble_row(as.list(feats))
}

# -------------------------
# 7) Build dataset
# -------------------------
data_list <- vector("list", length(wav_files))
for (i in seq_along(wav_files)) {
  file_path <- wav_files[i]
  filename <- basename(file_path)
  
  # label
  label <- NA
  if (!is.null(labels_df) && "fname" %in% names(labels_df)) {
    match_row <- labels_df %>% filter(grepl(filename, fname, ignore.case = TRUE))
    if (nrow(match_row) > 0) label <- match_row$label[1]
  }
  if (is.na(label) || label == "") label <- infer_label_from_path(file_path)
  
  feats <- extract_features_file(file_path, n_mfcc)
  
  data_list[[i]] <- bind_cols(tibble(file_path = file_path, label = label), feats)
  
  if (i %% 50 == 0) cat("Processed", i, "/", length(wav_files), "files...\n")
}

# -------------------------
# 8) Combine & save
# -------------------------
final_df <- bind_rows(data_list)
final_df$label[is.na(final_df$label) | final_df$label==""] <- "unknown"

cat("\n📊 Label distribution:\n")
print(table(final_df$label))

write_csv(final_df, features_file)
cat("✅ Saved feature dataset to:", features_file, "\n")
