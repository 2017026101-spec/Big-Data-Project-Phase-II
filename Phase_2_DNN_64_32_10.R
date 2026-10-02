# PHASE 2: DNN WITH THREE HIDDEN LAYERS (64, 32, 10)
# R >= 4.1. Run in RStudio from the folder containing the three CSV files.
# ONE-TIME INSTALLATION (run separately, then restart R):
# install.packages(c("keras3", "ggplot2"))
# keras3::install_keras(backend="tensorflow")
# Set run_dnn=FALSE to run the GLM and preprocessing first.
# The supplied validation windows overlap training target histories; validation
# scores are preliminary, not an independent chronological holdout assessment.
# No Taiwan dummy exists in these files, so no Taiwan models can be fitted.

# 1. SETTINGS ---------------------------------------------------------------
files <- c(
  train = file.choose(),
  validation = file.choose(),
  test = file.choose()
)
requested_countries <- c("AUS", "TWN")
output_dir <- "phase2_outputs_DNN_64_32_10"

#First run whole code with run_DNN=FALSE, then after running full code set run_dnn = TRUE## 
run_dnn <- TRUE
seed <- 2026L
max_epochs <- 300L
batch_size <- 64L
hidden_units <- c(64L,32L,10L)
dropout_rate <- 0.05
early_stopping_patience <- 25L
refit_with_validation <- FALSE # Keep FALSE until historical window dates verified.
log_base <- exp(1)             # ASSUMPTION: supplied log_mx is natural log.
# If the source uses base-10 logarithms, change log_base to 10.
# Log-scale scores do not depend on this assumption; rate-scale scores do.
# Test horizon labels follow the lecturer's file designation 2000-2019.
test_years <- 2000:2019
input_cols <- sprintf("input_log_mx_%02d", 1:20)
target_cols <- sprintf("target_log_mx_%02d", 1:20)
dir.create(output_dir, recursive=TRUE, showWarnings=FALSE)
set.seed(seed)
if (!requireNamespace("ggplot2", quietly=TRUE)) stop("Install ggplot2 first.")
library(ggplot2)
if (run_dnn) {
 if (!requireNamespace("keras3", quietly=TRUE)) stop("Install keras3 first.")
 library(keras3)
 use_backend("tensorflow")
 set_random_seed(seed)
}
out <- function(d, name) write.csv(d, file.path(output_dir,name), row.names=FALSE)
plot_out <- function(p,name,width=10,height=6) {
 dir.create(output_dir,recursive=TRUE,showWarnings=FALSE)
 grDevices::png(filename=file.path(output_dir,name),
                width=width,height=height,units="in",res=300,bg="white")
 on.exit(grDevices::dev.off(),add=TRUE)
 print(p)
}
to_rate <- function(x) {
 z <- log_base^x
 if (any(!is.finite(z)) || any(z<=0)) stop("Rate back-transformation overflow/underflow.")
 z
}

# 2. IMPORT AND DECODE COUNTRIES --------------------------------------------
read_split <- function(path) {
 if (!file.exists(path)) stop("File missing: ",path,". Edit files in Section 1.")
 d <- read.csv(path,check.names=FALSE)
 cc <- grep("^country_",names(d),value=TRUE)
 needed <- c(cc,"gender","age",input_cols,target_cols)
 if (!length(cc) || !all(needed %in% names(d))) stop("Unexpected CSV schema: ",path)
 if (!all(vapply(d[needed],is.numeric,logical(1)))) stop("Non-numeric data: ",path)
 if (anyNA(d[needed]) || any(!is.finite(as.matrix(d[needed]))))
  stop("Missing/non-finite data: ",path)
 if (!all(as.matrix(d[cc]) %in% c(0,1)) || any(rowSums(d[cc])>1))
  stop("Invalid country dummy encoding: ",path)
 if (!all(d$gender %in% c(0,1)) || any(d$age<0 | d$age>1))
  stop("Invalid gender or normalised age: ",path)
 labels <- sub("^country_","",cc)
 d$Country <- "AUS"
 for (i in seq_along(cc)) d$Country[d[[cc[i]]]==1] <- labels[i]
 d$Source_row <- seq_len(nrow(d))
 d
}
splits <- lapply(files,read_split)
country_cols <- grep("^country_",names(splits$train),value=TRUE)
for (s in names(splits)) {
 if (!identical(country_cols,grep("^country_",names(splits[[s]]),value=TRUE)))
  stop("Country columns differ between splits.")
}
# Preserve supplied one-hot encoding. AUS is the omitted reference category.
# A scalar ID registry is saved for interpretation; modelling is country-specific.
registry <- data.frame(Country=c("AUS",sub("^country_","",country_cols)),
                      Country_id=0:length(country_cols))
out(registry,"01_country_registry.csv")
coverage <- do.call(rbind,lapply(names(splits),function(s) {
 d <- splits[[s]]
 z <- as.data.frame(table(factor(d$Country,levels=registry$Country)))
 names(z) <- c("Country","Rows"); z$Split <- s; z
}))
out(coverage,"02_country_coverage.csv")
status <- data.frame(Country=requested_countries,
 Train=vapply(requested_countries,function(c) sum(splits$train$Country==c),integer(1)),
 Validation=vapply(requested_countries,function(c) sum(splits$validation$Country==c),integer(1)),
 Test=vapply(requested_countries,function(c) sum(splits$test$Country==c),integer(1)))
status$Available <- status$Train>0 & status$Validation>0 & status$Test>0
out(status,"03_baseline_status.csv"); print(status)
if (any(!status$Available)) warning("Unavailable countries skipped: ",
 paste(status$Country[!status$Available],collapse=", "))
if (!any(status$Available)) stop("No requested country is available in all splits.")

# 3. COLUMN-WISE MIN-MAX SCALING --------------------------------------------
# Fit on TRAINING rows only, including training response columns.
# Targets remain responses and are never fed into the predictors.
# Do not normalise age/gender/country dummies again.
# Validation/test values outside [0,1] are retained, never clipped.
fit_scaler <- function(d,cols) {
 lo <- vapply(d[cols],min,numeric(1)); hi <- vapply(d[cols],max,numeric(1))
 span <- hi-lo; constant <- span==0; span[constant] <- 1
 list(min=lo,max=hi,span=span,constant=constant)
}
scale_matrix <- function(d,cols,sc) {
 sweep(sweep(as.matrix(d[cols]),2,sc$min[cols],"-"),2,sc$span[cols],"/")
}
unscale_targets <- function(m,sc) {
 sweep(sweep(m,2,sc$span[target_cols],"*"),2,sc$min[target_cols],"+")
}
score <- function(actual,predicted) {
 if (!identical(dim(actual),dim(predicted)) || any(!is.finite(predicted)))
  stop("Invalid forecast dimensions or values.")
 a <- as.vector(actual); p <- as.vector(predicted)
 sst <- sum((a-mean(a))^2)
 data.frame(MAE=mean(abs(a-p)),MSE=mean((a-p)^2),
 R2=if(sst>0) 1-sum((a-p)^2)/sst else NA_real_)
}
# R2 is 1-SSE/SST, not squared correlation. Negative test R2 is possible.
score_both <- function(actual_log,predicted_log,country,model,split) {
 rbind(cbind(data.frame(Country=country,Model=model,Split=split,Scale="Log"),
             score(actual_log,predicted_log)),
       cbind(data.frame(Country=country,Model=model,Split=split,Scale="Rate"),
             score(to_rate(actual_log),to_rate(predicted_log))))
}

# 4. GLM: ONE GAUSSIAN GLM PER FORECAST HORIZON -------------------------------
# The response is a continuous log mortality rate, so Poisson is inappropriate.
# Gaussian identity-link GLMs are least-squares regression models on log rates.
# Do not describe this comparator as fundamentally different from regression.
# Age spline + gender + the 20 input histories predict each of 20 future values.
# Output coefficients can depend on horizon. No target columns enter predictors.
glm_predictors <- function(d,x) {
 z <- as.data.frame(x); names(z) <- input_cols
 z$age <- d$age; z$gender <- factor(d$gender,levels=c(0,1)); z
}
fit_glms <- function(d,x,y,age_df) {
 z <- glm_predictors(d,x)
 rhs <- paste(c(sprintf("splines::ns(age,df=%d,Boundary.knots=c(0,1))",age_df),
                "gender",input_cols),collapse=" + ")
 lapply(seq_len(20),function(h) {
  z$response <- y[,h]
  m <- glm(as.formula(paste("response ~",rhs)),data=z,family=gaussian())
  if (!m$converged || m$rank<length(coef(m)))
   stop("GLM convergence/rank issue; reduce predictor complexity before proceeding.")
  m
 })
}
predict_glms <- function(models,d,x) {
 z <- glm_predictors(d,x)
 vapply(models,function(m) as.numeric(predict(m,newdata=z)),numeric(nrow(d)))
}
# Inverse log transformation gives a point forecast; it is not automatically
# the conditional mean mortality rate. No smearing correction is claimed.

# 5. DNN: 22 INPUTS -> 64 -> 32 -> 10 -> 20 OUTPUTS ---------------------------
# A direct sequence-to-sequence forecast model. This differs from the paper's
# age/year/sex/population embedding architecture; describe it as an adaptation.
# Country dummies are constant within each selected country and are omitted.
dnn_predictors <- function(d,x) cbind(x,age=d$age,gender=d$gender)
build_dnn <- function(model_seed) {
 clear_session(); set_random_seed(model_seed)
 m <- keras_model_sequential(input_shape=c(22)) |>
  layer_dense(units=hidden_units[1],activation="relu") |>
  layer_dropout(rate=dropout_rate) |>
  layer_dense(units=hidden_units[2],activation="relu") |>
  layer_dropout(rate=dropout_rate) |>
  layer_dense(units=hidden_units[3],activation="relu") |>
  layer_dropout(rate=dropout_rate) |>
  layer_dense(units=20,activation="linear")
 compile(m,optimizer=optimizer_adam(learning_rate=0.001),loss="mse")
 m
}
# Linear outputs permit forecasts outside training min-max bounds.
# Training loss weights column-standardised target errors equally, not raw-rate MSE.

# 6. COUNTRY MODELLING ------------------------------------------------------
all_metrics <- list(); all_horizon_metrics <- list(); counter <- 0L
for (country in status$Country[status$Available]) {
 message("Fitting country: ",country)
 d <- lapply(splits,function(z) z[z$Country==country,,drop=FALSE])
 # Validate test grid: exactly one forecast window for each age/gender pair.
 if (anyDuplicated(d$test[c("age","gender")]) || nrow(d$test)!=200L ||
     length(unique(d$test$age))!=100L) stop("Unexpected test grid for ",country)
 sc <- fit_scaler(d$train,c(input_cols,target_cols))
 scaling <- data.frame(Column=names(sc$min),Min=sc$min,Max=sc$max,
                       Span=sc$span,Constant=sc$constant)
 out(scaling,paste0(country,"_04_scaling.csv"))
 saveRDS(sc,file.path(output_dir,paste0(country,"_scaler.rds")))
 x <- lapply(d,function(z) scale_matrix(z,input_cols,sc))
 y <- lapply(d,function(z) scale_matrix(z,target_cols,sc))
 for (s in names(d)) {
  z <- d[[s]]; z[input_cols] <- x[[s]]; z[target_cols] <- y[[s]]
  out(z,paste0(country,"_",s,"_scaled.csv"))
 }
 target_log <- lapply(d,function(z) as.matrix(z[target_cols]))
 # Select spline complexity using supplied validation only, log-scale MSE.
 dfs <- c(4L,8L)
 candidate_models <- lapply(dfs,function(df) fit_glms(d$train,x$train,y$train,df))
 glm_validation <- lapply(candidate_models,function(m)
  unscale_targets(predict_glms(m,d$validation,x$validation),sc))
 cv <- do.call(rbind,lapply(seq_along(dfs),function(i)
  cbind(data.frame(Age_df=dfs[i]),score(target_log$validation,glm_validation[[i]]))))
 out(cv,paste0(country,"_05_GLM_validation_candidates.csv"))
 best <- which.min(cv$MSE)
 selected <- candidate_models[[best]]
 predicted <- list(GLM=unscale_targets(predict_glms(selected,d$test,x$test),sc))
 counter <- counter+1L
 all_metrics[[counter]] <- score_both(target_log$validation,glm_validation[[best]],
                                     country,"GLM","Validation")
 best_epoch <- NA_integer_
 if (refit_with_validation) {
  selected <- fit_glms(rbind(d$train,d$validation),rbind(x$train,x$validation),
                      rbind(y$train,y$validation),dfs[best])
  predicted$GLM <- unscale_targets(predict_glms(selected,d$test,x$test),sc)
 }
 saveRDS(selected,file.path(output_dir,paste0(country,"_GLM_20_horizons.rds")))
 capture.output(lapply(selected,summary),
                file=file.path(output_dir,paste0(country,"_GLM_summaries.txt")))
 if (run_dnn) {
  model_seed <- seed+match(country,registry$Country)-1L
  m <- build_dnn(model_seed)
  learning <- fit(m,x=dnn_predictors(d$train,x$train),y=y$train,
   validation_data=list(dnn_predictors(d$validation,x$validation),y$validation),
   epochs=max_epochs,batch_size=batch_size,verbose=2,
   callbacks=list(callback_early_stopping(monitor="val_loss",patience=early_stopping_patience,
                                         restore_best_weights=TRUE)))
  lc <- data.frame(Epoch=seq_along(learning$metrics$loss),
                   Training=learning$metrics$loss,Validation=learning$metrics$val_loss)
  best_epoch <- which.min(lc$Validation)
  out(lc,paste0(country,"_06_DNN_learning.csv"))
  vp <- unscale_targets(as.matrix(predict(m,dnn_predictors(d$validation,x$validation),
                                          verbose=0)),sc)
  counter <- counter+1L
  all_metrics[[counter]] <- score_both(target_log$validation,vp,country,"DNN","Validation")
  if (refit_with_validation) {
   m <- build_dnn(model_seed)
   fit(m,x=rbind(dnn_predictors(d$train,x$train),dnn_predictors(d$validation,x$validation)),
       y=rbind(y$train,y$validation),epochs=best_epoch,batch_size=batch_size,verbose=2)
  }
  predicted$DNN <- unscale_targets(as.matrix(predict(m,dnn_predictors(d$test,x$test),
                                                    verbose=0)),sc)
  capture.output(summary(m),file=file.path(output_dir,paste0(country,"_DNN_summary.txt")))
  save_model(m,file.path(output_dir,paste0(country,"_DNN.keras")))
  lc_long <- rbind(data.frame(Epoch=lc$Epoch,Series="Training",MSE=lc$Training),
                   data.frame(Epoch=lc$Epoch,Series="Validation",MSE=lc$Validation))
  plot_out(ggplot(lc_long,aes(Epoch,MSE,colour=Series))+geom_line()+theme_minimal()+
   labs(title=paste(country,"DNN learning curves"),y="MSE on normalised log targets"),
   paste0(country,"_Figure_01_learning.png"))
 }
 forecasts <- list()
 for (model in names(predicted)) {
  pl <- predicted[[model]]
  counter <- counter+1L
  all_metrics[[counter]] <- score_both(target_log$test,pl,country,model,"Test")
  hm <- do.call(rbind,lapply(seq_len(20),function(h) {
   cbind(data.frame(Country=country,Model=model,Year=test_years[h],Horizon=h,Scale="Log"),
         score(matrix(target_log$test[,h],ncol=1),matrix(pl[,h],ncol=1)))
  }))
  hm_rate <- do.call(rbind,lapply(seq_len(20),function(h) {
   cbind(data.frame(Country=country,Model=model,Year=test_years[h],Horizon=h,Scale="Rate"),
    score(matrix(to_rate(target_log$test[,h]),ncol=1),matrix(to_rate(pl[,h]),ncol=1)))
  }))
  all_horizon_metrics[[paste(country,model)]] <- rbind(hm,hm_rate)
  # Row order here is row-major: all horizons for row 1, then row 2, etc.
  z <- data.frame(Country=country,Model=model,
   Source_row=rep(d$test$Source_row,each=20),
   Age_normalised=rep(d$test$age,each=20),Gender=rep(d$test$gender,each=20),
   Year=rep(test_years,times=nrow(d$test)),
   Actual_log=as.vector(t(target_log$test)),Predicted_log=as.vector(t(pl)))
  z$Actual_rate <- to_rate(z$Actual_log); z$Predicted_rate <- to_rate(z$Predicted_log)
  forecasts[[model]] <- z
 }
 f <- do.call(rbind,forecasts)
 out(f,paste0(country,"_07_test_forecasts.csv"))
 # Plot 6 representative observed age levels, preserving lecturer's age scaling.
 ages <- sort(unique(f$Age_normalised))
 chosen_ages <- ages[c(1,21,41,61,81,100)]
 fp <- f[f$Age_normalised %in% chosen_ages,]
 observed <- unique(fp[c("Age_normalised","Gender","Year","Actual_log")])
 plot_out(ggplot(fp,aes(Year,Predicted_log,colour=Model))+geom_line()+
  geom_line(data=observed,aes(Year,Actual_log),inherit.aes=FALSE,colour="black",linetype=2)+
  facet_grid(Gender~Age_normalised,scales="free_y")+theme_minimal()+
  labs(title=paste(country,"20-year mortality forecasts"),
       subtitle="Dashed black = observed; panels show gender code and supplied normalised age",
       y="Log central death rate"),paste0(country,"_Figure_02_forecasts.png"),12,7)
 saveRDS(list(scaler=sc,input_cols=input_cols,target_cols=target_cols,
  glm_age_df=dfs[best],best_dnn_epoch=best_epoch,log_base=log_base,
  refit_with_validation=refit_with_validation,seed=seed,
  hidden_units=hidden_units,dropout_rate=dropout_rate,
  early_stopping_patience=early_stopping_patience,max_epochs=max_epochs,
  batch_size=batch_size),
  file.path(output_dir,paste0(country,"_settings.rds")))
}

# 7. EXPORT COMPARISON TABLES -----------------------------------------------
results <- do.call(rbind,all_metrics); rownames(results) <- NULL
horizon_results <- do.call(rbind,all_horizon_metrics); rownames(horizon_results) <- NULL
out(results,"08_model_comparison.csv"); print(results)
out(horizon_results,"09_horizon_metrics.csv")
plot_out(ggplot(subset(horizon_results,Scale=="Log"),aes(Year,MSE,colour=Model))+
 geom_line()+facet_wrap(~Country)+theme_minimal()+
 labs(title="Forecast MSE by year",y="MSE on supplied log scale"),"Figure_03_year_MSE.png")
capture.output(sessionInfo(),file=file.path(output_dir,"sessionInfo.txt"))
out(data.frame(Split=names(files),File=unname(files),MD5=unname(tools::md5sum(files))),
    "10_input_checksums.csv")
writeLines(c(
 "Validation target sequences overlap training histories; scores are preliminary.",
 "Training/validation calendar years are not encoded in the supplied CSVs.",
 "Test years follow the supplied file designation 2000-2019; verify source window construction.",
 "Age is used exactly as supplied (observed range 0 to 0.99), with no further scaling.",
 "Gender codes are used as supplied; Female/Male labels remain unconfirmed.",
 paste("Assumed log base for rate back-transformation:",log_base),
 "GLM Gaussian identity response is normalised log mortality, not a Poisson death count.",
 "DNN hidden layers: 64, 32, 10 ReLU neurons; output: 20 linear neurons.",
 paste("Dropout after each hidden layer:",dropout_rate),
 paste("Early stopping patience:",early_stopping_patience,"epochs; best weights restored."),
 "DNN is a 20-output sequence forecast adaptation, not a Richman model replication.",
 "Taiwan is unavailable in this release; obtain its history or lecturer approval for another baseline.",
 "Do not select hyperparameters using the test scores.",
 "For the Python comparison preserve splits, scaling, predictors, targets and metric formulas."),
 file.path(output_dir,"modelling_notes.txt"))
message("Completed. Outputs in ",normalizePath(output_dir))
# To reuse a saved DNN, apply its saved training scaler and feed ONLY the 20
# scaled input columns plus supplied age/gender. Invert target scaling afterward.
# Static checks only: the modified architecture has not been trained here.
# Compare the new held-out scores with the earlier 128/64 run without choosing
# repeated changes based on test performance; validation overlap still applies.

