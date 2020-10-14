##   Rock mass classification by multivariate statistical techniques and artificial intelligence  ##
##   System for rock mass classification developed by: Santos, A.E.M.;Lana, M.S.;Pereira, T.M.    ##
##   Graduate Program in Mineral Engineering - PPGEM                                              ##
##   Federal University of Ouro Preto - UFOP                                                      ##
##   Mining Engineering DepartmentDEMIN                                                           ##

## Script to ANN model ##
## To run the script, it is necessary that the database has the weight  ##
## system adopted by the present research (according to Table 1).       ##
## Another important point to apply the neural network the database     ##
## must be in the form of dummies variables; the user can choose to     ##
## build the dummies manually or use a specific R package.              ##

## Clearing the R memory ##
rm(list = ls())
## choice of work directory ##
## Here the user chooses the folder where his database is and where the R history and the outputs will be saved ##
setwd("D:/Onedrive/revisao_paper_GeotechnicalGeologicalEngineering/supplementary_materials")
## Packages used ##
library(neuralnet)
library(gmodels)
library(ggsn)
library(caret)
library(plyr)
library(multiROC)
library(pROC)
library(UBL)
## Choose your database in csv format ##
## If there is any difficulty in reading the database, please consult the documentation at "help(read.csv2)" ##
data_base <- read.csv2("dataset.csv",header=TRUE,dec = ".")
data_base = data_base[,2:26]

## conversion of variables for the balancing process ##
data_base$O1intact_rock_strength  <- as.factor(data_base$O1intact_rock_strength ) 
data_base$O2intact_rock_strength  <- as.factor(data_base$O2intact_rock_strength ) 
data_base$O3intact_rock_strength  <- as.factor(data_base$O3intact_rock_strength ) 
data_base$O4intact_rock_strength  <- as.factor(data_base$O4intact_rock_strength ) 
data_base$O5intact_rock_strength  <- as.factor(data_base$O5intact_rock_strength ) 
data_base$O6intact_rock_strength  <- as.factor(data_base$O6intact_rock_strength ) 
data_base$O1weathering <- as.factor(data_base$O1weathering)
data_base$O2weathering <- as.factor(data_base$O2weathering)
data_base$O3weathering <- as.factor(data_base$O3weathering)
data_base$O4weathering <- as.factor(data_base$O4weathering)
data_base$O5weathering <- as.factor(data_base$O5weathering)
data_base$O1discontinuity_spacing  <- as.factor(data_base$O1discontinuity_spacing )
data_base$O2discontinuity_spacing  <- as.factor(data_base$O2discontinuity_spacing )
data_base$O3discontinuity_spacing  <- as.factor(data_base$O3discontinuity_spacing )
data_base$O4discontinuity_spacing  <- as.factor(data_base$O4discontinuity_spacing )
data_base$O1discontinuity_persistence <- as.factor(data_base$O1discontinuity_persistence)
data_base$O2discontinuity_persistence <- as.factor(data_base$O2discontinuity_persistence)
data_base$O3discontinuity_persistence <- as.factor(data_base$O3discontinuity_persistence)
data_base$O4discontinuity_persistence <- as.factor(data_base$O4discontinuity_persistence)
data_base$O1discontinuity_aperture <- as.factor(data_base$O1discontinuity_aperture)
data_base$O2discontinuity_aperture <- as.factor(data_base$O2discontinuity_aperture)
data_base$O3discontinuity_aperture <- as.factor(data_base$O3discontinuity_aperture)
data_base$O4discontinuity_aperture <- as.factor(data_base$O4discontinuity_aperture)
data_base$N1presence_of_water <- as.factor(data_base$N1presence_of_water)
data_base$classification <- as.factor(data_base$classification)
## Procedure for balancing the database against classes ##
## If there is any difficulty in applying Smote, please consult the documentation at "help(SmoteClassif)" ##
balanced_data <- SmoteClassif(classification ~., 
                                  data_base, 
                                  C.perc = "balance", 
                                  k = 5, 
                                  repl = FALSE,
                                  dist = "Overlap")

## Visualization of the balancing process ##
summary(balanced_data$classification)
plot(balanced_data$classification, ylim=c(0,1000))
## Assigning maximum interaction limit ##
maxit<-as.integer(1000000)
## separation of training and test samples ##
train_idx <- sample(nrow(balanced_data), 2/3 * nrow(balanced_data))
data_train <- balanced_data[train_idx, ]
data_test <- balanced_data[-train_idx, ]
## conversion of variables for the ANN training ##
data_train$O1intact_rock_strength  <- as.numeric(data_train$O1intact_rock_strength ) 
data_train$O2intact_rock_strength  <- as.numeric(data_train$O2intact_rock_strength ) 
data_train$O3intact_rock_strength  <- as.numeric(data_train$O3intact_rock_strength ) 
data_train$O4intact_rock_strength  <- as.numeric(data_train$O4intact_rock_strength ) 
data_train$O5intact_rock_strength  <- as.numeric(data_train$O5intact_rock_strength ) 
data_train$O6intact_rock_strength  <- as.numeric(data_train$O6intact_rock_strength ) 
data_train$O1weathering <- as.numeric(data_train$O1weathering)
data_train$O2weathering <- as.numeric(data_train$O2weathering)
data_train$O3weathering <- as.numeric(data_train$O3weathering)
data_train$O4weathering <- as.numeric(data_train$O4weathering)
data_train$O5weathering <- as.numeric(data_train$O5weathering)
data_train$O1discontinuity_spacing <- as.numeric(data_train$O1discontinuity_spacing)
data_train$O2discontinuity_spacing <- as.numeric(data_train$O2discontinuity_spacing)
data_train$O3discontinuity_spacing <- as.numeric(data_train$O3discontinuity_spacing)
data_train$O4discontinuity_spacing <- as.numeric(data_train$O4discontinuity_spacing)
data_train$O1discontinuity_persistence <- as.numeric(data_train$O1discontinuity_persistence)
data_train$O2discontinuity_persistence <- as.numeric(data_train$O2discontinuity_persistence)
data_train$O3discontinuity_persistence <- as.numeric(data_train$O3discontinuity_persistence)
data_train$O4discontinuity_persistence <- as.numeric(data_train$O4discontinuity_persistence)
data_train$O1discontinuity_aperture <- as.numeric(data_train$O1discontinuity_aperture)
data_train$O2discontinuity_aperture <- as.numeric(data_train$O2discontinuity_aperture)
data_train$O3discontinuity_aperture <- as.numeric(data_train$O3discontinuity_aperture)
data_train$O4discontinuity_aperture <- as.numeric(data_train$O4discontinuity_aperture)
data_train$N1presence_of_water <- as.numeric(data_train$N1presence_of_water)
data_test$O1intact_rock_strength  <- as.numeric(data_test$O1intact_rock_strength ) 
data_test$O2intact_rock_strength  <- as.numeric(data_test$O2intact_rock_strength ) 
data_test$O3intact_rock_strength  <- as.numeric(data_test$O3intact_rock_strength ) 
data_test$O4intact_rock_strength  <- as.numeric(data_test$O4intact_rock_strength ) 
data_test$O5intact_rock_strength  <- as.numeric(data_test$O5intact_rock_strength ) 
data_test$O6intact_rock_strength  <- as.numeric(data_test$O6intact_rock_strength ) 
data_test$O1weathering <- as.numeric(data_test$O1weathering)
data_test$O2weathering <- as.numeric(data_test$O2weathering)
data_test$O3weathering <- as.numeric(data_test$O3weathering)
data_test$O4weathering <- as.numeric(data_test$O4weathering)
data_test$O5weathering <- as.numeric(data_test$O5weathering)
data_test$O1discontinuity_spacing <- as.numeric(data_test$O1discontinuity_spacing)
data_test$O2discontinuity_spacing <- as.numeric(data_test$O2discontinuity_spacing)
data_test$O3discontinuity_spacing <- as.numeric(data_test$O3discontinuity_spacing)
data_test$O4discontinuity_spacing <- as.numeric(data_test$O4discontinuity_spacing)
data_test$O1discontinuity_persistence <- as.numeric(data_test$O1discontinuity_persistence)
data_test$O2discontinuity_persistence <- as.numeric(data_test$O2discontinuity_persistence)
data_test$O3discontinuity_persistence <- as.numeric(data_test$O3discontinuity_persistence)
data_test$O4discontinuity_persistence <- as.numeric(data_test$O4discontinuity_persistence)
data_test$O1discontinuity_aperture <- as.numeric(data_test$O1discontinuity_aperture)
data_test$O2discontinuity_aperture <- as.numeric(data_test$O2discontinuity_aperture)
data_test$O3discontinuity_aperture <- as.numeric(data_test$O3discontinuity_aperture)
data_test$O4discontinuity_aperture <- as.numeric(data_test$O4discontinuity_aperture)
data_test$N1presence_of_water <- as.numeric(data_test$N1presence_of_water)

## ANN training ##
nn <- neuralnet((classification == "class_I") 
                + (classification == "class_II") 
                + (classification == "class_III") 
                + (classification == "class_IV") 
                + (classification == "class_V")
                ~ O1intact_rock_strength + O2intact_rock_strength + O3intact_rock_strength + O4intact_rock_strength + O5intact_rock_strength + O6intact_rock_strength 
                + O1weathering + O2weathering + O3weathering + O4weathering + O5weathering 
                + O1discontinuity_spacing + O2discontinuity_spacing + O3discontinuity_spacing + O4discontinuity_spacing 
                + O1discontinuity_persistence + O2discontinuity_persistence + O3discontinuity_persistence + O4discontinuity_persistence 
                + O1discontinuity_aperture + O2discontinuity_aperture + O3discontinuity_aperture + O4discontinuity_aperture 
                + N1presence_of_water, 
                data = data_train, algorithm = "rprop+", err.fct = "sse", stepmax=maxit, 
                threshold =1, hidden = c(25), act.fct = "tanh", linear.output = FALSE)

## neural network plot ##
plot(nn, rep = "best",
     radius = 0.05, arrow.length = 0.2, intercept = TRUE,
    intercept.factor = 0.4, information = FALSE, information.pos = 0.1,
    col.entry.synapse = "blue4", col.entry = "black",
    col.hidden = "black", col.hidden.synapse = "black",
    col.out = "black", col.out.synapse = "brown2",
   col.intercept = "blue", fontsize = 14, dimension = 3,
   show.weights = FALSE)
## Overfitting and underfitting verification - Prediction of the model using the training sample ##
pred_train_nn <- predict(nn, data_train)
d_train_nn <- apply(pred_train_nn, 1, which.max)
d_train_nn <- mapvalues(d_train_nn,from = c(1,2,3,4,5), to = c("class_I", "class_II", "class_III", "class_IV", "class_V"))
d_train_nn <- as.factor(d_train_nn)
result_train_nn <- confusionMatrix(d_train_nn, data_train$classification)
roc_curve_train_nn <- multiclass.roc(response = data_train$classification, predictor = as.numeric(as.factor(d_train_nn)))
## Validation of the neural network trained with the test sample ##
pred_test_nn <- predict(nn, data_test)
d_test_nn <- apply(pred_test_nn, 1, which.max)
d_test_nn <- mapvalues(d_test_nn,from = c(1,2,3,4,5), to = c("class_I", "class_II", "class_III", "class_IV", "class_V"))
d_test_nn <- as.factor(d_test_nn)
result_test_nn <- confusionMatrix(d_test_nn, data_test$classification)
roc_curve_test_nn <- multiclass.roc(response = data_test$classification, predictor = as.numeric(as.factor(d_test_nn)))

## Confusion matrix - result in the training sample ##
result_train_nn
## Auc - result in the training sample ##
roc_curve_train_nn
## Confusion matrix - result in the test sample ##
result_test_nn
## Auc - result in the test sample ##
roc_curve_test_nn

## Geral metrics for model evaluation ##
accuracy_nn_train <- result_train_nn$overall['Accuracy']
lower_accuracy_nn_train <- result_train_nn$overall['AccuracyLower']
upper_accuracy_nn_train <- result_train_nn$overall['AccuracyUpper']
kappa_nn_train <- result_train_nn$overall['Kappa']
pvalue_accuracy_nn_train <- result_train_nn$overall['AccuracyPValue']
auc_nn_train <- roc_curve_train_nn$auc
vector_model_train_nn <- c("Artificial Neural Networks - Training sample performance",
                           round(accuracy_nn_train,3), 
                           round(lower_accuracy_nn_train,3),
                           round(upper_accuracy_nn_train,3),
                           round(kappa_nn_train,3),
                           round(auc_nn_train,3),
                           round(pvalue_accuracy_nn_train,10^20))

accuracy_nn_test <- result_test_nn$overall['Accuracy']
lower_accuracy_nn_test <- result_test_nn$overall['AccuracyLower']
upper_accuracy_nn_test <- result_test_nn$overall['AccuracyUpper']
kappa_nn_test <- result_test_nn$overall['Kappa']
pvalue_accuracy_nn_test <- result_test_nn$overall['AccuracyPValue']
auc_nn_test <- roc_curve_test_nn$auc
vector_model_test_nn <- c("Artificial Neural Networks - Test sample performance",
                          round(accuracy_nn_test,3), 
                          round(lower_accuracy_nn_test,3),
                          round(upper_accuracy_nn_test,3),
                          round(kappa_nn_test,3),
                          round(auc_nn_test,3),
                          round(pvalue_accuracy_nn_test,10^20))

compare_models <- rbind(vector_model_train_nn,
                         vector_model_test_nn)
rownames(compare_models) <- c("Artificial Neural Networks - Training sample performance", 
                               "Artificial Neural Networks - Test sample performance")
colnames(compare_models) <- c("Model",
                               "Accuracy", 
                               "Lower confidence interval - Accuracy",
                               "Upper confidence interval - Accuracy",
                               "kappa index",
                               "Auc value",
                               "p-Value")
compare_models <- as.data.frame(compare_models)
View(compare_models)
