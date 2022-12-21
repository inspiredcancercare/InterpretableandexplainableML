	Online supplementary materials and reproducible R code repository
# On the importance of Interpretable  Machine Learning Predictions to Inform Clinical Decision Making in Oncology

Table of contents
=================
<!--ts-->

 * [Model development](#model-development)
   * [Data preparation](#data-preparation)
   * [Model training and optimization](#model-training-and-optimization)
   
 * [Model interpretation](#model-interpretation)
    * [Model-specific approaches](#model-specific-approaches)
      * [Model coefficient](#coefficient-based-method)
      * [Decision trees](#decision-tree)
      * [Nearest neighbors](#nearest-neighbors)
    * [Model-agnostic approaches](#model-agnostic-approaches)
      * [Global interpretation](#global-interpretation)
        * [Variable importance](#variable-importance)
        * [Partial dependence plot](#partial-dependence-plot)
        * [Accumulated local effect](#accumulated-local-effect)
      * [Local interpretation](#local-interpretation)
        * [Break Down plot](#break-down-plot)
        * [Local surrogate](#local-surrogate)
        * [Shapely Additive Explanations](#shapely-additive-explanations)
        * [Ceteris-Paribus plot](#ceteris-paribus-plot)
   * [R session information](#r-session-information)
<!--te-->

Model development
==========================
Data preparation
-----------------
```
#Enviornment set-up 

ipak <- function(pkg){
  new.pkg <- pkg[!(pkg %in% installed.packages()[, "Package"])]
  if (length(new.pkg)) 
    install.packages(new.pkg, dependencies = TRUE)
  sapply(pkg, require, character.only = TRUE)
}

packages<-c("caret", "dplyr", "tensorflow", "recipes", "glmnet", 
            "earth","xgboost","rpart","RSNNS","DALEXtra","cowplot",
            "DALEX","lime","iml","rpart.plot","rattle","localModel")

ipak(packages)

#### (2) Load data and prepare ####
db<- read.csv(
  url("https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data")
)

dim(db)
names(db)

names(db)<- c("id", "Thickness", "Cell_Size" ,"Cell_Shape", "Adhesion", "Epithelial_Size", "Bare_Nuclei", 
              "Bland_Chromatin", "Normal_Nucleoli", "Normal_Mitoses","Class")


db<-db %>%
  mutate_all(~na_if(.,"?")) %>%
  mutate_if(is.integer, as.character) %>%
  mutate_if(is.character, as.factor)

db$Class <- as.factor(ifelse(db$Class==2, "benign", "malignant"))

#####Create training and testing data split ######
my_seed=2022
set.seed(my_seed)
split<-createDataPartition(
  y = db$Class,
  ## the outcome data are needed
  p = .75,
  ## The percentage of data in the
  ## training set
  list = FALSE
  )

db_development <- db[split,]
db_validation<-db[-split,]

#### define blueprint for data pre-processing #### 
outcome<-"Class"
predictors<- setdiff(names(db), c(outcome,"id"))
formula<-as.formula(paste(outcome, paste(predictors, collapse = "+"),sep="~"))

set.seed(my_seed)

recipe<-recipe(formula, db_development) %>%
  step_impute_knn(all_predictors(), neighbors = 5) %>% 
  #step_integer(all_nominal(),-all_outcomes())
  step_dummy(all_nominal())

db_prep =prep(recipe, db_development) 

```
 [back to top](#table-of-contents)
 
Model training and optimization
--------------------------------
```
#Train algorithms
#--------GLM--------
set.seed(my_seed)
glm_caret<-caret::train(
  recipe,
  data=db_development,
  method="glmnet",
  trControl = trainControl(savePredictions = "final")
)

#--------MARS--------

set.seed(my_seed)
mars_caret<-caret::train(
  recipe,
  data=db_development,
  method="earth",
  trControl = trainControl(savePredictions = "final"),
  tuneGrid=expand.grid(
    degree =2,
    nprune =12
  )
)

#-----decision tree------
set.seed(my_seed)
dt_caret<-caret::train(
  recipe,
  data=db_development,
  method="rpart",
  trControl = trainControl(savePredictions = "final")
)

#-----xgbt------
set.seed(my_seed)
xgbt_caret<-caret::train(
  recipe,
  data=db_development,
  method="xgbTree",
  trControl = trainControl(savePredictions = "final")
)

#-----knn------
set.seed(my_seed)
knn_caret<-caret::train(
  recipe,
  data=db_development,
  method="knn",
  trControl = trainControl(savePredictions = "final")
)

#-----nnet------
set.seed(my_seed)
nnet_caret<-caret::train(recipe,
                data=db_development,
                method="mlpWeightDecayML",
                trControl = trainControl(savePredictions = "final")
)

```
 [back to top](#table-of-contents)

Model interpretation
======================
Model-specific approaches
-------------------------
Model-specific approaches refer to model interpretation methods that are available as an inherent part of certain ML algorithms. In this paper, we provide introductions to model-specific interpretation apporoaches for models using three commonly used simple ML algorithms.

### Coefficient-based method
```
###GLM coefficient 
coef(glm_caret$finalModel, glm_caret$bestTune$lambda)

###Mars coefficient 
summary(mars_caret)$coefficients
```

### Decision tree
```
###decision rules
tree = dt_caret$finalModel

prp(tree, nn = TRUE)
fancyRpartPlot(tree, digits=1 )
rpart.rules(tree, roundint=1, clip.facs=TRUE, response.name="malignant", nn=T, cover=T)
```

### Nearest neighbors
```
####kNN
#get k first
k = knn_caret$finalModel$k
distanceToTraining<-as.matrix(dist(rbind(db_development[,c(predictors)], dplyr::select(obs,-outcome))))[nrow(db_development) + 1, 1:nrow(db_development)]

NearestNeighbor<-order(distanceToTraining)[1:k]
db_development[NearestNeighbor, 2:11]

x <- bind_rows(obs, db_development[NearestNeighbor, 2:11])
rownames(x)<-c("Predicting Instance (A)","Neighbor 1","Neighbor 2","Neighbor 3","Neighbor 4","Neighbor 5","Neighbor 6","Neighbor 7")

x$Class<-as.character(x$Class)
x$Class[1]<-"To Be Predicted"
x
```
 [back to top](#table-of-contents)
 
 Model-agnostic approaches
 --------------------------
 ```
 #Set-up environment, necessary functions, and model explainer
 
set.seed(my_seed)
df_train.explain.prep<-recipe(formula, db_validation) %>%
  step_impute_knn(all_predictors(), neighbors = 5) %>%
  prep()

set.seed(my_seed)
db_test =bake(df_train.explain.prep, db_validation)

pred_wrapper <-  function(object, newdata)  {
  results <- predict(object, newdata, type = "prob") %>% pull(malignant)
  return(results)                                                       
}


####xgbt explanier
set.seed(my_seed)
xgbt_explainer<-DALEX::explain(
  model= xgbt_caret,
  data = select(db_test,-outcome),
  y = as.numeric(ifelse(db_test[,outcome]=="malignant", 1, 0)),
  predict.fun = pred_wrapper,                          
  label = "XGBT",
  type = "classification"
)

####nnet explanier
set.seed(my_seed)
nnet_explainer<-DALEX::explain(
  model= nnet_caret,
  data = select(db_test,-outcome),
  y = as.numeric(ifelse(db_test[,outcome]=="malignant", 1, 0)),
  predict.fun = pred_wrapper,                          
  label = "Multiple Layer Perceptron",
  type = "classification"
)
 ```
 ### Global interpretation
 #### Variable importance
 ```
set.seed(my_seed)
xgbt_vi<-model_parts(xgbt_explainer)
plot(xgbt_vi, max_vars = 10,
     show_boxplots =F,
     subtitle="") + xlab("Features")+
  #scale_y_continuous(limits=c(0.015,0.065))
  theme_bw()+
  theme(axis.line = element_line(colour = "black"),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        panel.border = element_blank(),
        panel.background = element_blank()) +
  theme(legend.position = "none")+
  theme(axis.text.x=element_text(size = 12, color="black"),
        axis.text.y=element_text(size = 12, color="black"),
        axis.title.x=element_text(size = 14, color="black"),
        axis.title.y=element_text(size = 14, color="black"))
 ```
 #### Partial dependence plot
 ```
###PDP
set.seed(my_seed)
xgbt_pdp<-model_profile(explainer =xgbt_explainer,
                        type      = "partial",
                        variables  = c("Bare_Nuclei"))#,"Cell_Shape","Normal_Mitoses", "Epithelial_Size"))
xgbt_pdp$agr_profiles[[3]]<-factor(xgbt_pdp$agr_profiles[[3]], levels=c("1","2","3","4","5","6","7","8","9","10"))

plot(xgbt_pdp, subtitle="", geom = "aggregates")+
  theme_bw()+
  theme(axis.line = element_line(colour = "black"),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        panel.border = element_blank(),
        panel.background = element_blank()) +
  theme(legend.position = "none")+
  theme(axis.text.x=element_text(size = 12, color="black"),
        axis.text.y=element_text(size = 12, color="black"),
        axis.title.x=element_text(size = 14, color="black"),
        axis.title.y=element_text(size = 14, color="black"))+
  ylab("Average predictions")+
  xlab("Feature values")
 ```
 #### Accumulated local effect
 ```
###ALE
set.seed(my_seed)
xgbt_ale<-model_profile(explainer =xgbt_explainer,
                       type      = "accumulated",
                       variables  = c("Cell_Shape"))#,"Bland_Chromatin","Bare_Nuclei","Normal_Mitoses"))

xgbt_ale$agr_profiles[[3]]<-factor(xgbt_ale$agr_profiles[[3]], levels=c("1","2","3","4","5","6","7","8","9","10"))

plot(xgbt_ale, subtitle="")+
  theme_bw()+
  theme(axis.line = element_line(colour = "black"),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        panel.border = element_blank(),
        panel.background = element_blank()) +
  theme(legend.position = "none")+
  theme(axis.text.x=element_text(size = 12, color="black"),
        axis.text.y=element_text(size = 12, color="black"),
        axis.title.x=element_text(size = 14, color="black"),
        axis.title.y=element_text(size = 14, color="black"))+
  ylab("Average predictions")+
  xlab("Feature values")

 ```
 
  [back to top](#table-of-contents)
 ### Local interpretation
 ```
 ##select an individual from testing data for the demonstration
set.seed(my_seed)
obs = db_test[sample(1:length(db_test),1),] %>% as.data.frame()
obs= db_test[8,] %>% as.data.frame()
 ```
 #### Break Down plot
 ```
### Break down plot without considering variable interactions
set.seed(my_seed)
nnet_bd <- predict_parts(nnet_explainer, obs,  type = "break_down", keep_distributions=T, 
                         order = c("Bare_Nuclei","Cell_Shape","Normal_Nucleoli","Epithelial_Size"
                                   ,"Bland_Chromatin","Thickness","Cell_Size","Normal_Mitoses","Adhesion"))

nnet_bd_p<-plot(nnet_bd, digits  =2, plot_distributions=F, add_contributions=T,  title="", subtitle ="")+
  theme_bw()+
  theme(axis.line = element_line(colour = "black"),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        panel.border = element_blank(),
        panel.background = element_blank()) +
  theme(legend.position = "none")+
  theme(axis.text.x=element_text(size = 12, color="black"),
        axis.text.y=element_text(size = 12, color="black"),
        axis.title.x=element_text(size = 14, color="black"),
        axis.title.y=element_text(size = 14, color="black"))+
  xlab("Feature values")+
  ylab("Contribution")+
  facet_grid(labeller= list('Multiple Layer Perceptions'=""))

###Break down plot with variable interactions
set.seed(my_seed)
nnet_ibd <- predict_parts(nnet_explainer, obs,  type = "break_down_interactions", keep_distributions=T)

nnet_ibd_p<-plot(nnet_ibd, digits  =2, plot_distributions=F, add_contributions=T, title="", subtitle ="")+
  theme_bw()+
  theme(axis.line = element_line(colour = "black"),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        panel.border = element_blank(),
        panel.background = element_blank()) +
  theme(legend.position = "none")+
  theme(axis.text.x=element_text(size = 12, color="black"),
        axis.text.y=element_text(size = 12, color="black"),
        axis.title.x=element_text(size = 14, color="black"),
        axis.title.y=element_text(size = 14, color="black"))+
  facet_grid(labeller= list('Multiple Layer Perceptions'=""))
 
 
###Put the results from two versions of BDP side-by-side

plot_grid(nnet_bd_p, nnet_ibd_p, labels="AUTO")+
  draw_label("Contributions", x=0.5, y=  0, vjust=-0.5, angle= 0)+
  draw_label("Feature values", x=  0, y=0.5, vjust= 1.5, angle=90)
 ```
 #### Local surrogate
 ```
###surrogate model local lime

set.seed(my_seed)
nnet_lime_exp<-lime(db_development[,c(predictors, outcome)], nnet_caret)


set.seed(my_seed)
explanation <- lime::explain(obs, nnet_lime_exp, n_labels = 1, n_features = 9)

as.data.frame(explanation)
lime::plot_features(explanation)+  
  theme(axis.line = element_line(colour = "black"),
                                   panel.grid.major = element_blank(),
                                   panel.grid.minor = element_blank(),
                                   panel.border = element_blank(),
                                   panel.background = element_blank()) +
  theme(legend.position = "bottom")+
  theme(axis.text.x=element_text(size = 12, color="black"),
        axis.text.y=element_text(size = 12, color="black"),
        axis.title.x=element_text(size = 14, color="black"),
        axis.title.y=element_text(size = 14, color="black"))+
  geom_hline(yintercept = 0, linetype="dotted", color = "black", size=1)+
  ylab("Contribution")+
  xlab("Feature values")

exp_for_plot<-data.frame(Variables=paste(explanation$feature,explanation$feature_value, sep=' = '),
                         Weights = round(explanation$feature_weight,2),
                         Case =explanation$case,
                         Label= explanation$label,
                         Probability = round(explanation$label_prob,2),
                         R2= round(explanation$model_r2,2),
                         Intercept=round(explanation$model_intercept,2),
                         Direction = ifelse(explanation$feature_weight<0,"Contradicts","Supports")) %>% 
  arrange(abs(Weights))

exp_for_plot$Variables<- factor(exp_for_plot$Variables, levels=exp_for_plot$Variables)

subtitle= paste(
  paste("Label:", exp_for_plot$Label[[1]]),
  paste("Probability:", exp_for_plot$Probability[[1]]),
  paste("Model Intercept:",exp_for_plot$Intercept[[1]]),
  paste("Explanation Fit:", exp_for_plot$R2[[1]]),
  sep="\n"
)
ggplot(exp_for_plot, aes(x=Variables, y = Weights, fill=Direction)) +
  geom_col()+
  geom_text(aes(label = Weights),  hjust=ifelse(exp_for_plot$Weights>0,1,0))+
  coord_flip()+
  labs(subtitle=subtitle,
       fill="")+
  xlab("Feature values")+
  ylab("Contributions")+
  theme_classic()+
  theme(axis.text.x=element_text(size = 12, color="black"),
        axis.text.y=element_text(size = 12, color="black"),
        axis.title.x=element_text(size = 12, color="black"),
        axis.title.y=element_text(size = 12, color="black"),
        plot.subtitle = element_text(size=12, color="black"),
        legend.text  = element_text(size=12, color="black"),
        legend.position = "bottom")+
  geom_hline(yintercept = 0, linetype="dotted", color = "black", size=1)
 ```
 #### Shapely Additive Explanations
 ```
###SHAP
set.seed(my_seed)
nnet_shap <- DALEX::predict_parts(nnet_explainer, obs, type ="shap", B = 50)
plot(nnet_shap, show_boxplots =F)+  
  theme_bw()+
  theme(axis.line = element_line(colour = "black"),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        panel.border = element_blank(),
        panel.background = element_blank()) +
  theme(legend.position = "none")+
  theme(axis.text.x=element_text(size = 12, color="black"),
        axis.text.y=element_text(size = 12, color="black"),
        axis.title.x=element_text(size = 14, color="black"),
        axis.title.y=element_text(size = 14, color="black"))+
  geom_hline(yintercept = 0, linetype="dotted", color = "black", size=1)+
  xlab("Feature values")+
  ylab("Contribution")

exp_for_shap_df<- nnet_shap %>% 
  as.data.frame() %>%
  group_by(variable) %>% 
  summarise(contribution = mean(contribution), sign= max(sign)) %>%
  ungroup()%>%
  arrange(abs(contribution))

exp_for_shap_df$sign = ifelse(exp_for_shap_df$sign<0,"neg","pos")
 
exp_for_shap_df$variable<- factor(exp_for_shap_df$variable, levels=exp_for_shap_df$variable)

shap_plot_l<-ggplot(exp_for_shap_df, aes(x=variable, y = contribution, fill=sign)) +
  geom_col()+
  geom_text(aes(label = round(contribution,2)),  hjust=ifelse(exp_for_shap_df$contribution>0,1,0))+
  coord_flip()+
  xlab("Feature values")+
  ylab("Contributions")+
  theme_classic()+
  theme(axis.text.x=element_text(size = 12, color="black"),
        axis.text.y=element_text(size = 12, color="black"),
        axis.title.x=element_text(size = 12, color="black"),
        axis.title.y=element_text(size = 12, color="black"),
        plot.subtitle = element_text(size=12, color="black"),
        legend.text  = element_text(size=12, color="black"),
        legend.position = "none")+
  geom_hline(yintercept = 0, linetype="dotted", color = "black", size=1)

 ```
 #### Ceteris-Paribus plot
 ```
### Ceteris-paribus 
set.seed(my_seed)
nnet_cp <- predict_profile(nnet_explainer, obs)

plot(nnet_cp, variables=c("Cell_Shape"#, "Bland_Chromatin","Bare_Nuclei","Normal_Mitoses"
                          ), subtitle="",
     variable_type = "categorical", categorical_type = "bars")+
  theme_bw()+
  theme(axis.line = element_line(colour = "black"),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        panel.border = element_blank(),
        panel.background = element_blank()) +
  theme(legend.position = "none")+
  theme(axis.text.x=element_text(size = 12, color="black"),
        axis.text.y=element_text(size = 12, color="black"),
        axis.title.x=element_text(size = 14, color="black"),
        axis.title.y=element_text(size = 14, color="black"))+
  xlab("Feature values")+
  ylab("Predictions")
 ```
  [back to top](#table-of-contents)
  
  R session information
  =====================
  ```
> sessionInfo()
R version 4.2.1 (2022-06-23 ucrt)
Platform: x86_64-w64-mingw32/x64 (64-bit)
Running under: Windows 10 x64 (build 18362)

Matrix products: default

locale:
[1] LC_COLLATE=English_United States.utf8  LC_CTYPE=English_United States.utf8    LC_MONETARY=English_United States.utf8
[4] LC_NUMERIC=C                           LC_TIME=English_United States.utf8    

attached base packages:
[1] stats     graphics  grDevices utils     datasets  methods   base     

other attached packages:
 [1] localModel_0.5     rattle_5.5.1       bitops_1.0-7       tibble_3.1.8       rpart.plot_3.1.1   iml_0.11.1        
 [7] lime_0.5.3         DALEXtra_2.2.1     DALEX_2.4.2        RSNNS_0.4-14       Rcpp_1.0.9         rpart_4.1.16      
[13] xgboost_1.6.0.1    earth_5.3.1        plotmo_3.6.2       TeachingDemos_2.12 plotrix_3.8-2      Formula_1.2-4     
[19] glmnet_4.1-4       Matrix_1.5-1       recipes_1.0.1      tensorflow_2.9.0   caret_6.0-93       lattice_0.20-45   
[25] ggplot2_3.3.6      dplyr_1.0.10           

  ```
   [back to top](#table-of-contents)
