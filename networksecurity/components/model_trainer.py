import os
import sys
import mlflow

from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging

from networksecurity.entity.artifact_entity import (
    DataTransformationArtifact,
    ModelTrainerArtifact,
    ClassificationMetricArtifact
)
from networksecurity.entity.config_entity import ModelTrainerConfig

from networksecurity.utils.main_utils.utils import save_object,load_object,load_numpy_arr_data,evaluate_models
from networksecurity.utils.ml_utils.model.estimator import NetworkModel
from networksecurity.utils.ml_utils.metric.classification_metric import get_classification_score

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)

class ModelTrainer:
    def __init__(self,model_trainer_config : ModelTrainerConfig , data_transformation_artifact:DataTransformationArtifact):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise NetworkSecurityException(e,sys)
        
    def track_mlflow(self,best_model,classificationmetric):
        with mlflow.start_run():
            f1_score = classificationmetric.f1_score
            precision_score = classificationmetric.precision_score
            recall_score = classificationmetric.recall_score
            
            mlflow.log_metric("f1_score",f1_score)
            mlflow.log_metric("precision",precision_score)
            mlflow.log_metric("recall",recall_score)
            mlflow.sklearn.log_model(best_model,"model")
        
    def train_model(self,X_train,y_train,X_test,y_test):
        try:
            models = {
                "Random Forest" : RandomForestClassifier(verbose=1),
                "Decision Tree" : DecisionTreeClassifier(),
                "Gradient Boosting" : GradientBoostingClassifier(verbose=1),
                "Logistic Regression" : LogisticRegression(verbose=1),
                "AdaBoost" : AdaBoostClassifier(),
            }

            param_grids = {
                "Random Forest": {
                    "n_estimators": [50, 100],
                    "max_depth": [None, 10, 20],
                    "min_samples_split": [2, 5]
                },

                "Decision Tree": {
                    "criterion": ["gini", "entropy"],
                    "max_depth": [None, 5, 10],
                    "min_samples_split": [2, 5]
                },

                "Gradient Boosting": {
                    "n_estimators": [50, 100],
                    "learning_rate": [0.01, 0.1],
                    "max_depth": [3, 5]
                },

                "Logistic Regression": {},

                "AdaBoost": {
                    "n_estimators": [50, 100],
                    "learning_rate": [0.01, 0.1, 1.0]
                }
            }

            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models,params=param_grids)

            # To get best model score from dict
            best_model_score = max(sorted(list(model_report.values())))

            ## To get best model name from doct
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]
            
            y_train_pred = best_model.predict(X_train)

            classification_train_metric = get_classification_score(y_train,y_train_pred)

            ## Track the train experiments with mlflow
            self.track_mlflow(best_model,classification_train_metric)

            y_test_pred = best_model.predict(X_test)

            classification_test_metric = get_classification_score(y_test,y_test_pred)

            ## Track the text experiments with mlflow
            self.track_mlflow(best_model,classification_test_metric)

            preprocessor = load_object(self.data_transformation_artifact.transformed_object_file_path)

            model_dir_path = os.path.dirname(self.model_trainer_config.model_trainer_dir)
            os.makedirs(model_dir_path,exist_ok=True)

            Network_Model = NetworkModel(preprocessor=preprocessor,model=best_model)
            save_object(self.model_trainer_config.trained_model_file_path,obj=Network_Model)

            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                train_metric_artifact=classification_train_metric,
                test_metric_artifact=classification_test_metric
            )
            logging.info(f"Model trainer artifact: {model_trainer_artifact}")
            logging.info(f"The Model name is : {best_model_name}")

            return model_trainer_artifact
        
        except Exception as e:
            raise NetworkSecurityException(e,sys)
        
    def initiate_model_trainer(self)->ModelTrainerArtifact:
        try:
            train_file_path = self.data_transformation_artifact.transformed_train_file_path
            test_file_path = self.data_transformation_artifact.transformed_test_file_path

            ##loading training array and testing array
            train_arr = load_numpy_arr_data(train_file_path)
            test_arr = load_numpy_arr_data(test_file_path)

            X_train,y_train,X_test,y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1],
            )

            model = self.train_model(X_train,y_train,X_test,y_test)

            return model

        except Exception as e:
            raise NetworkSecurityException(e,sys)
