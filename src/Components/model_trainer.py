import os
import sys
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass


from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor


from src.utils import save_object,evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl')

class ModelTrainer:

    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info('Splitting train and test data ')

            X_train,y_train,X_test,y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]

            )

            models = {
                "RandomForest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose = False),
                "AdaBoost Regressor": AdaBoostRegressor()
            }

            model_report = dict = evaluate_models(X_train,X_test,y_train,y_test,models)

            best_model_name = max(model_report,key = model_report.get)
            best_model_score = model_report[best_model_name]
            if best_model_score < 0.6:
                raise CustomException("Best model not found")
            logging.info('Found best model returning the value')

            return best_model_name,best_model_score, model_report
        except Exception as e:
            raise CustomException(e,sys)

