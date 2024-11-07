import os
import sys
from src.exception import CustomException
from src.logger import logging
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
import pandas as pd
from src.Components.Data_transformation import DataTransformationConfig
from src.Components.Data_transformation import DataTransformation
from src.Components.model_trainer import ModelTrainerConfig
from src.Components.model_trainer import ModelTrainer


@dataclass
class DataIngestionConfig:
    train_data_path: str= os.path.join('artifacts','train.csv')
    test_data_path: str= os.path.join('artifacts','test.csv')
    raw_data_path: str = os.path.join('artifacts','raw.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
    
    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method of component")
        try:
            df = pd.read_csv('notebook\data\stud.csv')
            logging.info('Read the dataset as dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok = True)
            df.to_csv(self.ingestion_config.raw_data_path, index = False, header = True)

            logging.info('Train Test split initiated')
            train_set, test_set = train_test_split(df,test_size = 0.2,random_state = 42)

            train_set.to_csv(self.ingestion_config.train_data_path, index = False, header = True)
            test_set.to_csv(self.ingestion_config.test_data_path, index = False, header = True)

            logging.info("Ingestion of the data is completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )


        except Exception as e:
            raise CustomException(e,sys)

if __name__ == '__main__':
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    obj_trans = DataTransformation()
    train_arr, test_arr,_ = obj_trans.initiate_data_transformation(train_data, test_data)

    obj_train = ModelTrainer()
    best_name,best_score,total_report = obj_train.initiate_model_trainer(train_arr,test_arr)
    print(f'The best model name is {best_name}')
    print(f'The best model score is {best_score}')
    print(total_report)
