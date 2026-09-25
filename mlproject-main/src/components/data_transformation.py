import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


# Find the main project folder.
BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)


@dataclass
class DataTransformationConfig:
    # This file stores the fitted preprocessing pipeline.
    preprocessor_obj_file_path: str = os.path.join(
        BASE_DIR, "artifacts", "preprocessor.pkl"
    )


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        """
        Create the preprocessing pipeline.

        Numerical columns are filled and scaled.
        Categorical columns are filled, one-hot encoded and scaled.
        """
        try:
            numerical_columns = [
                "writing_score",
                "reading_score"
            ]

            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

            # Pipeline for numerical data.
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )

            # Pipeline for categorical data.
            cat_pipeline = Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(strategy="most_frequent")
                    ),
                    ("one_hot_encoder", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )

            logging.info(
                f"Categorical columns: {categorical_columns}"
            )
            logging.info(
                f"Numerical columns: {numerical_columns}"
            )

            # Apply the correct pipeline to each type of column.
            preprocessor = ColumnTransformer(
                [
                    (
                        "num_pipeline",
                        num_pipeline,
                        numerical_columns
                    ),
                    (
                        "cat_pipeline",
                        cat_pipeline,
                        categorical_columns
                    )
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            # Read the raw train and test files created by data ingestion.
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")
            logging.info("Obtaining preprocessing object")

            preprocessing_obj = self.get_data_transformer_object()

            # math_score is the value that we want the model to predict.
            target_column_name = "math_score"

            # Separate input features from the target value.
            input_feature_train_df = train_df.drop(
                columns=[target_column_name],
                axis=1
            )
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(
                columns=[target_column_name],
                axis=1
            )
            target_feature_test_df = test_df[target_column_name]

            logging.info(
                "Applying preprocessing object on training "
                "and testing data."
            )

            # Learn the transformations from training data.
            input_feature_train_arr = preprocessing_obj.fit_transform(
                input_feature_train_df
            )

            # Apply the same transformations to test data.
            input_feature_test_arr = preprocessing_obj.transform(
                input_feature_test_df
            )

            # Add the target column back to the transformed features.
            train_arr = np.c_[
                input_feature_train_arr,
                np.array(target_feature_train_df)
            ]

            test_arr = np.c_[
                input_feature_test_arr,
                np.array(target_feature_test_df)
            ]

            # Save the fitted preprocessing pipeline.
            # The transformed arrays themselves stay in memory and
            # are passed directly to ModelTrainer.
            logging.info("Saving preprocessing object")

            save_object(
                file_path=(
                    self.data_transformation_config
                    .preprocessor_obj_file_path
                ),
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            raise CustomException(e, sys)
