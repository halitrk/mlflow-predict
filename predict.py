import mlflow
import pandas as pd
import numpy as np
import argparse
import warnings

def load_model(model_path):
    # Load model as a PyFuncModel.
    return mlflow.pyfunc.load_model(model_path)

def make_prediction(model, data_path, limit):
    # Predict on a Pandas DataFrame.
    data = pd.read_csv(data_path)
    predictions = model.predict(pd.DataFrame(data.head(limit)))
    return predictions

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='MLflow Model Prediction')
    parser.add_argument('--model-path', type=str, default='./model',
                        help='Path to the MLflow model')
    parser.add_argument('--data-path', type=str, default='pred_data.csv',
                        help='Path to the input data file')
    parser.add_argument('--limit', type=int, default=100 )

    args = parser.parse_args()

    # Load the model
    loaded_model = load_model(args.model_path)

    # Make prediction
    predictions = make_prediction(loaded_model, args.data_path, args.limit)

    # Print predictions
    print(predictions)
