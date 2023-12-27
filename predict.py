import mlflow
logged_model = 'runs:/57511fc38ba849cea62f6b67a7b53143/model'
logged_model='/Users/halitfurkanturk/Documents/zpersonal_repo/MLflow/mlruns/949594437793911251/57511fc38ba849cea62f6b67a7b53143/artifacts/model'

# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)

# Predict on a Pandas DataFrame.
import pandas as pd
data=pd.read_csv('pred_data.csv')
predictions =loaded_model.predict(pd.DataFrame(data))
print(predictions)