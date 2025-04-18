import numpy as np
import mlflow
from mlflow.models import infer_signature

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

#print("Hello World");

from sklearn.datasets import load_iris

# Carregando o dataset iris
iris = load_iris()
X = iris.data
y = iris.target

# Dividindo os dados em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Define the model hyperparameters
params = {
    "solver": "lbfgs",
    "max_iter": 1000,
    "multi_class": "auto",
    "random_state": 8888,
}

# Criando o modelo
model = LogisticRegression(**params)

# Treinando o modelo
model.fit(X_train, y_train)

# Fazendo previsões no conjunto de teste
y_pred = model.predict(X_test)

# Calculando as métricas de avaliação
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# Imprimindo os resultados
print(f"Acurácia: {accuracy:.4f}")
print(f"Precisão: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")

mlflow.set_tracking_uri("http://localhost:8080")
mlflow.set_experiment("Tutorial Experiment")

with mlflow.start_run():
    mlflow.log_params(params)
    mlflow.log_metrics({
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    })

    mlflow.set_tag("Tutorial", "Logistic Regression for Iris Dataset")

    signature = infer_signature(X_test)

    model_info = mlflow.sklearn.log_model(
        sk_model=model,
        registered_model_name="logistic-regression-iris",
        signature=signature,
        input_example=X_train,
        artifact_path="iris-model-logistic-regression"
    )

    print(f"Modelo registrado com sucesso! ID: {model_info.model_uri}")
    
# Load the model back for predictions as a generic Python Function model
#loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)

loaded_model = mlflow.pyfunc.load_model(f'runs:/eab45198d7254d379a2b028d6dbecfdb/iris-model-logistic-regression')

predictions = loaded_model.predict(X_test)

iris_feature_names = load_iris().feature_names
import pandas as pd

result = pd.DataFrame(X_test, columns=iris_feature_names)
result["actual_class"] = y_test
result["predicted_class"] = predictions

print("\nPrimeiras 4 linhas do resultado:")
print(result[:4])
