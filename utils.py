import pandas as pd
import pickle

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def load_data():
    return pd.read_csv("titanic.csv")

def get_summary(df):
    """Returns basic summary statistics and metadata about the DataFrame."""
    return {
        "shape": df.shape,
        "data_types": df.dtypes,
        "missing_values": df.isnull().sum(),
        "unique_counts": df.nunique(),
        "summary_stats": df.describe()
    }

def preprocess_data(df):
    df = df.copy()

    # Fill missing values
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

    # Drop unnecessary text columns
    df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

    # Encode 'Sex': male -> 0, female -> 1
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

    # One-hot encode 'Embarked'
    df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

    # Separate features and target
    X = df.drop('Survived', axis=1)
    y = df['Survived']

    # Scale numerical features only
    scaler = MinMaxScaler()
    cols_to_scale = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
    X[cols_to_scale] = scaler.fit_transform(X[cols_to_scale])

    return X, y, scaler

def train_and_save_model(X, y):
    model = LogisticRegression()
    model.fit(X, y)

    # Save the model to a file
    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)

    return model

def load_model():
    """Loads a trained model from model.pkl."""
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

def evaluate_model(model, X, y):
    """Evaluates the model and returns accuracy, confusion matrix, and classification report."""
    predictions = model.predict(X)

    acc = accuracy_score(y, predictions)
    cm = confusion_matrix(y, predictions)
    report = classification_report(y, predictions)

    return acc, cm, report


def main():
    df = load_data()
    print(preprocess_data(df))
main()



