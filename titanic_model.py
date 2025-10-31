import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def load_data(train_path, test_path):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    return train, test


def preprocess_data(train, test):
    # Fehlende Werte im Trainingsdatensatz
    train['Age'].fillna(train['Age'].median(), inplace=True)
    train['Embarked'].fillna(train['Embarked'].mode()[0], inplace=True)
    train['Pclass'].fillna(train['Pclass'].median(), inplace=True)
    train['Cabin'].fillna('Unknown', inplace=True)
    train['Deck'] = train['Cabin'].str[0]
    
    # Fehlende Werte im Testdatensatz
    test['Age'].fillna(test['Age'].median(), inplace=True)
    test['Fare'].fillna(test['Fare'].median(), inplace=True)
    test['Pclass'].fillna(train['Pclass'].median(), inplace=True)
    test['Cabin'].fillna('Unknown', inplace=True)
    test['Deck'] = test['Cabin'].str[0]

    return train, test


def feature_engineering(train, test):
    for df in [train, test]:
        df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
        df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    return train, test


def encode_features(train, test):
    train = pd.get_dummies(train, columns=['Sex', 'Embarked'], drop_first=True)
    test = pd.get_dummies(test, columns=['Sex', 'Embarked'], drop_first=True)
    return train, test


def train_model(X_train, y_train):
    model = LogisticRegression(
        C=0.8,
        penalty='l2',
        solver='liblinear',
        max_iter=1000,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_val, y_val):
    y_pred = model.predict(X_val)
    print("Accuracy:", accuracy_score(y_val, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_val, y_pred))
    print(classification_report(y_val, y_pred))


def predict_and_save(model, test, features, filename="submission.csv"):
    X_test = test[features]
    test['Survived'] = model.predict(X_test)
    submission = test[['PassengerId', 'Survived']]
    submission.to_csv(filename, index=False)
    print(f"✅ Submission gespeichert als {filename}")


def main():
    # Pfade
    train_path = "/kaggle/input/titanic-machine-learning-from-disaster/train.csv"
    test_path = "/kaggle/input/titanic-machine-learning-from-disaster/test.csv"

    # Pipeline
    train, test = load_data(train_path, test_path)
    train, test = preprocess_data(train, test)
    train, test = feature_engineering(train, test)
    train, test = encode_features(train, test)

    # Features und Ziel definieren
    features = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'FamilySize', 'IsAlone',
                'Sex_male', 'Embarked_Q', 'Embarked_S']
    X = train[features]
    y = train['Survived']

    # Split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Training und Evaluation
    model = train_model(X_train, y_train)
    evaluate_model(model, X_val, y_val)

    # Testvorhersage speichern
    predict_and_save(model, test, features)


if __name__ == "__main__":
    main()
