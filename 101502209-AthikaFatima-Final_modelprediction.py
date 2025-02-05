import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

def data_loader(file_path):
    """Load data from a local CSV file."""
    df = pd.read_csv(file_path)
    return df

def data_cleaning(df):
    """Clean the dataset."""
    # Drop rows with any NaN values
    df = df.dropna()
    # Drop the unique identifier column as it's not useful for prediction
    df = df.drop(columns=['user_id'])
    return df

def feature_selection(df):
    """Select the best features for the model."""
    X = df.iloc[:, :-1]
    y = df['great_customer_class']
    
    # Identify categorical and numeric columns
    categorical_cols = X.select_dtypes(include=['object']).columns
    numeric_cols = X.select_dtypes(include=['number']).columns
    
    # Preprocessing: One-hot encode categorical columns and scale numeric columns
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_cols),
            ('cat', OneHotEncoder(), categorical_cols)
        ])
    
    # Apply feature selection
    X_preprocessed = preprocessor.fit_transform(X)
    selector = SelectKBest(f_classif, k=10)
    X_new = selector.fit_transform(X_preprocessed, y)
    
    # Get selected feature names
    numeric_feature_mask = selector.get_support()[:len(numeric_cols)]
    categorical_feature_mask = selector.get_support()[len(numeric_cols):]
    selected_numeric_features = numeric_cols[numeric_feature_mask]
    selected_categorical_features = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols)[categorical_feature_mask]
    selected_features = np.concatenate([selected_numeric_features, selected_categorical_features])
    
    return X_new, y, selected_features

def model_building(X, y):
    """Build and evaluate different models."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    models = {
        'Random Forest': RandomForestClassifier(random_state=42),
        'SVM': SVC(random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42),
        'Naive Bayes': GaussianNB(),
        'KNN': KNeighborsClassifier()
    }
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results[name] = accuracy_score(y_test, y_pred)
    return results

def ensemble_learning(X, y):
    """Apply ensemble learning to boost model accuracy."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    estimators = [
        ('rf', RandomForestClassifier(random_state=42)),
        ('svc', SVC(random_state=42, probability=True)),
        ('lr', LogisticRegression(random_state=42))
    ]
    ensemble = VotingClassifier(estimators, voting='soft')
    ensemble.fit(X_train, y_train)
    y_pred = ensemble.predict(X_test)
    return accuracy_score(y_test, y_pred)

def main(file_path):
    """Main function to execute the script."""
    df = data_loader(file_path)
    df = data_cleaning(df)
    X, y, selected_features = feature_selection(df)
    print("Selected Features:", selected_features)
    results = model_building(X, y)
    for model, accuracy in results.items():
        print(f"{model} Accuracy: {accuracy:.2f}")
    ensemble_accuracy = ensemble_learning(X, y)
    print(f"Ensemble Accuracy: {ensemble_accuracy:.2f}")

if __name__ == "__main__":
    # Replace 'great_customers.csv' with the path to your local CSV file
    file_path = 'great_customers.csv'
    main(file_path)