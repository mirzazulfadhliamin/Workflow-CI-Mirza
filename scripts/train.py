import pandas as pd
import numpy as np
import pickle
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

def main():
    print("=" * 60)
    print("TRAINING MODEL - GITHUB ACTIONS")
    print("=" * 60)
    
    print("\nLoading data...")
    
    data_path = '../Eksperimen_SML_Mirza_Zulfadhli_Amin/preprocessing/customer_churn_preprocessing.csv'
    
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        print(f"Data loaded from: {data_path}")
    else:
        print("File not found. Creating sample data...")
        np.random.seed(42)
        n_samples = 100
        df = pd.DataFrame({
            'tenure': np.random.randint(1, 72, n_samples),
            'monthly_charges': np.random.uniform(20, 150, n_samples),
            'total_charges': np.random.uniform(100, 8000, n_samples),
            'contract_type_One_year': np.random.randint(0, 2, n_samples),
            'contract_type_Two_year': np.random.randint(0, 2, n_samples),
            'payment_method_Credit_card': np.random.randint(0, 2, n_samples),
            'payment_method_Electronic_check': np.random.randint(0, 2, n_samples),
            'Churn': np.random.randint(0, 2, n_samples)
        })
        print(f"Sample data created: {len(df)} rows")
    
    print(f"Data shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    X = df.drop(columns=['Churn'])
    y = df['Churn']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nData split:")
    print(f"   Train: {len(X_train)} samples")
    print(f"   Test : {len(X_test)} samples")
    print(f"   Target distribution:")
    print(f"   Train - 0: {sum(y_train==0)}, 1: {sum(y_train==1)}")
    print(f"   Test  - 0: {sum(y_test==0)}, 1: {sum(y_test==1)}")
    
    print("\nTraining Random Forest Classifier...")
    
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    print("Model training complete!")
    
    print("\nEvaluating model...")
    
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='binary')
    
    print(f"   Accuracy: {accuracy:.4f}")
    print(f"   F1 Score : {f1:.4f}")
    
    print("\nSaving artifacts...")
    
    with open('random_forest_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("   Model saved: random_forest_model.pkl")
    
    metrics = {
        "accuracy": float(accuracy),
        "f1_score": float(f1),
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "model_type": "RandomForestClassifier",
        "n_estimators": 100,
        "max_depth": 10,
        "timestamp": str(pd.Timestamp.now())
    }
    
    with open('metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
    print("   Metrics saved: metrics.json")
    
    feature_importance = dict(zip(X.columns, model.feature_importances_.tolist()))
    
    with open('feature_importance.json', 'w') as f:
        json.dump(feature_importance, f, indent=4)
    print("   Feature importance saved: feature_importance.json")
    
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix - Random Forest')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300)
    print("   Confusion Matrix saved: confusion_matrix.png")
    plt.close()
    
    plt.figure(figsize=(10, 6))
    importance_df = pd.DataFrame({
        'feature': list(feature_importance.keys()),
        'importance': list(feature_importance.values())
    }).sort_values('importance', ascending=True)
    
    plt.barh(importance_df['feature'], importance_df['importance'])
    plt.title('Feature Importance')
    plt.xlabel('Importance Score')
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300)
    print("   Feature importance plot saved: feature_importance.png")
    plt.close()
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print(f"Accuracy : {accuracy:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"Artifacts saved: 5 files")
    print("=" * 60)
    
    files = os.listdir('.')
    print("\nGenerated files:")
    for f in files:
        if f.endswith('.pkl') or f.endswith('.json') or f.endswith('.png'):
            size = os.path.getsize(f)
            print(f"   - {f} ({size} bytes)")

if __name__ == "__main__":
    main()

