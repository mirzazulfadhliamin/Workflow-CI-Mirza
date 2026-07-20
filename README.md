# Workflow-CI-Mirza

CI/CD workflow untuk automated model training.

## Deskripsi

Repository ini berisi GitHub Actions workflow untuk:
1. Automatisasi preprocessing data (dari Eksperimen_SML)
2. Automatisasi model training
3. Upload artifacts (model, metrics, confusion matrix)

## Workflow

**train-model.yml**: Training model otomatis saat push ke main

## Hasil

Setiap workflow run menghasilkan artifacts:
- `random_forest_model.pkl` - Model yang sudah dilatih
- `metrics.json` - Accuracy dan F1 Score
- `feature_importance.json` - Feature importance
- `confusion_matrix.png` - Confusion matrix
- `feature_importance.png` - Feature importance plot

## Cara Menjalankan

1. Push ke branch main/master
2. Atau run manual dari tab Actions

## Author

Mirza Zulfadhli Amin

