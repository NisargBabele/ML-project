# End-to-End Student Performance ML Project

This project predicts a student's **math score** using demographic information,
lunch/test-preparation information, reading score, and writing score.

## Project flow

```text
stud.csv
   ↓
Data Ingestion
   ↓
artifacts/data.csv
artifacts/train.csv
artifacts/test.csv
   ↓
Data Transformation
   ↓
artifacts/preprocessor.pkl
   ↓
Model Training
   ↓
artifacts/model.pkl
   ↓
Flask Prediction App
```

## Important point about transformed data

The transformed train and test arrays are **not saved as CSV files** in this
project. They are kept in memory and passed directly from
`DataTransformation` to `ModelTrainer`.

The preprocessing pipeline is saved as:

```text
artifacts/preprocessor.pkl
```

The final trained model is saved as:

```text
artifacts/model.pkl
```

## Run the project

From the project root:

```bash
pip install -r requirements.txt
```

Then train the project:

```bash
python -m src.components.data_ingestion
```

This creates/updates the files in `artifacts/`.

Then start Flask:

```bash
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

## Main folders

- `notebook/data/` - original dataset
- `src/components/` - ingestion, transformation and training
- `src/pipeline/` - training/prediction pipeline code
- `artifacts/` - generated model files
- `templates/` - Flask HTML pages

The code comments are intentionally written in simple English to make the
ML pipeline easier to follow.
