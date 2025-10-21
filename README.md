# Sentiment Analysis & Predictive Score Notebook
 
This repository contains the Jupyter/Colab notebook `sentimentanalysis_predictivescore.ipynb`, which performs a simple end-to-end sentiment classification pipeline on a CSV dataset (commonly from Kaggle). The notebook loads the CSV, inspects columns, visualizes sentiment distribution, trains a baseline text classifier (CountVectorizer + LogisticRegression), evaluates it, and saves predicted labels to `predicted_sentiments.csv`.
 
---
 
## Notebook overview
 
The notebook performs the following steps:
 
1. Prompt the user to upload a CSV file (Colab `files.upload()`).
2. Load the CSV, attempting `utf-8` and falling back to `latin1` encoding.
3. Detect the text column and the sentiment label column (the example dataset used a a label column named `neutral` and a long column containing article text).
4. Display a bar chart of sentiment distribution (Plotly).
5. Convert text to numeric features using `CountVectorizer` (max_features=1000).
6. Train a `LogisticRegression` classifier.
7. Evaluate with accuracy and a classification report (precision, recall, f1).
8. Save a CSV of actual vs predicted labels (`predicted_sentiments.csv`) and trigger a download in Colab.
 
---
 
## Expected input CSV
 
The notebook expects a CSV file with at least:
- A sentiment label column containing categories such as `positive`, `negative`, and `neutral`.
  - In the provided notebook this column is named `neutral`. You can rename your label column to `neutral` or adjust the notebook variable `y = df['neutral']`.
- A text column containing the documents to classify (news articles, sentences, etc.).
  - The notebook attempts to auto-detect the text column; it looks for columns containing words such as "According" or "company" (based on the example dataset). It's recommended to rename your text column to a simple name like `text` for clarity.
 
Recommended column names (easy option):
- `text` — the input text
- `sentiment` — the label (`positive` / `neutral` / `negative`)
 
If you rename your columns to `text` and `sentiment`, update these lines in the notebook:
```python
X_text = df['text'].astype(str)
y = df['sentiment']
```
 
---
 
## How to run
 
### In Google Colab (recommended)
1. Open the notebook in Colab.
2. Run the first cell to install and import libraries (if necessary) and the cell that prompts file upload.
3. Upload your CSV using the upload widget.
4. Run the subsequent cells in order.
   - The notebook will try `utf-8`, then `latin1` encoding if needed.
   - It will train the model and show interactive Plotly charts.
5. The predictions are saved as `predicted_sentiments.csv` and the notebook triggers a download.
 
### Locally (Jupyter / VS Code)
1. Install dependencies (see next section).
2. Replace the `files.upload()` usage in the notebook with a direct file path, for example:
```python
file_name = "path/to/your-file.csv"
df = pd.read_csv(file_name, encoding='utf-8')  # or latin1 if needed
```
3. Run the notebook cells locally.
 
---
 
## Dependencies
 
The notebook uses common Python packages:
 
- Python 3.x
- pandas
- numpy
- scikit-learn
- plotly
- matplotlib
- (for Colab) google.colab
 
Install with pip, e.g.:
```bash
pip install pandas numpy scikit-learn plotly matplotlib
```
 
If running in Colab you do not need to install most packages as they are preinstalled.
 
---
 
## Outputs
 
- Interactive Plotly bar charts:
  - Sentiment distribution
  - Predicted vs Actual sentiment comparison
- Console output:
  - Data load confirmation
  - Model accuracy and classification report
- File:
  - `predicted_sentiments.csv` — contains columns `Actual` and `Predicted` for the test split.
