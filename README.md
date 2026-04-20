# Paper replication

This repository contains code and data to reproduce the results of this paper:

Jin, Z.; van Duijn, M.A.; Steglich, C. An Exploratory Review of Regional Perspectives on Social Capital and Occupational Studies. Social Science. 2026, 15, 221, https://doi.org/10.3390/socsci15040221. 

## How to run

Run the scripts from the repository root in the following order:

1. `analysis_model.py`
	- Fits the topic model for both the abstract and abstract-reference datasets.
	- This step creates the model outputs used by the next scripts.

2. `analysis_result.py`
	- Uses the fitted abstract-reference model to generate the final results.
	- The outputs are saved in the `results/` folder.

3. `analysis_comparison.py`
	- Compares the optimal results from the abstract and abstract-reference models.
	- Use this script after the first two steps are complete.

Example:

```bash
python scripts/analysis_model.py
python scripts/analysis_result.py
python scripts/analysis_comparison.py
```

Make sure the input data in `data/` is available before running the scripts.

## Reproducibility

The code was implemented in Spyder 6 with this Python environment:

- Python 3.11.12, packaged by conda-forge
- IPython 8.36.0

Environment details:

```text
Python 3.11.12 | packaged by conda-forge | (main, Apr 10 2025, 22:09:00) [MSC v.1943 64 bit (AMD64)]
IPython 8.36.0 -- An enhanced Interactive Python. Type '?' for help.
```

To reproduce the setup, create a conda environment with Python 3.11.12 and install the packages used by the scripts:

```bash
conda create -n thesis-replication python=3.11.12
conda activate thesis-replication
python -m pip install pandas numpy nltk scikit-learn wordcloud matplotlib gensim pyLDAvis python-docx requests tqdm scipy
```

The scripts also download several NLTK resources at runtime (`punkt`, `stopwords`, `wordnet`, `punkt_tab`, and `averaged_perceptron_tagger_eng`). 

If you prefer to work in Spyder, start Spyder from the same activated conda environment so it uses the same interpreter and installed packages.
