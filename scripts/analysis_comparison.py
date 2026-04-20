# -*- coding: utf-8 -*-
"""
Created on Mon Sep 29 09:26:24 2025

This is a script to compare the results from abstract model and 
abstract-reference model.

@author: Zhiyi Jin
"""

#%% Packages
from gensim.corpora import MmCorpus
from gensim.models import LdaModel
import numpy as np
import pandas as pd
from pathlib import Path

#%% 14 x 17 crosstable across TWO DIFFERENT CORPORA for the SAME DOC SET
# --- Config ---
path_model_abs      = "abs_model/model_12.model"       
path_model_absrefs  = "abs_ref model/model_15.model"    
path_corpus_abs     = "abs_model/ab_corpus.mm"
path_corpus_absrefs = "abs_ref model/combine_corpus.mm"

out_dir = Path("lda_compare_14x17_outputs")
out_dir.mkdir(exist_ok=True)

# --- Load ---
m_abs = LdaModel.load(path_model_abs)
m_absrefs = LdaModel.load(path_model_absrefs)

c_abs = MmCorpus(path_corpus_abs)
c_absrefs = MmCorpus(path_corpus_absrefs)

N_abs     = int(len(c_abs))
N_absrefs = int(len(c_absrefs))
print("Sizes seen at runtime:", N_abs, N_absrefs)

# --- Convert each doc into a dense topic vector ---
def get_topic_matrix(model, corpus):
    """
    Returns an (N_docs x K_topics) dense matrix of topic probabilities.
    """
    rows = []
    for bow in corpus:
        dist = model.get_document_topics(bow, minimum_probability=0.0)
        # dist is [(topic_id, prob), ...], sort by topic_id
        probs = [p for _, p in sorted(dist, key=lambda x: x[0])]
        rows.append(probs)
    return np.array(rows)

doc_topic_abs     = get_topic_matrix(m_abs, c_abs)         
doc_topic_absrefs = get_topic_matrix(m_absrefs, c_absrefs) 

# --- Get dominant topic with np.argmax ---
assign_abs     = np.argmax(doc_topic_abs, axis=1)     
assign_absrefs = np.argmax(doc_topic_absrefs, axis=1) 

# --- Build 14 x 17 crosstable ---
rows = pd.Categorical(assign_abs, categories=range(14), ordered=True)
cols = pd.Categorical(assign_absrefs, categories=range(17), ordered=True)
crosstab_counts = pd.crosstab(rows, cols, dropna=False)

crosstab_counts.index   = [f"Topic_{i+1}" for i in range(14)]
crosstab_counts.columns = [f"Topic_{j+1}" for j in range(17)]

print("\nCounts (14 x 17):")
print(crosstab_counts)

# --- Percentages ---
row_pct = crosstab_counts.div(crosstab_counts.sum(axis=1), axis=0).fillna(0) * 100
col_pct = crosstab_counts.div(crosstab_counts.sum(axis=0), axis=1).fillna(0) * 100

# --- Save ---
crosstab_counts.to_csv(out_dir / "crosstable_14x17_counts.csv")
row_pct.round(2).to_csv(out_dir / "crosstable_14x17_rowpct.csv")
col_pct.round(2).to_csv(out_dir / "crosstable_14x17_colpct.csv")

# --- Better visulization ---
def top_words(model, n=2):
    """
    Return dict {topic_id: 'word1, word2'} for each topic.
    """
    topic_terms = {}
    for t in range(model.num_topics):
        words = [w for w, p in model.show_topic(t, topn=n)]
        topic_terms[t] = ", ".join(words)
    return topic_terms

top_abs     = top_words(m_abs, n=2)     # for 15-topic model
top_absrefs = top_words(m_absrefs, n=2) # for 17-topic model

crosstab_counts.index = [f"T{i+1} ({top_abs[i]})" for i in range(14)]
crosstab_counts.columns = [f"T{j+1} ({top_absrefs[j]})" for j in range(17)]

crosstab_counts.to_csv(out_dir / "crosstable_14x17_counts_withwords.csv")
