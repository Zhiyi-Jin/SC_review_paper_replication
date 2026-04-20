# -*- coding: utf-8 -*-
"""
Created on Tue Aug 12 11:52:54 2025

This is a script to check the result from abstract-reference model.

@author: Zhiyi Jin
"""

#%% Packages
from gensim.models import LdaModel
from gensim.corpora import MmCorpus, Dictionary

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.cm import get_cmap

import pyLDAvis.gensim_models as gensimvis
import pyLDAvis

import glob, re
from pathlib import Path

#%% Load the 17-topic result
# Load the saved LDA model
model_17 = LdaModel.load("abs_ref model/model_15.model")

# Check the number of topics
print(f"Number of topics: {model_17.num_topics}")

# Get the original corpus
corpus = MmCorpus("abs_ref model/combine_corpus.mm")

#%% Check top words for each topic
def plot_top_words_gensim(lda_model, n_top_words=20):
    for topic_idx in range(lda_model.num_topics):
        top_words = lda_model.show_topic(topic_idx, topn=n_top_words)
        words, weights = zip(*top_words)

        plt.figure(figsize=(8, 4))
        plt.barh(words[::-1], weights[::-1])
        plt.title(f"Topic #{topic_idx}")
        plt.xlabel("Weight")
        plt.tight_layout()
        plt.show()

plot_top_words_gensim(model_17)

# better plots
import textwrap

def plot_selected_topics_gensim(
    lda_model,
    exclude=(8, 11, 14),
    n_top_words=15,
    rows=7,
    cols=2,
    wrap_width=16,          # helps long tokens fit
    width_per_col=3.6,
    height_per_row=2.3,
    font_size=8,
    title_size=10,
    dpi=300,
    savepath=None
):
    # topics to plot (keep original order)
    topic_ids = [k for k in range(lda_model.num_topics) if k not in set(exclude)]
    assert len(topic_ids) == rows * cols, (
        f"Grid {rows}×{cols} expects {rows*cols} topics, but got {len(topic_ids)}."
    )

    # collect data + global max for shared x-axis
    topics = []
    max_w = 0.0
    for k in topic_ids:
        top = lda_model.show_topic(k, topn=n_top_words)
        # ensure descending by weight; show_topic usually already is
        top = sorted(top, key=lambda x: x[1], reverse=True)
        words, weights = zip(*top)
        topics.append((k, list(words), list(weights)))
        max_w = max(max_w, max(weights))

    def wrap_label(s):
        return textwrap.fill(s.replace("_", " "), width=wrap_width)

    fig, axes = plt.subplots(
        rows, cols,
        figsize=(cols * width_per_col, rows * height_per_row),
        sharex=True,
        constrained_layout=True
    )
    axes = axes.ravel()

    for i, (topic_id, words, weights) in enumerate(topics):
        ax = axes[i]
        y = np.arange(len(words))
        ax.barh(y, weights)

        # Show labels on BOTH columns
        ax.set_yticks(y)
        ax.set_yticklabels([wrap_label(w) for w in words], fontsize=font_size)

        # Highest weight at the top
        ax.invert_yaxis()

        ax.set_xlim(0, max_w * 1.05)
        ax.set_title(f"Topic {topic_id}", fontsize=title_size, pad=3)
        ax.tick_params(axis="x", labelsize=font_size)
        ax.grid(axis="x", linestyle=":", linewidth=0.5, alpha=0.5)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    # turn off any unused axes (shouldn't happen here)
    for j in range(len(topics), len(axes)):
        axes[j].axis("off")

    fig.supxlabel("Weight", fontsize=title_size)
    if savepath:
        fig.savefig(savepath, dpi=dpi, bbox_inches="tight")

    return fig

# Example usage:
fig = plot_selected_topics_gensim(model_17, exclude=(8,11,14), n_top_words=15,
                                   rows=7, cols=4, savepath="topics_selected.pdf", dpi=600)
plt.show()


#%% Dist of topic probability over document
K = model_17.num_topics
n_docs = len(corpus)

# Build a dense doc-topic matrix (n_docs x K) with full distributions
doc_topic = np.zeros((n_docs, K), dtype=float)
for d, dist in enumerate(model_17.get_document_topics(corpus, 
                                                      minimum_probability=0)):
    for tid, p in dist:
        doc_topic[d, tid] = p

# Make 17 histograms: distribution of each topic's probability across documents
for t in range(K):
    plt.figure()
    plt.hist(doc_topic[:, t], bins=20)  
    plt.title(f"Topic {t}: probability distribution across documents")
    plt.xlabel("Topic probability in document")
    plt.ylabel("Number of documents")
    plt.tight_layout()
    plt.savefig(f"topic_{t}_distribution.png", dpi=150)

# Make one plot with 17 subplots
n_cols = 4
n_rows = int(np.ceil(K / n_cols))
bins = np.linspace(0, 1, 31)  # same bins for comparability

fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 10))
axes = axes.ravel()

for t in range(K):
    ax = axes[t]
    ax.hist(doc_topic[:, t], bins=bins)
    ax.set_title(f"Topic {t}")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Topic probability")
    ax.set_ylabel("Doc count")

# Hide any unused subplots (since 17 != 4*5)
for j in range(K, n_rows * n_cols):
    fig.delaxes(axes[j])

fig.suptitle("Topic probability distributions across documents", y=1.02, fontsize=14)
fig.tight_layout()
plt.show()

#%% Compute the metrics
# ---- Compute metrics ----
dominant_topic = np.argmax(doc_topic, axis=1)

def get_margin(row):
    sorted_probs = np.sort(row)[::-1]
    return sorted_probs[0] - sorted_probs[1]

margin = np.apply_along_axis(get_margin, 1, doc_topic)

def get_entropy(row):
    p = row[row > 0]
    return -(p * np.log(p)).sum()

entropy = np.apply_along_axis(get_entropy, 1, doc_topic)

# Top topic probability
top_topic_prob = doc_topic[np.arange(n_docs), dominant_topic]

# ---- Create DataFrame ----
df_results = pd.DataFrame({
    "doc_id": range(n_docs),
    "dominant_topic": dominant_topic,
    "top_topic_prob": top_topic_prob,
    "margin": margin,
    "entropy": entropy
})

# ---- Save to CSV ----
output_path = "document_topic_confidence.csv"
df_results.to_csv(output_path, index=False)
  
# RC: how much the set of documents assigned to that topic 
# (via dominant-topic = argmax) overlaps with the top-probability 
# documents for that same topic
def calculate_distribution_overlap(K, doc_topic, dominant_topic, percentiles=[10, 25, 50]):
    """Calculate how much the dominant topic documents overlap with top percentiles"""
    
    overlap_results = []
    
    for t in range(K):
        dominant_docs = set(np.where(dominant_topic == t)[0])
        topic_probs = doc_topic[:, t]
        total_docs = len(topic_probs)
        
        percentile_overlaps = {}
        
        for p in percentiles:
            # Get top p% documents by P(topic|doc)
            n_top = int(total_docs * p / 100)
            top_docs = set(np.argsort(topic_probs)[::-1][:n_top])
            
            # Calculate overlap
            overlap = len(dominant_docs & top_docs) / len(dominant_docs)
            percentile_overlaps[f'top_{p}pct'] = overlap
        
        overlap_results.append({
            'topic': t,
            'n_dominant': len(dominant_docs),
            **percentile_overlaps
        })
        
        print(f"Topic {t}: {len(dominant_docs):3d} docs | " + 
              " | ".join([f"Top {p}%: {overlap:.3f}" 
                         for p, overlap in percentile_overlaps.items()]))
    
    return overlap_results

# Run this analysis
overlap_results = calculate_distribution_overlap(K, doc_topic, dominant_topic)

# Visulize it
def create_assignment_heatmap_with_names(overlap_results, topic_names):
    """
    overlap_results: your existing results list
    topic_names: list of topic names (top words) for each topic
    """
    # Prepare data
    topics = [f'{topic_names[r["topic"]]}' for r in overlap_results]
    metrics = ['top_10pct', 'top_25pct', 'top_50pct']
    metric_labels = ['Top 10%', 'Top 25%', 'Top 50%']
    
    # Create data matrix
    data = np.array([[r[metric] for metric in metrics] for r in overlap_results])
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Create heatmap manually
    im = ax.imshow(data, cmap='Blues', aspect='auto')
    
    # Set ticks and labels
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_yticks(np.arange(len(topics)))
    ax.set_xticklabels(metric_labels)
    ax.set_yticklabels(topics)
    
    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add text annotations with white text for contrast
    for i in range(len(topics)):
        for j in range(len(metrics)):
            text = ax.text(j, i, f'{data[i, j]:.3f}',
                          ha="center", va="center", color='white', 
                          fontsize=10, fontweight='bold')
    
    ax.set_title("Dominant Topic Assignment Coverage by Percentile", fontsize=14, pad=20)
    fig.colorbar(im, ax=ax, label='Coverage Proportion')
    plt.tight_layout()
    plt.savefig('topic_assignment_coverage_named.png', dpi=300, bbox_inches='tight')
    plt.show()

topic_names = {
    0: "information",
    1: "community",  
    2: "economic",  
    3: "class", 
    4: "child",   
    5: "gender",    
    6: "urban",      
    7: "participation",     
    8: "search",     
    9: "migrant",    
    10: "mobility",   
    11: "tie",        
    12: "care",       
    13: "risk",      
    14: "organization", 
    15: "support",     
    16: "skill" 
}

create_assignment_heatmap_with_names(overlap_results, topic_names)

#%% Check with metadata
# Load the metadata
file_path = "cleaned_doi_data_2.csv"
meta_df = pd.read_csv(file_path)

# Merge with the metric table
df_combined = pd.concat([meta_df.reset_index(drop=True), df_results], axis=1)

# Select papers with topic 6 as dominant topic
df_topic6 = df_combined[df_combined["dominant_topic"] == 6].copy()

#  Save to CSV 
output_path = "documents_with_dominant_topic_6.csv"
df_topic6.to_csv(output_path, index=False)

# ---- Dominant topic prevalence over time
df_dom = pd.DataFrame({
    "year": meta_df["year"].astype(int),
    "dominant_topic": dominant_topic
})

# Counts
topic_counts = df_dom.groupby(["year", "dominant_topic"]).size().unstack(fill_value=0)
for t in range(K):
    if t not in topic_counts.columns:
        topic_counts[t] = 0
topic_counts = topic_counts.reindex(sorted(topic_counts.columns), axis=1)

# Sort topics by average count
topic_order = topic_counts.mean(axis=0).sort_values(ascending=False).index
topic_counts = topic_counts[topic_order]

# Color
cmap = get_cmap('tab20')
colors = [cmap(i) for i in range(K)]

# Plot count instead of share
plt.figure(figsize=(12, 6))
for idx, col in enumerate(topic_counts.columns):
    plt.plot(topic_counts.index, topic_counts[col], label=f"topic_{col}", color=colors[idx])
plt.title("Dominant-topic counts over years")
plt.xlabel("Year")
plt.ylabel("Number of documents")
plt.legend(ncol=3, fontsize=8)
plt.tight_layout()
plt.show()

# Stacked area chart
plt.figure(figsize=(12, 6))
plt.stackplot(topic_counts.index, topic_counts.T, labels=[f"topic_{col}" for col in topic_counts.columns], colors=colors)
plt.title("Dominant-topic counts over years (stacked)")
plt.xlabel("Year")
plt.ylabel("Number of documents")
plt.legend(ncol=3, fontsize=8, loc='upper left')
plt.tight_layout()
plt.show()

# Plot the annual distribution of pulisded articles
year_counts = meta_df['year'].value_counts().sort_index()
year_counts = year_counts[year_counts.index != 2025]

plt.figure(figsize=(12, 6))
plt.plot(year_counts.index, year_counts.values, marker='o', linestyle='-')

plt.title("Distribution of Annual Publications")
plt.xlabel("Year")
plt.ylabel("Number of Articles")
plt.tight_layout()
plt.show()

# Calculate shares (proportions) instead of counts
year_totals = topic_counts.sum(axis=1)
topic_shares = topic_counts.div(year_totals, axis=0) * 100  # Convert to percentages

# Sort topics by average share
topic_order = topic_shares.mean(axis=0).sort_values(ascending=False).index
topic_shares = topic_shares[topic_order]

# Get topic names (most frequent terms)
# Get topic names from the LDA model - using only ONE term
def get_topic_names(lda_model, n_terms=1):
    topic_names = []
    for topic_idx in range(lda_model.num_topics):
        # Get top term for this topic
        top_terms = lda_model.show_topic(topic_idx, topn=n_terms)
        # Extract just the first word
        top_word = top_terms[0][0]  # Get the word from the first (word, prob) tuple
        topic_names.append(top_word)
    return topic_names

# Generate topic names from your model (single term only)
topic_names = get_topic_names(model_17, n_terms=1)

# Create a mapping from topic number to topic name
topic_name_mapping = {i: topic_names[i] for i in range(model_17.num_topics)}

cmap = get_cmap('tab20')
colors = [cmap(i) for i in range(model_17.num_topics)]

# Plot shares with topic names
plt.figure(figsize=(12, 6))
for idx, col in enumerate(topic_shares.columns):
    topic_label = topic_name_mapping[col]
    
    if col == 6:
        # Make topic 6 more visible
        plt.plot(topic_shares.index, topic_shares[col], 
                label=f"{topic_label}*",  # Use single term with asterisk
                color=colors[idx], 
                linewidth=3,  # Thicker line
                marker='o',   # Add markers
                markersize=6,
                linestyle='-',
                markeredgecolor='black',  # Black border for markers
                markeredgewidth=1)
    else:
        plt.plot(topic_shares.index, topic_shares[col], 
                label=topic_label,  # Use single term
                color=colors[idx],
                linewidth=1.5,
                alpha=0.8)

plt.title("Dominant Topic Shares Over Years")
plt.xlabel("Year")
plt.ylabel("Percentage of Documents (%)")

# Place legend inside the plot frame
plt.legend(ncol=3, fontsize=9, loc='upper right', framealpha=0.9)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


#%% Check some of those articles for the writting

# Select papers 
df_topic13_15 = df_combined[df_combined["dominant_topic"].isin([13, 15])]
df_topic13_15_year = df_topic13_15[(df_topic13_15["year"] >= 1985) & (df_topic13_15["year"] <= 1995)]

# Save to CSV 
output_path = "documents_with_dominant_topic_13_15.csv"
df_topic13_15_year.to_csv(output_path, index=False)


df_topic1 = df_combined[df_combined["dominant_topic"] == 1]
df_topic1_year = df_topic1[df_topic1["year"].isin([2000, 2002])]
output_path = "documents_with_dominant_topic_1.csv"
df_topic1_year.to_csv(output_path, index=False)


#%% Check other models
# load the combined dictionary
dictionary = Dictionary.load("combine_dictionary.dict")

# visulize the result of other models
model_paths = sorted(
    glob.glob("model_*.model"),
    key=lambda p: int(re.search(r"model_(\d+)\.model$", p).group(1))
)

out_dir = Path("pyldavis_exports")
out_dir.mkdir(exist_ok=True)

for p in model_paths:
    m = LdaModel.load(p)
    K = m.num_topics  
    vis = gensimvis.prepare(m, corpus, dictionary) 
    out_path = out_dir / f"pyldavis_k{K}.html"
    pyLDAvis.save_html(vis, str(out_path))
    print(f"Saved {out_path}")


#%% Check book and chapters cited by articles in topic 6
df_topic6 = pd.read_csv("documents_with_dominant_topic_6.csv")

books = {
    "Bourdieu, P., 1986": r"BOURDIEU.*1986.*(FORMS OF CAPITAL|HANDBOOK OF THEORY|RICHARDSON)",
    
    "Burt, R. S., 1992": r"BURT.*1992.*STRUCTURAL HOLES",
    
    "Coleman, J. S., 1990": r"COLEMAN.*1990.*FOUNDATIONS OF SOCIAL THEORY",
    
    "Putnam, R. D., 2000": r"PUTNAM.*2000.*BOWLING ALONE",
    
    "Putnam, R. D., 1993": r"PUTNAM.*1993.*MAKING DEMOCRACY WORK",
    
    "Lin, N., 2001": r"LIN.*2001.*SOCIAL CAPITAL"
}

def extract_books_precise(cr):
    if pd.isna(cr):
        return []
    cr = cr.upper()
    return [
        book for book, pattern in books.items()
        if re.search(pattern, cr)
    ]

df_topic6["cr_book"] = df_topic6["CR"].apply(extract_books_precise)

output_path = "documents_with_dominant_topic_6_book.csv"
df_topic6.to_csv(output_path, index=False)

exploded = df_topic6.explode("cr_book").dropna(subset=["cr_book"])

freq_table_t6 = (
    exploded["cr_book"]
    .value_counts()
    .reset_index()
    .rename(columns={"index": "book", "cr_book": "frequency"})
)

freq_table_t6




 