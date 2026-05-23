
# HATCom: Advanced Community Detection Framework for Complex Network Analysis

HATCom is a self-supervised graph learning framework designed for community detection in **Heterogeneous Attributed Temporal Networks (HATNs)**.

The framework jointly models:

- Heterogeneous graph structures
- Relation semantics
- Temporal evolution
- Multi-hop interactions

to generate stable and meaningful communities from complex real-world interaction data.

This project was developed as our final year B.Tech project and focuses on advanced graph representation learning using:

- RTCA (Relation-Time Co-Attention)
- GTN (Graph Transformer Network)
- EMA-Based Temporal Memory
- Self-Supervised Link Prediction

---

# Features

- Heterogeneous graph construction using users, movies, and genres
- Relation-aware and time-aware attention learning using RTCA
- Automatic multi-hop semantic learning using GTN
- EMA-based temporal memory for stable embedding evolution
- Self-supervised learning using link prediction
- Community detection using K-Means clustering
- Overlapping community analysis
- Temporal stability analysis across graph snapshots
- Structural and semantic evaluation metrics
- Genre classification using KNN classifier

---

# Dataset

We used the **MovieLens 100K** dataset.

## Dataset Statistics

| Feature | Value |
|---|---|
| User-Movie Ratings | 100,000 |
| Users | 943 |
| Movies | 1,682 |
| Genre Categories | 18 |
| Interaction Type | Timestamped Interactions |

Timestamped interactions were important because the framework models temporal graph evolution.

---

# Framework Overview

The complete HATCom pipeline consists of:

1. Data Preprocessing  
2. Heterogeneous Graph Construction  
3. Type-Aware Feature Encoding  
4. Relation-Time Co-Attention (RTCA)  
5. Graph Transformer Network (GTN)  
6. EMA-Based Temporal Memory  
7. Self-Supervised Link Prediction  
8. Community Detection using K-Means  
9. Genre Classification using KNN  
10. Structural and Semantic Evaluation  

---

# Core Modules

## 1. Heterogeneous Graph Construction

A multi-relational graph was constructed using:

- User nodes
- Movie nodes
- Genre nodes

### Edge Types

- User → Movie interactions
- Movie → Genre relations
- User → User similarity
- Movie → Movie similarity

This creates a heterogeneous multi-relational graph.

---

# 2. Type-Aware Encoder

Different node types contain different semantic information.

## Implemented Encoding

- Movie titles and genres embedded using Sentence Transformers
- User categorical information encoded separately
- Features projected into a unified embedding space

## Final Embedding Dimension

```text
128-dimensional embeddings
```

---

# 3. Relation-Time Co-Attention (RTCA)

RTCA captures:

- Relation-aware attention
- Temporal importance of interactions

Recent and semantically important interactions receive higher attention scores.

## RTCA Attention Equation

```math
e_{ij} = LeakyReLU(W_a[h_i || h_j || e_r || \phi(t)])
```

## Softmax Normalization

```math
\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{k \in N(j)} \exp(e_{kj})}
```

## Embedding Aggregation

```math
h_j^{RTCA} = \sum_{i \in N(j)} \alpha_{ij} h_i
```

## Temporal Encoding

```math
\phi(t) = \exp\left(-\frac{t_{max}-t}{\tau}\right)
```

### RTCA Functionality

- Learns semantic relation importance
- Gives higher weight to recent interactions
- Updates node embeddings using weighted neighborhood aggregation

---

# 4. Graph Transformer Network (GTN)

GTN automatically learns:

- Soft meta-paths
- Multi-hop semantic relationships

Instead of manually designing graph paths, GTN discovers important heterogeneous semantic structures automatically.

## GTN Aggregation Equation

```math
H^{GTN} = \parallel_{k=1}^{K} \sum_{j \in N(i)} \alpha_{ij}^{(k)} W^{(k)} h_j
```

### GTN Functionality

- Learns indirect semantic relationships
- Captures multi-hop graph structures
- Performs multi-channel neighborhood aggregation

---

# 5. EMA-Based Temporal Memory

The full graph timeline was divided into:

```text
5 Temporal Snapshots
```

EMA combines:

- Current embeddings
- Historical embeddings

to stabilize temporal embedding evolution.

## EMA Equation

```math
M_t = \alpha H_t + (1-\alpha)M_{t-1}
```

## Purpose of EMA

- Prevent sudden embedding changes
- Preserve historical behavior
- Maintain smooth temporal evolution
- Stabilize communities across snapshots

---

# 6. Self-Supervised Link Prediction

The framework was trained using:

- Positive user-movie interactions
- Randomly sampled negative pairs

This allowed the graph itself to generate supervision without manually labeled communities.

## BCE Loss Equation

```math
L = - \sum_{(u,v)\in E^+} \log \sigma(h_u \cdot h_v)
    - \sum_{(u,v)\in E^-} \log(1 - \sigma(h_u \cdot h_v))
```

Where:

- \(E^+\) → Real user-movie interactions
- \(E^-\) → Random negative pairs

## Link Prediction Functionality

- Learns meaningful embeddings
- Brings similar nodes closer
- Separates fake interactions from real interactions
- Enables self-supervised learning

---

# 7. Community Detection

Final movie embeddings were clustered using:

# K-Means Clustering

## Number of Clusters

```text
18 Communities
```

Communities were formed based on embedding similarity.

### Community Meaning

Movies with:
- similar genres
- similar semantics
- similar interaction patterns

became grouped into the same community.

---

# 8. Genre Classification using KNN

To evaluate semantic embedding quality, we performed genre classification using a K-Nearest Neighbors (KNN) classifier.

The learned movie embeddings were used as classification features.

## Genre Classification Statistics

| Parameter | Value |
|---|---|
| Total Movie Embeddings | 1682 |
| Embedding Dimension | 128 |
| Movies with Primary Genre | 1680 |
| Movies Used After Filtering | 1679 |
| Number of Genres | 17 |
| Train Set Size | 1343 |
| Test Set Size | 336 |
| KNN Accuracy | 66.67% |

## Observation

The KNN classifier achieved around **66.67% accuracy**, showing that the learned embeddings captured meaningful semantic genre information.

---

# Evaluation Metrics

## Structural & Semantic Metrics

| Metric | Result |
|---|---|
| Silhouette Score | 0.6368 |
| Calinski-Harabasz Index | 41,157.85 |
| Davies-Bouldin Index | 0.5284 |
| Modularity | 0.8579 |
| Avg Conductance | 0.0604 |
| Multi-label Purity | 0.4479 |
| Avg Temporal Stability | 0.3725 |
| ROC-AUC | 0.7950 |
| Validation Accuracy | 71.82% |

---

# Metric Explanations

## Silhouette Score

Measures how compact and well-separated the clusters are.

Higher value indicates better clustering quality.

---

## Calinski-Harabasz Index

Measures cluster compactness and separation.

Higher value indicates stronger cluster structure.

---

## Davies-Bouldin Index

Measures overlap between clusters.

Lower value indicates better community separation.

---

## Modularity

Measures strength of community structure.

Higher value indicates stronger internal community connectivity.

## Modularity Equation

```math
Q = \frac{1}{2m} \sum_{ij} \left(A_{ij} - \frac{k_i k_j}{2m}\right)\delta(c_i,c_j)
```

---

## Conductance

Measures how much a community connects outside itself.

Lower conductance indicates better community separation.

## Conductance Equation

```math
\phi(S)=\frac{cut(S,\bar S)}{\min(vol(S),vol(\bar S))}
```

---

## Cluster Purity

Measures semantic consistency inside clusters.

## Purity Equation

```math
Purity = \frac{1}{N}\sum_k \max_j |c_k \cap t_j|
```

---

## Temporal Stability

Measures whether communities remain stable across temporal snapshots.

## Temporal Stability Equation

```math
TS = \frac{1}{T-1}\sum_{t=1}^{T-1} NMI(C_t,C_{t+1})
```

---

# Key Observations

- Strong semantic grouping of movie embeddings was observed
- Communities showed high structural coherence
- EMA improved temporal embedding stability
- The framework captured overlapping semantic communities
- RTCA successfully learned relation-aware temporal attention
- GTN learned meaningful multi-hop semantic structures

---

# Technologies Used

## Programming

- Python
- Google Colab

## Deep Learning & Graph Learning

- PyTorch
- PyTorch Geometric

## Data Processing

- NumPy
- Pandas
- Scikit-learn

## Graph Processing & Visualization

- NetworkX
- Matplotlib

---

# Baseline Comparisons

HATCom was compared against:

- DeepWalk + K-Means
- Node2Vec + Louvain
- Simplified GTN

The proposed framework achieved superior semantic clustering and community quality.

---

# Publication

Our research paper based on HATCom has been accepted at:

# ICT4SD 2026

## International Conference on ICT for Sustainable Development

### Publisher

Springer

---

# Future Scope

- Real-time dynamic graph processing
- Distributed large-scale graph learning
- Transformer-based temporal attention
- Cross-domain applications in:
  - Recommendation Systems
  - Fraud Detection
  - Healthcare Analytics
  - Cybersecurity Networks

---
