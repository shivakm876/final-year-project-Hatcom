HATCom: An Advanced Community Detection Framework for Complex Network Analysis

HATCom is a self-supervised graph learning framework designed for community detection in Heterogeneous Attributed Temporal Networks (HATNs). The framework jointly models heterogeneous graph structures, semantic relationships, and temporal evolution to generate stable and meaningful communities from complex real-world interaction data.

This project was developed as our final year B.Tech project and focuses on advanced graph representation learning using RTCA, GTN, EMA-based temporal memory, and self-supervised link prediction.

Features
Heterogeneous graph construction using users, movies, and genres
Relation-aware and time-aware attention learning using RTCA
Automatic multi-hop semantic learning using GTN
EMA-based temporal memory for stable embedding evolution
Self-supervised learning using link prediction
Community detection using K-Means clustering
Overlapping community analysis
Temporal stability analysis across graph snapshots
Multiple structural and semantic evaluation metrics
Dataset

We used the MovieLens 100K dataset.

Dataset statistics:

100,000 user-movie ratings
943 users
1,682 movies
18 genre categories
Timestamped interactions

The timestamped interactions were important for modeling temporal graph evolution.

Framework Overview

The complete HATCom pipeline consists of:

Data preprocessing
Heterogeneous graph construction
Type-aware feature encoding
Relation-Time Co-Attention (RTCA)
Graph Transformer Network (GTN)
EMA-based temporal memory integration
Self-supervised link prediction training
Community detection using K-Means
Structural, semantic, and temporal evaluation
Core Modules
1. Heterogeneous Graph Construction

A multi-relational graph was constructed using:

User nodes
Movie nodes
Genre nodes

Different edge types included:

User → Movie ratings
Movie → Genre relations
User similarity edges
Movie similarity edges
2. Type-Aware Encoder

Different node types contain different features.

Movie titles and genres were embedded using Sentence Transformers
User occupations and categorical information were encoded separately
Features were projected into a unified 128-dimensional embedding space
3. Relation-Time Co-Attention (RTCA)

RTCA captures:

semantic importance of relations
temporal recency of interactions

Recent and semantically important interactions receive higher attention weights.

RTCA updates node embeddings using relation-aware and time-aware neighborhood aggregation.

4. Graph Transformer Network (GTN)

GTN automatically learns:

soft meta-paths
multi-hop semantic relationships

Instead of manually defining graph paths, GTN discovers important heterogeneous graph patterns automatically.

5. EMA-Based Temporal Memory

The dataset timeline was divided into 5 temporal snapshots.

EMA memory combines:

current embeddings
historical embeddings

to maintain stable temporal evolution.

EMA Equation:

M
t
	​

=αH
t
	​

+(1−α)M
t−1
	​

6. Self-Supervised Link Prediction

The model was trained using:

positive user-movie interactions
randomly sampled negative pairs

Binary Cross Entropy (BCE) loss was used for training.

This allowed the model to learn meaningful embeddings without labeled community datasets.

7. Community Detection

Final movie embeddings were clustered using:

K-Means Clustering
Number of clusters: 18
Communities were formed based on embedding similarity
Overlapping semantic communities were also analyzed
Evaluation Metrics

The framework was evaluated using multiple metrics.

Metric	Value
Silhouette Score	0.6368
Calinski-Harabasz Index	41157.85
Davies-Bouldin Index	0.5284
Modularity	0.8579
Average Conductance	0.0604
Multi-label Cluster Purity	0.4479
Average Temporal Stability	0.3725
ROC-AUC	0.7950
Validation Accuracy	71.82%
Key Observations
Strong semantic grouping of movie embeddings was observed
Communities showed high structural coherence
EMA improved temporal embedding stability
The framework captured overlapping semantic communities
RTCA successfully learned relation-aware temporal attention
Technologies Used
Programming
Python
Google Colab
Deep Learning & Graph Learning
PyTorch
PyTorch Geometric
Data Processing
NumPy
Pandas
Scikit-learn
Graph Processing & Visualization
NetworkX
Matplotlib
Baseline Comparisons

HATCom was compared against:

DeepWalk + K-Means
Node2Vec + Louvain
Simplified GTN

The proposed framework achieved superior community quality and semantic clustering performance.

Publication

Our research paper based on HATCom has been accepted at:

ICT4SD 2026 — International Conference on ICT for Sustainable Development

Publisher: Springer

Future Scope
Real-time dynamic graph processing
Large-scale distributed graph learning
Transformer-based temporal attention
Cross-domain applications in:
recommendation systems
fraud detection
cybersecurity
healthcare analytics
