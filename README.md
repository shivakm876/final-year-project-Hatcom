
# Genre Classification using KNN

To evaluate whether the learned movie embeddings captured meaningful semantic information, we performed genre classification using a K-Nearest Neighbors (KNN) classifier.

The learned movie embeddings were used as input features for classification.

## Classification Details

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

The KNN classifier achieved around **66.67% accuracy**, showing that the learned embeddings successfully captured meaningful genre-based semantic information.
