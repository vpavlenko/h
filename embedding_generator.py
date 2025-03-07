import json
import numpy as np
import os
from sklearn.manifold import TSNE, MDS
from sklearn.decomposition import PCA, NMF
import umap
from sklearn.preprocessing import normalize
import random

# Load the data
with open("histograms.json", "r") as f:
    data = json.load(f)

# Extract keys and vectors
keys = list(data.keys())
vectors = np.array([data[key] for key in keys])

# Step 1: Normalize each vector (divide by sum)
normalized_vectors = np.array([vec / np.sum(vec) for vec in vectors])

# Step 1b: Create augmented vectors by concatenating original with binary indicators
# This enhances the contrast between zero and non-zero components
binary_indicators = (normalized_vectors > 0).astype(float)
augmented_vectors = np.hstack([normalized_vectors, binary_indicators])

# Define embedding methods and parameters
embedding_methods = {
    "UMAP_n5": lambda data: umap.UMAP(
        n_neighbors=5, min_dist=0.1, metric="cosine"
    ).fit_transform(data),
    "UMAP_n15": lambda data: umap.UMAP(
        n_neighbors=15, min_dist=0.1, metric="cosine"
    ).fit_transform(data),
    "UMAP_n30": lambda data: umap.UMAP(
        n_neighbors=30, min_dist=0.1, metric="cosine"
    ).fit_transform(data),
    "UMAP_n50": lambda data: umap.UMAP(
        n_neighbors=50, min_dist=0.1, metric="cosine"
    ).fit_transform(data),
    "UMAP_euclidean": lambda data: umap.UMAP(metric="euclidean").fit_transform(data),
    "UMAP_correlation": lambda data: umap.UMAP(metric="correlation").fit_transform(
        data
    ),
    "tSNE_p5": lambda data: TSNE(
        n_components=2, perplexity=5, metric="cosine"
    ).fit_transform(data),
    "tSNE_p30": lambda data: TSNE(
        n_components=2, perplexity=30, metric="cosine"
    ).fit_transform(data),
    "tSNE_p50": lambda data: TSNE(
        n_components=2, perplexity=50, metric="cosine"
    ).fit_transform(data),
    "tSNE_p100": lambda data: TSNE(
        n_components=2, perplexity=100, metric="cosine"
    ).fit_transform(data),
    "PCA": lambda data: PCA(n_components=2).fit_transform(data),
    "MDS_euclidean": lambda data: MDS(n_components=2, metric=True).fit_transform(data),
    "MDS_cosine": lambda data: MDS(
        n_components=2, dissimilarity="precomputed"
    ).fit_transform(1 - normalize(data, norm="l2") @ normalize(data, norm="l2").T),
    "Augmented_UMAP": lambda data: umap.UMAP(
        n_neighbors=15, min_dist=0.1, metric="euclidean"
    ).fit_transform(data),
    "Augmented_tSNE": lambda data: TSNE(n_components=2, perplexity=30).fit_transform(
        data
    ),
    "Augmented_UMAP_cosine": lambda data: umap.UMAP(metric="cosine").fit_transform(
        data
    ),
    "Augmented_NMF": lambda data: NMF(
        n_components=2, init="random", random_state=0
    ).fit_transform(data),
    "Binary_Weighted_1.5x": lambda data: umap.UMAP(n_neighbors=15).fit_transform(
        np.hstack([data[:, :12], 1.5 * data[:, 12:]])
    ),
    "Binary_Weighted_2x": lambda data: umap.UMAP(n_neighbors=15).fit_transform(
        np.hstack([data[:, :12], 2.0 * data[:, 12:]])
    ),
    "Binary_Weighted_3x": lambda data: umap.UMAP(n_neighbors=15).fit_transform(
        np.hstack([data[:, :12], 3.0 * data[:, 12:]])
    ),
}

# Add more random UMAP configurations to reach 20 embeddings
for i in range(1, 5):
    n_neighbors = random.randint(5, 100)
    min_dist = random.uniform(0.01, 0.5)
    metric = random.choice(["cosine", "correlation", "euclidean", "manhattan"])
    embedding_methods[f"UMAP_random{i}"] = (
        lambda data, n=n_neighbors, d=min_dist, m=metric: umap.UMAP(
            n_neighbors=n, min_dist=d, metric=m
        ).fit_transform(data)
    )

# Step 2: Generate embeddings
embeddings = {}

# First apply methods that use the original normalized vectors
print("Generating embeddings using original normalized vectors...")
for method_name in [
    "UMAP_n5",
    "UMAP_n15",
    "UMAP_n30",
    "UMAP_n50",
    "UMAP_euclidean",
    "UMAP_correlation",
    "tSNE_p5",
    "tSNE_p30",
    "tSNE_p50",
    "tSNE_p100",
    "PCA",
    "MDS_euclidean",
    "MDS_cosine",
    "UMAP_random1",
    "UMAP_random2",
    "UMAP_random3",
    "UMAP_random4",
]:
    method_func = embedding_methods[method_name]
    print(f"  - {method_name}...")
    try:
        # Generate 2D embedding
        embedding_2d = method_func(normalized_vectors)

        # Scale to 0..1000 range
        min_vals = embedding_2d.min(axis=0)
        max_vals = embedding_2d.max(axis=0)
        range_vals = max_vals - min_vals
        scaled_embedding = (embedding_2d - min_vals) / range_vals * 1000

        # Create a list of [x, y] points
        points = scaled_embedding.tolist()

        # Add to embeddings dict
        embeddings[method_name] = {"points": dict(zip(keys, points))}
    except Exception as e:
        print(f"    Error with {method_name}: {e}")

# Now apply methods specifically designed for the augmented vectors
print("\nGenerating embeddings using augmented vectors (with binary indicators)...")
for method_name in [
    "Augmented_UMAP",
    "Augmented_tSNE",
    "Augmented_UMAP_cosine",
    "Augmented_NMF",
    "Binary_Weighted_1.5x",
    "Binary_Weighted_2x",
    "Binary_Weighted_3x",
]:
    method_func = embedding_methods[method_name]
    print(f"  - {method_name}...")
    try:
        # For weighted methods, use augmented_vectors directly (weights applied in lambda)
        if method_name.startswith("Binary_Weighted_"):
            embedding_2d = method_func(augmented_vectors)
        else:
            embedding_2d = method_func(augmented_vectors)

        # Scale to 0..1000 range
        min_vals = embedding_2d.min(axis=0)
        max_vals = embedding_2d.max(axis=0)
        range_vals = max_vals - min_vals
        scaled_embedding = (embedding_2d - min_vals) / range_vals * 1000

        # Create a list of [x, y] points
        points = scaled_embedding.tolist()

        # Add to embeddings dict
        embeddings[method_name] = {"points": dict(zip(keys, points))}
    except Exception as e:
        print(f"    Error with {method_name}: {e}")

# Save embeddings to JSON file
with open("embeddings.json", "w") as f:
    json.dump(embeddings, f, indent=2)

print(f"\nGenerated {len(embeddings)} embeddings and saved to embeddings.json")
print(
    "Augmented vector methods use both original values and binary indicators (0/1) to emphasize zero vs non-zero patterns."
)
