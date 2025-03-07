# 2D Embeddings Generator and Viewer

This project generates 2D embeddings for 12-dimensional vectors using various dimensionality reduction techniques, with a focus on preserving local topology. It's particularly useful for vectors where each dimension has a distinct semantic meaning that should not be mixed through rotations.

## Features

- Multiple dimensionality reduction techniques (UMAP, t-SNE, PCA, MDS)
- Various parameter configurations to capture different aspects of the data
- Interactive visualization with ability to switch between embeddings
- Each point links to relevant external URL (`rawl.rocks/f/<key>`)
- Normalization of input vectors

## Requirements

- Python 3.6+
- Modern web browser

## Installation

1. Clone this repository
2. Install the required Python packages:

```bash
pip install -r requirements.txt
```

## Usage

### Step 1: Generate Embeddings

Run the Python script to generate the embeddings:

```bash
python embedding_generator.py
```

This will:

1. Load data from `histograms.json`
2. Normalize each vector (divide by sum)
3. Apply 20 different embedding methods
4. Save results to `embeddings.json`

### Step 2: View Embeddings

Open `embedding_viewer.html` in a web browser to interact with the generated embeddings:

- Select different embedding methods from the dropdown
- Adjust point size with the slider
- Toggle labels on/off
- Hover over points to see details
- Click on points to open the corresponding link

## Technical Details

### Embedding Methods

The script uses several dimensionality reduction techniques with different parameters:

1. **UMAP** (Uniform Manifold Approximation and Projection)

   - Various neighborhood sizes (5, 15, 30, 50)
   - Different distance metrics (cosine, euclidean, correlation)
   - Random parameter combinations

2. **t-SNE** (t-Distributed Stochastic Neighbor Embedding)

   - Various perplexity values (5, 30, 50, 100)
   - Cosine distance metric to respect the semantic nature of dimensions

3. **PCA** (Principal Component Analysis)

   - Linear dimensionality reduction

4. **MDS** (Multidimensional Scaling)
   - Both euclidean and cosine distance metrics

These techniques prioritize local similarities in different ways, allowing you to explore which best captures the cluster structure in your data.

## Customization

- Add new embedding methods in `embedding_generator.py`
- Modify the HTML/JS visualization in `embedding_viewer.html`
- Adjust normalization or preprocessing steps in the Python script
