````md
# Recommender Engine — Collaborative Filtering with PyTorch

Neural recommender system built with **matrix factorization** in PyTorch to learn
user–movie embeddings from the MovieLens dataset and generate personalized
recommendations.

## What it does
- Learns **latent factors** for users and movies using gradient descent
- Predicts missing ratings from user–item interactions
- Groups similar movies via **KMeans on embeddings**
- Discovers genres and taste clusters without using movie metadata

## Dataset
MovieLens (`movies.csv`, `ratings.csv`)

## Model
- Matrix Factorization (user + movie embeddings)
- Loss: MSE
- Optimizer: Adam
- Trained on CPU (8GB RAM friendly)

## Training
```bash
python train.py
````

## Output

* Trained user & movie embeddings
* Movie clusters showing similar tastes (e.g. sci-fi, drama, action)

## Tech Stack

PyTorch · Pandas · NumPy · Scikit-learn · MovieLens

```