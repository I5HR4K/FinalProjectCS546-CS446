🚨 Machine Learning–Driven Network Intrusion Detection

Reproducing & Improving Academic Research with Real-World ML Pipelines

Modern cyberattacks evolve faster than signature-based defenses can keep up. This project demonstrates how machine learning can close that gap—and how thoughtful engineering decisions can outperform published research.

🔍 Project Overview

This project is a full end-to-end reproduction and enhancement of a peer-reviewed machine learning research paper focused on improving Network Intrusion Detection Systems (NIDS). Instead of treating this as a theoretical exercise, we rebuilt the models from scratch, engineered a production-style ML pipeline, and critically analyzed why our results matched—and in some cases outperformed—the original study.

We applied multiple machine learning classifiers to two industry-standard intrusion detection datasets:

UNSW-NB15

NSL-KDD

Our goal wasn’t just high accuracy—it was balanced, trustworthy detection using precision, recall, and F1-score to avoid misleading results common in cybersecurity ML research.

📈 What We Did Better Than the Paper

Balanced Metrics Over Inflated Accuracy

The original paper achieved high accuracy on NSL-KDD but suffered in precision/recall.

Our SMOTE-based pipeline produced more realistic, deployable results.

Model-Specific Feature Engineering

Applied binarization for Bernoulli Naive Bayes, aligning data with mathematical assumptions.

Result: significantly improved F1 scores compared to the paper.

Cleaner Data = Better Generalization

Isolation Forest removed noisy outliers that degrade ML performance in real networks.

This demonstrates critical thinking, not blind replication.

🛠️ Tech Stack

Python

Scikit-learn

Pandas / NumPy

Jupyter Notebook
