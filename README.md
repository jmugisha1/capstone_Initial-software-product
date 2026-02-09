# Disease Prediction System

A machine learning system that predicts diseases from symptom descriptions using ClinicalBERT embeddings and an Artificial Neural Network classifier.

## Live Demo

[https://huggingface.co/spaces/mkkarekezi/Capstone-Initial-Software-Product](https://huggingface.co/spaces/mkkarekezi/Capstone-Initial-Software-Product)

## How It Works

- **ClinicalBERT**: Converts symptom text into medical embeddings (768 dimensions)
- **ANN Classifier**: 3-layer neural network for disease classification
- **Performance**: 65.1% accuracy, 61.1% F1-score

## Local Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Run application
python app.py
```

## Requirements
```txt
gradio
tensorflow
sentence-transformers
scikit-learn
```

## Usage

Enter symptoms in natural language (e.g., "I have blackheads and pimples") to get disease predictions with confidence scores.

## Deployment

Deployed on Hugging Face Spaces with automatic building and hosting.

## video

https://youtu.be/w7ED1PhxWi4
