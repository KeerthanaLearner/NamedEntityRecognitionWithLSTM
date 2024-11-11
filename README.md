# Named Entity Recognition Using Bidirectional LSTM Networks

### Authors
- **Dr. David Raj Micheal** - Department of Mathematics, Vellore Institute of Technology, Chennai
- **Keerthana V** - Department of Mathematics, Vellore Institute of Technology, Chennai

### Overview
This project explores the implementation of a Named Entity Recognition (NER) system using a Bidirectional Long Short-Term Memory (BiLSTM) network. NER is essential in Natural Language Processing (NLP) for identifying and categorizing entities such as people, organizations, locations, and dates within text. The project leverages the IOB (Inside-Outside-Beginning) tagging schema to classify tokens, enhancing the model’s accuracy in differentiating entity boundaries and types, which is beneficial for information extraction and NLP tasks.

### Key Features
- **Deep Learning Model**: Utilizes a BiLSTM model to capture contextual information from both past and future tokens, enhancing the accuracy of named entity recognition.
- **IOB Tagging Schema**: Implements token-level classification, distinguishing entity boundaries effectively.
- **Performance Evaluation**: Evaluates model accuracy and cross-entropy loss across different entity types.
- **Comparison**: Assesses the BiLSTM model against traditional rule-based methods, demonstrating the advantages of contextual learning.

### Table of Contents
- [Introduction](#introduction)
- [Model Architecture](#model-architecture)
- [Data Preprocessing](#data-preprocessing)
- [Evaluation Metrics](#evaluation-metrics)
- [Future Work](#future-work)
- [References](#references)

---

## Introduction
Named Entity Recognition (NER) aims to identify and classify named entities within text into categories like Person, Location, and Organization. NER finds applications in search engines, content categorization, and knowledge base construction. Traditional rule-based and statistical approaches are limited in handling complex and varied linguistic patterns. This project presents a BiLSTM-based model that captures contextual information bidirectionally, achieving superior results compared to traditional models.

---

## Model Architecture
The model architecture is built around:
1. **Embedding Layer**: Maps words into dense vector representations.
2. **Bidirectional LSTM Layer**: Processes sequences in both forward and backward directions to capture full context.
3. **TimeDistributed Dense Layer**: Outputs probability distributions for each word’s classification into NER tags.

Key settings include:
- Embedding vector size of 50.
- 64 LSTM units with a recurrent dropout rate of 0.1.
- Softmax activation function for multi-class classification.

---

## Data Preprocessing
1. **Dataset**: An entity-annotated corpus using IOB tagging with entity categories like `geo`, `org`, `per`, `gpe`, and `tim`.
2. **Steps**:
   - **Tokenization and Index Mapping**: Assigns a unique index to each token and tag.
   - **Padding**: Ensures uniform sequence lengths for efficient batch processing.
   - **Embedding**: Maps indexed tokens into vector space, capturing semantic relationships.
3. **Structure**: Each word and tag is converted into a numerical index, facilitating efficient deep learning processing.

---

## Evaluation Metrics
- **Training Accuracy**: 99.7%
- **Validation Accuracy**: 99.3%
  
Evaluation was conducted using accuracy and cross-entropy loss. The model performs well across entity types, with minor performance drops in rare categories, such as Natural Phenomena.

---

## Future Work
Future improvements include:
- Expanding the dataset to improve generalization.
- Integrating pre-trained embeddings like GloVe or BERT to leverage semantic knowledge.
- Experimenting with Transformer-based architectures (e.g., BERT, RoBERTa) for enhanced performance in sequence tagging tasks.

---

## References
Key references include foundational and recent works in NER, BiLSTM networks, and sequence tagging.

---

This project illustrates the potential of BiLSTM networks in NER tasks, offering insights into deep learning approaches for enhanced text sequence classification in NLP.
