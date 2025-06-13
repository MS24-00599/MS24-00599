# JFE: MS24-00599

This repository contains the code and documentation accompanying the research paper MS24-00599. The primary objective of this project is to develop and train neural network models that map the text of a patent to:

1. An expected citation score
2. A propensity score indicating the likelihood that the lead author is female

## Overview

The core modeling framework employs a state-of-the-art language encoder—Longformer—to extract representations from patent texts. These embeddings are then processed by three distinct components:

### Citation Estimation Heads

Two separate fully connected layers with ReLU activation are used to predict expected citations, conditional on the gender of the lead author. One head is trained on male-authored patents, and the other on female-authored patents.

### Gender Propensity Head

A third fully connected layer predicts the probability that a patent's lead author is female, yielding a score in the interval [0, 1].

### Loss Function

The overall loss function is a weighted sum of four components:

- **Mean Squared Error (MSE)** for the male citation head
- **Mean Squared Error (MSE)** for the female citation head
- **Binary Cross-Entropy (BCE)** for the gender propensity head
- **Masked Language Modeling (MLM) Loss**: Cross-entropy loss based on the prediction of a randomly masked token in each input sequence

By default, all loss terms are weighted equally, but these weights may be adjusted by the user to emphasize particular components as needed.

## Dataset and Sampling Strategy

Our analysis is based on a subset of United States utility patents granted from 1976 to 2021. To mitigate the gender imbalance inherent in the raw data, we construct a balanced sample as follows:

- **Female-authored patents**: All patents with a female-identified lead author are included
- **Male-authored patents**: An equal number of male-authored patents are randomly selected (without replacement)

This balanced design is crucial for training an unbiased gender propensity predictor. If the model were trained on the unfiltered population, where over 90% of lead authors are male, the propensity output would be dominated by the majority class and thus uninformative.

### Data Preparation Example

To prepare the training set:

1. Collect all patents with an identified female lead author
2. Randomly sample an equal number of patents with male lead authors
3. Combine these patents into a single, balanced dataset

## Repository Structure

```
.
├── data/
│   └── <sampled_patent_data_files>
├── models/
│   └── <model_checkpoint_files>
├── src/
│   └── <training_and_evaluation_scripts>
├── README.md
└── ...
```

## Usage

1. Clone the repository
2. Prepare the dataset using the outlined balanced sampling approach
3. Train the model with the scripts provided in the `src/` directory

Detailed instructions for data formatting and model training are provided in the `src/` directory.

