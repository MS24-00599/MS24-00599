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

## Datasets

*data/synthetic_1k_sample.parquet* --
A random sample of 1000k patents with randomly assigned T groups and synthetic outcomes. The synthetic outcomes are created by taking a random linear combination of all the text embeddings produced by the pre-trained Longformer model. The T=1 group receives one higher citation than T=0.

*data/positive_balanced_sample_output.parquet* -- The main output of the manual script -- gender balanced subset of patents with positive citations. This parquet file contains the expected citation counts and female author propensity estimated by our model. To replicate our paper, please use the expected citations by the male model and exponentiate the raw numbers with exp(x)-1, because when fitting the model, we apply a log(1+x) transformation to the citation counts.

## codes
*examples/example_fit_data.ipynb* --
A general notebook tutorial that can be used to fit an I-TEXT model with a chosen text encoder (the default one in the script is Longformer) and properly formatted data containing text (input), target (output), and group indicator.

*examples/run_small_synthetic_check.ipynb* --
An example code that uses the I-TEXT architecture to estimate the average difference between the citation counts estimated in two groups. This example uses real patent data, random group assignment, and synthetic targets (outcomes) where the ground-truth difference between groups 1 and 0 is 1. The estimated average difference in this dataset is 1.02.


## Repository Structure

```
.
├── data/
│   └── <sampled_patent_data_files>
├── examples/
│   └── <sample codes>
├── README.md
└── ...
```

## Using this repo
To clone the repository, use the following command:

```bash
git clone https://github.com/MS24-00599/MS24-00599.git
```

### Fitting a model

1. Clone the repository
2. Prepare the dataset as a pandas dataframe containing text (input), target (output), and group indicator.
3. Fit the model with the scripts provided in the `code/example_fit_data.ipynb` directory.

### Using the dataset as a control variable for patent quality

```python
import pandas as pd
import numpy as np
dfcit=pd.read_parquet("../data/positive_balanced_sample_output.parquet")
dfcit["text_quality"]=np.exp(dfcit.log_female_expected_citation)-1
```
