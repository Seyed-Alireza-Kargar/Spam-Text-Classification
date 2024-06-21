# SMS Spam Classifier

This project aims to classify SMS text messages as spam or ham (non-spam) using natural language processing (NLP) techniques and a machine learning model.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model and Evaluation](#model-and-evaluation)
- [Results](#results)
- [Contributing](#contributing)

## Introduction

This project demonstrates a basic implementation of a spam text message classifier using a Naive Bayes model. The goal is to filter out spam messages from legitimate ones, thereby reducing unwanted messages.

## Dataset

The dataset used for this project is a collection of SMS messages labeled as either 'spam' or 'ham'. The dataset is sourced from Kaggle and is available in a CSV file named `spam-text-message-data.csv`.

You can download the dataset from the following link:
[SMS Spam Collection Dataset on Kaggle](https://www.kaggle.com/datasets/team-ai/spam-text-message-classification)

The dataset consists of two columns:
- `Category`: Label indicating whether the message is 'spam' or 'ham'.
- `Message`: The content of the SMS message.

## Installation

To run this project, you need to have Python and the following libraries installed:

- pandas
- matplotlib
- nltk
- scikit-learn

You can install these dependencies using pip:

```sh
pip install pandas matplotlib nltk scikit-learn
```
Additionally, you need to download the NLTK stopwords:

``` python
import nltk
nltk.download('stopwords')
```
<br>
<br>

## Usage
Clone this repository to your local machine :
``` sh
git clone https://github.com/yourusername/spam-text-classification.git
cd spam-text-classification
```
Place the spam-text-message-data.csv file in the project directory.

Run the main.py script to execute the spam classification process:

``` sh
python main.py
```
<br>
<br>

# Model and Evaluation

The script performs the following steps:

Reads the dataset.
Preprocesses the data by cleaning and stemming the text messages.
Converts the text messages into TF-IDF features.
Splits the dataset into training and testing sets.
Trains a Naive Bayes classifier on the training set.
Evaluates the model using accuracy, precision, and recall metrics.

<br>
<br>

## Results
The confusion matrix and performance metrics (accuracy, precision, recall) are printed to the console after the model evaluation. Here is an example of the output:


``` sh
Confusion Matrix:
[[955   0]
 [ 34 126]]

Accuracy: 0.9695
Precision: 1.0
Recall: 0.7875
```

<br>
<br>

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

