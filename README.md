# Text Classification with Persian Text and Text Augmentation

This repository contains code for text classification using BERT in PyTorch, specifically designed for Persian text. It also includes text augmentation techniques to increase the diversity of the training data.

## Installation

To use this code, please follow the instructions below:

1. Clone the repository:

   ```
   git clone https://github.com/armansouri9/Text-Classification-Persian-text-and-text-augmentation.git
   ```

2. Install the required dependencies:

   ```
   pip install pandas nlpaug transformers nltk matplotlib torch tqdm
   ```

## Usage

1. Import the necessary libraries:

   ```python
   import pandas as pd
   import torch
   import numpy as np
   import nlpaug.augmenter.word as naw
   from transformers import BertTokenizer
   import nltk

   nltk.download('punkt')
   ```

2. Load the dataset:

   ```python
   df = pd.read_csv('/path/to/dataset.csv')
   ```

3. Perform data preprocessing:

   ```python
   def delete_Stopword(txt):
       # Function implementation

   def normalizer(txt):
       # Function implementation

   for i in range(len(df)):
       delete_Stopword(df['Text'][i])
       normalizer(df['Text'][i])
       df['Text'][i].split()
       " ".join(df['Text'][i])
   ```

4. Define the dataset class:

   ```python
   tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

   labels = {'WithoutContent': 0, 'WithContent': 1}

   class Dataset(torch.utils.data.Dataset):
       # Dataset class implementation
   ```

5. Split the dataset into train, validation, and test sets:

   ```python
   np.random.seed(112)
   df_train, df_val, df_test = np.split(df.iloc[:, :2].sample(frac=1, random_state=42),
                                        [int(.7 * len(df)), int(.9 * len(df))])
   ```

6. Define the BERT classifier model:

   ```python
   class BertClassifier(nn.Module):
       # BERT classifier model implementation
   ```

7. Train and evaluate the model:

   ```python
   # Train and evaluate the model
   ```

## License

This project is licensed under a Free License.
