# English Word Segmentation
=============

This is a simple implementation of a word segmenter for English. This model is based on Bi-LSTM CRF and created in pytorch. It uses character n-grams (1 and 2) for training and inference. The dataset used in this example is the Brown Corpus plus 10 full wikipedia articles  obtained with the [Wikipedia API](https://github.com/martin-majlis/Wikipedia-API)

Outline
-----
The unique file contains the creation of the dataset (1) the annotation function in the BIES scheme ``def conll_tagging`` (2), the model architecture and training ``def train`` and ``def evaluate`` (3) and the inference functions ``def predict`` at the end. 

Hyperparameters
-----
* Adam optimizer 
* Learning rate:1e-5
* Weight_decay_rate:0.001
* Dropout: 0.5 
* Loss function: CrossEntropyLoss


Results
-----
This approach obtained 85,43%  F1- score in the self annotated dataset in 20 Epochs. 

Example:

  ``“Theyhadwonpromotionthepreviousseason”``
    Result:
  ``"They had won promotion the previous season" ``
