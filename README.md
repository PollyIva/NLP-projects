# NLP-projects

### [1.Spam Classifier](https://github.com/PollyIva/NLP-projects/blob/main/Spam%20Classifier/Spam_classifier.ipynb)

This project aims to create a spam classifier using natural language processing techniques. The dataset used for this project contains a collection of SMS messages labeled as spam or ham.

Given 2 ways to find spam:

  1. using length and punctuation
  2. using message text processing

Data set: [smsspamcollection.tsv](https://github.com/PollyIva/NLP-projects/blob/main/smsspamcollection.tsv)

Libraries: NumPy, pandas, scikit-learn, NLTK, Matplotlib, Seaborn


### [2.Detecting Hate tweets](https://github.com/PollyIva/NLP-projects/blob/main/Detecting%20Hate%20tweets/toxify_ML_05.03.ipynb)

Data set: [test.csv](https://www.kaggle.com/datasets/julian3833/jigsaw-toxic-comment-classification-challenge) 

For the XGBRegressor model, the following steps were taken:

1.   Data preprocessing: A dataset of tweets labeled as hate speech or not hate speech was collected and preprocessed by cleaning the text data, removing stop words, and converting the text into numerical features using techniques such as TF-IDF.

2.   Model training: The XGBRegressor model was trained using the preprocessed data. Since XGBRegressor is a gradient boosting algorithm, it was trained using multiple weak learners to minimize the mean squared error (MSE) between the predicted hate speech probability and the actual hate speech probability.

3.  Model evaluation: The XGBRegressor model was evaluated using metrics such as accuracy, precision, recall, and F1-score. The model was also tested on a holdout dataset to assess its generalization performance.

For the TensorFlow model, the following steps were taken:

1.  Data preprocessing: The same hate speech dataset was preprocessed in the same way as for the XGBRegressor model.

2.  Model architecture: A deep learning architecture consisting of an embedding layer, two LSTM layers, and a dense output layer was built using TensorFlow. The embedding layer learned a dense representation of the input text, while the LSTM layers captured the sequence information. The dense output layer predicted the probability of hate speech.

3.  Model training: The TensorFlow model was trained using the preprocessed data. The loss function used was binary cross-entropy, and the optimizer used was Adam.

4.  Model evaluation: The TensorFlow model was evaluated using the same metrics as for the XGBRegressor model. The model was also tested on a holdout dataset to assess its generalization performance.

Overall, both models were able to accurately recognize hate speech with high precision and recall. However, the TensorFlow model was able to achieve slightly higher accuracy and F1-score than the XGBRegressor model.
