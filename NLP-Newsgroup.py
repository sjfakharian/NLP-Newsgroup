'''
#Step 1: Download the dataset
You can download the dataset using the scikit-learn library, which provides a function to fetch the 20 Newsgroups dataset:
http://qwone.com/~jason/20Newsgroups/
'''

from sklearn.datasets import fetch_20newsgroups

categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
newsgroups_train = fetch_20newsgroups(subset='train', categories=categories)
newsgroups_test = fetch_20newsgroups(subset='test', categories=categories)

'''
This code downloads the training and testing subsets of the dataset, and only includes posts from four categories (atheism, religion, graphics, and medicine). 
You can modify the categories variable to include other categories if you'd like.

#Step 2: Preprocess the data
Next, you'll need to preprocess the text data. Here's an example of how you can do this using the nltk library:
'''

import nltk
nltk.download('punkt')

def preprocess(text):
    tokens = nltk.word_tokenize(text.lower())
    tokens = [token for token in tokens if len(token) > 2]  # Remove short words
    tokens = [token for token in tokens if token.isalpha()]  # Remove punctuation and numbers
    return ' '.join(tokens)

X_train = [preprocess(text) for text in newsgroups_train.data]
y_train = newsgroups_train.target

X_test = [preprocess(text) for text in newsgroups_test.data]
y_test = newsgroups_test.target

'''
This code tokenizes the text data using the word_tokenize() function from the nltk library, removes short words and non-alphabetic tokens, and joins the remaining tokens back into a string.

#Step 3: Extract features
Now that you've preprocessed the data, you'll need to extract features from it. One way to do this is to use the CountVectorizer class from the sklearn library, which converts a collection of text documents into a matrix of token counts:
'''

from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()
X_train_counts = vectorizer.fit_transform(X_train)
X_test_counts = vectorizer.transform(X_test)

'''
This code fits the CountVectorizer on the training data and transforms both the training and testing data into matrices of token counts.

#Step 4: Train the model
Finally, you can train a machine learning model on the feature matrix using the MultinomialNB class from the sklearn library:
'''

from sklearn.naive_bayes import MultinomialNB

clf = MultinomialNB()
clf.fit(X_train_counts, y_train)

'''
This code trains a Naive Bayes classifier on the training data and feature matrix.

Step 5: Evaluate the model
Once you've trained the model, you can evaluate its performance on the testing data using the predict() method and the accuracy_score() function from the sklearn library:
'''
from sklearn.metrics import accuracy_score

y_pred = clf.predict(X_test_counts)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

'''
This code predicts the target labels for the testing data using the trained model, and calculates the accuracy score of the predictions.
'''
