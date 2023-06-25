import re
from joblib import dump

import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score




nltk.download("stopwords")


df = pd.read_csv("data/train.csv")
df.shape

df.isnull().sum()
df = df.fillna("")

# merging author name and news title for easier use
df["content"] = df["author"] + " " + df["title"]



def stemming(content):
    ps = PorterStemmer()
    stemmed_content = re.sub("[^a-zA-Z]", " ", content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [
        ps.stem(word)
        for word in stemmed_content
        if word not in stopwords.words("english")
    ]
    stemmed_content = " ".join(stemmed_content)
    return stemmed_content


df["content"] = df["content"].apply(stemming)
x = df["content"].values
y = df["label"].values

vectorizer = TfidfVectorizer()
vectorizer.fit(x)

x = vectorizer.transform(x)
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, stratify=y, random_state=2
)

# model = LogisticRegression()
# model.fit(x_train, y_train)

# x_train_pred = model.predict(x_train)
# training_data_acc = accuracy_score(x_train_pred, y_train)
# print(f"Accuracy score: {training_data_acc}")

# x_test_pred = model.predict(x_test)
# test_data_acc = accuracy_score(x_test_pred, y_test)
# print(f"Accuracy score: {test_data_acc}")

# dump(model, "model.joblib")


x_new = x_test[125]
print(x_new)

# prediction = model.predict(x_new)
# print(prediction)

# if prediction[0] == 0:
#     print("The news is Real")
# else:
#     print("The news is Fake")
