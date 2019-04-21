import pandas as pd
import os
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, f1_score
from sklearn.model_selection import cross_val_score
#
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfTransformer

SPAM = 1
HAM = 0
TEST_SIZE = 0.3

FILE_PATHS = [(os.path.abspath(
    os.path.dirname(__file__)+'/spam_and_ham_mails/spam_2/'), SPAM),
    (os.path.abspath(
        os.path.dirname(__file__)+'/spam_and_ham_mails/easy_ham/'), HAM),
    (os.path.abspath(
        os.path.dirname(__file__)+'/spam_and_ham_mails/hard_ham/'), HAM)
]
SKIP_FILES = ['cmd']
result = []
for path, mail_type in FILE_PATHS:
    file_list = os.listdir(path)
    for f in file_list:
        if f in SKIP_FILES:
            continue
        with open(path + '/' + f, 'r', encoding="latin-1") as mail:
            content = []
            for line in mail:
                content.append(line)
            content = '\n'.join(content)
        result.append({'data': content, 'type': mail_type})
data = pd.DataFrame(result)
# X: Data y: Tag
X_train, X_test, y_train, y_test = train_test_split(data['data'], data['type'],
                                                    test_size=TEST_SIZE, random_state=42)
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)
tfidf_transformer = TfidfTransformer()
X_train = tfidf_transformer.fit_transform(X_train_counts)
neigh = KNeighborsClassifier(n_neighbors=5)
neigh.fit(X_train, y_train)
cv_scores = cross_val_score(neigh, X_train, y_train, cv=5)
print(cv_scores)
X_test_counts = count_vect.transform(X_test)
X_test = tfidf_transformer.transform(X_test_counts)
start = datetime.now()
predicted = neigh.predict(X_test)
end = datetime.now()
print((end - start).total_seconds())  # 0.82
print(confusion_matrix(y_test, predicted))
print('accuracy_score', accuracy_score(y_test, predicted))
print('f1_score', f1_score(y_test, predicted))
print('recall_score', recall_score(y_test, predicted))
