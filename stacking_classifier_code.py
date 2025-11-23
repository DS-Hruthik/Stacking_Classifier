from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np
import pandas as pd
import pickle

iris = load_iris()

df_features = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df_target = pd.DataFrame(data=iris.target, columns=['species'])
final = pd.concat([df_features, df_target], axis=1)

X = np.array(final.iloc[:, :4])
Y = np.array(final['species'])

X_train, X_test, y_train, y_test = train_test_split(X, Y, stratify=Y, random_state=42)

estimators = [
    ('rf', RandomForestClassifier(n_estimators=10, random_state=42)),
    ('svr', make_pipeline(StandardScaler(), LinearSVC(random_state=42)))
]

clf = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())

stacking = clf.fit(X_train, y_train)

accuracy = stacking.score(X_test, y_test)
print("Accuracy:", accuracy)

pickle.dump(stacking, open('stacking_iris.pkl', 'wb'))

model = pickle.load(open('stacking_iris.pkl', 'rb'))

test = pd.read_csv(r"C:\Users\hruth\Desktop\self pace learning\my work\Ensemble Technique\Voting and Stacking Material\4.d.Ensemble Models\Stacking Classifier\iris_test.csv")

pred = model.predict(test)
print("Predictions:", pred)
