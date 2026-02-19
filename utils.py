from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer 
import re

class PreProcessor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.ageImputer = SimpleImputer(strategy='median')
        self.ageImputer.fit(X[["Age"]])
        return self
    def transform(self, X, y=None):
        X = X.copy()
        X["Age"] = self.ageImputer.transform(X[["Age"]])
        X["CabinClass"] = X["Cabin"].fillna("M").astype(str).str.replace(r"[^a-zA-Z]", "", regex=True)
#         X["CabinNumber"] = X["Cabin"].fillna(0).astype(str).str.replace(r"[^0-9]", "", regex=True).astype(float)
        X['CabinNumber'] = X['Cabin'].fillna('M').apply(lambda x: str(x).replace(" ", "")).apply(lambda x: re.sub(r'[^0-9]', '', x)).replace('', 0) 

        X['Embarked'] = X['Embarked'].fillna('M')
        X = X.drop(['PassengerId', 'Name', 'Ticket','Cabin'], axis=1, errors='ignore')
        return X

columns = ['PassengerId', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch','Ticket', 'Fare', 'Cabin', 'Embarked']