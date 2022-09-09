import numpy as np
import sys
sys.path.insert(0, './pipelines/lr_model/')
sys.path.insert(0, './models/')
from joblib import dump
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from pipelines.lr_model.steps import (Disqualify, FeatureCalculator)

LEARNED_INTERCEPT = -1
LEARNED_COEFFICIENTS = [2, 1, 0, 0, 1]
lr_model = LogisticRegression()
lr_model.classes_ = np.array([0, 1])
lr_model.intercept_ = LEARNED_INTERCEPT
lr_model.coef_ = np.array([LEARNED_COEFFICIENTS])

pipeline = Pipeline(
    steps=[
        ("step1", Disqualify()),
        (
            "step2",
            FeatureCalculator(check_qualified=True),
        ),
        ("model", lr_model),
    ]
)

from bson.objectid import ObjectId

SID0 = ObjectId("abcdef123456abcdef123453")
SID1 = ObjectId("abcdef123456abcdef123453")
SID2 = ObjectId("abcdef123456abcdef123454")
SID3 = ObjectId("abcdef123456abcdef123454")
SID4 = ObjectId("abcdef123456abcdef543210")

BUILDER = {
    "feature1": SID0,
    "feature2": SID1,
    "feature3": SID4,
    "feature4": SID2,
    "feature5": SID1,
}

ROLE = {
    "feature1": SID0,
    "feature2": SID1,
    "feature3": SID1,
    "feature4": SID3,
    "feature5": SID4,
}

pipeline.predict_proba([(x, ROLE) for x in [BUILDER]])

# somehow I should upload this pipeline and the necessary local dependencies to baseten