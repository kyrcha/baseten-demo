from typing import Dict, List



import numpy as np
import sys
sys.path.insert(0, './pipelines/lr_model/')
sys.path.insert(0, './models/')

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from model.pipelines.lr_model.steps import (Disqualify, FeatureCalculator)

# MODEL_BASENAME = 'model'
# MODEL_EXTENSIONS = ['.joblib', '.pkl', '.pickle']


class Model:
    def __init__(self, **kwargs) -> None:
        self._data_dir = kwargs['data_dir']
        config = kwargs['config']
        model_metadata = config['model_metadata']
        self._supports_predict_proba = model_metadata['supports_predict_proba']
        self._model_binary_dir = model_metadata['model_binary_dir']
        self._model = None

    def load(self):
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
        self._model = pipeline

    def preprocess(self, request: Dict) -> Dict:
        """
        Incorporate pre-processing required by the model if desired here.

        These might be feature transformations that are tightly coupled to the model.
        """
        return request

    def postprocess(self, request: Dict) -> Dict:
        """
        Incorporate post-processing required by the model if desired here.
        """
        return request

    def predict(self, request: Dict) -> Dict[str, List]:
        response = {}
        inputs = request['inputs']
        result = self._model.predict(inputs)
        response['predictions'] = result
        if self._supports_predict_proba:
            response['probabilities'] = self._model.predict_proba(inputs).tolist()
        return response
