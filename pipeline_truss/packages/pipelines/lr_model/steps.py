from sklearn.base import TransformerMixin
from typing import Tuple
import numpy as np

from models.role import Role
from models.builder import Builder

class QualifiedBuilder(Role):
    qualified: bool


class Disqualify(TransformerMixin):
    def transform(self, X):
        print(X)
        return [self._disqualify(*x) for x in X]

    @staticmethod
    def _disqualify(
        builder: Builder,
        role: Role,
    ) -> Tuple[Builder, QualifiedBuilder]:
        disqualified = Disqualify.check_if_disqualified(
            builder,
            role,
        )

        return (
            builder,
            {**role, "qualified": not disqualified},
        )

    @staticmethod
    def check_if_disqualified(
        builder: Builder, role: Role
    ) -> bool:
        if builder.get("feature1") is None:
            return True
        else:
            return False

class FeatureCalculator(TransformerMixin):
    
    DISQUALIFIED_FEATURES = [-1e3] * 5

    def __init__(self, check_qualified=False):
        super().__init__()
        self._check_qualified = check_qualified

    @staticmethod
    def _get_features(
        builder: Builder,
        role: Role,
    ) -> np.array:

        return [
            1.0 if builder.get("feature1") == role.get("feature1") else 0.0,
            1.0 if builder.get("feature2") == role.get("feature2") else 0.0,
            1.0 if builder.get("feature3") == role.get("feature3") else 0.0,
            1.0 if builder.get("feature4") == role.get("feature4") else 0.0,
            1.0 if builder.get("feature5") == role.get("feature5") else 0.0
        ]

    def transform(self, X):
        if not self._check_qualified:
            return np.array([self._get_features(*x) for x in X])

        return np.array(
            [
                self._get_features(role, builder)
                if builder.get("qualified", True)
                else self.DISQUALIFIED_FEATURES
                for role, builder in X
            ]
        )