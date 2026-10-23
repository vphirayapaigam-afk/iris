import unittest

from classifier import (
    find_best_k,
    load_data,
    predict_flower,
    split_and_scale,
    train_model,
)


class ClassifierTests(unittest.TestCase):
    def setUp(self):
        self.df, self.X, self.y, self.names = load_data()
        self.X_train, self.X_test, self.y_train, self.y_test, self.scaler = split_and_scale(
            self.X, self.y
        )

    def test_dataset_shape(self):
        self.assertEqual(self.X.shape, (150, 4))
        self.assertEqual(len(self.names), 3)

    def test_split_is_80_20(self):
        self.assertEqual(len(self.X_train), 120)
        self.assertEqual(len(self.X_test), 30)

    def test_scaling_uses_train_stats(self):
        self.assertAlmostEqual(float(self.X_train.mean()), 0.0, places=6)

    def test_model_accuracy_is_high(self):
        model = train_model(self.X_train, self.y_train, 5)
        self.assertGreater(model.score(self.X_test, self.y_test), 0.85)

    def test_best_k_in_range(self):
        k, ks, _ = find_best_k(self.X_train, self.y_train, range(1, 11))
        self.assertIn(k, ks)

    def test_predict_flower(self):
        model = train_model(self.X_train, self.y_train, 5)
        self.assertEqual(predict_flower(model, self.scaler, self.names, [5.0, 3.4, 1.5, 0.2]), "setosa")


if __name__ == "__main__":
    unittest.main()
