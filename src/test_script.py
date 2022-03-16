import pandas as pd


# функции для тестирования
from train_predict_model import convert_examples_to_inputs



def test_convert_examples_to_inputs(train_path, test_path):
    assert read_train_test_data(train_path, test_path) 
    assert 
