# Установим необходимые библиотеки
!pip install fastai opendatasets

import opendatasets as od
import pandas as pd
from fastai.text.all import *

# Скачаем датасет
od.download("https://www.kaggle.com/competitions/word2vec-nlp-tutorial/")

# Распакуем датасет
!unzip /content/word2vec-nlp-tutorial/labeledTrainData.tsv.zip -d /content/

# Загрузим данные в датафрейм
train = pd.read_csv("/content/word2vec-nlp-tutorial/labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)

# Посмотрим на данные
train.head()

# Загрузим данные в TextDataLoaders
dls = TextDataLoaders.from_df(train, text_col='review', label_col='sentiment', valid_pct=0.2)

# Посмотрим на случайный набор данных
dls.show_batch(max_n=5)

# Создадим классификатор на базе AWD_LSTM
learn = text_classifier_learner(dls, AWD_LSTM, metrics=accuracy)

# Найдем оптимальный learning rate
learn.lr_find()

# Обучим модель на 3 эпохах
learn.fine_tune(3, base_lr=1e-2)

# Посмотрим результаты
learn.show_results()

# Пример предобработки данных
train['review'] = train['review'].str.lower()
train['review'] = train['review'].str.replace(r'<[^<>]*>', '', regex=True)

# Пересоздаем DataLoaders с новыми данными
dls = TextDataLoaders.from_df(train, text_col='review', label_col='sentiment', valid_pct=0.2)

# Пересоздадим и обучим модель снова
learn = text_classifier_learner(dls, AWD_LSTM, metrics=accuracy)
learn.fine_tune(3, base_lr=1e-2)
learn.show_results()
