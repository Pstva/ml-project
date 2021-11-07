
## Используемые инструменты

+ линтер pycodestyle
+ линтер PyFlakes
+ форматтер Black


### pycodestyle (2.6.0)


Этот инструмент был выбран для того, чтобы проверить, насколько мой код по стилю соответсвует стандарту PEP. 

Запускался так:

```sh
pycodestyle --statistics -qq src
```

#### Вывод:

```sh
20      E101 indentation contains mixed spaces and tabs
2       E117 over-indented
2       E122 continuation line missing indentation or outdented
11      E128 continuation line under-indented for visual indent
5       E225 missing whitespace around operator
10      E251 unexpected spaces around keyword / parameter equals
1       E303 too many blank lines (2)
17      E501 line too long (82 > 79 characters)
121     W191 indentation contains tabs
11      W291 trailing whitespace
7       W293 blank line contains whitespace
```

Выведено довольно много нарушений PEP, что неудивительно, учитывая тот факт, что изначально код писался в гугл колабе, где это не отслеживается.

### pyflakes (2.4.0)

Инструмент выбран, чтобы посмотреть на возможные ошибки в скрипте

```sh
pyflakes src/
```
#### Вывод:

```sh
src/train_predict_model.py:12:1 'torch.utils.data.SequentialSampler' imported but unused
src/train_predict_model.py:13:1 'os' imported but unused
src/train_predict_model.py:14:1 'tqdm.trange' imported but unused
src/train_predict_model.py:15:1 'sklearn.metrics.precision_recall_fscore_support' imported but unused
src/train_predict_model.py:18:1 redefinition of unused 'classification_report' from line 15
src/train_predict_model.py:18:1 'seqeval.metrics.classification_report' imported but unused
src/train_predict_model.py:18:1 'seqeval.metrics.f1_score' imported but unused
src/train_predict_model.py:19:1 'torch.nn.functional as F' imported but unused
```

Как можно заметить, у меня было импортировано несколько функций, которые я не использую в дальнейшем. 
Я убрала их импорт из скрипта, после чего инструмент больше не выдавал мне ошибок.

### Black (19.10b0)


Так как у меня было довольно много стилистических нарушений в коде, я решила использовать форматтер Black
с дефолтными настройками для того, чтобы это исправить.

```sh
black src/train_predict_model.py
```

#### Вывод:

```sh
reformatted src/train_predict_model.py
All done! ✨ 🍰 ✨
1 file reformatted.
```

После чего, снова решила проверить вывод инструмента pycodestyle, изменив максимальную длину строки на 88
(так как это дефолтное значение Black, я поверила ему, что так будет лучше)

```sh
pycodestyle --max-line-lenght=88 train_predict_model.py
```
Инструмент не выдал мне ошибок.

