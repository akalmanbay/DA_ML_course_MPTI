# C2W3 - Задача регрессии

### 1. Линейные модели: статистический взгляд
#### 1.1 Метод максимального правдоподобия
Максимизируем вероятность получения именно такой выборки на находим значение нужной для нас переменной. (оценка максимального правдоподобия)
#### 1.2 Регрессия как максимизация правдоподобия
Предполагая, что шум в регресионном уравнении распределен нормально (Gaussian Distribution), мы приходим к тому, что само уравнение также распределено нормально со средним a(x). 

>$$y=\Theta_0+\Theta_1x+\varepsilon =a(x)+\varepsilon, where\ \varepsilon\sim N(0,\sigma^2 )$$
 Следственно,
$$ y\sim N(a(x), \sigma^2)$$

С помощью ММП, максимизируем вероятность/мат.ожидание от у и находим веса регрессии а(х).
#### 1.4 Регуляризация
Решения переобучения: больше данных, меньше признаков (L1), ограничение весов (L1 и L2-регуляризаторы).
>L1,L2 регуляризаторы больше смещают, но уменьшают дисперсию, L1 отбирает признаки.
#### 1.5 Логрег как оценивание веротяностей (logloss/entropy)
Логрег как одна из общих линейных моделей (GLM).
Максимизируем вероятность получения такой выборки: $$L(x)=\prod_{i=y_i=1} P(x_i) \prod_{i=y_i=0}[1-P(x_i)]$$ и находим $P(x)  \sim \frac{e^{<w,x>}}{1+e^{<w,x>}}$

### 2. Практические рекомендации по линейным моделям
* **Масштабирование:** (i) Нормализация, (ii) Масштабирование на [0,1]
* **Справляющие пространства**: нахождение новых признаков по исходным (Н-р, полиномы, логарифмы, итд)
* **Работа с категориальными признаками**: бинарное кодирование
* **Несбалансированные данные**: Undersampling, Oversampling, Stratification
* **Многоклассовая классификация**: one-vs-all, матрица ошибок, точность/полнота/F-мера 

### 3. Библиотека sci-kit learn
#### 3.1 Обучение линейной модели через обычный grid search и без pipeline 
    from sklearn import model_selection, linear_model, metrics, pipeline, preprocessing
    
    train_data, test_data, train_labels, test_labels = model_selection.train_test_split(data, target, test_size = 0.3,random_state = 0)                                            
Задание модели

    classifier = linear_model.SGDClassifier(random_state = 0, tol=1e-3)
   Генерация сетки

    classifier.get_params().keys()
    parameters_grid = {
        'loss' : ['hinge', 'log', 'squared_hinge', 'squared_loss'],
        'penalty' : ['l1', 'l2'],
        'max_iter' : np.arange(5,10),
        'alpha' : np.linspace(0.0001, 0.001, num = 5),
    }
    cv = model_selection.StratifiedShuffleSplit(n_splits=10, test_size = 0.2, random_state = 0)

   Grid search
  

    grid_cv = model_selection.GridSearchCV(classifier, parameters_grid, scoring = 'accuracy', cv = cv)
    
    grid_cv.fit(train_data, train_labels)
    
    grid_cv.best_estimator_
    grid_cv.best_score_)
    grid_cv.best_params_)
    grid_cv.cv_results_[:10]
   #### 3.2 Обучение линейной модели через  grid search и без pipeline 
    from sklearn import model_selecti
Scaling

    scaler = StandardScaler()
    scaler.fit(train_data, train_labels)
    scaled_train_data = scaler.transform(train_data)
    scaled_test_data = scaler.transform(test_data)

Pipeline

    regressor = linear_model.SGDRegressor(random_state = 0, n_iter = 3, loss = 'squared_loss', penalty = 'l2')
    
    estimator = pipeline.Pipeline(steps = [       
        ('feature_processing', pipeline.FeatureUnion(transformer_list = [        
                #binary
                ('binary_variables_processing', preprocessing.FunctionTransformer(lambda data: data[:, binary_data_indices])), 
                        
            #numeric
            ('numeric_variables_processing', pipeline.Pipeline(steps = [
                ('selecting', preprocessing.FunctionTransformer(lambda data: data[:, numeric_data_indices])),
                ('scaling', preprocessing.StandardScaler(with_mean = 0))            
                        ])),
        
            #categorical
            ('categorical_variables_processing', pipeline.Pipeline(steps = [
                ('selecting', preprocessing.FunctionTransformer(lambda data: data[:, categorical_data_indices])),
                ('hot_encoding', preprocessing.OneHotEncoder(handle_unknown = 'ignore'))            
                        ])),
        ])),
    ('model_fitting', regressor)])
    #estimator.fit(train_data, train_labels)
Grid search

    estimator.get_params().keys()
    parameters_grid = {
        'model_fitting__alpha' : [0.0001, 0.001, 0,1],
        'model_fitting__eta0' : [0.001, 0.05],
    }
    grid_cv = model_selection.GridSearchCV(estimator, parameters_grid, scoring = 'neg_mean_absolute_error', cv = 4)
    grid_cv.fit(train_data, train_labels)
    print(grid_cv.best_score_)
    print(grid_cv.best_params_)
    metrics.mean_absolute_error(test_labels, test_predictions)
