# Импортируем необходимые библиотеки
import pandas as pd
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
# Загрузим данные из CSV файла
data = pd.read_csv('popular.csv')
# Подготовим признаки и целевую переменную
X = data[['region', 'year']]  # Признаки (регион и год)
y = data['value']  # Целевая переменная (число людей)
# Разделим данные на обучающий и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Создадим и обучим модель CatBoost
model = CatBoostRegressor(iterations=100, depth=6, learning_rate=0.1, loss_function='RMSE')
# Преобразуем признаки и целевую переменную в объекты CatBoost
train_pool = Pool(X_train, label=y_train, cat_features=[0])
test_pool = Pool(X_test, label=y_test, cat_features=[0])
# Обучим модель
model.fit(train_pool, eval_set=test_pool, verbose=10)
# Предскажем значения на тестовом наборе
y_pred = model.predict(test_pool)
# Оценим производительность модели
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
print(f'R-squared (R2): {r2}')
