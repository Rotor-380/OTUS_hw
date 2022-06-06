import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

'''
____________ I. Numpy ____________
'''

'''
Создайте одномерный массив размера 10, заполненный нулями и пятым элемент равным 1. 
Трансформируйте в двумерный массив.
'''
x = np.zeros(10)
x[4] = 1
x = x.reshape(2, 5)
# print(x)

'''
Создайте одномерный массив со значениями от 10 до 49 и разверните его (первый элемент становится последним). 
Найдите в нем все четные элементы.
'''
mass = np.arange(10, 50)  # [::-1] #как самый простой вариант
mass = np.flipud(mass)  # np.flip(mass, 0) #как альтернатива
mass = mass[mass % 2 == 0]
# print(mass)

'''
Создайте двумерный массив 3x3 со значениями от 0 до 8
'''
m = np.arange(9).reshape(3, 3)
# print(m)

'''
Создайте массив 4x3x2 со случайными значениями. Найти его минимум и максимум.
'''
ms = np.random.rand(4, 3, 2)
ms_min = np.min(ms)
ms_max = np.max(ms)
# print(ms_min, ms_max)

'''
Создайте два двумерных массива размерами 6x4 и 4x3 и произведите их матричное умножение.
'''
A = np.ones((6, 4))
B = np.ones((4, 3))
AB = A.dot(B)
# print(AB)

'''
Создайте случайный двумерный массив 7x7, найти у него среднее и стандартное оклонение. 
Нормализуйте этот массив.
'''
mr = np.random.rand(7, 7)
mr_mean = mr.mean()  # среднее
mr_std = mr.std()  # стандартное оклонение
mr_norm = np.linalg.norm(mr)  # Нормализуйте этот массив.
# print(mr_mean)
# print(mr_std)
# print(mr_norm)

'''
____________ II. Pandas ____________
'''

df = pd.read_csv('tips.csv', sep=',')

'''Посмотрите на первые 5 строчек'''

# print(df.head(5))

'''Узнайте сколько всего строчек и колонок в данных'''

# print(df.shape)

'''Проверьте есть ли пропуски в данных'''
# print(df.isna().sum())
# print(df.isna().mean())

'''
Посмотрите на распределение числовых признаков
'''
# print(df.describe())
'''
Найдите максимальное значение 'total_bill'
'''
# print(df['total_bill'].max())

'''
Найдите количество курящих людей
'''
# print(df.query('smoker == "Yes"').smoker.count())

'''
Узнайте какой средний 'total_bill' в зависимости от 'day' 
'''
# print(df.groupby('day')['total_bill'].mean())

'''
Отберите строчки с 'total_bill' больше медианы и узнайте какой средний 'tip' в зависимости от 'sex'
'''
# print(df.query('total_bill > total_bill.median()').groupby('sex')['tip'].mean())

'''
Преобразуйте признак 'smoker' в бинарный (0-No, 1-Yes)
'''
# df['smoker'] = df['smoker'].apply(lambda i: 1 if i == 'Yes' else 0)   # способ через лямбду
# df['smoker'] = df['smoker'].map({'Yes': 1, 'No': 0})  # альтернативно через map()
# print(df.smoker)

'''____________ III. Visualization ____________'''

'''
Постройте гистограмму распределения признака 'total_bill'
'''
# hiss = df['total_bill'].hist()
# hist = hiss.get_figure()
# plt.show()

'''
Постройте scatterplot, представляющий взаимосвязь между признаками 'total_bill' и 'tip'
'''
# sc = sns.scatterplot(data=df, x='total_bill', y='tip')
# plt.show()

'''Постройте pairplot'''
# pt = sns.load_dataset("tips")
# sns.pairplot(pt)
# plt.show()

'''Постройте график взаимосвязи между признаками 'total_bill' и "day" '''
# gr = sns.barplot(y=df.total_bill, x=df['day'].values, data=df)
# plt.show()

'''
Постройте две гистограммы распределения признака 'tip' в зависимости от категорий "time" 
'''
# hiss1 = df.query('time == "Dinner"').tip.hist()
# hist1 = hiss1.get_figure()
# hiss2 = df.query('time == "Lunch"').tip.hist()
# hist2 = hiss2.get_figure()
# plt.show()

# мне кажется, что такой график нагляднее демонстрирует распределение
# gr = sns.distplot(df.query('time == "Dinner"').tip, kde=False, color='red')
# gr = sns.distplot(df.query('time == "Lunch"').tip, kde=False, color='green')
# plt.show()

'''
Постройте два графика scatterplot, представляющих взаимосвязь между признаками 'total_bill' и 'tip' один для Male, 
другой для Female и раскрасьте точки в зависимоти от признака 'smoker'
'''

# sc = df.query('sex == "Male"').plot.scatter(x='total_bill', y='tip') #это попытка через scatter, но не удалось раскрасить
# sc2 = df.query('sex == "Female"').plot.scatter(x='total_bill', y='tip') #это попытка через scatter, но не удалось раскрасить
# plt.show()

#df.plot.scatter(x=0, y=0)
'''
к сожалению, pycharm мнгновенно закрывает график вызываемый следующим кодом, 
если не разкаменчена предыдущая строка df.plot.scatter
'''
# g = sns.FacetGrid(df, col="sex", hue="smoker", height=4)
# g.map(sns.scatterplot, "total_bill", "tip", alpha=.7)
# g.add_legend()
# plt.show()

'''
Сделайте выводы по анализу датасета и построенным графикам. 
По желанию можете продолжить анализ данных и также отразить это в выводах.
'''
# print(df.head(12))
# print(df.shape)
# print(df.day.unique())
# print(df.total_bill.describe())

'''
count    244.000000
mean      19.785943
std        8.902412
min        3.070000
25%       13.347500
50%       17.795000
75%       24.127500
max       50.810000
'''

# gr = sns.barplot(y=df.total_bill, x=df['tip'].values, data=df)
# gr = sns.barplot(y=df.total_bill, x=df['sex'].values, hue='day' , data=df)

# gr = sns.barplot(y=df.total_bill, x=df['smoker'].values, hue="sex", data=df)
# gr = sns.barplot(y=df.tip, x=df['smoker'].values, hue="sex", data=df)

# gr = sns.barplot(y=df.total_bill, x=df['day'].values, data=df)
# gr = sns.barplot(y=df.total_bill, x=df['time'].values, data=df)
gr = sns.barplot(y=df.total_bill, x=df['size'].values, hue="sex", data=df)
plt.show()

max_bill = df.sort_values('total_bill', ascending=False).head(60) #отбираем ~24% самых больших счетов

# gr = sns.barplot(y=df.tip, x=df['total_bill'].values, data=max_bill)
# gr = sns.barplot(y=df.total_bill, x=df['sex'].values, data=max_bill)

# gr = sns.barplot(y=df.total_bill, x=df['smoker'].values, data=max_bill)
# gr = sns.barplot(y=df.tip, x=df['smoker'].values, data=max_bill)

# gr = sns.barplot(y=df.total_bill, x=df['day'].values, data=max_bill)
# gr = sns.barplot(y=df.total_bill, x=df['time'].values, data=max_bill)
# plt.show()

# df.tip.describe()
'''
count    244.000000
mean       2.998279
std        1.383638
min        1.000000
25%        2.000000
50%        2.900000
75%        3.562500
max       10.000000
'''

'''
выводы по датасету tips:
- если смотреть в целом - то нужно отметить, что датасет содержит мало данных (как пример - дней недели представленно только 4) 
    и большинство метрик получаются усреднёнными и сделать по ним однозначный вывод затруднительно.
- если принять за ключевой показатель total_bill, то какой-либо значимой зависимости от дня недели не наблюдается, 
    хотя показатель в воскресение немного больше.
- по признаку пола можно сказать, что счета и чаевые мужчин немного больше
- аналогичная ситуация наблюдается в зависимости счёта от того, курит человек или нет - курящие мужчины заказывают немного больше
- такую же картину можно наблюдать относительно распределения чаевых в зависимости пола и пристрастия к курению - показатели распредилились равномерно
- анализ максимальных счетов нам показал, что:
    - максимальные чаевые сосредаточены в 25% самых больших счетов, а распределение остальной части чаевых равномерное.
    - зависимости размера счёта и размера чаевых от пристрастия не наблюдается, обе группы заказывают примерно одинаково
    - зависимость от дней недели не поменялась относительно общей группы
    - за ужином эта группа, как и в изначальном датасете оставляет больше денег
'''
