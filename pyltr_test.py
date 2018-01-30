import pyltr
import pickle

with open('lambda_test.txt') as evalfile:
    EX, Ey, Eqids, _ = pyltr.data.letor.read_dataset(evalfile)

modelfile = open('ltr_model')
model = pickle.load(modelfile)
metricfile = open('metric')
metric = pickle.load(metricfile)

Epred = model.predict(EX)
print(Epred)
print('Random ranking:', metric.calc_mean_random(Eqids, Ey))
print('Our model:', metric.calc_mean(Eqids, Ey, Epred))
