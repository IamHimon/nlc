import pyltr
import pickle

with open('lambda_train.txt') as trainfile, open('lambda_test.txt') as evalfile:
    # open('vali.txt') as valifile, \
    TX, Ty, Tqids, _ = pyltr.data.letor.read_dataset(trainfile)
    # VX, Vy, Vqids, _ = pyltr.data.letor.read_dataset(valifile)
    EX, Ey, Eqids, _ = pyltr.data.letor.read_dataset(evalfile)

metric = pyltr.metrics.NDCG(k=10)

# Only needed if you want to perform validation (early stopping & trimming)
# monitor = pyltr.models.monitors.ValidationMonitor(VX, Vy, Vqids, metric=metric, stop_after=250)

model = pyltr.models.LambdaMART(
    metric=metric,
    n_estimators=100,
    learning_rate=0.02,
    max_features=0.5,
    query_subsample=0.5,
    max_leaf_nodes=10,
    min_samples_leaf=64,
    verbose=1,
)

model.fit(TX, Ty, Tqids)

modelfile = open('ltr_model', 'w')
metricfile = open('metric', 'w')
pickle.dump(model, modelfile)
pickle.dump(metric, metricfile)

# Epred = model.predict(EX)
# print Epred
# print 'Random ranking:', metric.calc_mean_random(Eqids, Ey)
# print 'Our model:', metric.calc_mean(Eqids, Ey, Epred)
