# gridSearch

this module includes one method find_hyper_params(train_set, target, MLAs, grid_params). we can find appropriate hyper parameters of machine learning algorithm according to given data set.

train_set is our data set cleaned train set
target is our data set target
MLAs is a list of tuple of machine learning algorithm like this :  [('abc', AdaBoostClassifier()),('bc', BaggingClassifier())]
grid_params is list of dictionary of possible hyper parameter of machine learning algorithm respectively to MLAs like this:
[{'n_estimators': n_estimator,'learning_rate': grid_learn, 'algorithm': alg ,'random_state': grid_seed},#ada
{'n_estimators': n_estimator,'max_samples': grid_ratio,'max_features': grid_ratio , 'bootstrap': boolean, 'bootstrap_features':boolean, 'oob_score':boolean, 'warm_start':boolean, 'n_jobs':number,'random_state': grid_seed, 'verbose': number},#bagging
              
