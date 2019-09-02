import time
from sklearn import model_selection
import warnings
warnings.filterwarnings('ignore')

def find_hyper_params(train_set, target, MLAs, grid_params):
    start_total = time.perf_counter()
    cv_split = model_selection.ShuffleSplit(n_splits = 10, test_size = .3, train_size = .6, random_state = 0 ) # run model 10x with 60/30 split intentionally leaving out 10%
    
    for mla, param in zip (MLAs, grid_params):
        best_search = model_selection.GridSearchCV(estimator = mla[1], param_grid = param, cv = cv_split, scoring = 'roc_auc')
        best_search.fit(train_set, target)
        best_param = best_search.best_params_
        print('The best parameter for {} is {}.'.format(mla[1].__class__.__name__, best_param))
        mla[1].set_params(**best_param) 

    run_total = time.perf_counter() - start_total
    print('Total optimization time: {:.2f} minutes.'.format(run_total/60))

    print('-'*10)