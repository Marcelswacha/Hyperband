from hyperopt import hp
from hyperopt.pyll.stochastic import sample

from hyperband import Hyperband
import kerasmodel as km

space = {
    'lr' : hp.uniform('lr', 0.0001, 0.001),
    'do1' : hp.uniform('do1', 0.2, 0.3),
    'do2' : hp.uniform('do2', 0.2, 0.3),
    'do3' : hp.uniform('do3', 0.4, 0.5),
    'extra_first_layers' : hp.choice('extra_first_layers', [1, 2, 3]),
    'extra_second_layers' : hp.choice('extra_first_layers', [1, 2]),
    }

dummy_space = {
    'x' : hp.uniform('x', 0.2, 0.9),
    }

def dummy_get_params():
    return sample(dummy_space)

def dummy_try_params(n, p):
    acc = p['x'] * n;
    return {'acc' : acc}

def get_params():
    params = sample(space)
    return params

def try_params(n, p):
    km.train_model(n, p)
    
hb = Hyperband(dummy_get_params, dummy_try_params)
results = hb.run()
hb.print_best_results(5)
 
print (hb.get_best_config())
