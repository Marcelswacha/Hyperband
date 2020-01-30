import numpy as np

from random import random
from math import log, ceil
from time import time, ctime

class Hyperband:
    
    def __init__( self, get_params_function, try_params_function ):
        self.get_params = get_params_function
        self.try_params = try_params_function
        
        self.max_iter = 81      # maximum iterations per configuration
        self.eta = 3            # defines configuration downsampling rate (default = 3)

        self.logeta = lambda x: log( x ) / log( self.eta )
        self.s_max = int( self.logeta( self.max_iter ))
        self.B = ( self.s_max + 1 ) * self.max_iter

        self.results = []    # list of dicts
        self.counter = 0
        self.best_acc = 0
        self.best_counter = -1
    
    def get_best_results(self, n):
        best = sorted( self.results, key = lambda x: x['acc'], reverse=True )[:n]
        return best
    
    def get_best_config(self):
        return self.get_best_results(1)[0]['params']
    
    def print_best_results(self, n):
        for r in self.get_best_results(n):
            print ("acc: {:.2%} | {} seconds | {:.1f} iterations | run {} ".format( 
                r['acc'], r['seconds'], r['iterations'], r['counter'] ))
        

    # can be called multiple times
    def run( self, skip_last = 0):
        
        for s in reversed( range( self.s_max + 1 )):
            
            # initial number of configurations
            n = int( ceil( self.B / self.max_iter / ( s + 1 ) * self.eta ** s ))    
            
            # initial number of iterations per config
            r = self.max_iter * self.eta ** ( -s )        

            # n random configurations
            T = [ self.get_params() for _ in range( n )] 
            
            for i in range(( s + 1 ) - int( skip_last )):    # changed from s + 1
                
                # Run each of the n configs for <iterations> 
                # and keep best (n_configs / eta) configurations
                
                n_configs = n * self.eta ** ( -i )
                n_iterations = r * self.eta ** ( i )
                
                print("\n*** {} configurations x {:.1f} iterations each".format( 
                    n_configs, n_iterations ))
                
                val_accs = []
                #early_stops = []
                
                for t in T:
                    
                    self.counter += 1
                    print ("\n{} | {} | highest acc so far: {:.4f} (run {})\n".format( 
                        self.counter, ctime(), self.best_acc, self.best_counter ))
                    
                    start_time = time()
                    
                    result = self.try_params( n_iterations, t )        # <---
                        
                    assert( type( result ) == dict )
                    assert( 'acc' in result )
                    
                    seconds = int( round( time() - start_time ))
                    print ("\n{} seconds.".format( seconds ))
                    
                    acc = result['acc']    
                    val_accs.append( acc )
                    
                    #early_stop = result.get( 'early_stop', False )
                    #early_stops.append( early_stop )
                    
                    # keeping track of the best result so far (for display only)
                    # could do it be checking results each time, but hey
                    if acc > self.best_acc:
                        self.best_acc = acc
                        self.best_counter = self.counter
                    
                    result['counter'] = self.counter
                    result['seconds'] = seconds
                    result['params'] = t
                    result['iterations'] = n_iterations
                    
                    self.results.append( result )
                
                # select a number of best configurations for the next loop
                # filter out early stops, if any
                #indices = np.argsort( val_accs )
                #T = [ T[i] for i in indices if not early_stops[i]]
                #T = T[ int( n_configs / self.eta ) + 1:]
                T = [ T[i] for i in np.argsort(val_accs)[int( n_configs / self.eta ):] ]
        
        return self.results