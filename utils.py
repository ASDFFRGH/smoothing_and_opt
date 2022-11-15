import matplotlib.pyplot as plt 
import numpy as np
from dataclasses import dataclass
from tqdm import tqdm
import copy

level = 1.2

def g(v, conf):
    p = conf.p
    delta = conf.delta
    return np.where(np.abs(v) < delta, -delta*p* np.exp(1-(1/(1-(v/delta)**2))), 0)

def f(v, conf):
    return 0.5*(v-conf.p)**2 + g(v, conf)

def grad(v, conf):
    p = conf.p
    delta = conf.delta
    return np.where(np.abs(v) < delta ,(v-p) + ((2*v*p)/delta)*(1-(v/delta)**2)**(-2)*np.exp(1-(1/(1-(v/delta)**2))), (v-p))

def gd(x, lr, conf):
    x -= lr * grad(x, conf)
    return x

def sgd(x, lr, conf, flag):
    delta = conf.delta
    if flag == 'l':
        r = conf.r1
    elif flag == 's':
        r = conf.r2
    noise = np.random.uniform(-r, r)
    x -= lr * (grad(x, conf) + noise)
    return x

def averaged_sgd(x, x_ave, lr, t, conf, flag):
    beta = 1/(t+1)
    delta = conf.delta
    if flag == 'l':
        r = conf.r1
    elif flag == 's':
        r = conf.r2
    noise = np.random.uniform(-r, r)
    x -= lr * (grad(x, conf) + noise)
    x_ave *= beta*t
    x_ave += beta*x
    return x, x_ave


def optimize(configs, opt):
    sol_list1 = []
    
    for i in tqdm(range(configs.sample_num)):
        #x = configs.init + 10 * np.random.randn()
        x = -1.0
        
        if 'Ave_SGD' in opt:
            x_ave = copy.deepcopy(x)       

        if opt == 'SGD_large':            
            for j in range(configs.iter_num): 
                flag = 'l'
                x = sgd(x, configs.lr_large, configs, flag)
        elif opt == 'SGD_small':
            for j in range(configs.iter_num): 
                flag = 's'
                x = sgd(x, configs.lr_small, configs, flag)
        elif opt == 'Ave_SGD_large':
            for j in range(configs.iter_num):  
                flag = 'l'
                x, x_ave = averaged_sgd(x, x_ave, configs.lr_large, j, configs, flag)
        elif opt == 'Ave_SGD_small':
            for j in range(configs.iter_num):  
                flag = 's'
                x, x_ave = averaged_sgd(x, x_ave, configs.lr_small, j, configs, flag)
        elif opt == 'GD_large':
            for j in range(configs.iter_num): 
                x = gd(x, configs.lr_large, configs)
        elif opt == 'GD_small':
            for j in range(configs.iter_num): 
                x = gd(x, configs.lr_small, configs)
                
        else:
            print('opt is not define')
            
        if 'Ave_SGD' in opt:
            sol_list1.append(x_ave)
          
        else:
            sol_list1.append(x)
         
    sol_arr1 = np.array(sol_list1)
   
    print('complete : ',opt)
    return sol_arr1

@dataclass
class config():
    init: float = 5.0
    iter_num: int = 100
    sample_num: int = 10
    lr_large: float = 0.1
    lr_small: float = 0.01
    p: float = 10.0
    delta: float = 0.1
    r1: float = 1.0
    r2: float = 10.0