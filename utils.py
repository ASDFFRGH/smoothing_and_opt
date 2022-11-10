import matplotlib.pyplot as plt 
import numpy as np
from dataclasses import dataclass
from tqdm import tqdm
import copy

level = 0.8

def g(x, m, p):
    return (m*(x-p)**2-m*p**2)

def c(x):
    return np.where(x<200, 100*(x+100)**2, 100*(200+100)**2)
    
def objective(x, conf):
    return g(x, conf.m, conf.p)

def grad_c(x):
    return np.where(x<200, 200*(x+100), 200*(200+100))
    
def grad_g(x, conf):
    m = conf.m
    p = conf.p
    return 2*m*(x-p)

def grad_y(x, y):
    return 2*c(x)

def sgd(x, y, lr, conf):
    noise = level * np.random.uniform(-1,1)
    print(y)
    print(noise)
    x -= lr * grad_g(x, conf) + grad_c(x) * (y + lr * noise)**2
    y -= lr * grad_y(x, y)*(y + lr * noise)
    #print((y + lr * noise))
    return x, y

def averaged_sgd(x, x_ave, y, y_ave, lr, t, conf):
    beta = 1/(t+1)
    noise = level * np.random.uniform(-1,1)
    x -= lr * grad_g(x, conf) + grad_c(x) * (y + lr * noise)**2
    y -= lr * grad_y(x, y)*(y + lr * noise)
    x_ave *= beta*t
    x_ave += beta*x
    y_ave *= beta*t
    y_ave += beta*x
    return x, x_ave, y, y_ave


def optimize(configs, opt):
    sol_list1 = []
    sol_list2 = []
    for i in tqdm(range(configs.sample_num)):
        x = configs.init + 10 * np.random.randn()
        y = 0
        
        if 'Ave_SGD' in opt:
            x_ave = copy.deepcopy(x)
            y_ave = copy.deepcopy(y)

        if opt == 'SGD_large':            
            for j in range(configs.iter_num):            
                x, y = sgd(x, y, configs.lr_large, configs)
        elif opt == 'SGD_small':
            for j in range(configs.iter_num):            
                x, y = sgd(x, y, configs.lr_small, configs)
        elif opt == 'Ave_SGD_large':
            for j in range(configs.iter_num):            
                x, x_ave, y, y_ave = averaged_sgd(x, x_ave, y, y_ave, configs.lr_large, j, configs)
        elif opt == 'Ave_SGD_small':
            for j in range(configs.iter_num):            
                x, x_ave, y, y_ave = averaged_sgd(x, x_ave, y, y_ave, configs.lr_small, j, configs)
       
        else:
            print('opt is not define')
            
        if 'Ave_SGD' in opt:
            sol_list1.append(x_ave)
            sol_list2.append(y_ave)
        else:
            sol_list1.append(x)
            sol_list2.append(y)
    sol_arr1 = np.array(sol_list1)
    sol_arr2 = np.array(sol_list2)
    print('complete : ',opt)
    return sol_arr1, sol_arr2

@dataclass
class config():
    init: float = 5.0
    iter_num: int = 100
    sample_num: int = 10
    lr_large: float = 0.1
    lr_small: float = 0.00001
    rho: float = 1.0
    m: float = 5.0
    p: float = 10.0