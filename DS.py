import warnings
import numpy as np
import pandas as pd
import scipy as sp
import scipy.stats as sps
import scipy.linalg as spl
import matplotlib.pyplot as plt
import cvxopt as cvx
import cvxopt.solvers as solv

expit = lambda x: 1/(1+np.exp(-x))
def add_intercept(x_vars,y_var,df):
    newdf = df.copy()
    x_vars = ['(intercept)'] + x_vars
    newdf['(intercept)'] = np.ones((df.shape[0],))
    return(x_vars,y_var,newdf)
def newton(gradient,hessian,init_x:np.ndarray,max_reps:int=100,tolerance:float=1e-6):
    x = init_x.copy()
    for i in range(max_reps):
        update = -np.linalg.solve(hessian(x),gradient(x))
        x += update
        if np.abs(update).sum()<tolerance:
            return (x,i)
    raise Exception('Newton did not converge')
def gradient_descent(func,gradient,init_x:np.ndarray,learning_rate:float=0.05,max_reps:int=1000,maximize=False):
    x = init_x.copy()
    for i in range(max_reps):
        gx = gradient(x)
        x0 = x.copy()
        flast = func(x)
        x += gx*learning_rate if maximize else -gx*learning_rate
        if (func(x)<flast and maximize and i>2) or (func(x)>flast and (not maximize) and i>2): 
            x = x0
            break
    return x
def solve_lasso(x,y,thresh):
    n,r = x.shape
    P = np.kron(np.array([[1,-1],[-1,1]]),x.T@x)
    q = -np.kron(np.array([[1],[-1]]),x.T@y.reshape(-1,1))
    G_1 = -np.eye(2*r)
    h_1 = np.zeros((2*r,1))
    G_2 = np.ones((1,2*r))
    h_2 = np.array([[thresh]])
    G = np.vstack((G_1,G_2))
    h = np.vstack((h_1,h_2))
    opt = solv.qp(cvx.matrix(P),cvx.matrix(q),cvx.matrix(G),cvx.matrix(h))
    opt = np.array(opt['x'])
    return opt[:r,0]-opt[r:,0]
def mean_squared_error(model,x,y):
    return ((y-model.predict(x))**2).mean()
def k_fold_cross_validation(model,x,y,folds:int=5,seed=None,statistic=mean_squared_error):
    n,r = x.shape
    deck = np.arange(n)
    outp = []
    if seed is not None: np.random.seed(seed)
    np.random.shuffle(deck)
    for i in range(folds):
        test = deck[int(i*n/folds):int((i+1)*n/folds)]
        train_lower = deck[:int(i*n/folds)]
        train_upper = deck[int((i+1)*n/folds):]
        train = np.concatenate((train_lower,train_upper))
        modl = model(x[train],y[train])
        mspe = statistic(modl,x[test],y[test])
        outp += [mspe]
    return np.array(outp)

class learning_model(object):
    def __init__(self,x):
        self.x = x
        (self.n_obs,self.n_feat) = self.x.shape
class unsupervised_model(learning_model):
    pass
class supervised_model(learning_model):
    def __init__(self,x,y):
        super(supervised_model,self).__init__(x)
        self.y = y
        
class prediction_model(supervised_model):
    def predict(self,target): raise NotImplementedError()
    @property
    def fitted(self): return self.predict(self.x)
    @property
    def residuals(self): return self.y-self.fitted
class clustering_model(unsupervised_model):
    def cluster(self,target): raise NotImplementedError
        
        
        
        
        
class likelihood_model(learning_model):
    def evaluate_lnL(self,pred): raise NotImplementedError
    @property
    def lnL(self): return self.evaluate_lnL(self.fitted)
    @property
    def aic(self): return 2*self.n_feat-2*self.lnL
    @property
    def bic(self): return np.log(self.n_obs)*self.n_feat-2*self.lnL
    @property
    def deviance(self): return 2*self.lnL-2*self._null_lnL_()
    def _gradient_(self,coef): raise NotImplementedError
    def _hessian_(self,coef): raise NotImplementedError
    def _null_lnL_(self): return self.evaluate_lnL(np.ones(self.y.shape)*self.y.mean())
    def __vcov_params_lnL__(self): return -np.linalg.inv(self._hessian_(self.params))
    def __max_likelihood__(self,init_params,gradient=None,hessian=None):
        if gradient==None: gradient=self._gradient_
        if hessian==None: hessian=self._hessian_
        return newton(gradient,hessian,init_params)

class linear_model(prediction_model,likelihood_model):
    def __init__(self,x,y,*args,**kwargs):
        super(linear_model,self).__init__(x,y)
        self.params = self.__fit__(x,y,*args,**kwargs)
    def __fit__(self,x,y,*args,**kwargs): return np.linalg.solve(x.T@x,x.T@y)
    def predict(self,target): return target@self.params
    def evaluate_lnL(self,pred): return -self.n_obs/2*(np.log(2*np.pi*(self.y-pred).var())+1)
    @property
    def r_squared(self):
        return 1-self.residuals.var()/self.y.var()
    @property
    def adjusted_r_squared(self):
        return 1-(1-self.r_squared)*(self.n_obs-1)/(self.n_obs-self.n_feat)
    @property
    def degrees_of_freedom(self):
        return self.n_obs-self.n_feat
    @property
    def ssq(self):
        return self.residuals.var()*(self.n_obs-1)/self.degrees_of_freedom
class least_squares_regressor(linear_model):
    def __init__(self,x,y,white:bool=False,hc:int=3,*args,**kwargs):
        super(least_squares_regressor,self).__init__(x,y,*args,**kwargs)
        self.white = white
        self.hc = hc
    @property
    def vcov_params(self): 
        if self.white: 
            return self._white_(self.hc)
        return np.linalg.inv(self.x.T@self.x)*self.residuals.var()
    def _white_(self,hc):
        e = self.resid.values.reshape(-1,1) if type(e)==pd.Series else self.resid
        e = self.__hc_correction__(e**2,hc)
        meat = np.diagflat(e.flatten())
        bread = np.linalg.inv(self.x.T@self.x)@self.x.T
        return bread@meat@bread.T
    def __hc_correction__(self,ressq,hc):
        mx = 1-np.diagonal(self.x@np.linalg.solve(self.x.T@self.x,self.x.T)).reshape(-1,1)
        p = int(np.round((1-mx).sum()))
        if hc==1: ressq *= self.n_obs/(self.n_obs-self.n_feat)
        elif hc==2: ressq /= mx
        elif hc==3: ressq /= mx**2
        elif hc==4: 
            delta = 4*np.ones((self.n_obs,1))
            delta = np.hstack((delta,self.n_obs*(1-mx)/p))
            delta = delta.min(1).reshape(-1,1)
            ressq /= np.power(mx,delta)
        elif hc==5:
            delta = max(4,self.n_obs*0.7*(1-mx).max()/p)*np.ones((self.n_obs,1))
            delta = np.hstack((delta,self.n_obs*(1-mx)/p))
            delta = delta.min(1).reshape(-1,1)/2
            ressq /= np.power(mx,delta)
        return ressq    
class logistic_regressor(linear_model):
    def __fit__(self,x,y):
        params,self.iters = self.__max_likelihood__(np.zeros(self.n_feat))
        return params
    @property
    def vcov_params(self):return self.__vcov_params_lnL__()
    def evaluate_lnL(self,pred):return self.y.T@np.log(pred)+(1-self.y).T@np.log(1-pred)
    def _gradient_(self,coefs):return self.x.T@(self.y-expit(self.x@coefs))
    def _hessian_(self,coefs):
        Fx = expit(self.x@coefs)
        return -self.x.T@np.diagflat((Fx*(1-Fx)).values)@self.x
    def predict(self,target):return expit(target@self.params)
class l1_regularization_regressor(least_squares_regressor):
    def __init__(self,x,y,thresh:float,*args,**kwargs):
        super(l1_regularization_regressor,self).__init__(x,y,thresh=thresh,*args,**kwargs)
        self.threshold=thresh
    def __fit__(self,x,y,thresh:float,*args,**kwargs):
        if x[:,0].var()==0:
            dx = x[:,1:]-x[:,1:].mean(0)
            dy = y-y.mean(0)
            outp = solve_lasso(dx,dy,thresh)
            intc = y.mean(0)-x[:,1:].mean(0)@outp.reshape(-1,1)
            return np.concatenate([intc,outp])
        else:
            return solve_lasso(x,y,thresh)
class l1_cross_validation_regressor(l1_regularization_regressor):
    def __init__(self,x,y,max_thresh=None,folds:int=5,statistic=mean_squared_error,seed=None,*args,**kwargs):
        default_state = solv.options.get('show_progress',True)
        solv.options['show_progress'] = False
        if max_thresh==None: max_thresh = np.abs(least_squares_regressor(x,y).params[1:]).sum()
        outp = []
        for lam in np.linspace(0,1,100):
            model = lambda x,y: l1_regularization_regressor(x,y,thresh=lam*max_thresh)
            mse = k_fold_cross_validation(model,x,y,folds=folds,statistic=statistic,seed=seed).mean()
            outp += [(mse,lam)]
        outp = np.array(outp)
        lam = outp[outp[:,0].argmin(),1]
        thresh = lam*max_thresh
        solv.options['show_progress'] = default_state
        super(l1_cross_validation_regressor,self).__init__(x,y,thresh=thresh,*args,**kwargs)
        self.max_threshold = max_thresh
        self.lambda_value = lam
class tree_based_model(prediction_model):
    def __init__(self,x,y,level:str='',max_level=None,random_x:bool=False,prune:bool=True,conf_level=0.95):
        super(tree_based_model,self).__init__(x,y)
        if max_level is not None and len(level)>=max_level:
            self.__init_terminal__(y)
            return
        xvars = np.random.permutation(self.n_feat)[:int(np.sqrt(self.n_feat))] if random_x else np.arange(self.n_feat)
        RSS = [self.__calc_RSS_and_split__(x[:,i]) for i in xvars]
        RSS = [i for i in RSS if i is not None]
        RSS = np.array(RSS)
        if len(RSS)==0: 
            self.__init_terminal__(y)
            return
        self.split_variable = RSS[:,0].argmin()
        self.RSS = RSS[self.split_variable,0]
        self.split_value = RSS[self.split_variable,1]
        dummy = (x[:,self.split_variable]>=self.split_value).astype(int)
        if dummy.var(0) == 0:
            self.lower = None
            self.upper = None
            return
        model = least_squares_regressor(np.hstack((np.ones((self.n_obs,1)),dummy.reshape(-1,1))),y)
        self.statistic = model.params[1]/np.sqrt(model.vcov_params[1,1])
        self.p_value = 2*sp.stats.t.cdf(-np.abs(self.statistic),df=self.n_obs-2)
        if prune and self.p_value>(1-conf_level)/2**(len(level)+1):
            self.lower = None
            self.upper = None
            return
        self.lower = tree_based_model(x[dummy==0,:],y[dummy==0],level+'L',
                                      max_level=max_level,random_x=random_x,conf_level=conf_level)
        self.upper = tree_based_model(x[dummy==1,:],y[dummy==1],level+'R',
                                      max_level=max_level,random_x=random_x,conf_level=conf_level)
    def __init_terminal__(self,y):
        self.split_variable = None
        self.RSS = ((y-y.mean())**2).sum()
        self.split_value = None
        self.lower = None
        self.upper = None
    def __calc_RSS_and_split__(self,x):
        y = self.y
        outmat = []
        for item in np.unique(x):
            lowery = y[x<item].astype(float)
            uppery = y[x>=item].astype(float)
            if lowery.shape[0]<1 or uppery.shape[0]<1: continue
            lowery -= lowery.mean()
            uppery -= uppery.mean()
            RSS = (lowery**2).sum()+(uppery**2).sum()
            outmat += [(RSS,item)]
        outmat = np.array(outmat)
        if len(outmat)==0: return
        return outmat[outmat[:,0].argmin(0),:]
    def predict(self,target): 
        if self.lower == None and self.upper == None:
            return np.full(shape=target.shape[0],fill_value=self.y.mean())
        outmat = np.zeros(target.shape[0])
        lowerx = target[target[:,self.split_variable]<self.split_value]
        upperx = target[target[:,self.split_variable]>=self.split_value]
        outmat[target[:,self.split_variable]<self.split_value] = self.lower.predict(lowerx)
        outmat[target[:,self.split_variable]>=self.split_value] = self.upper.predict(upperx)
        return outmat

    
class plot_model(object):
    @property
    def plot(self): raise NotImplementedError
class broom_model(learning_model):
    @property
    def vcov_params(self)->np.ndarray: raise NotImplementedError()
    def _glance_dict_(self)->dict: raise NotImplementedError()
    @property
    def std_error(self): return np.sqrt(np.diagonal(self.vcov_params))
    @property
    def t(self): return self.params/self.std_error
    @property
    def p(self): return 2*sp.stats.t.cdf(-np.abs(self.t),df=self.n_obs-self.n_feat)
    def conf_int(self,level:float):
        spread = -self.std_error*sp.stats.t.ppf((1-level)/2,df=self.n_obs-self.n_feat)
        return np.vstack([self.params-spread,self.params+spread])
    @property
    def tidy(self,ci=False,level=0.95):return self.tidyci(ci=False)
    def tidyci(self,level=0.95,ci=True):
        n = len(self.x_vars)
        df = [self.x_vars,self.params[:n],self.std_error[:n],self.t[:n],self.p[:n]]
        cols = ['variable','estimate','std.error','t.statistic','p.value']
        if ci:
            df += [self.conf_int(level)[:,:n]]
            cols += ['ci.lower','ci.upper']
        df = pd.DataFrame(np.vstack(df).T)
        df.columns = cols
        return df
    @property
    def glance(self):
        return pd.DataFrame(self._glance_dict_(),index=[''])
    @property
    def augment(self):
        return pd.concat((self.data[self.x_vars],
                          self.data[self.y_var],
                          self.fitted,
                          self.residuals,
                          sp.stats.zscore(self.residuals)),1)
class LeastSquaresRegressor(least_squares_regressor,broom_model,plot_model):
    def __init__(self,x_vars:list,y_var:str,data:pd.DataFrame,*args,**kwargs):
        super(LeastSquaresRegressor,self).__init__(data[x_vars],data[y_var],*args,**kwargs)
        self.x_vars = x_vars
        self.y_var = y_var
        self.data = data
    def _glance_dict_(self):
        return {'r.squared':self.r_squared,
                'adjusted.r.squared':self.adjusted_r_squared,
                'self.df':self.n_feat,
                'resid.df':self.degrees_of_freedom,
                'aic':self.aic,
                'bic':self.bic,
                'log.likelihood':self.lnL,
                'deviance':self.deviance,
                'resid.var':self.ssq}
    @property
    def plot(self):
        figs = self._gen_diagnostic_plots_()
        figs.update(self._gen_variable_plots_())
        return figs
    def _gen_diagnostic_plots_(self):
        figs = {}
        plt.ioff()
        for i in range(4):
            if i == 0: x,y = self.fitted,self.residuals
            if i == 1: x,y = self.fitted,self.y
            if i == 2: x,y = self.fitted,np.sqrt(sp.stats.zscore(self.residuals)**2)
            if i == 3: 
                x = sps.norm.ppf(np.linspace(1/(y.shape[0]+2),1-1/(y.shape[0]+2),y.shape[0]))
                y = sp.stats.zscore(self.residuals)
                y.sort()
            xbnds = np.linspace(x.min(),x.max(),1000)
            ybnds = least_squares_regressor(ostk(x),y.values if type(y)==pd.Series else y).predict(ostk(xbnds))
            fig = plt.figure()
            plt.scatter(x,y)
            plt.plot(xbnds,ybnds,color="black")
            if i == 0: 
                plt.xlabel('fitted values')
                plt.ylabel('residuals')
                figs['fit.resid'] = fig
            if i == 1: 
                plt.xlabel('fitted values')
                plt.ylabel('actual y values')
                figs['fit.actual'] = fig
            if i == 2: 
                plt.xlabel('fitted values')
                plt.ylabel('absolute residual z score')
                figs['fit.sq_resid'] = fig
            if i == 3: 
                plt.xlabel('normal quantiles')
                plt.ylabel('residual z scores')
                figs['QQ.plot'] = fig
        return figs
    def _gen_variable_plots_(self):
        figs = {}
        plt.ioff()
        for i in range(self.n_feat):
            if self.x_vars[i].strip()=='(intercept)':
                continue
            fig = plt.figure()
            x = self.x.values[:,i]
            null_x = np.ones([x.shape[0],1])@self.x.values.mean(0).reshape(1,-1)
            y = null_x.copy()
            y[:,i] = x
            y = self.predict(y)
            xbnds = np.linspace(x.min(),x.max(),1000)
            ybnds = least_squares_regressor(ostk(x),y).predict(ostk(xbnds))
            plt.scatter(x,self.y)
            plt.plot(xbnds,ybnds,color="black")
            plt.ylabel(self.y_var)
            plt.xlabel(self.x_vars[i])
            figs[self.x_vars[i]] = fig
        return figs
class LogisticRegressor(logistic_regressor,broom_model):
    def __init__(self,x_vars:list,y_var:str,data:pd.DataFrame,*args,**kwargs):
        super(LogisticRegressor,self).__init__(data[x_vars],data[y_var],*args,**kwargs)
        self.x_vars = x_vars
        self.y_var = y_var
        self.data = data
    def _glance_dict_(self):
        return {'mcfadden.r.squared':self.r_squared,
                'adjusted.r.squared':self.adjusted_r_squared,
                'self.df':self.n_feat,
                'resid.df':self.degrees_of_freedom,
                'aic':self.aic,
                'bic':self.bic,
                'log.likelihood':self.lnL,
                'deviance':self.deviance,
                'resid.var':self.ssq}