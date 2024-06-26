import pandas as pd
import numpy as np
import statsmodels.stats.power as smp

# calculating pooled standard deviation
def calculate_pooled_std(nobs, std):   
    pooled_var = ((nobs[0] - 1)*std[0]**2 + (nobs[1]-1)*std[1]**2)/(nobs[0] + nobs[1] -2)
    pooled_std = np.sqrt(pooled_var)
    return pooled_std

# calculating Cohen's d to quantify the difference between means of two groups
def calculate_cohen_d(mean, pooled_std):
    d = (mean[0] - mean[1])/pooled_std
    return d
    
# calculating the power of t-test 
def calculate_power_ttest(mean, std, nobs,alpha=.05,alternative = "two-sided"):
    pooled_std = calculate_pooled_std(nobs, std)
    d = calculate_cohen_d(mean, pooled_std)
    power_ttest = smp.ttest_power(effect_size=d, nobs=nobs[0], alpha=alpha, alternative=alternative)
    return pooled_std,d,power_ttest


def main():
    cats = pd.read_csv('../cookie_cats.csv') # you might have to replace relative with absolute path

    # preparing the data
    retention_gate_30 = cats.loc[cats['version'] == 'gate_30', 'retention_7']
    retention_gate_40 = cats.loc[cats['version'] == 'gate_40', 'retention_7']
    nobs = np.array([retention_gate_30.shape[0],retention_gate_40.shape[0]])
    mean = np.array([retention_gate_30.mean(), retention_gate_40.mean()])
    std = np.array([retention_gate_30.std(), retention_gate_40.std()])

    # calculating power for t-test
    pooled_std,d,power_ttest = calculate_power_ttest(mean, std, nobs)
    
    print('Number of observations:', nobs)
    print('Mean:', mean)
    print('Standard deviation:', std)
    print('Pooled standard deviation:', round(pooled_std, 3))
    print("Cohen's d:", round(d, 3))
    print('Power for t_test:', round(power_ttest, 3)) 
      
if __name__ == '__main__':
    main()