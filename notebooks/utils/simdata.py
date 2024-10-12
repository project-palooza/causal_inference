import pandas as pd
import numpy as np

def email_collections(n=5000):
    """https://github.com/matheusfacure/python-causality-handbook/blob/master/Simulated-Data.ipynb"""
    np.random.seed(24)
    email = np.random.binomial(1, 0.5, n)

    credit_limit = np.random.gamma(6, 200, n)
    risk_score = np.random.beta(credit_limit, credit_limit.mean(), n)

    opened = np.random.normal(5 + 0.001*credit_limit - 4*risk_score, 2)
    opened = (opened > 4).astype(float) * email


    agreement = np.random.normal(30 +(-0.003*credit_limit - 10*risk_score), 7) * 2 * opened
    agreement = (agreement > 40).astype(float)

    payments = (np.random.normal(500 + 0.16*credit_limit - 40*risk_score + 11*agreement + email, 75).astype(int) // 10) * 10

    data = pd.DataFrame(dict(payments=payments,
                            email=email,
                            opened=opened,
                            agreement=agreement,
                            credit_limit=credit_limit,
                            risk_score=risk_score))
    return data

def hospital_treatment(n=80):
    """https://github.com/matheusfacure/python-causality-handbook/blob/master/Simulated-Data.ipynb"""
    np.random.seed(24)
    hospital = np.random.binomial(1, 0.5, n)

    treatment = np.where(hospital.astype(bool),
                        np.random.binomial(1, 0.9, n),
                        np.random.binomial(1, 0.1, n))

    severity = np.where(hospital.astype(bool), 
                        np.random.normal(20, 5, n),
                        np.random.normal(10, 5, n))

    days = np.random.normal(15 + -5*treatment + 2*severity, 7).astype(int)

    hospital = pd.DataFrame(dict(hospital=hospital,
                                treatment=treatment,
                                severity=severity,
                                days=days))
    
    return hospital

if __name__ == "__main__":
    import sys
    dataset_name = sys.argv[1]
    if dataset_name == "collections":
        data = email_collections()
        data.to_csv(f"../../data/{dataset_name}.csv",index = False)
    else:
        data = hospital_treatment()
        data.to_csv(f"../../data/{dataset_name}.csv",index = False)