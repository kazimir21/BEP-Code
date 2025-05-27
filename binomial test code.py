from scipy.stats import binomtest


n_trials = 120
n_successes = 80        
p_null = 0.5             

# One-sided binomial test
binom_result = binomtest(k=n_successes, n=n_trials, p=p_null, alternative='greater')
observed_proportion = n_successes / n_trials

# C.I.
z = 1.96  # for 95% CI
se = (observed_proportion * (1 - observed_proportion) / n_trials) ** 0.5
ci_lower = observed_proportion - z * se
ci_upper = observed_proportion + z * se

# Print results
print("Observed proportion:", round(observed_proportion, 3))
print("p-value:", round(binom_result.pvalue, 5))
print("95% Confidence Interval:", (round(ci_lower, 3), round(ci_upper, 3)))
