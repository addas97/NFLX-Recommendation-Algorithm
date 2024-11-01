import numpy as np
import kmeans
import common
import naive_em
import em

X = np.loadtxt("toy_data.txt")
X_netflix = np.loadtxt("netflix_incomplete.txt")

# Initilizations 
K = [1, 2, 3, 4]
seeds = [0, 1, 2, 3, 4]
K_em = [1, 12]

def seed_test(X, K, seeds):

    '''
    Test various seeds

    Args:
        X: dataset of size (n,d)
        K: list of clusters
        seeds: list of seeds to test
    '''
    '''
    total_cost = []
    costs = []

    for k in range(1, len(K) + 1):
        for s in range(len(seeds)):
            mixture, post = common.init(X, k, s)
            cost = kmeans.run(X, mixture, post)[2]
            costs.append(cost)
        total_cost.append(min(costs))
    '''
    
    total_cost = [min(kmeans.run(X, *common.init( X, k, s))[2] for s in seeds)
                  for k in K]
    
    log_likelihoods = [max(naive_em.run(X, *common.init(X, k, s))[2] for s in seeds)
                  for k in K]
    
    em_log_likelihoods = [max(em.run(X, *common.init(X, k, s))[2] for s in seeds) for k in K_em]

    return total_cost, log_likelihoods, em_log_likelihoods

def find_best_seed(X, seeds):
    ll = []
    for s in seeds:
        initialMixture, initialPost = common.init(X,4,s)
        mixtureEM, postEM, lls = naive_em.run(X,initialMixture,initialPost)
        ll.append(lls)
    return ll, ll.index(max(ll))

#print(find_best_seed(X, seeds)[0])
#print(find_best_seed(X, seeds)[1])

# for K = 1, seed = 0
# for K = 2, seed = 2
# for K = 3, seed = 0
# for K = 4, seed = 4

for k in K:
    for s in seeds:
        mixture, post = common.init(X, 3, seed=s)

        # Run the EM algorithm
        final_mixture, final_post, log_likelihood = naive_em.run(X, mixture, post)

        # Calculate BIC
        bic_value = common.bic(X, final_mixture, log_likelihood)

        print(f"Best BIC for K={k}, seed value s = {s}: {bic_value}")

def test_em_seeds(X, K):
	print("\n############## EM K=" + str(K) + " ###############")

	mixture0, post0 = common.init(X,K,0)
	mixture1, post1 = common.init(X,K,1)
	mixture2, post2 = common.init(X,K,2)
	mixture3, post3 = common.init(X,K,3)
	mixture4, post4 = common.init(X,K,4)

	cost0 = em.run(X,mixture0,post0)[2]
	cost1 = em.run(X,mixture1,post1)[2]
	cost2 = em.run(X,mixture2,post2)[2]
	cost3 = em.run(X,mixture3,post3)[2]
	cost4 = em.run(X,mixture4,post4)[2]

	print("K=" + str(K) + " seed=0 : likelihood=" + str(cost0))
	print("K=" + str(K) + " seed=1 : likelihood=" + str(cost1))
	print("K=" + str(K) + " seed=2 : likelihood=" + str(cost2))
	print("K=" + str(K) + " seed=3 : likelihood=" + str(cost3))
	print("K=" + str(K) + " seed=4 : likelihood=" + str(cost4))

X_netflix = np.loadtxt("netflix_incomplete.txt")
test_em_seeds(X_netflix, 1)
test_em_seeds(X_netflix, 12)

X_gold = np.loadtxt('netflix_complete.txt')
mixture4, post4 = common.init(X_netflix,12,1)
mixture, post, cost4 = em.run(X_netflix,mixture4,post4)
X_pred = em.fill_matrix(X_netflix,mixture)

rmse_result = common.rmse(X_gold, X_pred)
print("RMSE between prediction and GOLD is : " + str(rmse_result))

# ==========

# K-means: determining the centroids by comparing the cost
k_dict = dict()
total_cost_dict = dict()
for seed in range(5):
    total_cost = 0
    for k in range(1, 5):
        mixture, post = common.init(X=X, K=k, seed=seed)
        cost = kmeans.run(X, mixture, post)[2]
        total_cost += cost
        k_dict.update({(seed, k): cost})
    total_cost_dict.update({seed: total_cost})

### get the best seed and the best k size that minimizes the cost

## Best seed
# Get the lowest cost
optimal_seed_cost = total_cost_dict[0]
for k, v in total_cost_dict.items():
    if v < optimal_seed_cost:
        optimal_seed_cost = v
    else:
        optimal_seed_cost = optimal_seed_cost
# Get the seed associated with the lowest cost
for k, v in total_cost_dict.items():
    if v == optimal_seed_cost:
        optimal_seed = k

## Best k size
# Get the lowest cost
optimal_k_cost = k_dict[(optimal_seed, 1)]
# Create a new dictionary for k size
optimal_k_dict = dict()
for i in range(1, 5):
    optimal_k_dict.update({(optimal_seed, i): k_dict[(optimal_seed, i)]})
for k, v in optimal_k_dict.items():
    if v < optimal_k_cost:
        optimal_k_cost = v
    else:
        optimal_k_cost = optimal_k_cost
# Get the seed associated with the lowest cost
for k, v in optimal_k_dict.items():
    if v == optimal_k_cost:
        optimal_k = k[1]

### Plotting the k clusters
optimal_seed_k = list()
optimal_seed_k_post = list()
title_list = list()
for i in range(1, 5):
    initial_mixture, initial_post = common.init(X, i, seed = optimal_seed)
    mixture, post, cost = kmeans.run(X, initial_mixture, initial_post)
    optimal_seed_k.append(mixture)
    optimal_seed_k_post.append(post)
    title_list.append(("K-means: The mixture plot when k = {}".format(i)))

for i in range(4):
    common.plot(X, optimal_seed_k[i], optimal_seed_k_post[i], title_list[i])

####### Compare k-means with EM

# K-means: determining the centroids by comparing the cost
em_k_dict = dict()
em_total_likelihood_dict = dict()
for seed in range(5):
    em_total_likelihood = 0
    for k in range(1, 5):
        mixture, post = common.init(X=X, K=k, seed=seed)
        likelihood = naive_em.run(X, mixture, post)[2]
        em_total_likelihood += likelihood
        em_k_dict.update({(seed, k): likelihood})
    em_total_likelihood_dict.update({seed: em_total_likelihood})

### get the best seed and the best k size that minimizes the cost

## Best seed
# Get the lowest cost
optimal_seed_cost = em_total_likelihood_dict[0]
for k, v in em_total_likelihood_dict.items():
    if v > optimal_seed_cost:
        optimal_seed_cost = v
    else:
        optimal_seed_cost = optimal_seed_cost
# Get the seed associated with the lowest cost
for k, v in em_total_likelihood_dict.items():
    if v == optimal_seed_cost:
        optimal_seed = k
print(em_k_dict)

### Plotting the k clusters
em_optimal_seed_k = list()
em_optimal_seed_k_post = list()
em_title_list = list()
for i in range(1, 5):
    initial_mixture, initial_post = common.init(X, i, seed = optimal_seed)
    mixture, post, likelihood = naive_em.run(X, initial_mixture, initial_post)
    em_optimal_seed_k.append(mixture)
    em_optimal_seed_k_post.append(post)
    em_title_list.append(("Gaussian Mixture: The mixture plot when k = {}".format(i)))

for i in range(4):
    common.plot(X, em_optimal_seed_k[i], em_optimal_seed_k_post[i], title_list[i])

### Seed with best BIC
BIC_k_dict = dict()
BIC_total_likelihood_dict = dict()
for seed in range(5):
    BIC_total_likehood = 0
    for k in range(1, 5):
        mixture, post = common.init(X=X, K=k, seed=seed)
        log_likelihood = naive_em.run(X, mixture, post)[2]
        BIC = common.bic(X, mixture, log_likelihood)
        BIC_total_likehood += BIC
        BIC_k_dict.update({(seed, k): BIC})
    total_cost_dict.update({seed: BIC_total_likehood})
print(BIC_k_dict)


### Determining the initialization

### run EM algorithm on X