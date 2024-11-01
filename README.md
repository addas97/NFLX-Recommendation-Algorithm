# NFLX-Recommendation-Algorithm

Using a Gaussian mixture model, we build a collaborative filtering model to predict moview preferences of users from the NFLX movie rating database.

The dataset contains movie ratings made by users. Each user has rated a fraction of the total movies available, so the data is only partially filled.

We sample the user's ratings via a Gaussian mixture model, i.e., we sample a user type and then the rating profile from the Gaussian distribution associated with the type. In this way, we employ the  Expectation Maximization (EM) algorithm to estimate this mixture from a partially observed rating matrix. With this mixture, we use it to predict values for the missing entries in the data matrix.
