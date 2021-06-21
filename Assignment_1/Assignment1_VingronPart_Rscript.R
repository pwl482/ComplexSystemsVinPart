#Complex Systems Assignment 1
#by Yiftach ?, Paul Vogler
#for Problem 4:

#install.packages("infotheo")
library(infotheo)

# Generate two vectors of length 300 from the continuous uniform distribution:
vec1 <- runif(300, min = 0, max = 1)
vec2 <- runif(300, min = 0, max = 1)
# Discretize the data to the bins of length 0.1:
h1 <- hist(vec1, breaks = seq(from=0, to=1, by=0.1))
h2 <- hist(vec2, breaks = seq(from=0, to=1, by=0.1))
# Compute the mutual information:
# We expect the mutual information to be 0 if the variables are independent.
# If they are completely dependent, the mutual information should converge
# towards the entropy of one of the inputs.
mutinformation(h1$counts, h2$counts, method="emp")
# output: 1.695743
entropy(h1$counts, method="emp")
# output: 1.834372
entropy(h2$counts, method="emp")
# output: 2.163956
# We got an output close to the entropies of either of the two inputs, 
# therefore they seem to be dependent variables.
# When the number of samples is increased, this also becomes more clear:

# Generate two vectors of length 300 from the continuous uniform distribution:
vec3 <- runif(10000, min = 0, max = 1)
vec4 <- runif(10000, min = 0, max = 1)
# Discretize the data to the bins of length 0.1:
h3 <- hist(vec3, breaks = seq(from=0, to=1, by=0.1))
h4 <- hist(vec4, breaks = seq(from=0, to=1, by=0.1))
# Compute the mutual information:
mutinformation(h3$counts, h4$counts, method="emp")
# output: 2.302585
entropy(h3$counts, method="emp")
# output: 2.302585
entropy(h4$counts, method="emp")
# output: 2.302585
# Here the mutual information converged to the entropy of both variables