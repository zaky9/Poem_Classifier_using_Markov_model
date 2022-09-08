# Poem classifier using Markov Model

This repo is part of my notes/exercise from the Udemy course : [Natural Language Processing in Python (V2)](https://www.udemy.com/course/natural-language-processing-in-python/). In this repo, We take a look at how we can classify text based on which authors. A model that take input a string of text and output a prediction about which authors it belong to. Text classification is supervised learning, but Markov models are unsupervised (training data is sequence of text and no labels). Thus, we must apply Baye's rules (using Bayes Classifier) $$p(y|x) = \frac{p(x|y)p(y)}{p(x)}$$

- **Workflow for text classifier**: 
    1. train a seperate Markov model for each class (sequence of text). Where each model give us $p(x|class=k) for all k (class)
    2. Bayes rule is used to create the decision rule:
$$K^\* = arg\underset{k}{max}p(class = k|x)$$ Where k = class, x= input

    3. This posterior probability can be simplified since we dont need the actual value and we only need the $argmax$
    4. We can reduce this decision rule by using Maximum a posteriori method. By taking the sum of the log of likelihood with log of the prior: $$K^* = arg\underset{k}{max}logp(class = k|x)+logp(class=k)$$
    5. if the prior is uniform (where all classes have equal chance of being choosen) then our problem  reduce to maximum likelihood.
      maximum likelihood: $$K^* = arg\underset{k}{max}logp(class = k|x)$$
