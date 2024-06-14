# Project Palooza

## causal inference series

fitting linear regression models is 3 lines of code, but what are we actually doing? from a pure ML point of view, you're mainly focused on building a model `f` that minimizes the difference between predicted and observed values, probably to build some kind of decision system `d`. 

`predicted outcome = f(data)`

`decision = d(predicted outcome)`

if making correct decisions depends on having good predictions, then whether or not the relationship `outcome ~ data` is *causal* is kind of an afterthought.

i'm being a little unfair to machine learning here, but the point i'm making is this:

**building an error minimizing model is not the same as investigating causal relationships**

and the latter is really important sometimes...especially if you're designing *policy* or influencing *strategy*

so we will explore how to do causal inference. the perspective we will take is heavily influenced by econometrics (data analysis the way it is taught to economists).

### high-level setup:

reminder: correlation (more generally, *association*) is not causation

...unless...maybe.. it is.

suppose we have an intervention/experimental condition $X \in \{0,1\}$

and a numeric outcome variable $Y$

deep down, we want to believe that receiving the intervention makes $Y$ increase (or decrease, it depends what $Y$ is).

we can easily calculate the difference between average $Y$ under intervention ($X = 1$) vs. no intervention ($X = 0$).

`avg(Y | X = 1) - avg(Y | X = 0)`

if the difference is greater than 0 then it's time to pull out the champagne, right?

not so fast - there's two issues we need to address first:

- how do we know the people in ($X = 1$) and ($X = 0$) are not fundamentally different and incomparable?
- even if they are comparable, how do we know the difference we see is not due to chance?



