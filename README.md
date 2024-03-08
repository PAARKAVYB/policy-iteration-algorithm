# POLICY ITERATION ALGORITHM

## AIM:
To develop a Python program to find the optimal policy for the given MDP using the policy iteration algorithm.

## PROBLEM STATEMENT:
The problem statement is a Five stage slippery walk where there are five stages excluding goal and hole.The problem is stochastic thus doesnt allow transition probability of 1 for each action it takes.It changes according to the state and policy.

### STATE:
The states include two terminal states: 0-Hole[H] and 6-Goal[G].

It has five non terminal states including starting state.

### TRANSITION pROBABILITY;
The transition probabilities for the problem statement is:

50% - The agent moves in intended direction.

33.33% - The agent stays in the same state.

16.66% - The agent moves in orthogonal direction.

### REWARDS:
The agent receives a reward of +1 for reaching the goal state (State 7). The agent receives a reward of 0 for all other states (State 0 - State 6).

### GRAPHICAL REPRESENTATION:
![EXP3](https://github.com/PAARKAVYB/policy-iteration-algorithm/assets/93509383/a02e0135-89d2-4134-bea8-943077b0c0dd)

## POLICY ITERATION ALGORITHM:
The algorithm implemented in the policy_iteration is a method used to find the optimal policy in a Markov decision process (MDP). Here's a step-by-step explanation of the algorithm:

Initialize the policy pi. In this implementation, a random action is chosen for each state s in the MDP P. The initial policy is represented by the lambda function pi=lambda s:{s:a for s,a in enumerate(random_actions)}[s], where random_actions is a list of randomly chosen actions for each state.

Enter a loop that continues until the policy pi is no longer changing. This is determined by comparing the previous policy (old_pi) with the current policy computed in the loop.

Store the previous policy as old_pi for comparison later.

Perform policy evaluation using the function policy_evaluation. This step calculates the state-values (V) for each state s given the current policy pi. The state-values represent the expected cumulative rewards starting from state s following policy pi and discounting future rewards by a factor of gamma. The function policy_evaluation is called with the arguments pi, P, gamma, and theta.

Perform policy improvement using the function policy_improvement. This step updates the policy pi based on the current state-values V. The function policy_improvement is called with the arguments V, P, and gamma.

Check if the policy has converged by comparing the previous policy old_pi with the current policy {s:pi(s) for s in range(len(P))}. If they are the same for all states s, the loop is exited.

Return the final state-values V and the optimal policy pi.

To summarize, policy iteration iteratively improves the policy by alternating between policy evaluation and policy improvement steps until convergence is reached. The algorithm guarantees to find the optimal policy for the given MDP P with a discount factor gamma.

## PROGRAM:
```
Developed by : Paarkavy B
Reg No : 212221230072
```
### POLICY IMPROVEMENT FUNCTION:
```
def policy_improvement(V, P, gamma=1.0):
    Q = np.zeros((len(P), len(P[0])), dtype=np.float64)
    # Write your code here to implement policy improvement algorithm
    for s in range(len(P)):
      for a in range(len(P[s])):
        for prob, next_state,reward, done in P[s][a]:
          Q[s][a]+= prob*(reward+gamma*V[next_state]*(not done))
          new_pi = lambda s: {s:a for s, a in enumerate(np.argmax(Q, axis=1))}[s]

    return new_pi
```

### POLICY ITERATION FUNCTION:
```
def policy_iteration(P, gamma=1.0,theta=1e-10):
  random_actions=np.random.choice(tuple(P[0].keys()),len(P))
  pi = lambda s: {s:a for s, a in enumerate(random_actions)}[s]
  while True:
    old_pi = {s:pi(s) for s in range(len(P))}
    V = policy_evaluation(pi, P,gamma,theta)
    pi = policy_improvement(V,P,gamma)
    if old_pi == {s:pi(s) for s in range(len(P))}:
      break
  return V,pi
```

## OUTPUT:
### OPTIMAL POLICY:
![op1](https://github.com/PAARKAVYB/policy-iteration-algorithm/assets/93509383/292fd7fd-2161-4f92-b20c-aad506b2ae78)

### OPTIMAL VALUE FUNCTION:
![op2](https://github.com/PAARKAVYB/policy-iteration-algorithm/assets/93509383/0c59bb91-50cf-4ac1-a0d4-fcfecd7ac682)

### SUCCESS RATE:
![op3](https://github.com/PAARKAVYB/policy-iteration-algorithm/assets/93509383/d9125322-292a-49b2-abfa-6ca1b19e280e)

## RESULT:
Thus, a program is developed to perform policy iteration for the given MDP.
