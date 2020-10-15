import numpy as np
import matplotlib.pyplot as plt
import IPython

FLOAT_MAX = 1e30

class MaxEntIRL:
    def __init__(self, P, feat_map, trajs_idx):

        self.P = P
        self.feat_map = feat_map
        self.trajs = trajs_idx

        self.end = np.unique([traj[-1] for traj in self.trajs])

        self.n_states,_ = np.shape(self.P[0])
        self.n_actions = len(self.P)

        t = np.random.uniform(size=(self.feat_map.shape[1]))
        self.theta = t
        self.theta_history = [self.theta]

        self.error_history = []

        self.policy = None
        self.policy_history = []

        self.values = None
        self.values_history = []

        self.svf = None
        self.svf_history = []

        self.rewards = np.dot(self.feat_map, self.theta)
        self.rewards_history = [self.rewards]

    def value_iteration(self, error=0.01, max_iter=100):
        """
        static value iteration function. Perhaps the most useful function in this repo

        inputs:
        error       float - threshold for a stop
        max_iter    int - threshold for stop

        returns:
        values    Nx1 matrix - estimated values
        policy    Nx1 matrix - policy
        """

        n = 0
        values = np.ones([self.n_states])* -FLOAT_MAX
        qvalues = np.ones((self.n_states, self.n_actions))* -FLOAT_MAX
        policy = np.zeros((self.n_states, self.n_actions))

        # estimate values
        while True:
            values_tmp = values.copy()
            values[self.end] = 0 # goal

            for s in range(self.n_states):
                qvalues[s] = [sum([self.P[a][s, s1]*(self.rewards[s] + values[s1]) for s1 in range(self.n_states)]) for a in range(self.n_actions)]

                softmax = max(qvalues[s]) + np.log(1.0 + np.exp(min(qvalues[s]) - max(qvalues[s])))

                values[s] = self.rewards[s] + softmax
                policy[s,:] = np.exp(qvalues[s]-values[s])/sum(np.exp(qvalues[s]-values[s]))

            if max([abs(values[s] - values_tmp[s]) for s in range(self.n_states)]) < error:
                break
            n += 1
            # max iteration
            if n > max_iter:
                print("    WARNING: max number of iterations", max_iter)
                break

        self.values = values
        self.values_history.append(values)
        self.policy = policy
        self.policy_history.append(policy)

    def compute_state_visition_freq(self, nb_step=50):
        """compute the expected states visition frequency p(s| theta, T)
        using dynamic programming
        inputs:
        P_a     NxNxN_ACTIONS matrix - transition dynamics
        gamma   float - discount factor
        start_idx   idx of start position
        nb_step idx - nb of step to iterate
        policy  Nx1 vector - policy

        returns:
        p       Nx1 vector - state visitation frequencies
        """
        # mu[s, t] is the prob of visiting state s at time t
        mu = np.zeros([self.n_states, nb_step])
        for traj in self.trajs:
            mu[traj[0], 0] += 1
        mu[:,0] = mu[:,0]/len(self.trajs)

        for s in range(self.n_states):
            for t in range(nb_step-1):
                mu[s, t+1] = sum([sum([mu[pre_s, t]*self.P[a1][pre_s, s]*self.policy[pre_s, a1] for a1 in range(self.n_actions)]) for pre_s in range(self.n_states)])

        p = np.sum(mu, 1)
        self.svf = p
        self.svf_history.append(p)

    def maxent_irl(self, lr=0.05, error=0.1, max_iter=10):
        """
        Maximum Entropy Inverse Reinforcement Learning (Maxent IRL)
        inputs:
        feat_map    NxD matrix - the features for each state
        P_a         NxNxN_ACTIONS matrix - P_a[s0, s1, a] is the transition prob of
                                           landing at state s1 when taking action
                                           a at state s0
        gamma       float - RL discount factor
        trajs       a list of demonstrations
        lr          float - learning rate
        n_iters     int - number of optimization steps
        returns
        rewards     Nx1 vector - recoverred state rewards
        """
        # calc feature expectations
        feat_exp = np.zeros([self.feat_map.shape[1]])
        for episode in self.trajs:
            for step in episode:
                feat_exp += self.feat_map[step,:]
        feat_exp = feat_exp/len(self.trajs)

        n = 0
        # training
        while True:

            if n % (max_iter/20) == 0:
                print('iteration: {}/{}'.format(n, max_iter))

            # compute reward function
            self.rewards = np.dot(self.feat_map, self.theta)

            # compute policy
            self.value_iteration(error=0.01, max_iter=100)

            # compute state visition frequences
            self.compute_state_visition_freq(nb_step=50)

            # compute gradients
            grad = feat_exp - self.feat_map.T.dot(self.svf)

            # update params
            t = self.theta.copy()
            # t *= np.exp(lr * grad)
            # t /= np.sum(t)
            t += lr * grad
            self.theta = t
            self.theta_history.append(t)

            self.error_history.append(sum(grad**2))

            plt.plot(self.error_history)
            IPython.display.clear_output(wait=True)
            IPython.display.display(plt.show())

            if sum(grad**2) < error:
                break
            # max iteration
            if n > max_iter:
                print("    WARNING: max number of iterations", max_iter)
                break

            n += 1

        self.rewards = np.dot(self.feat_map, self.theta)
