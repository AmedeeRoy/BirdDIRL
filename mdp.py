import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import dok_matrix

class MDP:
    """
    Grid world environment
    """

    def __init__(self, length_max, height, width, start_end_pos):
        """
            input:
            height - idx : height of the spatial grid
            width - idx : width of the spatial grid
            length - idx : temporal length of a trip

            start_pos 2-tuple : coordinates within the state_space (height x width)

        """
        # dimensions
        self.height = height
        self.width = width
        self.length_max = length_max
        self.n_states = self.height*self.width*self.length_max

        # goal
        self.start = (0, start_end_pos[0], start_end_pos[1])
        self.start_idx = self.state2idx(self.start)
        self.end = (length_max-1, start_end_pos[0], start_end_pos[1])
        self.end_idx = self.state2idx(self.end)

        # actions
        self.actions = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        self.n_actions = len(self.actions)
        self.neighbors = [(0, 0),(-1, 0),(-1, 1),(0, 1),(1, 1),(1, 0),(1, -1),(0, -1),(-1, -1)]
        self.dirs = {0: 'stay', 1: 'n', 2: 'ne', 3: 'e', 4: 'se', 5: 's', 6: 'sw', 7: 'w', 8: 'nw'}

    def get_grid_idx(self):
        return np.array(range(self.n_states)).reshape((self.length_max, self.height, self.width))

    def get_list_state(self):
        return [(i,j,k) for i in range(self.length_max) for j in range(self.height) for k in range(self.width)]

    def state2idx(self, state):
        """
        input:
          2d state
        returns:
          1d index
        """
        return self.get_grid_idx()[state]

    def idx2state(self, idx):
        """
        input:
          1d idx
        returns:
          2d state
        """
        return self.get_list_state()[idx]

    def get_next_state(self, state, action):
        """
        get next state with [action] on [state]
        args
          state     (z, y, x)
          action    int
        returns
          new state
        """
        if state[0] >= self.length_max-1:
            return state
        else :
            inc = self.neighbors[action]
            nei_s = (state[1] + inc[0], state[2] + inc[1])
            if nei_s[0] >= 0 and nei_s[0] < self.height and nei_s[1] >= 0 and nei_s[1] < self.width:
                next_state = (state[0] + 1, nei_s[0], nei_s[1])
            else:
                next_state = (state[0] + 1, state[1], state[2])
            return next_state

    def get_list_previous_state(self, state):
        """
        args
          state     (z, y, x)
        returns
          tuple
              - previous state (z, y, x)
              - associated action int
        """
        previous = []
        for a in self.actions:
            inc = self.neighbors[a]
            nei_s = (state[1] - inc[0], state[2] - inc[1])

            if nei_s[0] >= 0 and nei_s[0] < self.height and nei_s[1] >= 0 and nei_s[1] < self.width:
                previous_state = (state[0] - 1, nei_s[0], nei_s[1])
                previous.append((previous_state,a))
        return previous

    def get_transition_mat(self):
        """
        get transition dynamics of the gridworld
        return:
          P_a         NxN matrix in list of N_ACTIONS transition probabilities matrix -
                        P_a[a][s0, s1] is the transition prob of
                        landing at state s1 when taking action
                        a at state s0
        """

#         P_a = [dok_matrix((self.n_states, self.n_states), dtype='uint8') for i in range(self.n_actions)]
        P_a = [np.zeros((self.n_states, self.n_states), dtype='uint8') for i in range(self.n_actions)]

        for i in range(self.n_states):
            si = self.idx2state(i)
            for a in range(self.n_actions):
                sj = self.get_next_state(si,a)
                j = self.state2idx(sj)
                P_a[a][i, j] = 1
        return P_a

    def get_feature_mat(self, feature):
        feat_map = []
        for k in feature.keys():
            A = feature[k].reshape(self.height*self.width)
            B = np.zeros((self.height*self.width, self.length_max))
            B[:,0] = A
            f = B
            for i in range(1, self.length_max):
                B = np.zeros((self.height*self.width, self.length_max))
                B[:,i] = A
                f = np.vstack([f, B])
            feat_map.append(f)

        return np.hstack(feat_map)

    # def get_feature_mat(self, feature):
    #     f = []
    #     for k in feature.keys():
    #         A = feature[k].reshape(self.height*self.width)
    #         f.append(A)
    #     ff = np.array(f).T
    #
    #     feat_map = ff
    #     for t in range(self.length_max-1):
    #         ff[:, len(feature)-1]  = t/(self.length_max-2)
    #         feat_map = np.vstack([feat_map, ff])
    #
    #     return feat_map

    def get_trajs_idx(self, trajs):
        trajectories = []
        trajectories_idx = []

        for _,traj in trajs.items():
            t = [(self.length_max-len(traj) + i, traj[i][0], traj[i][1]) for i in range(len(traj))]
            t_idx = [self.state2idx(t[j]) for j in range(len(t))]

            trajectories.append(t)
            trajectories_idx.append(t_idx)

        return trajectories_idx

    def generate_demonstrations(self, policy, start, n_trajs=10, len_traj=5):
        """gatheres expert demonstrations
        inputs:
        policy      Nx1 matrix
        n_trajs     int - number of trajectories to generate
        rand_start  bool - randomly picking start position or not
        start_pos   2x1 list - set start position, default [0,0]
        returns:
        trajs       a list of trajectories - each element in the list is a list of Steps representing an episode
        """

        trajs = []
        for i in range(n_trajs):

            episode = []
            state = start
            idx = self.state2idx(state)
            episode.append(idx)

            # while not is_done:
            for _ in range(len_traj-1):

                act = np.random.choice(self.n_actions, p= policy[idx,:]/np.sum(policy[idx,:]))
                next_state = self.get_next_state(state, act)
                next_idx = self.state2idx(next_state)
                episode.append(next_idx)
                state = next_state
                idx = next_idx

            trajs.append(episode)
        return trajs

    def get_rewards_grid(self, rewards):
        return rewards.reshape((self.length_max, self.height, self.width))
