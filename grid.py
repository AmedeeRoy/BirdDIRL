import numpy as np
import matplotlib.pyplot as plt
from itertools import groupby
import imageio
import os

class Grid:
    def __init__(self, lon, lat):

        self.lon = lon
        self.lat = lat[::-1]

        self.height = len(self.lat)
        self.width = len(self.lat)

        self.size = (len(lat), len(lon))

        self.grid = np.array([[(lon, lat) for lon in self.lon] for lat in self.lat])
        self.grid_lon = np.array([[lon for lon in self.lon] for lat in self.lat])
        self.grid_lat = np.array([[lat for lon in self.lon] for lat in self.lat])

        self.feature = {}

        self.trajs = {}


    # UTILS ----------------------------------------
    def coord2idx(self, coord):
        return (np.argmin(abs(self.lat - coord[1])), np.argmin(abs(self.lon - coord[0])))

    def idx2coord(self, idx):
        return self.grid[idx]

     # ADDING TRAJECTORY ----------------------------------------
    def add_trajectory(self, time, traj, name, save=True):
        # get position on grid
        state = [(np.argmin(abs(self.lat - traj[i,1])), np.argmin(abs(self.lon - traj[i,0])))
                 for i in range(traj.shape[0])]

        # group identical consecutive states
        grouped_state = [(k, sum(1 for i in j)) for k,j in groupby(state)]

#         # test if there is gaps in data
#         indexes = [s[1] for s in grouped_state]
#         step_length = np.zeros(len(indexes))
#         step_start = 0
#         step_end = indexes[0]
#         for i in range(len(indexes)-1):
#             step_length[i] = seconds_between( time[step_start], time[step_end])
#             step_start = np.sum(indexes[0:i+1])
#             step_end = np.sum(indexes[0:i+2])
#         # create trajectory
#         traj_new = [grouped_state[i][0] for i in range(len(grouped_state)) for j in range(int(np.round(step_length[i]/60)))]

        traj_new = [grouped_state[i][0] for i in range(len(grouped_state))]

        ## check if start = end
        if traj_new[0] == traj_new[-1] and len(traj_new)>2:
            ## continuous
            if np.max(abs(np.array(traj_new[1:len(traj_new)]) - np.array(traj_new[0:len(traj_new)-1])))<=1:
                self.trajs[name] = traj_new
                print('Adding trajectory', name, ': ok')

    def create_gif_trajectory(self, path):
        for name in self.trajs.keys():
            # test if folder exist, else create it
            if not os.path.isdir(path+'/'+name):
                os.mkdir(path+'/'+name)
            if not os.path.isdir(path+'/'+name+'/step'):
                os.mkdir(path+'/'+name+'/step')

            images = []
            n = 0
            m = np.zeros(self.size)
            # loop over each position and save associated image
            for step in self.trajs[name]:
                m = m/2
                m[step] += 1
                path_im = path+'/'+name+'/step/'+name+'_'+str(self.width)+'x'+ str(self.width)+'_'+str(n)
                # Create a new figure, plot into it, then close it so it never gets displayed
                fig = plt.figure()
                plt.imshow(m)
                plt.savefig(path_im)
                plt.close(fig)
                images.append(imageio.imread(path_im+'.png'))
                n+=1
            # save gif
            path_gif = path+'/'+name+'/'+name+'_'+str(self.width)+'x'+ str(self.width)+'.gif'
            imageio.mimsave(path_gif, images)

    def show_trajectory_density(self, name):
        grid = 0*self.grid_lon
        for step in self.trajs[name]:
            grid[step[0], step[1]] += 1
        return grid

    def show_trajectory_density_all(self):
        grid = 0*self.grid_lon
        for k,v in self.trajs.items():
            for step in v:
                grid[step[0], step[1]] += 1
        return grid

    # ADDING FEATURE MAPS ----------------------------------------
    def format_feature(self, lon, lat, data):

        lon_min = min(self.lon)
        lon_max = max(self.lon)
        lat_min = min(self.lat)
        lat_max = max(self.lat)

        j_min = np.argmin(abs(lon[:] - lon_min)) - 1
        j_max = np.argmin(abs(lon[:] - lon_max)) + 1
        i_max = np.argmin(abs(lat[:] - lat_min)) + 1
        i_min = np.argmin(abs(lat[:] - lat_max)) - 1

        y = lon[j_min:j_max]
        x = lat[i_min:i_max]
        data = data[i_min:i_max, j_min:j_max]

        feature = np.zeros(self.size)
        xx = np.array([[self.lat[np.argmin(abs(self.lat - i))] for j in y] for i in x])
        yy = np.array([[self.lon[np.argmin(abs(self.lon - j))] for j in y] for i in x])

        for i in range(len(self.lat)):
            for j in range(len(self.lon)):
                feature[i,j] = np.mean(data[(yy == self.lon[j]) * (xx == self.lat[i])])

        return feature

    def normalize_feature(self):
        for key,value in self.feature.items():
            if not np.max(value) - np.min(value) == 0:
                self.feature[key] = (value - np.max(value)) / (np.max(value) - np.min(value))
            else:
                self.feature[key] = -0.5*np.ones(self.size)


    def add_feature(self, x, y, data, name):
        self.feature[name] = self.format_feature(x,y,data)

    def show_feature(self, name):
        plt.imshow(self.feature[name])
        plt.show()

    def show_feature_all(self):
        fig, axs = plt.subplots(1,len(self.feature))
        i = 0
        for k in self.feature.keys():
            axs[i].imshow(self.feature[k])
            i += 1
        plt.show()
