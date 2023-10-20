import numpy as np
from numba import jit
import pandas as pd
from scipy.optimize import curve_fit
import h5py
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import trackpy as tp

# Define the Gaussian function with an additive background term
def gaussian_bg(x, mu, sigma, amplitude, background):
    return amplitude * np.exp(-((x - mu) / sigma)**2 / 2) + background

def gaussian(x, mu, sigma, amplitude):
    return amplitude * np.exp(-((x - mu) / sigma)**2 / 2)



class DNAFluctuationMSD:

    def __init__(self, h5path):
        self.h5path = h5path
    
    def read(self):
        with h5py.File(self.h5path, 'r') as h5:
            self.num_colors = h5['parameters']['Number of Colors'][...]
            self.pix_size = h5['parameters']['Pixel Size'][...] * 1e-3 # in µm
            self.frame_time = self.num_colors * h5['parameters']['Aquisition Time'][...]
            self.DNA_length = h5['parameters']['MultiPeak']['DNAlength'][...]
            self.img_arr_0 = h5['img_arr_0'][:]
            self.region_dna = h5['parameters']['region3_Loop'][:]
            self.dna_ends = h5['parameters']['dna ends'][:]
            if self.num_colors==2:
                self.img_arr_1 = h5['img_arr_1'][:]

    def animate_gauss_fitting_single_frame(self, frame=0):
        region_dna_0 = int(self.region_dna[0])
        region_dna_1 = int(self.region_dna[1])
        dna_pos_start = int(self.dna_ends[0])
        dna_pos_end = int(self.dna_ends[1])
        self.img_arr_0_dna = self.img_arr_0[region_dna_0:region_dna_1, ::]
        image = self.img_arr_0_dna[frame, ::]
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].imshow(image)
        axs[0].set_title('Image')
        ax0line = axs[0].axhline(y=dna_pos_start, color='r')
        # Define the update function for the animation
        def update(i):
            # Update the value of dna_pos_start
            dna_pos_start = int(self.dna_ends[0] + i)
            # Update the line position
            ax0line.set_ydata(dna_pos_start)
            # Clear the previous plot
            axs[1].cla()
            y = image[dna_pos_start, :]
            x = np.arange(0, len(y), 1)
            # Plot the new data
            axs[1].plot(x, y, 'k*')
            # Fit the data to the Gaussian function
            mu_0 = np.argmax(y)
            sigma_0 = 3
            amplitude_0 = np.max(y)
            background_0 = np.min(y)
            try:
                params, cov = curve_fit(gaussian_bg, x, y, p0=[mu_0, sigma_0, amplitude_0, background_0])
                mu, sigma, amplitude, background = params
                axs[1].plot(x, gaussian_bg(x, mu, sigma, amplitude, background), 'r-')
                params, cov = curve_fit(gaussian, x, y, p0=[mu_0, sigma_0, amplitude_0])
                mu, sigma, amplitude = params
                axs[1].plot(x, gaussian(x, mu, sigma, amplitude), 'b-')
                axs[1].set_title(f'Plot for dna_pos_start = {dna_pos_start}')
                axs[1].set_ylim([np.min(y), np.max(y)])
                axs[1].set_xlim([0, len(y)])
                axs[1].set_xlabel('Pixel')
                axs[1].set_ylabel('Intensity')
            except:
                print(f"Error occurred at dna_pos_start = {dna_pos_start}")
                pass
        ani = FuncAnimation(fig, update, frames=dna_pos_end-dna_pos_start, interval=100)
        ani_filename = self.h5path[:-5] + f'_gaussfit_frame{frame}.mp4'
        ani.save(ani_filename)

    def fit_gaussian(self, image, frame, skip_dna_pixs=1):
        # create an empty dataframe
        df = pd.DataFrame(columns=['frame', 'mu', 'sigma', 'amplitude', 'background'])

        # loop through the frames and fit the data to the Gaussian function
        for i in range(0, image.shape[0], skip_dna_pixs):
            dna_pos = i
            y = image[i, :]
            x = np.arange(0, len(y), 1)
            # Fit the data to the Gaussian function
            mu_0 = np.argmax(y)
            sigma_0 = 3
            amplitude_0 = np.max(y)
            background_0 = np.min(y)
            try:
                params, cov = curve_fit(gaussian_bg, x, y, p0=[mu_0, sigma_0, amplitude_0, background_0])
                mu, sigma, amplitude, background = params
            except:
                # print(f"Error occurred at frame:{frame}, dna position = {i}")
                mu, sigma, amplitude, background = mu_0, sigma_0, amplitude_0, background_0
                pass

            # add the values to the dataframe
            df = pd.concat([df, pd.DataFrame({'frame': frame, 'dna_pos': [dna_pos], 'mu': [mu], 'sigma': [sigma], 'amplitude': [amplitude], 'background': [background]})], ignore_index=True)

        # return the dataframe
        return df

    def fit_gaussian_all_frames(self, skip_dna_pixs=1, rewrite=False):
        with h5py.File(self.h5path, 'r') as h5:
            if 'msd_analysis' in h5.keys():
                if 'gauss_fit' in h5['msd_analysis'].keys() and rewrite==False:
                    df_combined = pd.DataFrame(h5['msd_analysis']['gauss_fit'][...])
                    return df_combined
        region_dna_0 = int(self.region_dna[0])
        region_dna_1 = int(self.region_dna[1])
        dna_pos_start = int(self.dna_ends[0])
        dna_pos_end = int(self.dna_ends[1])
        self.img_arr_0_dna = self.img_arr_0[region_dna_0:region_dna_1, ::]       
        # create an empty dataframe
        df_combined = pd.DataFrame(columns=['frame', 'dna_pos', 'mu', 'sigma', 'amplitude', 'background'])
        
        # loop through the frames and concatenate the dataframes
        for frame in range(self.img_arr_0_dna.shape[0]):
            image = self.img_arr_0_dna[frame, ::]
            df = self.fit_gaussian(image, frame, skip_dna_pixs=skip_dna_pixs)
            df_combined = pd.concat([df_combined, df], ignore_index=True)
        df_combined = df_combined.astype('float64')
        # return and save the combined dataframe
        with h5py.File(self.h5path, 'r+') as h5:
            if 'msd_analysis' in h5.keys() and rewrite==False:
                msd_group = h5['msd_analysis']
            elif 'msd_analysis' in h5.keys() and rewrite==True:
                print("msd_analysis already exists. Overwriting...")
                del h5['msd_analysis']
                msd_group = h5.create_group("msd_analysis")
            else:
                msd_group = h5.create_group("msd_analysis")
            if 'gauss_fit' in msd_group.keys():
                print("gauss_fit already exists. Overwriting...")
                del msd_group['gauss_fit']
                msd_group['gauss_fit'] = df_combined.to_records()
                # msd_group.create_dataset('gauss_fit', data=df_combined.to_records())
            else:
                msd_group['gauss_fit'] = df_combined.to_records()
                # msd_group.create_dataset('gauss_fit', data=df_combined.to_records())
        return df_combined

    def animate_gaussfit_all_frames(self):
        fig_2 = plt.figure(figsize=(10, 10))
        # create a grid with two rows and two columns
        gs = fig_2.add_gridspec(2, 5)
        # create the subplots within the new grid
        axs = [fig_2.add_subplot(gs[0, 0]),
            fig_2.add_subplot(gs[0, 1:]),
            fig_2.add_subplot(gs[1, :])]

        dna_pos = 10
        dna_positions = np.linspace(10, self.img_arr_0_dna.shape[1]-20, 3).astype(int)
        df_combined = self.fit_gaussian_all_frames(skip_dna_pixs=1, rewrite=False)
        mean_dna_axis = df_combined['mu'].mean()
        # create a rainbow color map
        cmap = plt.get_cmap('rainbow')
        for idx, dna_pos in enumerate(dna_positions):
            df_i = df_combined[df_combined['dna_pos'] == dna_pos]
            df_i['mu'] = df_i['mu'] - mean_dna_axis
            axs[2].plot(df_i['frame'].values, df_i['mu'].values,
                        color=cmap(idx/len(dna_positions)),
                        alpha=0.9,)
        axs[2].set_ylim(-5, 5)
        line_frame = axs[2].axvline(x=dna_positions[0], color='k', linestyle='--')
        # axs[1].set_title(f'Plot for frame = {i}')
        axs[2].set_xlabel('Frame')
        axs[2].set_ylabel('Gassian center\nposition (pixel)')
        def update(i):
            image = self.img_arr_0_dna[i, ::]
            # Clear the previous plot
            axs[0].cla()
            axs[0].imshow(image)
            df_i = df_combined[df_combined['frame'] == i]
            axs[0].scatter(df_i['mu'], df_i['dna_pos'], c='r', s=1)
            axs[1].cla()
            for idx, dna_pos in enumerate(dna_positions):
                axs[0].axhline(y=dna_pos, color=cmap(idx/len(dna_positions)))
                frame, _, mu, sigma, amplitude, background = df_i[df_i['dna_pos'] == dna_pos].values[0]
                y = image[dna_pos, :]
                x = np.arange(0, len(y), 1)
                # Plot the new data
                axs[1].plot(x, y, '*', color=cmap(idx/len(dna_positions)))
                # Fit the data to the Gaussian function
                x = np.linspace(0, len(y), 100)
                axs[1].plot(x, gaussian_bg(x, mu, sigma, amplitude, background), '-', color=cmap(idx/len(dna_positions)))
                # axs[1].axvline(x=mu, color=cmap(idx/len(dna_positions)))
                line_frame.set_xdata(i)
                axs[0].set_xlim([0, len(y)])
                axs[1].set_xlim([0, len(y)])
                # set some axis parameters
                axs[0].set_xticks([])
                axs[0].set_yticks([])
                axs[0].set_xticklabels([])
                axs[0].set_yticklabels([])
        ani = FuncAnimation(fig_2, update, frames=self.img_arr_0_dna.shape[0], interval=100)# frames=img_arr_0_loop.shape[0]
        ani_filename = self.h5path[:-5] + f'_gaussfit_all_frames.mp4'
        ani.save(ani_filename)

    def msd_all_dna_lines(self):
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        dna_pos = 30
        cmap = plt.get_cmap('rainbow')
        skip_dna_pixs = 3 # skip every 5 pixels
        start_dna_pix = 4
        dna_half_width = int(-start_dna_pix+self.img_arr_0_dna.shape[1]-start_dna_pix)
        dna_pos_list = []
        msd_list = [] # at lagtime = 3 e.g.
        msd_at = 3
        pixel_size = self.pix_size
        fps = 1/self.frame_time
        max_lagtime = 100

        df_combined = self.fit_gaussian_all_frames()
        df_msds = pd.DataFrame(columns=['lagtimes'])

        # plot the second subplot
        for i in range(start_dna_pix, dna_half_width, skip_dna_pixs):
            dna_pos = i
            df_i = df_combined[df_combined['dna_pos'] == dna_pos]
            df_i['particle'] = 1
            df_i['x'] = df_i['mu']
            df_i['y'] = 1
            imsd = tp.imsd(df_i, mpp=pixel_size, fps=fps, max_lagtime=max_lagtime)
            dna_pos_list.append(dna_pos)
            msd_list.append(imsd.iloc[msd_at, 0])# at lagtime = 3 e.g.
            lagtimes = imsd.index.values
            msds = imsd.iloc[:, 0].values
            df_msds[str(dna_pos)] = msds
            axs[1].plot(lagtimes, msds, '-*',
                    color=cmap(i/dna_half_width), alpha=0.5,
                    label=f'dna_pos = {dna_pos}')
        df_msds['lagtimes'] = lagtimes
        with h5py.File(self.h5path, 'r+') as h5:
            if 'msd_analysis' in h5.keys():
                if 'msds' in h5['msd_analysis'].keys():
                    del h5['msd_analysis']['msds']
                h5['msd_analysis']['msds'] = df_msds.to_records()
        axs[1].set_xscale('log')
        axs[1].set_yscale('log')
        axs[1].set_xlabel('lagtime (s)')
        axs[1].set_ylabel('MSD (µm/s²)')
        # plot the first subplot
        axs[0].scatter(dna_pos_list, msd_list, c=dna_pos_list, cmap='rainbow')
        axs[0].set_ylim([0, 2.5*np.mean(msd_list)])
        axs[1].set_ylim([0, 2.5*np.mean(msd_list)])
        axs[0].set_xlabel('DNA position')
        axs[0].set_ylabel('MSD (µm/s²)')
        plt.show()
        
        return df_msds


if __name__ == '__main__':
    h5path = r"\\cifs1.bpcentral.biophys.mpg.de\msdata\kimlab\biswajit\tud_archive\data\20201216_plectonominDNA\S151d12Dec20-CH6_A3_50mW561nm_5msAqt_200nMSxO_analysis\Plectoneme.leads.hdf5"
    dna_fluctuation_msd = DNAFluctuationMSD(h5path)
    dna_fluctuation_msd.read()
    dna_fluctuation_msd.animate_gauss_fitting_single_frame(frame=0)
    dna_fluctuation_msd.fit_gaussian_all_frames(skip_dna_pixs=1, rewrite=False)
    # dna_fluctuation_msd.animate_gaussfit_all_frames()
    _ = dna_fluctuation_msd.msd_all_dna_lines()