import tensorflow as tf
from IPython.display import clear_output
import matplotlib.pylab as plt
# import seaborn as sns
import pathlib

class TrainingPlot_(tf.keras.callbacks.Callback):
    def __init__(self, nrows=1, ncols=2, figsize=(10, 5), title=None, save_file=None, old_logs_path=None, every_epoch=1):
      self.nrows=nrows
      self.ncols=ncols
      self.figsize=figsize
      self.title=title
      self.old_logs_path=old_logs_path
      self.old_logs = []
      self.old_log_file_names = []

      if self.old_logs_path is not None:
        p = pathlib.Path(self.old_logs_path)
        self.old_logs_files = p.parent.glob(p.name)

        for f in self.old_logs_files:
          try:
            self.old_logs.append(pd.read_csv(f))
            self.old_log_file_names.append(pathlib.Path(f).stem)
          except:
            continue

      self.save_file = save_file
      
      self.metrics = []
      self.logs = []

      self.every_epoch = every_epoch
        
    def add(self, row, col, name, color=None, vmin=None, vmax=None, show_min=False, show_max=False):
      self.metrics.append({'row': row, 'col': col, 'name': name, 'color': color, 'vmin': vmin, 'vmax': vmax, 'show_min': show_min, 'show_max': show_max})
    
    def on_train_begin(self, logs={}):
      self.logs = []
      
      for m in self.metrics:
          m['values'] = []

    def on_epoch_end(self, epoch, logs={}):
      for m in self.metrics:
          v = m['values']
          v.append(logs.get(m['name']))

      if epoch % self.every_epoch != 0:
        return

      clear_output(wait=True)
      plt.style.use("seaborn")
      # sns.set_style("whitegrid")
      fig, ax = plt.subplots(self.nrows, self.ncols, figsize=self.figsize)

      if self.title:
        fig.suptitle(self.title)
      
      for m in self.metrics:
          v = m['values']

          if len(ax.shape) > 1:
              a = ax[m['row'], m['col']]
          else:
              a = ax[m['row'] + m['col']]
          
          if m['name'] == 'off':
            a.axis('off')
            continue
                              
          # old logs
          for i, old_log in enumerate(self.old_logs):
            if m['name'] not in old_log.keys():
              continue

            old_values = old_log[m['name']].values
            a.plot(np.arange(len(old_values)), old_values, 
                    '-', 
                    color=m['color'], 
                  #  label=self.old_log_file_names[i], # gets crowded
                    alpha=0.1, lw=1)

          # new log
          a.plot(np.arange(len(v)), v, '-o', color=m['color'], label=m['name'], lw=2, markersize=3)
          a.set_xlabel('Epoch #', size=14)
          
          yname = m['name']
          if yname.startswith('val_'):
            yname = m['name'][4:]
          a.set_ylabel(yname, size=14)

          xdist = a.get_xlim()[1] - a.get_xlim()[0]
          ydist = a.get_ylim()[1] - a.get_ylim()[0]
          if m['show_max']:
            x = np.argmax(v)
            y = np.max(v)
            if y is not None:
              a.scatter(x, y, s=200, color=m['color'], alpha=0.5)
              a.text(x-0.03*xdist, y-0.13*ydist, f'{round(y, 4)}', size=14)
          if m['show_min']:
            x = np.argmin(v)
            y = np.min(v)
            if y is not None:
              a.scatter(x, y, s=200, color=m['color'], alpha=0.5)
              a.text(x-0.03*xdist, y+0.05*ydist, f'{round(y, 4)}', size=14)
          if m['vmin'] is not None:
            a.set_ylim(m['vmin'], m['vmax'])

          a.legend()

      plt.show()

      if self.save_file is not None:
        fig.savefig(self.save_file)