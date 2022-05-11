import glob, os, pathlib
import numpy as np
import librosa
import tensorflow as tf
import spec_augment_tensorflow
import matplotlib.pylab as plt
import matplotlib as mpl
import random

mpl.use("agg")                           # matplot backend for memory management
os.environ["CUDA_VISIBLE_DEVICES"]="0"  # Use only if your GPU is powerful
#t_shape= 1792 # scale all arrays to 10-seconds size (around 2240 values)

def file_grab(directory):
	os.chdir(directory)
	files= glob.glob("*.wav")
	return files

def file_grab_paths(directory):
	files= glob.glob((directory+"/*.wav"))
	return files

def random_augment(files,output_path,names,batch_num,lenght_sound,overlap):
	#-------------------------------------------------------------------	
	def augment_loop(sounds,output_path,t_shape,names,batch_num,minimum):
		count=0
		for sound in sounds:
			count += 1
			if (len(sound)/sampling_rate) >= minimum:
				#Randoms
				cmap= random.choice(['viridis','plasma','inferno','magma','cividis','bone','binary', 'gist_yarg', 'gist_gray', 'gray', 'bone',
                      'pink', 'spring', 'summer', 'autumn', 'winter', 'cool',
                      'Wistia', 'hot', 'afmhot', 'gist_heat', 'copper'])
				#fmax= random.choice([1000,1200,1500])
				n_mels= random.choice([128])

				#cmap= random.choice(['magma'])
				fmax= random.choice([1000])
				fmin=50
				#n_mels= random.choice([64])

				mel_spectrogram = librosa.feature.melspectrogram(y=sound,sr=sampling_rate,hop_length=16,n_mels=n_mels,fmin=fmin,fmax=fmax)
				shape = mel_spectrogram.shape
				#mel_spectrogram[:,(shape[1]-20):shape[1]]=0 # get rid of some edges
				#mel_spectrogram[:,0:5]=0                    # get rid of some edges

				#Scale the array to 10-seconds (t_shape= 2240)
				scale_difference = t_shape-shape[1]
				if scale_difference > 0:
					mel_spectrogram = np.append(mel_spectrogram, [np.zeros(scale_difference) for _ in range(shape[0])] , axis=1)
					shape = mel_spectrogram.shape
				mel_spectrogram = np.reshape(mel_spectrogram, (-1, shape[0], shape[1], 1))
				mel_spectrogram = tf.cast(mel_spectrogram, dtype=tf.float32) #Cast to tf for better performance 


				#Augmentation
				mel_spectrogram = spec_augment_tensorflow.spec_augment(mel_spectrogram)
				#mel_spectrogram = tf.cast(mel_spectrogram, dtype=np.float32)
				mel_spectrogram = mel_spectrogram.numpy()
				mel_spectrogram = np.reshape(mel_spectrogram, (shape[0], shape[1]))
				S_dB = librosa.power_to_db(mel_spectrogram, ref=np.max)
				S_dB = 2.*(S_dB - np.min(S_dB))/np.ptp(S_dB) -1

				#Plot
				plt.figure(figsize=(2.99, 2.99))
				#fig.subplots_adjust(left=-0.05,right=1.05,bottom=-0.05,top=1.05)
				plt.axes([0., 0., 1., 1.])
				librosa.display.specshow(S_dB,cmap=cmap,y_axis='mel',fmax=fmax,sr=sampling_rate)
				file_path=pathlib.PurePosixPath(output_path,(os.path.splitext(names[x])[0] + ("_part %d" % count) + batch_num + ".png"))
				plt.savefig(str(file_path),format='png')
				plt.close('all'), plt.clf(), plt.cla()
		#print("Augment done!")

	#------------------------------------------------------------------------
	def non_augment_loop(sounds,output_path,t_shape,names,batch_num,minimum):
		count=0
		for sound in sounds:
			count += 1
			if (len(sound)/sampling_rate) >= minimum:
				#Randoms
				cmap= random.choice(['viridis','plasma','inferno','magma','cividis','bone','binary', 'gist_yarg', 'gist_gray', 'gray', 'bone',
                      'pink', 'spring', 'summer', 'autumn', 'winter', 'cool',
                      'Wistia', 'hot', 'afmhot', 'gist_heat', 'copper'])
				#fmax= random.choice([1000,1200,1500])
				n_mels= random.choice([128])

				#cmap= random.choice(['magma'])
				fmax= random.choice([1000])
				fmin=50
				#n_mels= random.choice([64])

				mel_spectrogram = librosa.feature.melspectrogram(y=sound,sr=sampling_rate,hop_length=16,n_mels=n_mels,fmin=fmin,fmax=fmax)

				#Scale the array to 10-seconds (t_shape= 2240)
				shape = mel_spectrogram.shape
				scale_difference = t_shape-shape[1]
				if scale_difference > 0:
					mel_spectrogram = np.append(mel_spectrogram, [np.zeros(scale_difference) for _ in range(shape[0])] , axis=1)
					shape = mel_spectrogram.shape
				
				S_dB = librosa.power_to_db(mel_spectrogram, ref=np.max)
				S_dB = 2.*(S_dB - np.min(S_dB))/np.ptp(S_dB) -1

				plt.figure(figsize=(2.99, 2.99))
				plt.axes([0., 0., 1., 1.])

				librosa.display.specshow(S_dB,cmap=cmap,y_axis='mel',fmax=fmax,sr=sampling_rate)
				file_path=pathlib.PurePosixPath(output_path,(os.path.splitext(names[x])[0] + ("_part %d" % count) + batch_num + ".png"))
				plt.savefig(str(file_path),format='png')
				plt.close('all'), plt.clf(), plt.cla()
		#print("Non_augment done!")
	#--------------------------------------------------------------------------
	t_shape=np.round(1380*lenght_sound) #345 for hop=64 ,  172 for hop=128
	minimum=np.round(lenght_sound/2)
	for x in range(len(files)):
		sound_original, sampling_rate = librosa.load(files[(x)])
		sounds=[]
		sound_temp=[1]
		i=0
		while len(sound_temp) != 0 and sum(sound_temp) !=0: # Cut the file into samples
			sound_temp= sound_original[int((len(sound_original)/(len(sound_original)/sampling_rate))*(i*(lenght_sound - overlap))):
			int((len(sound_original)/(len(sound_original)/sampling_rate))*(i*(lenght_sound - overlap) + lenght_sound))]
			if len(sound_temp) != 0 and sum(sound_temp) !=0:
				sound_temp = 2.*(sound_temp - np.min(sound_temp))/np.ptp(sound_temp) -1 # scale to [-1 1]
				sounds.append(sound_temp)
				i += 1
		rnd= random.choice(['augment','non_augment'])
		if rnd == 'augment':
			augment_loop(sounds,output_path,t_shape,names,batch_num,minimum)
		elif rnd == 'non_augment':
			non_augment_loop(sounds,output_path,t_shape,names,batch_num,minimum)

def validation_non_augment(files,output_path,names,batch_num,lenght_sound,overlap):
	t_shape=np.round(1380*lenght_sound)
	minimum=np.round(lenght_sound/2)
	for x in range(len(files)):
		sound_original, sampling_rate = librosa.load(files[(x)])
		sounds=[]
		sound_temp=[1]
		i=0
		while len(sound_temp) != 0 and sum(sound_temp) !=0:
			sound_temp= sound_original[int((len(sound_original)/(len(sound_original)/sampling_rate))*(i*(lenght_sound - overlap))):
			int((len(sound_original)/(len(sound_original)/sampling_rate))*(i*(lenght_sound - overlap) + lenght_sound))]
			if len(sound_temp) != 0 and sum(sound_temp) !=0:
				sound_temp = 2.*(sound_temp - np.min(sound_temp))/np.ptp(sound_temp)-1 # scale to [-1 1]
				sounds.append(sound_temp)
				i += 1
		count=0
		for sound in sounds:
			count += 1
			if (len(sound)/sampling_rate) >= minimum:
				#Randoms
				#cmap= random.choice(['viridis','plasma','inferno','magma','cividis','bone'])
				#fmax= random.choice([3500,2500,1500])
				#n_mels= random.choice([64,128,256])

				cmap= random.choice(['magma'])
				fmax= random.choice([1000])
				fmin=50
				n_mels= random.choice([128])

				mel_spectrogram = librosa.feature.melspectrogram(y=sound,sr=sampling_rate,hop_length=16,n_mels=n_mels,fmin=fmin,fmax=fmax)
				shape = mel_spectrogram.shape

				#Scale the array to 10-seconds (t_shape= 2240)
				scale_difference = t_shape-shape[1]
				if scale_difference > 0:
					mel_spectrogram = np.append(mel_spectrogram, [np.zeros(scale_difference) for _ in range(shape[0])] , axis=1)
					shape = mel_spectrogram.shape

				S_dB = librosa.power_to_db(mel_spectrogram, ref=np.max)
				S_dB = 2.*(S_dB - np.min(S_dB))/np.ptp(S_dB) -1

				plt.figure(figsize=(2.99, 2.99))
				plt.axes([0., 0., 1., 1.])

				librosa.display.specshow(S_dB,cmap=cmap,y_axis='mel',fmax=fmax,sr=sampling_rate)
				file_path=pathlib.PurePosixPath(output_path,(os.path.splitext(names[x])[0] + ("_part %d" % count) + batch_num + ".png"))
				plt.savefig(str(file_path),format='png')
				plt.close('all'), plt.clf(), plt.cla()
	#print("Validation done!")



#input_paths=file_grab("C:/Users/behno/OneDrive/Documents")
#output_paths="C:/Users/behno/OneDrive/Documents"

#non_augment_loop(input_paths,output_paths)