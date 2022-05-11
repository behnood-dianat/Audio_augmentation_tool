import glob, os, pathlib
import numpy as np
import librosa
import matplotlib.pylab as plt
import matplotlib as mpl
import random
import tensorflow as tf
import spec_augment_tensorflow

mpl.use("agg")                           # matplot backend for memory management
#os.environ["CUDA_VISIBLE_DEVICES"]="-1"  # Use only if your GPU is powerful
#t_shape= 1792 # scale all arrays to 10-seconds size (around 2240 values)

def file_grab(directory):
	os.chdir(directory)
	files= glob.glob("*.wav")
	return files

def file_grab_paths(directory):
	files= glob.glob((directory+"/*.wav"))
	return files

def random_augment(files,output_path,names,batch_num,lenght_sound,overlap):
	#------------------------------------------------------------------------
	def augment_loop(sounds1,sounds2,sounds3,sounds4,output_path,names,batch_num,minimum):
		count=0
		for j in range(len(sounds1)):
			count += 1
			if (len(sounds1[j])/sampling_rate) >= minimum:
				#Randoms
				cmap= random.choice(['viridis','plasma','inferno','magma','cividis','bone'])
				#fmax= random.choice([3500,2500,1500])
				#n_mels= random.choice([64,128,256])

				#cmap= random.choice(['magma'])
				fmax= random.choice([1600])
				fmin=0
				#n_mels= random.choice([64])


				mel_spectrogram_1 = librosa.feature.melspectrogram(y=sounds1[j],sr=sampling_rate,hop_length=32,n_mels=16,fmin=fmin,fmax=100)
				#mel_spectrogram_2 = librosa.feature.melspectrogram(y=sounds2[j],sr=sampling_rate,hop_length=32,n_mels=32,fmin=50,fmax=300)
				#mel_spectrogram_3 = librosa.feature.melspectrogram(y=sounds3[j],sr=sampling_rate,hop_length=32,n_mels=32,fmin=100,fmax=600)
				#mel_spectrogram_4 = librosa.feature.melspectrogram(y=sounds4[j],sr=sampling_rate,hop_length=32,n_mels=48,fmin=400,fmax=fmax)

				mfccs2 = librosa.feature.mfcc(y=sounds2[j],sr=sampling_rate,hop_length=32,n_mels=32,n_mfcc=32,fmin=50,fmax=300)
				mfccs3 = librosa.feature.mfcc(y=sounds3[j],sr=sampling_rate,hop_length=32,n_mels=32,n_mfcc=32,fmin=100,fmax=600)
				mfccs4 = librosa.feature.mfcc(y=sounds4[j],sr=sampling_rate,hop_length=32,n_mels=48,n_mfcc=48,fmin=400,fmax=fmax)

				#Scale the array to 10-seconds (t_shape= 2240)
				shape = mel_spectrogram_1.shape
				scale_difference = t_shape-shape[1]
				if scale_difference > 0:
					#mel_spectrogram_1 = np.append(mel_spectrogram_1, [np.zeros(scale_difference) for _ in range(mel_spectrogram_1.shape[0])] , axis=1)
					#mel_spectrogram_2 = np.append(mel_spectrogram_2, [np.zeros(scale_difference) for _ in range(mel_spectrogram_2.shape[0])] , axis=1)
					#mel_spectrogram_3 = np.append(mel_spectrogram_3, [np.zeros(scale_difference) for _ in range(mel_spectrogram_3.shape[0])] , axis=1)
					#mel_spectrogram_4 = np.append(mel_spectrogram_4, [np.zeros(scale_difference) for _ in range(mel_spectrogram_4.shape[0])] , axis=1)

					mfccs2 = np.append(mfccs2, [np.zeros(scale_difference) for _ in range(mfccs2.shape[0])] , axis=1)
					mfccs3 = np.append(mfccs3, [np.zeros(scale_difference) for _ in range(mfccs3.shape[0])] , axis=1)
					mfccs4 = np.append(mfccs4, [np.zeros(scale_difference) for _ in range(mfccs4.shape[0])] , axis=1)

				#S_dB_1 = librosa.power_to_db(mel_spectrogram_1, ref=np.max)
				#S_dB_2 = librosa.power_to_db(mel_spectrogram_2, ref=np.max)
				#S_dB_3 = librosa.power_to_db(mel_spectrogram_3, ref=np.max)
				#S_dB_4 = librosa.power_to_db(mel_spectrogram_4, ref=np.max)

				#S_dB_1 = 2.*(S_dB_1 - np.min(S_dB_1))/np.ptp(S_dB_1) -1
				#S_dB_2 = 2.*(S_dB_2 - np.min(S_dB_2))/np.ptp(S_dB_2) -1
				#S_dB_3 = 2.*(S_dB_3 - np.min(S_dB_3))/np.ptp(S_dB_3) -1
				#S_dB_4 = 2.*(S_dB_4 - np.min(S_dB_4))/np.ptp(S_dB_4) -1

				mfccs2 = np.delete(mfccs2,0,0)
				mfccs2 = np.delete(mfccs2,0,0)
				mfccs3 = np.delete(mfccs3,0,0)
				mfccs3 = np.delete(mfccs3,0,0)
				mfccs4 = np.delete(mfccs4,0,0)
				mfccs4 = np.delete(mfccs4,0,0)

				mfccs2 = 2.*(mfccs2 - np.min(mfccs2))/np.ptp(mfccs2) -1
				mfccs3 = 2.*(mfccs3 - np.min(mfccs3))/np.ptp(mfccs3) -1
				mfccs4 = 2.*(mfccs4 - np.min(mfccs4))/np.ptp(mfccs4) -1

				concatenated = np.concatenate((mfccs2,mfccs3,mfccs4))
				shape = concatenated.shape

				rnd= random.choice(['augment','non_augment'])
				if rnd=='augment':
					# Reshape before calling spec_augment------------------------------
					concatenated = np.reshape(concatenated, (-1, shape[0], shape[1], 1))
					concatenated = tf.cast(concatenated, dtype=tf.float32) #Cast to tf for better performance

					#Augmentation
					concatenated = spec_augment_tensorflow.spec_augment(concatenated)
					concatenated = concatenated.numpy()
					concatenated = np.reshape(concatenated, (shape[0], shape[1]))

				#Plot
				plt.figure(figsize=(2.99, 2.99))
				plt.axes([0., 0., 1., 1.])
				librosa.display.specshow(concatenated,fmax=fmax,sr=sampling_rate,cmap=cmap)
				file_path=pathlib.PurePosixPath(output_path,(names[x].split('.')[0] + ("_part %d" % count) + batch_num + ".png"))
				plt.savefig(str(file_path),format='png')
				plt.close('all'), plt.clf(), plt.cla()
	#--------------------------------------------------------------------------------------
	t_shape=np.round(690*lenght_sound) # 690 for hop=32
	minimum=np.round(lenght_sound/2)
	for x in range(len(files)):
		temp_name=names[x].split('.')
		if temp_name[1]=='mode1':
			sound_original1, sampling_rate = librosa.load(str(os.path.split(files[x])[0]).replace('\\','/')+'/'+temp_name[0]+'.mode1.'+temp_name[2])
			sound_original2, sampling_rate = librosa.load(str(os.path.split(files[x])[0]).replace('\\','/')+'/'+temp_name[0]+'.mode2.'+temp_name[2])
			sound_original3, sampling_rate = librosa.load(str(os.path.split(files[x])[0]).replace('\\','/')+'/'+temp_name[0]+'.mode3.'+temp_name[2])
			sound_original4, sampling_rate = librosa.load(str(os.path.split(files[x])[0]).replace('\\','/')+'/'+temp_name[0]+'.mode4.'+temp_name[2])
			sounds1=[]
			sounds2=[]
			sounds3=[]
			sounds4=[]
			sound_temp1=[1]
			sound_temp2=[1]
			sound_temp3=[1]
			sound_temp4=[1]

			i=0
			while len(sound_temp1) != 0 and sum(sound_temp1) !=0 and sum(sound_temp2) !=0 and sum(sound_temp3) !=0 and sum(sound_temp4) !=0: # Cut the file into samples
				sound_temp1= sound_original1[int((len(sound_original1)/(len(sound_original1)/sampling_rate))*(i*(lenght_sound - overlap))):
				int((len(sound_original1)/(len(sound_original1)/sampling_rate))*(i*(lenght_sound - overlap) + lenght_sound))]

				sound_temp2= sound_original2[int((len(sound_original2)/(len(sound_original2)/sampling_rate))*(i*(lenght_sound - overlap))):
				int((len(sound_original2)/(len(sound_original2)/sampling_rate))*(i*(lenght_sound - overlap) + lenght_sound))]

				sound_temp3= sound_original3[int((len(sound_original3)/(len(sound_original3)/sampling_rate))*(i*(lenght_sound - overlap))):
				int((len(sound_original3)/(len(sound_original3)/sampling_rate))*(i*(lenght_sound - overlap) + lenght_sound))]

				sound_temp4= sound_original4[int((len(sound_original4)/(len(sound_original4)/sampling_rate))*(i*(lenght_sound - overlap))):
				int((len(sound_original4)/(len(sound_original4)/sampling_rate))*(i*(lenght_sound - overlap) + lenght_sound))]

				if len(sound_temp1) != 0 and sum(sound_temp1) !=0 and sum(sound_temp2) !=0 and sum(sound_temp3) !=0 and sum(sound_temp4) !=0:
					sound_temp1 = 2.*(sound_temp1 - np.min(sound_temp1))/np.ptp(sound_temp1) -1 # scale to [-1 1]
					sounds1.append(sound_temp1)
					sound_temp2 = 2.*(sound_temp2 - np.min(sound_temp2))/np.ptp(sound_temp2) -1 # scale to [-1 1]
					sounds2.append(sound_temp2)
					sound_temp3 = 2.*(sound_temp3 - np.min(sound_temp3))/np.ptp(sound_temp3) -1 # scale to [-1 1]
					sounds3.append(sound_temp3)
					sound_temp4 = 2.*(sound_temp4 - np.min(sound_temp4))/np.ptp(sound_temp4) -1 # scale to [-1 1]
					sounds4.append(sound_temp4)
					i += 1
			augment_loop(sounds1,sounds2,sounds3,sounds4,output_path,names,batch_num,minimum)

def validation_non_augment(files,output_path,names,batch_num,lenght_sound,overlap):
	t_shape=np.round(690*lenght_sound)
	minimum=np.round(lenght_sound/2)
	for x in range(len(files)):
		temp_name=names[x].split('.')
		if temp_name[1]=='mode1':
			sound_original1, sampling_rate = librosa.load(str(os.path.split(files[x])[0]).replace('\\','/')+'/'+temp_name[0]+'.mode1.'+temp_name[2])
			sound_original2, sampling_rate = librosa.load(str(os.path.split(files[x])[0]).replace('\\','/')+'/'+temp_name[0]+'.mode2.'+temp_name[2])
			sound_original3, sampling_rate = librosa.load(str(os.path.split(files[x])[0]).replace('\\','/')+'/'+temp_name[0]+'.mode3.'+temp_name[2])
			sound_original4, sampling_rate = librosa.load(str(os.path.split(files[x])[0]).replace('\\','/')+'/'+temp_name[0]+'.mode4.'+temp_name[2])
			sounds1=[]
			sounds2=[]
			sounds3=[]
			sounds4=[]
			sound_temp1=[1]
			sound_temp2=[1]
			sound_temp3=[1]
			sound_temp4=[1]
			i=0
			while len(sound_temp1) != 0 and sum(sound_temp1) !=0 and sum(sound_temp2) !=0 and sum(sound_temp3) !=0 and sum(sound_temp4) !=0: # Cut the file into samples
				sound_temp1= sound_original1[int((len(sound_original1)/(len(sound_original1)/sampling_rate))*(i*(lenght_sound - overlap))):
				int((len(sound_original1)/(len(sound_original1)/sampling_rate))*(i*(lenght_sound - overlap) + lenght_sound))]

				sound_temp2= sound_original2[int((len(sound_original2)/(len(sound_original2)/sampling_rate))*(i*(lenght_sound - overlap))):
				int((len(sound_original2)/(len(sound_original2)/sampling_rate))*(i*(lenght_sound - overlap) + lenght_sound))]

				sound_temp3= sound_original3[int((len(sound_original3)/(len(sound_original3)/sampling_rate))*(i*(lenght_sound - overlap))):
				int((len(sound_original3)/(len(sound_original3)/sampling_rate))*(i*(lenght_sound - overlap) + lenght_sound))]

				sound_temp4= sound_original4[int((len(sound_original4)/(len(sound_original4)/sampling_rate))*(i*(lenght_sound - overlap))):
				int((len(sound_original4)/(len(sound_original4)/sampling_rate))*(i*(lenght_sound - overlap) + lenght_sound))]

				if len(sound_temp1) != 0 and sum(sound_temp1) !=0 and sum(sound_temp2) !=0 and sum(sound_temp3) !=0 and sum(sound_temp4) !=0:
					sound_temp1 = 2.*(sound_temp1 - np.min(sound_temp1))/np.ptp(sound_temp1) -1 # scale to [-1 1]
					sounds1.append(sound_temp1)

					sound_temp2 = 2.*(sound_temp2 - np.min(sound_temp2))/np.ptp(sound_temp2) -1 # scale to [-1 1]
					sounds2.append(sound_temp2)

					sound_temp3 = 2.*(sound_temp3 - np.min(sound_temp3))/np.ptp(sound_temp3) -1 # scale to [-1 1]
					sounds3.append(sound_temp3)

					sound_temp4 = 2.*(sound_temp4 - np.min(sound_temp4))/np.ptp(sound_temp4) -1 # scale to [-1 1]
					sounds4.append(sound_temp4)
				i += 1
			count=0
			for j in range(len(sounds1)):
				count += 1
				if (len(sounds1[j])/sampling_rate) >= minimum:
					#Randoms
					#cmap= random.choice(['viridis','plasma','inferno','magma','cividis','bone'])
					#fmax= random.choice([3500,2500,1500])
					#n_mels= random.choice([64,128,256])

					cmap= random.choice(['magma'])
					fmax= random.choice([1600])
					fmin=0
					#n_mels= random.choice([64])


					mel_spectrogram_1 = librosa.feature.melspectrogram(y=sounds1[j],sr=sampling_rate,hop_length=32,n_mels=16,fmin=fmin,fmax=100)
					#mel_spectrogram_2 = librosa.feature.melspectrogram(y=sounds2[j],sr=sampling_rate,hop_length=32,n_mels=32,fmin=50,fmax=300)
					#mel_spectrogram_3 = librosa.feature.melspectrogram(y=sounds3[j],sr=sampling_rate,hop_length=32,n_mels=32,fmin=100,fmax=600)
					#mel_spectrogram_4 = librosa.feature.melspectrogram(y=sounds4[j],sr=sampling_rate,hop_length=32,n_mels=48,fmin=400,fmax=fmax)

					mfccs2 = librosa.feature.mfcc(y=sounds2[j],sr=sampling_rate,hop_length=32,n_mels=32,n_mfcc=32,fmin=50,fmax=300)
					mfccs3 = librosa.feature.mfcc(y=sounds3[j],sr=sampling_rate,hop_length=32,n_mels=32,n_mfcc=32,fmin=100,fmax=600)
					mfccs4 = librosa.feature.mfcc(y=sounds4[j],sr=sampling_rate,hop_length=32,n_mels=48,n_mfcc=48,fmin=400,fmax=fmax)

					#Scale the array to 10-seconds (t_shape= 2240)
					shape = mel_spectrogram_1.shape
					scale_difference = t_shape-shape[1]
					if scale_difference > 0:
							#mel_spectrogram_1 = np.append(mel_spectrogram_1, [np.zeros(scale_difference) for _ in range(mel_spectrogram_1.shape[0])] , axis=1)
							#mel_spectrogram_2 = np.append(mel_spectrogram_2, [np.zeros(scale_difference) for _ in range(mel_spectrogram_2.shape[0])] , axis=1)
							#mel_spectrogram_3 = np.append(mel_spectrogram_3, [np.zeros(scale_difference) for _ in range(mel_spectrogram_3.shape[0])] , axis=1)
							#mel_spectrogram_4 = np.append(mel_spectrogram_4, [np.zeros(scale_difference) for _ in range(mel_spectrogram_4.shape[0])] , axis=1)

							mfccs2 = np.append(mfccs2, [np.zeros(scale_difference) for _ in range(mfccs2.shape[0])] , axis=1)
							mfccs3 = np.append(mfccs3, [np.zeros(scale_difference) for _ in range(mfccs3.shape[0])] , axis=1)
							mfccs4 = np.append(mfccs4, [np.zeros(scale_difference) for _ in range(mfccs4.shape[0])] , axis=1)

					#S_dB_1 = librosa.power_to_db(mel_spectrogram_1, ref=np.max)
					#S_dB_2 = librosa.power_to_db(mel_spectrogram_2, ref=np.max)
					#S_dB_3 = librosa.power_to_db(mel_spectrogram_3, ref=np.max)
					#S_dB_4 = librosa.power_to_db(mel_spectrogram_4, ref=np.max)

					#S_dB_1 = 2.*(S_dB_1 - np.min(S_dB_1))/np.ptp(S_dB_1) -1
					#S_dB_2 = 2.*(S_dB_2 - np.min(S_dB_2))/np.ptp(S_dB_2) -1
					#S_dB_3 = 2.*(S_dB_3 - np.min(S_dB_3))/np.ptp(S_dB_3) -1
					#S_dB_4 = 2.*(S_dB_4 - np.min(S_dB_4))/np.ptp(S_dB_4) -1
					mfccs2 = np.delete(mfccs2,0,0)
					mfccs2 = np.delete(mfccs2,0,0)
					mfccs3 = np.delete(mfccs3,0,0)
					mfccs3 = np.delete(mfccs3,0,0)
					mfccs4 = np.delete(mfccs4,0,0)
					mfccs4 = np.delete(mfccs4,0,0)

					mfccs2 = 2.*(mfccs2 - np.min(mfccs2))/np.ptp(mfccs2) -1
					mfccs3 = 2.*(mfccs3 - np.min(mfccs3))/np.ptp(mfccs3) -1
					mfccs4 = 2.*(mfccs4 - np.min(mfccs4))/np.ptp(mfccs4) -1

					concatenated = np.concatenate((mfccs2,mfccs3,mfccs4))


					#Plot
					plt.figure(figsize=(2.99, 2.99))
					plt.axes([0., 0., 1., 1.])
					librosa.display.specshow(concatenated,fmax=fmax,sr=sampling_rate,cmap=cmap)
					file_path=pathlib.PurePosixPath(output_path,(names[x].split('.')[0] + ("_part %d" % count) + batch_num + ".png"))
					plt.savefig(str(file_path),format='png')
					plt.close('all'), plt.clf(), plt.cla()


#input_paths=file_grab("C:/Users/behno/OneDrive/Documents")
#output_paths="C:/Users/behno/OneDrive/Documents"

#non_augment_loop(input_paths,output_paths)