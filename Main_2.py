import timeit, shutil
import os, pathlib
import First_augment  
import mfcc_augment
#import triple_mel_augment as tma
import triple_mfcc_augment as tma
from PIL import Image
from pathlib import Path
from sklearn.model_selection import train_test_split

#t_shape = 1033 # 6 seconds
os.environ["CUDA_VISIBLE_DEVICES"]="0"  

start = timeit.default_timer()

def names(file):
	file_list=[]
	for i in range(len(file)):
		file_list.append(file[i].name)
	return file_list

def file_func(directory):
	file_list=[]
	for file in Path(directory).rglob('*.wav'):
		file_list.append(file)
	return file_list

# mel or mfcc-------------------------------------------------------------------------------------------
run_mode='mel'

# Directories-------------------------------------------------------------------------------------------
input_dir1='G:/My Drive/ILD-Project/Batch preparation - CTD/Mode_3_high_freqs/Un_separated'
#output_dir1="G:/My Drive/ILD-Project/Batch preparation - CTD/Training_Validation_Test_mode_5_"+run_mode

#input_dir1='D:/Super_PC_files/CTD_Wave_files/Wave_ICBHI_Raw'
output_dir1="D:/Super_PC_files/CTD_dataset/mode3_mel_v11_final"

#input_dir1='G:/My Drive/ILD-Project/Covid_19/Sgolay_filtered'
#output_dir1="D:/Super_PC_files/COVID19/sgolay_v1_4_1"

#inputs=[input_dir1,input_dir2]
#outputs=[output_dir1,output_dir2]

inputs=[input_dir1]
outputs=[output_dir1]

for location in range(len(inputs)):
	input_dir=inputs[location]
	output_dir=outputs[location]

	os.makedirs(output_dir,exist_ok=True)

	for filename in os.listdir(output_dir):
		file_path = os.path.join(output_dir, filename)
		try:
			if os.path.isfile(file_path) or os.path.islink(file_path):
				os.unlink(file_path)
			elif os.path.isdir(file_path):
				shutil.rmtree(file_path)
				print('deleted')
		except Exception as e:
			print('Failed to delete %s. Reason: %s' % (file_path, e))

	

	total_negative=pathlib.PurePosixPath(input_dir, 'Negative')
	total_positive=pathlib.PurePosixPath(input_dir, 'Positive')

	negative_paths=file_func(str(total_negative))
	positive_paths=file_func(str(total_positive))

	n_train, n_val = train_test_split(negative_paths,test_size=0.2, shuffle=True)
	p_train, p_val = train_test_split(positive_paths,test_size=0.2, shuffle=True)

	n_train, n_test = train_test_split(n_train,test_size=0.1, shuffle=True)
	p_train, p_test = train_test_split(p_train,test_size=0.1, shuffle=True)

	# Negative Paths and names-----------------------------------------------------------------------------------------
	list_1=['Training','Validation','Test']
	list_2=['Negative','Positive']
	output_n=[]
	output_p=[]
	for i in list_1:
		for j in list_2:
			path_make=pathlib.PurePosixPath(output_dir, i, j)
			os.makedirs(path_make,exist_ok=True)
			if j == 'Negative':
				output_n.append(path_make)
			elif j == 'Positive':
				output_p.append(path_make)

	names_n=[]
	paths_n=[]
	names_p=[]
	paths_p=[]
	list_1=[n_train,n_val,n_test]
	list_2=[p_train,p_val,p_test]
	for i in range(3):
		names_n.append(names(list_1[i]))
		paths_n.append(list_1[i])
		names_p.append(names(list_2[i]))
		paths_p.append(list_2[i])

	# Parameters---------------------------------------------------------------------
	overlap=1
	
	lenght_sounds=[4,8,12,20]
	

	# Validation and Test sets-------
	for lenght_sound in lenght_sounds:
		print(lenght_sound)
		for iter in range(1,3):
			if run_mode=='mfcc':
				# Non_Augment----------------------------------------------------------------
				print('Negatives---------')
				mfcc_augment.validation_non_augment(paths_n[iter], output_n[iter], names_n[iter],
				"_run_%d_%d" % (iter,lenght_sound) ,lenght_sound,overlap)
				print('Positives---------')
				mfcc_augment.validation_non_augment(paths_p[iter], output_p[iter], names_p[iter],
				"_run_%d_%d" % (iter,lenght_sound) ,lenght_sound,overlap)
			elif run_mode=='mel':
				# Non_Augment----------------------------------------------------------------
				First_augment.validation_non_augment(paths_n[iter], output_n[iter], names_n[iter],
				"_run_%d_%d" % (iter,lenght_sound) ,lenght_sound,overlap)
				First_augment.validation_non_augment(paths_p[iter], output_p[iter], names_p[iter],
				"_run_%d_%d" % (iter,lenght_sound) ,lenght_sound,overlap)
			elif run_mode=='tma':
				# Non_Augment----------------------------------------------------------------
				tma.validation_non_augment(paths_n[iter], output_n[iter], names_n[iter],
				"_run_%d_%d" % (iter,lenght_sound) ,lenght_sound,overlap)
				tma.validation_non_augment(paths_p[iter], output_p[iter], names_p[iter],
				"_run_%d_%d" % (iter,lenght_sound) ,lenght_sound,overlap)
	overlap=1

	lenght_sounds=[4,8,12,20]
	iter=0 # Training=0 validation=1 test=2

	print('Trainings----------')

	# Training sets-------------------
	for lenght_sound in lenght_sounds:
		print(lenght_sound)
		for loop_num in range(0,1): 
			if run_mode=='mfcc':
			# ----Augment----------------------------------------------------------------
				mfcc_augment.random_augment(paths_n[iter], output_n[iter], names_n[iter],
				"_run_%d_%d" % (loop_num,lenght_sound) ,lenght_sound,overlap)
				mfcc_augment.random_augment(paths_p[iter], output_p[iter], names_p[iter],
				"_run_%d_%d" % (loop_num,lenght_sound) ,lenght_sound,overlap)
			elif run_mode=='mel':
			# ----Augment----------------------------------------------------------------
				First_augment.random_augment(paths_n[iter], output_n[iter], names_n[iter],
				"_run_%d_%d" % (loop_num,lenght_sound) ,lenght_sound,overlap)
				First_augment.random_augment(paths_p[iter], output_p[iter], names_p[iter],
				"_run_%d_%d" % (loop_num,lenght_sound) ,lenght_sound,overlap)
			elif run_mode=='tma':
			# Non_Augment----------------------------------------------------------------
				tma.random_augment(paths_n[iter], output_n[iter], names_n[iter],
				"_run_%d_%d" % (loop_num,lenght_sound) ,lenght_sound,overlap)
				tma.random_augment(paths_p[iter], output_p[iter], names_p[iter],
				"_run_%d_%d" % (loop_num,lenght_sound) ,lenght_sound,overlap)

	stop = timeit.default_timer()
	print('Time: ', (stop - start)/3600 )

	def convert_RGB():
		folder = Path(output_dir)
		file_list=[]
		for file in folder.rglob('*.png'):
			file_list.append(file)

		for i in range(len(file_list)):
				im1 = Image.open(file_list[i]).convert('RGB')
				os.remove(file_list[i])
				im1.save(str(pathlib.PurePosixPath(file_list[i].parent,(os.path.splitext(file_list[i].name)[0] + ".png"))))

	convert_RGB()

