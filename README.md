# physical_band_potentials
This repository contains code and data for "Full prediction of band potentials in semiconductor materials" manuscript.

if you use finding of this work, kindly cite the following paper.



# Traning
for training use the root directory to store slab cif files. also provide the targets with cif file name as id_prop.csv file.
using following command, training from scrach can be done.


~ python3.9 main.py --train-size 4000 --val-size 673 --test-size 673 --epochs 3000 --batch-size 64  --optim 'Adam' --lr 0.0024156  --n-conv 3 --n-h 5 --atom-fea-len 64 --h-fea-len 64  --disable-cuda --task 'regression' ./root_dir/




# Transfer learning
to retrain the model, after preparing the input cif file and targes, run follwing commands. \n
\n

~ python3.9 main.py --train-size 4000 --val-size 673 --test-size 673 --epochs 3000 --start-epoch 1000 --batch-size 64  --optim 'Adam' --lr 0.0024156  --n-conv 3 --n-h 5 --atom-fea-len 64 --h-fea-len 64  --disable-cuda --task 'regression' --resume checkpoint.pth.tar ./root_dir/ 


mind that start-epoch means the mximum number of epochs that base model was trained. --epochs means total number of epochs starting from 0.
--resume is path to the checkpoint.pth.tar file which belongs to the base model.



# Prediction
the prediction folder contains everything require for prediction of band diagram. 
uses the primary (or conventional) cif file of the desired structure and run following command. 
specify the address of the cif file to be read at the end of the band_diagram_prediction.py screipt to be loaded using ase.io.read module.
\n

~ python3.9 band_diagram_prediction.py

