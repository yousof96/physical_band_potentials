
def ml_predict_vb(structure, predictor_path, model_path, root_path, *args, **kwargs):
    
    """
        This function run the prediction using an input structure
    
    structure: pymatgen structure of slab
    predictor_path: path to predict.py file:   band_potential/cgcnn_vb_model/predict.py
    model_path: path to model:        band_potential/cgcnn_vb_model/model_best.pth.tar path to this file.    
    root_path: path to root where you have atom_init_file.: cgcnn_prediction_dir path to this file.
    
    return vb in e.v. respect to vacuum
    """
    
    import os
    import ast
    import csv
    import subprocess    
    from pymatgen import io 
    from pymatgen.core import Structure
    
    # # writing cif file in root_path:
    structure.to(filename=os.path.join(root_path, "mp-1.cif"))
    structure.to(filename=os.path.join(root_path, "1.cif"))
    
    # import csv
    with open(os.path.join(root_path, "id_prop.csv"), 'w') as f:
        writer = csv.writer(f)
        writer.writerow((1, 0))
    
    return_code = subprocess.check_output(
        f"python3.9 {predictor_path} {model_path} {root_path}", 
        universal_newlines=True,
        shell=True
    )

    
    return ast.literal_eval(return_code.split("\n")[-2].split(":")[-1].strip())[0]



def my_slab_generator(ase_structure, miller_index, num_layers, vacuum=15, slab_dim=(2, 2, 1)):
    
    """
    ase_structure: bukl structure in ase format
    miller_index:  miller index in tuple format
    num_layers: number of slab layers (slab thickness)
    
    return the slab in ase format.
    """

    from ase.build import surface, make_supercell
    
    slab_structure = surface(lattice=ase_structure, indices=miller_index, layers=num_layers)

    # add vacume to top and bottom.
    slab_structure.center(vacuum=vacuum/2, axis=2)  # this put molecume in the center. thats why i used half of vacume.

    slab_dim = np.array(
        [
            [slab_dim[0], 0, 0],  # Repeat in x direction 
            [0, slab_dim[1], 0],  # Repeat in y direction 
            [0, 0, slab_dim[2]]   # Repeat in z direction
        ]
    )
    
    return make_supercell(slab_structure, slab_dim)



def predict_band_diagram(my_struc):
    
    ml_predict_vb
    my_slab_generator
    import predict_bandgap
    from pymatgen.io.ase import AseAtomsAdaptor
    from ase import io as ase_io
    import itertools
    import numpy as np

    Set following path according to saved locations.
    BG2_MODEL_PATH = "path_to_model_file" 
    BG2_PREDICTION_DIR = "path_to_predictor_directory"
    VB_3D_MODEL_PATH = "path_to_model_file"
    VB_3D_PREDICTION_DIR = "path_to_predictor_directory"
    VB_3D_PREDICTOR_FILE_PATH = "path_to_predict.py_file"


    list_of_miller_indexes = [(0, 0, 1), (1, 0, 0), (1, 2, 0), (0, 2, 0)] # Add more if wanted
    miller_index = (0, 0, 1)
    num_layers = 30  # To be caonsidered as 3D 
    vacuum = 15 # Angestrum (half on top of slab and half on bottom of slab)
    slab_dim = (2, 2, 1)

    
    _,bulk_bg_M2_relax = predict_bandgap.BG(
        AseAtomsAdaptor.get_structure(my_struc),
        model = BG2_MODEL_PATH,
        predict_dir = BG2_PREDICTION_DIR
    )

    vbm_3d_slab = []
    for miller_index_ in list_of_miller_indexes:
        try:
        
            my_3d_slab = my_slab_generator(
                ase_structure=my_struc,
                miller_index=miller_index_,
                num_layers=num_layers,
                vacuum=vacuum,
                slab_dim=slab_dim
            )
            # calculate VBM
            vbm_3d_slab.append(ml_predict_vb(
                structure=AseAtomsAdaptor.get_structure(my_3d_slab),
                predictor_path=VB_3D_PREDICTOR_FILE_PATH,
                model_path=VB_3D_MODEL_PATH,
                root_path=VB_3D_PREDICTION_DIR
            ))

            user_message = "Successful 3D slab bs calcualtion."
            
            

        except BaseException as e:
            user_message = "Error occured."
            print("3D slab VBM calculations Error: ")
            print(e)
            pass

    try:           

        # # Calculate  SALB BG
        _,bg_M2_relax = predict_bandgap.BG(
            AseAtomsAdaptor.get_structure(my_3d_slab),
            model = BG2_MODEL_PATH,
            predict_dir = BG2_PREDICTION_DIR
        )
    except BaseException as e:
        print("BG calcultions failed. ")
        print(e)
        pass


    print("="*50)
    print(
        "\n ML predicted band diagram with reference to Vacuum potential", 
        "\nBG M2 bulk (eV): ", bulk_bg_M2_relax, 
        "\nBG M2 SLAB (eV): ", bg_M2_relax[0],
        "\nVBM 3d mean (V vs vacuum): ", np.mean(vbm_3d_slab),
        "CBM mean with bulk BG (V vs vacuum): ", np.mean(vbm_3d_slab) + bulk_bg_M2_relax,
        "CBM mean with SLAB BG (V vs vacuum): ", np.mean(vbm_3d_slab) + bg_M2_relax[0],
    )



if __name__ == '__main__':
    
    # my_struc = Ase Atom object of bulk primitive structure (reas using ase.io.read)
    predict_band_diagram(my_struc)
