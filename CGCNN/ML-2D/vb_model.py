def predict_vb(structure, predictor_path, model_path, root_path, *args, **kwargs):
    
    """
        This function run the prediction using an input structure
    
    structure: pymatgen structure
    predictor_path: path to predict.py file:   band_potential/cgcnn_vb_model/predict.py
    model_path: path to model:        band_potential/cgcnn_vb_model/model_best.pth.tar     
    root_path: path to root where you have atom_init_file.: band_potential/cgcnn_vb_model/test_prediction_root_dir
    
    return vb
    """
    
    import os
    import ast
    import csv
    import subprocess    
    from pymatgen import io 
    from pymatgen.core import Structure
    
    # writing cif file in root_dir:
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
