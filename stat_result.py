import os
import glob
import pickle
import multiprocessing as mp
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import ndimage
from scipy.ndimage.morphology import binary_erosion, binary_dilation

# import user defined library
from Util.data_process import load_3d_volume_as_array
from shape_analysis import get_contact_area, get_surface_area


def get_contact(volume):
    contact_pairs, contact_areas = get_contact_area(volume)

    return contact_pairs, contact_areas


def save_pd(pd, save_file):
    if not os.path.isdir(os.path.dirname(save_file)):
        os.makedirs(os.path.dirname(save_file))
    pd.to_csv(save_file)


# ==========================================
# codes with nucleus
# ==========================================
def stat_embryo(embryo_name):
    preresult_folder = "./ResultCell/BinaryMembPostseg"
    postresult_folder = "./ResultCell/BinaryMembPostseg"

    preresult_files = glob.glob(os.path.join(preresult_folder, embryo_name+"Cavity", "*.nii.gz"))
    preresult_files.sort()
    postresult_files = glob.glob(os.path.join(postresult_folder, embryo_name+"LabelUnified", "*.nii.gz"))
    postresult_files.sort()
    for index, postresult_file in enumerate(tqdm(postresult_files, desc="Process {}".format(embryo_name))):
        base_name = "_".join(os.path.basename(postresult_file).split("_")[:-1])

        preresult_file = os.path.join(preresult_folder, embryo_name+"Cavity", base_name + "_segCavity.nii.gz")
        postresult = load_3d_volume_as_array(postresult_file)

        if os.path.isfile(preresult_file):
            presult = load_3d_volume_as_array(preresult_file)
            # errors = np.logical_and(presult != 0, ~(postresult != 0)).astype(np.uint8)
            extra_label_volume = ndimage.label(presult)[0]
            extra_labels = np.unique(extra_label_volume).tolist()
            extra_labels.remove(0)

            # combine labels
            extra_label_init = 10000
            for idx, extra_label in enumerate(extra_labels):
                postresult[extra_label_volume == extra_label] = extra_label_init + idx

        # get surface, volume
        labels = []
        volumes = []
        surfaces = []
        postlabels = np.unique(postresult).tolist()
        postlabels.remove(0)
        for label in postlabels:
            surface = get_surface_area(postresult == label)

            volume = (postresult == label).sum()

            labels.append(label)
            volumes.append(volume)
            surfaces.append(surface)
        contact_pairs, contact_areas = get_contact(postresult)
        contact_surfaces = {contact_pairs[i][0]: {contact_pairs[i][1]: contact_areas[i]} for i in range(len(contact_areas))}

        # save to csv file
        save_volme_file = os.path.join("./Tem/Stat", embryo_name, base_name + "_surface_volume.csv")
        save_contact_file = os.path.join("./Tem/Stat", embryo_name, base_name + "_contact.csv")

        pd_surface_volume = pd.DataFrame.from_dict({"Label": labels, "Surface": surfaces, "Volume": volumes})
        pd_surface_volume.set_index("Label", inplace=True)
        save_pd(pd=pd_surface_volume, save_file=save_volme_file)
        pd_contact = pd.DataFrame.from_dict({i: contact_surfaces[i] for i in contact_surfaces.keys()}, orient="index")
        save_pd(pd=pd_contact, save_file=save_contact_file)

def stat_embryo_new_programe(embryo_name):
    postresult_folder = "/home/jeff/ProjectCode/LearningCell/MembProjectCode/output"

    postresult_files = glob.glob(os.path.join(postresult_folder, embryo_name, "SegCell", "*.nii.gz"))
    postresult_files.sort()
    for index, postresult_file in enumerate(tqdm(postresult_files, desc="Process analysis {}".format(embryo_name))):
        base_name = "_".join(os.path.basename(postresult_file).split("_")[:-1])

        postresult = load_3d_volume_as_array(postresult_file)

        # get surface, volume
        labels = []
        volumes = []
        surfaces = []
        postlabels = np.unique(postresult).tolist()
        postlabels.remove(0)
        for label in postlabels:
            surface = get_surface_area(postresult == label)

            volume = (postresult == label).sum()

            labels.append(label)
            volumes.append(volume)
            surfaces.append(surface)
        contact_pairs, contact_areas = get_contact(postresult)
        contact_surfaces = {contact_pairs[i][0]: {contact_pairs[i][1]: contact_areas[i]} for i in range(len(contact_areas))}

        # save to csv file
        save_volme_file = os.path.join("./Tem/Stat", embryo_name, base_name + "_surface_volume.csv")
        save_contact_file = os.path.join("./Tem/Stat", embryo_name, base_name + "_contact.csv")

        pd_surface_volume = pd.DataFrame.from_dict({"Label": labels, "Surface": surfaces, "Volume": volumes})
        pd_surface_volume.set_index("Label", inplace=True)
        save_pd(pd=pd_surface_volume, save_file=save_volme_file)
        pd_contact = pd.DataFrame.from_dict({i: contact_surfaces[i] for i in contact_surfaces.keys()}, orient="index")
        save_pd(pd=pd_contact, save_file=save_contact_file)

# ==========================================
# codes without nucleus
# ==========================================
def stat_embryo_no_nucleus(embryo_name):
    preresult_folder = "ResultCell/BinaryMembPostseg"

    preresult_files = glob.glob(os.path.join(preresult_folder, embryo_name, "*.nii.gz"))
    preresult_files.sort()
    for index, preresult_file in enumerate(tqdm(preresult_files, desc="Process analysis {}".format(embryo_name))):
        base_name = "_".join(os.path.basename(preresult_file).split("_")[:-1])

        preresult = load_3d_volume_as_array(preresult_file)

        # get surface, volume
        labels = []
        volumes = []
        surfaces = []
        postlabels = np.unique(preresult).tolist()
        postlabels.remove(0)


        for label in postlabels:
            surface = get_surface_area(preresult == label)

            volume = (preresult == label).sum()

            labels.append(label)
            volumes.append(volume)
            surfaces.append(surface)
        contact_surfaces = {}
        if len(postlabels) > 1:
            contact_pairs, contact_areas = get_contact(preresult)
            contact_surfaces = {contact_pairs[i][0]: {contact_pairs[i][1]: contact_areas[i]} for i in range(len(contact_areas))}

        # save to csv file
        save_volme_file = os.path.join("./Tem/Stat", embryo_name, base_name + "_surface_volume.csv")
        save_contact_file = os.path.join("./Tem/Stat", embryo_name, base_name + "_contact.csv")

        pd_surface_volume = pd.DataFrame.from_dict({"Label": labels, "Surface": surfaces, "Volume": volumes})
        pd_surface_volume.set_index("Label", inplace=True)
        save_pd(pd=pd_surface_volume, save_file=save_volme_file)
        pd_contact = pd.DataFrame.from_dict({i: contact_surfaces[i] for i in contact_surfaces.keys()}, orient="index")
        save_pd(pd=pd_contact, save_file=save_contact_file)


if __name__ == "__main__":
    # embryo_names = ["200314plc1p1", "181210plc1p3", "200314plc1p2"] + \
    #     "200309plc1p2, 200309plc1p3, 200310plc1p2, 200311plc1p1, 200315plc1p2, 200315plc1p3, 200316plc1p1, 200316plc1p2".split(",") + \
    #     "200309plc1p1, 200312plc1p2".split(",") + \
    #     "200311plc1p2, 200311plc1p3, 200312plc1p1, 200312plc1p3, 200314plc1p3, 200315plc1p1, 200316plc1p3, 181210plc1p1".split(",") + \
    #     "181210plc1p2, 170704plc1p1".split(",") + \
    #     ["200315plc1p1", "200311plc1p2", "200310plc1p2"]
    # embryo_names = [embryo_name.replace(" ", "") for embryo_name in embryo_names]

    # embryo_names = ["200710hmr1plc1p1", "200710hmr1plc1p2", "200710hmr1plc1p3"]
    # embryo_names = ["170704plc1p1", "181210plc1p1", "181210plc1p2", "181210plc1p3", "200309plc1p1",
    #                 "200309plc1p2", "200309plc1p3", "200310plc1p2", "200311plc1p1", "200311plc1p2",
    #                 "200311plc1p3", "200312plc1p1", "200312plc1p2", "200312plc1p3", "200314plc1p1",
    #                 "200314plc1p2", "200314plc1p3", "200315plc1p1", "200315plc1p2", "200315plc1p3",
    #                 "200316plc1p1", "200316plc1p2", "200316plc1p3"]
    #
    # embryo_names = ["181210plc1p3", "200309plc1p1",
    #                 "200309plc1p2", "200314plc1p1",
    #                 "200314plc1p2"]
    # stat_embryo_no_nucleus(embryo_names[0])
    # with mp.Pool(processes=16) as p:
    #     p.map(stat_embryo_new_programe, embryo_names)
    embryo_names = ["200113plc1p2"]
    for embryo_name in embryo_names:
        stat_embryo_new_programe(embryo_name)

    # ==========================================
    # Combine CSV files (volume and surface)
    # ==========================================
    for embryo_name in embryo_names:
        csv_folder = os.path.join("./Tem/Stat", embryo_name)

        save_folder = os.path.join("./Tem/Combined", os.path.basename(csv_folder))
        if not os.path.isdir(save_folder):
            os.makedirs(save_folder)

        # with open("ShapeUtil/name_dictionary.txt", "rb") as f:
        #     name_dict = pickle.load(f)

        # ----> for embryo_names = ["200113plc1p2"]
        with open("/home/jeff/ProjectCode/LearningCell/MembProjectCode/dataset/number_dictionary.txt", "rb") as f:
            name_dict = pickle.load(f)
        # combine surface and volumes
        csv_files = glob.glob(os.path.join(csv_folder, "*surface_volume.csv"))
        csv_files.sort()

        pd_surfaces = []
        pd_volumes = []
        for t, csv_file in enumerate(tqdm(csv_files, desc="Combine surface / volume {}".format(embryo_name))):
            pd_data = pd.read_csv(csv_file)

            cell_names = []
            for label in pd_data["Label"].values.tolist():
                if label in name_dict.keys():
                    cell_names.append(name_dict[label])
                else:
                    cell_names.append(label)

            pd_surface = pd.DataFrame(data=np.expand_dims(pd_data["Surface"], axis=0), index=[t+1], columns=cell_names)
            pd_volume = pd.DataFrame(data=np.expand_dims(pd_data["Volume"], axis=0), index=[t+1], columns=cell_names)

            pd_surfaces.append(pd_surface)
            pd_volumes.append(pd_volume)

        surface = pd.concat(pd_surfaces, axis=0, join="outer")
        surface.to_csv(os.path.join(save_folder, "_".join([os.path.basename(csv_folder), "surface.csv"])))
        volume = pd.concat(pd_volumes, axis=0, join="outer")
        volume.to_csv(os.path.join(save_folder, "_".join([os.path.basename(csv_folder), "volume.csv"])))


    # ==========================================
    # Combine CSV files (contact)
    # ==========================================
    for embryo_name in embryo_names:
        csv_folder = os.path.join("./Tem/Stat", embryo_name)

        save_folder = os.path.join("./Tem/Combined", os.path.basename(csv_folder))
        if not os.path.isdir(save_folder):
            os.makedirs(save_folder)

        # with open("ShapeUtil/name_dictionary.txt", "rb") as f:
        #     name_dict = pickle.load(f)

        # ----> for embryo_names = ["200113plc1p2"]
        with open("/home/jeff/ProjectCode/LearningCell/MembProjectCode/dataset/number_dictionary.txt", "rb") as f:
            name_dict = pickle.load(f)
        # # combine surface and volumes
        csv_files = glob.glob(os.path.join(csv_folder, "*contact.csv"))
        csv_files.sort()

        pd_contacts = []
        for t, csv_file in enumerate(tqdm(csv_files, desc="Combining contact {}".format(embryo_name))):
            pd_data = pd.read_csv(csv_file, header=0, index_col=0)
            columns = list(map(lambda x: int(float(x)), pd_data.columns.values))
            rows = list(map(lambda x: int(float(x)), pd_data.index.values))
            name_columns = [name_dict[i] if i < 10000 else i for i in columns]
            name_rows = [name_dict[i] if i < 10000 else i for i in rows]
            pd_data.columns = name_columns
            pd_data.index = name_rows

            pd_data = pd_data.stack().to_frame().T
            pd_data.index = [t+1]

            pd_contacts.append(pd_data)

        surface = pd.concat(pd_contacts, axis=0, join="outer")
        surface.to_csv(os.path.join(save_folder, "_".join([os.path.basename(csv_folder), "contact.csv"])))




# pd_name = pd.read_csv("/home/jeff/ProjectCode/LearningCell/MembProjectCode/dataset/number_dictionary.csv", names=["Name", "Label"])
# pd_name = pd_name.drop(index=0).set_index("Label", drop=True)
# pd_dict = pd_name.to_dict()["Name"]
# with open("/home/jeff/ProjectCode/LearningCell/MembProjectCode/dataset/number_dictionary.txt", "wb") as f:
#     pickle.dump(pd_dict, f)





