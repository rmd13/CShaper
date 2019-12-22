

import os
import nibabel
import time
import numpy as np
import random
import codecs, json
from scipy import ndimage
from Util.post_lib import get_seconde_largest
from scipy import ndimage, stats
from skimage import morphology
import SimpleITK as sitk
import nibabel as nib

def search_file_in_folder_list(folder_list, file_name):
    """
    Find the full filename from a list of folders
    inputs:
        folder_list: a list of folders
        file_name:  filename
    outputs:
        full_file_name: the full filename
    """
    file_exist = False
    for folder in folder_list:
        full_file_name = os.path.join(folder, file_name)
        if(os.path.isfile(full_file_name)):
            file_exist = True
            break
    if(file_exist == False):
        raise ValueError('{0:} is not found in {1:}'.format(file_name, folder))
    return full_file_name

def load_3d_volume_as_array(filename):
    if('.nii' in filename):
        return load_nifty_volume_as_array(filename)
    elif('.mha' in filename):
        return load_mha_volume_as_array(filename)
    raise ValueError('{0:} unspported file format'.format(filename))

def load_mha_volume_as_array(filename):
    img = sitk.ReadImage(filename)
    nda = sitk.GetArrayFromImage(img)
    return nda

def load_nifty_volume_as_array(filename, with_header = False):
    """
    load nifty image into numpy array, and transpose it based on the [z,y,x] axis order
    The output array shape is like [Depth, Height, Width]
    inputs:
        filename: the input file name, should be *.nii or *.nii.gz
        with_header: return affine and hearder infomation
    outputs:
        Data: a numpy Data array
    """
    img = nibabel.load(filename)
    data = img.get_data()
    data = np.transpose(data, [2,1,0])
    if(with_header):
        return data, img.affine, img.header
    else:
        return data

def save_array_as_nifty_volume(data, filename, reference_name = None):
    """
    save a numpy array as nifty image
    inputs:
        Data: a numpy array with shape [Depth, Height, Width]
        filename: the ouput file name
        reference_name: file name of the reference image of which affine and header are used
    outputs: None
    """
    img = sitk.GetImageFromArray(data)
    folder_name = os.path.dirname(filename)
    if '.gz.gz' in filename:
        filename = filename[:-3]  # prevent possible input with '*.nii.gz'
    if not os.path.isdir(folder_name):
        os.makedirs(folder_name)
    if(reference_name is not None):
        img_ref = sitk.ReadImage(reference_name)
        img.CopyInformation(img_ref)
    sitk.WriteImage(img, filename)


def itensity_normalize_one_volume(volume):
    """
    normalize the itensity of an nd volume based on the mean and std of nonzeor region
    inputs:
        volume: the input nd volume
    outputs:
        out: the normalized nd volume
    """
    
    pixels = volume[volume > 0]
    mean = pixels.mean()
    std  = pixels.std()
    out = (volume - mean)/std
    out_random = np.random.normal(0, 1, size = volume.shape)
    out[volume == 0] = out_random[volume == 0]
    return out

def get_ND_bounding_box(label, margin):
    """
    get the bounding box of the non-zero region of an ND volume
    """
    input_shape = label.shape
    if(type(margin) is int ):
        margin = [margin]*len(input_shape)  # Extend the margin to multi-dimension
    assert(len(input_shape) == len(margin))
    indxes = np.nonzero(label)  # Get the location of all non-zero pixels
    idx_min = []
    idx_max = []
    for i in range(len(input_shape)):  # Get the boundary on each dimension
        idx_min.append(indxes[i].min())
        idx_max.append(indxes[i].max())

    for i in range(len(input_shape)):  # Extend the region by 5 pixels
        idx_min[i] = max(idx_min[i] - margin[i], 0)
        idx_max[i] = min(idx_max[i] + margin[i], input_shape[i] - 1) # index from 0
    return idx_min, idx_max

def crop_ND_volume_with_bounding_box(volume, min_idx, max_idx):
    """
    crop/extract a subregion form an nd image.
    """
    dim = len(volume.shape)
    assert(dim >= 2 and dim <= 5)  # only deal with 2,3,4 dimensions
    if(dim == 2):
        output = volume[np.ix_(range(min_idx[0], max_idx[0] + 1),
                               range(min_idx[1], max_idx[1] + 1))]
    elif(dim == 3):
        output = volume[np.ix_(range(min_idx[0], max_idx[0] + 1),
                               range(min_idx[1], max_idx[1] + 1),
                               range(min_idx[2], max_idx[2] + 1))]
    elif(dim == 4):
        output = volume[np.ix_(range(min_idx[0], max_idx[0] + 1),
                               range(min_idx[1], max_idx[1] + 1),
                               range(min_idx[2], max_idx[2] + 1),
                               range(min_idx[3], max_idx[3] + 1))]
    elif(dim == 5):
        output = volume[np.ix_(range(min_idx[0], max_idx[0] + 1),
                               range(min_idx[1], max_idx[1] + 1),
                               range(min_idx[2], max_idx[2] + 1),
                               range(min_idx[3], max_idx[3] + 1),
                               range(min_idx[4], max_idx[4] + 1))]
    else:
        raise ValueError("the dimension number shoud be 2 to 5")
    return output  # Return cropped volume from in the box

def set_ND_volume_roi_with_bounding_box_range(volume, bb_min, bb_max, sub_volume):
    """
    set a subregion to an nd image.
    """
    dim = len(bb_min)
    out = volume
    if(dim == 2):
        out[np.ix_(range(bb_min[0], bb_max[0] + 1),
                   range(bb_min[1], bb_max[1] + 1))] = sub_volume
    elif(dim == 3):
        out[np.ix_(range(bb_min[0], bb_max[0] + 1),
                   range(bb_min[1], bb_max[1] + 1),
                   range(bb_min[2], bb_max[2] + 1))] = sub_volume
    elif(dim == 4):
        out[np.ix_(range(bb_min[0], bb_max[0] + 1),
                   range(bb_min[1], bb_max[1] + 1),
                   range(bb_min[2], bb_max[2] + 1),
                   range(bb_min[3], bb_max[3] + 1))] = sub_volume
    else:
        raise ValueError("array dimension should be 2, 3 or 4")
    return out

def convert_label(in_volume, label_convert_source, label_convert_target):
    """
    convert the label value in a volume
    inputs:
        in_volume: input nd volume with label set label_convert_source
        label_convert_source: a list of integers denoting input labels, e.g., [0, 1, 2, 4]
        label_convert_target: a list of integers denoting output labels, e.g.,[0, 1, 2, 3]
    outputs:
        out_volume: the output nd volume with label set label_convert_target
    """
    mask_volume = np.zeros_like(in_volume)
    convert_volume = np.zeros_like(in_volume)
    for i in range(len(label_convert_source)):
        source_lab = label_convert_source[i]
        target_lab = label_convert_target[i]
        if(source_lab != target_lab):
            temp_source = np.asarray(in_volume == source_lab)
            temp_target = target_lab * temp_source
            mask_volume = mask_volume + temp_source
            convert_volume = convert_volume + temp_target
    out_volume = in_volume * 1
    out_volume[mask_volume>0] = convert_volume[mask_volume>0]  # Only change value that changed
    return out_volume
        
def get_random_roi_sampling_center(input_shape, output_shape, sample_mode, bounding_box = None):
    """
    get a random coordinate representing the center of a roi for sampling
    inputs:
        input_shape: the shape of sampled volume
        output_shape: the desired roi shape
        sample_mode: 'valid': the entire roi should be inside the input volume
                     'full': only the roi centre should be inside the input volume
        bounding_box: the bounding box which the roi center should be limited to
    outputs:
        center: the output center coordinate of a roi
    """
    center = []
    for i in range(len(input_shape)):
        if(sample_mode[i] == 'full'):
            if(bounding_box):
                x0 = bounding_box[i*2]; x1 = bounding_box[i*2 + 1]
            else:
                x0 = 0; x1 = input_shape[i]
        else:
            if(bounding_box):
                x0 = bounding_box[i*2] + int(output_shape[i]/2)   
                x1 = bounding_box[i*2+1] - int(output_shape[i]/2)   
            else:
                x0 = int(output_shape[i]/2)   
                x1 = input_shape[i] - x0
        if(x1 <= x0):
            centeri = int((x0 + x1)/2)
        else:
            centeri = random.randint(x0, x1)
        center.append(centeri)
    return center    

def transpose_volumes(volume, slice_direction):
    """
    transpose a list of volumes
    inputs:
        volumes: a list of nd volumes
        slice_direction: 'axial', 'sagittal', or 'coronal'
    outputs:
        tr_volumes: a list of transposed volumes
    """
    if (slice_direction == 'axial'):
        tr_volumes = volume
    elif(slice_direction == 'sagittal'):
        tr_volumes = np.transpose(volume, (2, 0, 1))
    elif(slice_direction == 'coronal'):
        tr_volumes = np.transpose(volume, (1, 0, 2))
    else:
        print('undefined slice direction:', slice_direction)
        tr_volumes = volume
    return tr_volumes


def transpose_volumes_reverse(volume, slice_direction):
    """
    transpose a list of volumes
    inputs:
        volumes: a list of nd volumes
        slice_direction: 'axial', 'sagittal', or 'coronal'
    outputs:
        tr_volumes: a list of transposed volumes
    """
    if (slice_direction == 'axial'):
        tr_volumes = volume
    elif(slice_direction == 'sagittal'):
        tr_volumes = np.transpose(volume, (1, 2, 0))
    elif(slice_direction == 'coronal'):
        tr_volumes = np.transpose(volume, (1, 0, 2))
    else:
        print('undefined slice direction:', slice_direction)
        tr_volumes = volume
    return tr_volumes



def resize_ND_volume_to_given_shape(volume, out_shape, order = 3):
    """
    resize an nd volume to a given shape
    inputs:
        volume: the input nd volume, an nd array
        out_shape: the desired output shape, a list
        order: the order of interpolation
    outputs:
        out_volume: the reized nd volume with given shape
    """
    shape0=volume.shape
    assert(len(shape0) == len(out_shape))
    scale = [(out_shape[i] + 0.0)/shape0[i] for i in range(len(shape0))]
    out_volume = ndimage.interpolation.zoom(volume, scale, order = order)
    return out_volume

def resize_ND_thin_mask_to_given_shape(in_volume, out_shape):
    """
    Resize thin segMemb without distortion
    inputs:
        in_volume: input Data segMemb
        out_shape: target segMemb shape
    outputs:
        out_volume: resized segMemb
    """
    shape0 = in_volume.shape
    out_volume = np.zeros(out_shape)
    assert(len(shape0) == len(out_shape))

    # Get white pixels idx after resize
    scale = [(out_shape[i] + 0.0)/shape0[i] for i in range(len(shape0))]
    [s_idxes, r_idxes, z_idxes] = np.nonzero(in_volume)
    s_resized_idxes = np.floor(s_idxes * scale[0]).astype(int)
    s_resized_idxes[s_resized_idxes < 0] = 0
    s_resized_idxes[s_resized_idxes > out_shape[0]-1] = out_shape[0] - 1

    r_resized_idxes = np.floor(r_idxes * scale[1]).astype(int)
    r_resized_idxes[r_resized_idxes < 0] = 0
    r_resized_idxes[r_resized_idxes > out_shape[1]-1] = out_shape[1] - 1

    z_resized_idxes = np.floor(z_idxes * scale[2]).astype(int)
    z_resized_idxes[z_resized_idxes < 0] = 0
    z_resized_idxes[z_resized_idxes > out_shape[2]-1] = out_shape[2] - 1

    # Use linear index for speed
    linex = np.ravel_multi_index(np.vstack((s_resized_idxes, r_resized_idxes, z_resized_idxes)), out_shape)
    flatted_volcume = out_volume.flatten()
    flatted_volcume[linex] = 1
    out_volume = np.reshape(flatted_volcume, out_shape)

    return out_volume


def extract_roi_from_volume(volume, in_center, output_shape, fill = 'random'):
    """
    extract a roi from a 3d volume
    inputs:
        volume: the input 3D volume
        in_center: the center of the roi
        output_shape: the size of the roi
        fill: 'random' or 'zero', the mode to fill roi region where is outside of the input volume
    outputs:
        output: the roi volume
    """
    input_shape = volume.shape   
    if(fill == 'random'):
        output = np.random.normal(0, 1, size = output_shape)
    else:
        output = np.zeros(output_shape)
    r0max = [int(x/2) for x in output_shape]  # If slicer center number is out of the range, it should be sliced based on shape
    r1max = [output_shape[i] - r0max[i] for i in range(len(r0max))]  # r0max=r1max when shape is even
    r0 = [min(r0max[i], in_center[i]) for i in range(len(r0max))]
    r1 = [min(r1max[i], input_shape[i] - in_center[i]) for i in range(len(r0max))]
    out_center = r0max

    # If there are valid layers in the volume, we sample with the center locating at the label_shape center. Otherwise,
    # layers outside of the volume are filled with random noise. In_center should always be the center at the new volume.
    output[np.ix_(range(out_center[0] - r0[0], out_center[0] + r1[0]),
                  range(out_center[1] - r0[1], out_center[1] + r1[1]),
                  range(out_center[2] - r0[2], out_center[2] + r1[2]))] = \
        volume[np.ix_(range(in_center[0] - r0[0], in_center[0] + r1[0]),
                      range(in_center[1] - r0[1], in_center[1] + r1[1]),
                      range(in_center[2] - r0[2], in_center[2] + r1[2]))]
    return output

def set_roi_to_volume(volume, center, sub_volume):
    """
    set the content of an roi of a 3d/4d volume to a sub volume
    inputs:
        volume: the input 3D/4D volume
        center: the center of the roi
        sub_volume: the content of sub volume
    outputs:
        output_volume: the output 3D/4D volume
    """
    volume_shape = volume.shape   
    patch_shape = sub_volume.shape
    output_volume = volume
    for i in range(len(center)):
        if(center[i] >= volume_shape[i]):  # If the length on any dimension is bigger than the shape, return the whole
            return output_volume
    r0max = [int(x/2) for x in patch_shape]
    r1max = [patch_shape[i] - r0max[i] for i in range(len(r0max))]
    r0 = [min(r0max[i], center[i]) for i in range(len(r0max))]
    r1 = [min(r1max[i], volume_shape[i] - center[i]) for i in range(len(r0max))]
    patch_center = r0max

    if(len(center) == 3):
        output_volume[np.ix_(range(center[0] - r0[0], center[0] + r1[0]),
                             range(center[1] - r0[1], center[1] + r1[1]),
                             range(center[2] - r0[2], center[2] + r1[2]))] = \
            sub_volume[np.ix_(range(patch_center[0] - r0[0], patch_center[0] + r1[0]),
                              range(patch_center[1] - r0[1], patch_center[1] + r1[1]),
                              range(patch_center[2] - r0[2], patch_center[2] + r1[2]))]
    elif(len(center) == 4):
        output_volume[np.ix_(range(center[0] - r0[0], center[0] + r1[0]),
                             range(center[1] - r0[1], center[1] + r1[1]),
                             range(center[2] - r0[2], center[2] + r1[2]),
                             range(center[3] - r0[3], center[3] + r1[3]))] = \
            sub_volume[np.ix_(range(patch_center[0] - r0[0], patch_center[0] + r1[0]),
                              range(patch_center[1] - r0[1], patch_center[1] + r1[1]),
                              range(patch_center[2] - r0[2], patch_center[2] + r1[2]),
                              range(patch_center[3] - r0[3], patch_center[3] + r1[3]))]
    else:
        raise ValueError("array dimension should be 3 or 4")        
    return output_volume  

def get_largest_two_component(img, print_info = False, threshold = None):
    """
    Get the largest two components of a binary volume
    inputs:
        img: the input 3D volume
        threshold: a size threshold
    outputs:
        out_img: the output volume 
    """
    s = ndimage.generate_binary_structure(3,2) # iterate structure
    labeled_array, numpatches = ndimage.label(img,s) # labeling
    sizes = ndimage.sum(img,labeled_array,range(1,numpatches+1)) 
    sizes_list = [sizes[i] for i in range(len(sizes))]
    sizes_list.sort()
    if(print_info):
        print('component size', sizes_list)
    if(len(sizes) == 1):
        out_img = img
    else:
        if(threshold):
            out_img = np.zeros_like(img)
            for temp_size in sizes_list:
                if(temp_size > threshold):
                    temp_lab = np.where(sizes == temp_size)[0] + 1
                    temp_cmp = labeled_array == temp_lab
                    out_img = (out_img + temp_cmp) > 0
            return out_img
        else:    
            max_size1 = sizes_list[-1]
            max_size2 = sizes_list[-2]
            max_label1 = np.where(sizes == max_size1)[0] + 1
            max_label2 = np.where(sizes == max_size2)[0] + 1
            component1 = labeled_array == max_label1
            component2 = labeled_array == max_label2
            if(max_size2*10 > max_size1):
                component1 = (component1 + component2) > 0
            out_img = component1
    return out_img

def fill_holes(img):
    """
    filling small holes of a binary volume with morphological operations
    """
    neg = 1 - img
    s = ndimage.generate_binary_structure(3,1) # iterate structure
    labeled_array, numpatches = ndimage.label(neg,s) # labeling
    sizes = ndimage.sum(neg,labeled_array,range(1,numpatches+1)) 
    sizes_list = [sizes[i] for i in range(len(sizes))]
    sizes_list.sort()
    max_size = sizes_list[-1]
    max_label = np.where(sizes == max_size)[0] + 1
    component = labeled_array == max_label
    return 1 - component


def remove_external_core(lab_main, lab_ext):
    """
    remove the core region that is outside of whole tumor
    """
    
    # for each component of lab_ext, compute the overlap with lab_main
    s = ndimage.generate_binary_structure(3,2) # iterate structure
    labeled_array, numpatches = ndimage.label(lab_ext,s) # labeling
    sizes = ndimage.sum(lab_ext,labeled_array,range(1,numpatches+1)) 
    sizes_list = [sizes[i] for i in range(len(sizes))]
    new_lab_ext = np.zeros_like(lab_ext)
    for i in range(len(sizes)):
        sizei = sizes_list[i]
        labeli =  np.where(sizes == sizei)[0] + 1
        componenti = labeled_array == labeli
        overlap = componenti * lab_main
        if((overlap.sum()+ 0.0)/sizei >= 0.5):
            new_lab_ext = np.maximum(new_lab_ext, componenti)
    return new_lab_ext

def binary_dice3d(s,g):
    """
    dice score of 3d binary volumes
    inputs: 
        s: segmentation volume
        g: ground truth volume
    outputs:
        dice: the dice score
    """
    assert(len(s.shape)==3)
    [Ds, Hs, Ws] = s.shape
    [Dg, Hg, Wg] = g.shape
    assert(Ds==Dg and Hs==Hg and Ws==Wg)
    prod = np.multiply(s, g)
    s0 = prod.sum()
    s1 = s.sum()
    s2 = g.sum()
    dice = (2.0*s0 + 1e-10)/(s1 + s2 + 1e-10)
    return dice


def discrete_transform(continuous_img, non_linear = True, num_samples = 16):
    """
    Discretization on continous image.
    Input:
        continuous_img: rawMemb image with continous value on pixel
        num_samples: the number of discretization
    Return:
        discrete_img: discreted image
    """
    if non_linear:
        continuous_img  = np.exp(continuous_img)
    min_value = np.amin(continuous_img)
    max_value = np.amax(continuous_img)
    bins = np.linspace(min_value, max_value + np.finfo(float).eps, num=num_samples+1)
    discrete_img = np.digitize(continuous_img, bins) - 1


    return discrete_img


def binary_to_EDT_3D(binary_image, valid_edt_width, discrete_num_bins = 0):
    """
    Transform binary 3D segMemb into distance transform segMemb.
    inputs:
        binary_image: 3D bin
    outputs:
        EDT_image: distance transoformation of the image

    """
    assert len(binary_image.shape)==3, 'Input for EDT shoulb be 3D volume'
    if (discrete_num_bins==2):
        return binary_image
    edt_image = ndimage.distance_transform_edt(binary_image==0)

    # Cut out two large EDT far away from the binary segMemb
    original_max_edt = np.max(edt_image)
    target_max_edt = min(original_max_edt, valid_edt_width)  # Change valid if given is too lagre
    valid_revised_edt = np.maximum(target_max_edt - edt_image, 0) / target_max_edt
    if(discrete_num_bins):
        discrete_revised_edt = discrete_transform(valid_revised_edt, non_linear=True, num_samples=discrete_num_bins)
    else:
        discrete_revised_edt = valid_revised_edt

    return discrete_revised_edt

def post_process_on_edt(edt_image):
    """
    Threshold the distance map and get the presegmentation results
    Input:
        edt_image: distance map from EDT or net prediction
        edt_threshold: threshold value
    Output:
        threshold_segmentation: result after threshold
    """
    max_in_map = np.max(edt_image)
    assert max_in_map, 'Given threshold should be smaller the maximum in EDT'
    post_segmentation = np.zeros_like(edt_image, dtype=np.uint16)
    post_segmentation[edt_image == max_in_map] = 1
    largestCC = get_seconde_largest(post_segmentation)

    # Close operation on the thresholded image
    struct = ndimage.generate_binary_structure(3, 2)  # Generate binary structure for morphological operations
    final_seg = ndimage.morphology.binary_closing(largestCC, structure=struct).astype(np.uint16)

    return final_seg

def delete_isolate_labels(discrete_edt):

    # delete all unconnected binary segMemb
    label_structure = np.ones((3, 3, 3))
    [labelled_edt, _]= ndimage.label(discrete_edt, label_structure)

    # get the largest connected label
    [sr, sc, sz] = discrete_edt.shape
    sample_area = labelled_edt[np.ix_(range(int(sr/2) - 10, int(sr/2) + 10),
                                      range(int(sc/2) - 10, int(sc/2) + 10),
                                      range(int(sz/2) - 10, int(sz/2) + 10))]
    [most_label, _] = stats.mode(sample_area, axis=None)

    valid_edt_mask0 = (labelled_edt == most_label[0])
    valid_edt_mask = ndimage.morphology.binary_closing(valid_edt_mask0, iterations=2)
    filtered_edt = np.copy(discrete_edt)
    filtered_edt[valid_edt_mask == 0] = 0


    return filtered_edt


#===================================================================================#
#                               library for web GUI Data
#===================================================================================#
def save_numpy_as_json(np_data, save_file, surface_only = True):
    """
    Save python numpy Data as json for web GUI
    :param np_data: numpy variable (should be cell segMemb embedded with embryo)
    :param save_file: save file name
    :param surface_only: whether exact the surface first and save surface points as json file
    :return:
    """
    if surface_only:
        np_data = get_cell_surface_mask(np_data)
    nonzero_loc = np.nonzero(np_data)
    nonzero_value = np_data[np_data!=0]
    loc_and_val = np.vstack(nonzero_loc + (nonzero_value,)).transpose().tolist()
    loc_and_val.insert(0, list((-1,) + np_data.shape))  # write volume size at the first location
    json.dump(loc_and_val, codecs.open(save_file, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4)


def get_cell_surface_mask(cell_volume):
    """
    Extract cell surface segMemb from the volume segmentation
    :param cell_volume: cell volume segMemb with the membrane embedded
    :return cell_surface: cell surface with only surface pixels
    """
    cell_mask = cell_volume == 0
    strel = morphology.ball(2)
    dilated_cell_mask = ndimage.binary_dilation(cell_mask, strel, iterations=1)
    surface_mask = np.logical_and(~cell_mask, dilated_cell_mask)
    surface_seg = cell_volume
    surface_seg[~surface_mask] = 0

    return surface_seg

#===================================================================#
#                    For testing library function
#===================================================================#
if __name__=="__main__":
    start_time = time.time()
    seg = nib.load("/home/jeff/ProjectCode/LearningCell/DMapNet/ResultCell/test_embryo_robust/BothWithRandomnetPostseg/181210plc1p2_volume_recovered/membT4CellwithMmeb.nii.gz").get_fdata()
    save_numpy_as_json(seg, "/home/jeff/ProjectCode/LearningCell/DMapNet/jsonSample4.json")
    print("runing time: {}s".format(time.time() - start_time))
