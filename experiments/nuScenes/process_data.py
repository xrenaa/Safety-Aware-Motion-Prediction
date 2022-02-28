import os
import sys
import time
import multiprocessing
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib import cm
from pyquaternion import Quaternion
import argparse

# import nuscenes
sys.path.append('./nuscenes-devkit/python-sdk')
from nuscenes import NuScenes
from nuscenes.prediction import PredictHelper
from nuscenes.prediction.helper import convert_local_coords_to_global
from nuscenes.eval.prediction.splits import get_prediction_challenge_split
from nuscenes.prediction.input_representation.static_layers import StaticLayerRasterizer
from nuscenes.prediction.input_representation.agents import AgentBoxesWithFadedHistory
from nuscenes.prediction.input_representation.interface import InputRepresentation
from nuscenes.prediction.input_representation.combinators import Rasterizer
from nuscenes.prediction.helper import quaternion_yaw
from nuscenes.prediction.input_representation.utils import convert_to_pixel_coords, get_rotation_matrix, get_crops

def crop_img(img, meters_ahead = 40, meters_behind = 10, meters_left = 25, meters_right = 25, resolution = 0.1):
    buffer = max([meters_ahead, meters_behind, meters_left, meters_right]) * 2
    image_side_length = int(buffer/resolution)
    row_crop, col_crop = get_crops(meters_ahead, meters_behind, meters_left, meters_right, resolution, image_side_length)
    return img[row_crop, col_crop].astype('uint8')

def isPinRect(xP, yP, xA, yA, xB, yB, xC, yC, xD, yD):

    ABCD = 0.5 * abs((yA - yC)*(xD - xB) + (yB - yD)*(xA - xC))

    ABP = 0.5 * abs(xA*(yB - yP) + xB*(yP - yA) + xP*(yA - yB))
    BCP = 0.5 * abs(xB*(yC - yP) + xC*(yP - yB) + xP*(yB - yC))
    CDP = 0.5 * abs(xC*(yD - yP) + xD*(yP - yC) + xP*(yC - yD))
    DAP = 0.5 * abs(xD*(yA - yP) + xA*(yP - yD) + xP*(yD - yA))

    return not (ABCD + 0.1 < (ABP + BCP + CDP + DAP))

def resize_input(img, target = (128, 128), method = Image.NEAREST):
    # Image.BILINEAR
    img = Image.fromarray(img)
    img = img.resize(target, method) 
    return img

def visual_worst_case_map(worst_case_map):
    worst_case_map = 255 * (worst_case_map - 0)/ np.max(worst_case_map)
    plt.imshow(worst_case_map.astype(np.uint8))
    plt.show()
    
def paint_frame_at_time(sample_in_frame, center_agent_yaw, show = False):
    base_image = np.zeros((800,800,3)).astype(np.uint8)
    for sample_instance_token in sample_in_frame:
        sample_annotation = sample_in_frame[sample_instance_token]
        sample_xy = sample_annotation["sample_xy"]
        center_xy = sample_annotation["center_xy"]
        yaw_in_radians = sample_annotation["yaw_in_radians"]
        width_in_pixels = sample_annotation["width_in_pixels"]
        length_in_pixels = sample_annotation["length_in_pixels"]
        row_pixel, column_pixel = convert_to_pixel_coords(sample_xy,
                                                          center_xy,
                                                          (400, 400), 0.1)
        coord_tuple = ((column_pixel, row_pixel), (length_in_pixels, width_in_pixels), -yaw_in_radians * 180 / np.pi)
        box = cv2.boxPoints(coord_tuple)
        cv2.fillPoly(base_image, pts=[np.int0(box)], color=color)
    rotation_mat = get_rotation_matrix(base_image.shape, center_agent_yaw)
    rotated_image = cv2.warpAffine(base_image, rotation_mat, (base_image.shape[1], base_image.shape[0]))
    if show:
        plt.imshow(rotated_image)
        plt.show()
    return rotated_image
    
def simple_interpolation(object_1, object_2, slice_num = 5):
    object_1 = np.array(object_1)
    object_2 = np.array(object_2)
    data_list = []
    delta = (object_2 - object_1)/slice_num
    for i in range(slice_num):
        data_list.append(object_1 + i*delta)
    data_list = np.stack(data_list)    
    return data_list

def make_up_frame_dict(frame_dict):
    # this function make up the time frame that some_instance is not in some time
    time = len(frame_dict.keys()) - 1
    for i in range(time - 1):
        present = frame_dict[str(i)]
        future = frame_dict[str(i+1)]
        for sample_instance_token in present.keys():
            try: 
                future[sample_instance_token]
            except KeyError:
                frame_dict[str(i+1)][sample_instance_token] = present[sample_instance_token]
    return frame_dict

def interpolate_frame_dict(frame_dict, slice_num = 5):
    # first_make_up the frame
    frame_dict = make_up_frame_dict(frame_dict)
    # then make up the slice
    new_frame_dict = {}
    time = len(frame_dict.keys()) - 1
    for i in range(time - 1):
        # first create the time
        for j in range(slice_num):
            new_frame_dict[str(i*slice_num + j)] = {}
        # then make interpolation
        present = frame_dict[str(i)]
        future = frame_dict[str(i+1)]
        for sample_instance_token in present.keys():
            inter_xy = simple_interpolation(present[sample_instance_token]["sample_xy"], future[sample_instance_token]["sample_xy"], slice_num)
            inter_yaw = simple_interpolation(present[sample_instance_token]["yaw_in_radians"], future[sample_instance_token]["yaw_in_radians"], slice_num)
            center_xy = present[sample_instance_token]["center_xy"]
            width_in_pixels = present[sample_instance_token]["width_in_pixels"]
            length_in_pixels = present[sample_instance_token]["length_in_pixels"]
            
            for j_ in range(slice_num):
                if sample_instance_token not in new_frame_dict[str(i*slice_num + j_)].keys():
                    new_frame_dict[str(i*slice_num + j_)][sample_instance_token] = {}
                
                new_frame_dict[str(i*slice_num + j_)][sample_instance_token]["sample_xy"] = inter_xy[j_]
                new_frame_dict[str(i*slice_num + j_)][sample_instance_token]["yaw_in_radians"] = inter_yaw[j_]
                new_frame_dict[str(i*slice_num + j_)][sample_instance_token]["center_xy"] = center_xy
                new_frame_dict[str(i*slice_num + j_)][sample_instance_token]["width_in_pixels"] = width_in_pixels
                new_frame_dict[str(i*slice_num + j_)][sample_instance_token]["length_in_pixels"] = length_in_pixels
                
    return new_frame_dict

def draw_stack_imgs(new_frame_dict, center_agent_yaw):
    imgs = []
    time = len(new_frame_dict.keys())
    if "center_agent_yaw" in new_frame_dict.keys():
        time = time - 1
    
    for i in range(time):
        sample_in_frame = new_frame_dict[str(i)]
        rotated_image = paint_frame_at_time(sample_in_frame, center_agent_yaw)
        imgs.append(crop_img(rotated_image))
        
    return imgs

def my_func(index, split, img_size):
    instance_token, sample_token = mini_train[index].split("_")
    center_annotation = helper.get_sample_annotation(instance_token, sample_token)

    # set the unseen
    unseen_tokens = []

    ## first plot the rotate ones
    coord_tuple = ((400, 250), (500, 500), 0)
    box = cv2.boxPoints(coord_tuple)

    local_box = (box*np.array([1, -1]) - np.array([400, -400]))*0.1
    global_box = convert_local_coords_to_global(local_box, center_annotation['translation'], center_annotation['rotation'])

    sample_in_frame = helper.get_annotations_for_sample(sample_token)

    for sample_annotation in sample_in_frame:
        sample_instance_token = sample_annotation["instance_token"]
        sample_xy = sample_annotation["translation"][:2]

        if("vehicle" not in sample_annotation["category_name"]):
            continue

        if(isPinRect(sample_xy[0], sample_xy[1], global_box[0,0], global_box[0,1], global_box[1,0], global_box[1,1],\
                        global_box[2,0], global_box[2,1], global_box[3,0], global_box[3,1]) == False):

            future = helper.get_future_for_agent(sample_instance_token, sample_token, seconds=future_length, in_agent_frame=False)

            for i in range(future.shape[0]):
                temp_xy = future[i]
                if(isPinRect(temp_xy[0], temp_xy[1], global_box[0,0], global_box[0,1], global_box[1,0], global_box[1,1],\
                        global_box[2,0], global_box[2,1], global_box[3,0], global_box[3,1]) == True):
                    if sample_instance_token not in unseen_tokens:
                        unseen_tokens.append(sample_instance_token)
                    break

    # begin to generate worst case map
    frame_dict = {}
    frame_dict["0"] = {}

    map_layers = 255 - static_layer_rasterizer.make_map_representation(instance_token, sample_token)
    center_annotation = helper.get_sample_annotation(instance_token, sample_token)
    center_xy = center_annotation["translation"][:2]
    sample_in_frame = helper.get_annotations_for_sample(sample_token)

    for sample_annotation in sample_in_frame:
        sample_instance_token = sample_annotation["instance_token"]
        sample_xy = sample_annotation["translation"][:2]

        yaw_in_radians = quaternion_yaw(Quaternion(sample_annotation['rotation']))
        width_in_pixels = sample_annotation['size'][0]/0.1
        length_in_pixels = sample_annotation['size'][1]/0.1

        row_pixel, column_pixel = convert_to_pixel_coords(sample_xy,
                                                          center_xy,
                                                          (400,400), 0.1)

        coord_tuple = ((column_pixel, row_pixel), (length_in_pixels, width_in_pixels), -yaw_in_radians * 180 / np.pi)

        frame_dict["0"][sample_instance_token] = {}
        frame_dict["0"][sample_instance_token]["sample_xy"] = sample_xy
        frame_dict["0"][sample_instance_token]["center_xy"] = center_xy
        frame_dict["0"][sample_instance_token]["yaw_in_radians"] = yaw_in_radians
        frame_dict["0"][sample_instance_token]["width_in_pixels"] = width_in_pixels
        frame_dict["0"][sample_instance_token]["length_in_pixels"] = length_in_pixels

    center_agent_yaw = quaternion_yaw(Quaternion(center_annotation['rotation']))
    frame_dict["center_agent_yaw"] = center_agent_yaw

    future = helper.get_future_for_sample(sample_token, future_length, in_agent_frame=False, just_xy=False)

    for time in range(2*future_length):
        frame_dict[str(time+1)] = {}

        for sample_instance_token in future:

            sample_all = future[sample_instance_token]

            try:
                sample_annotation = sample_all[time]
            except IndexError:
                continue

            sample_xy = sample_annotation["translation"][:2]

            yaw_in_radians = quaternion_yaw(Quaternion(sample_annotation['rotation']))

            width_in_pixels = sample_annotation['size'][0]/0.1
            length_in_pixels = sample_annotation['size'][1]/0.1

            row_pixel, column_pixel = convert_to_pixel_coords(sample_xy,
                                                              center_xy,
                                                              (400,400), 0.1)
            coord_tuple = ((column_pixel, row_pixel), (length_in_pixels, width_in_pixels), -yaw_in_radians * 180 / np.pi)

            frame_dict[str(time+1)][sample_instance_token] = {}
            frame_dict[str(time+1)][sample_instance_token]["sample_xy"] = sample_xy
            frame_dict[str(time+1)][sample_instance_token]["center_xy"] = center_xy
            frame_dict[str(time+1)][sample_instance_token]["yaw_in_radians"] = yaw_in_radians
            frame_dict[str(time+1)][sample_instance_token]["width_in_pixels"] = width_in_pixels
            frame_dict[str(time+1)][sample_instance_token]["length_in_pixels"] = length_in_pixels


    input_img = mtp_input_representation.make_input_representation(instance_token, sample_token)
    resize_input_img = resize_input(input_img, (img_size, img_size))

    new_frame_dict = interpolate_frame_dict(frame_dict, slice_num)
    imgs = draw_stack_imgs(new_frame_dict, frame_dict["center_agent_yaw"])

    rasterize_img = imgs.copy()
    rasterize_img.append(map_layers)
    full_image = Rasterizer().combine(rasterize_img)

    worst_case_map = (interpolate_length + 1) * np.ones((img_size,img_size)).astype(np.uint8)
    resize_map = np.array(resize_input(map_layers, (img_size, img_size)))
    for x in range(img_size):
        for y in range(img_size):
            occupancy = resize_map[x, y]
            if occupancy[0] == 255:
                worst_case_map[x, y] = 0

    # then for the future occupancy
    for time, occupancy_map in enumerate(imgs):
        occupancy_map = np.array(resize_input(occupancy_map, (img_size, img_size)))
        for x in range(img_size):
            for y in range(img_size):
                occupancy = occupancy_map[x, y]
                if occupancy[0] == 255:
                    if time < worst_case_map[x, y]:
                        worst_case_map[x, y] = time

    # begin to generate unseen mask
    base_image = np.zeros((800,800,3)).astype(np.uint8)

    center_xy = center_annotation["translation"][:2]
    center_agent_yaw = quaternion_yaw(Quaternion(center_annotation['rotation']))
    rotation_mat = get_rotation_matrix(base_image.shape, center_agent_yaw)

    future = helper.get_future_for_sample(sample_token, future_length, in_agent_frame=False, just_xy=False)

    for unseen_instance_token in unseen_tokens:
        sample_all = future[unseen_instance_token]

        # for time in range(2*future_length):
        for time in range(len(sample_all)):

            sample_annotation = sample_all[time]
            
            sample_xy = sample_annotation["translation"][:2]
            yaw_in_radians = quaternion_yaw(Quaternion(sample_annotation['rotation']))
            width_in_pixels = sample_annotation['size'][0]/0.1
            length_in_pixels = sample_annotation['size'][1]/0.1

            row_pixel, column_pixel = convert_to_pixel_coords(sample_xy,
                                                              center_xy,
                                                              (400,400), 0.1)

            coord_tuple = ((column_pixel, row_pixel), (length_in_pixels, width_in_pixels), -yaw_in_radians * 180 / np.pi)
            box = cv2.boxPoints(coord_tuple)
            cv2.fillPoly(base_image, pts=[np.int0(box)], color=color)

    rotated_image = cv2.warpAffine(base_image, rotation_mat, (base_image.shape[1],
                                                              base_image.shape[0]))
    rotated_image = (crop_img(rotated_image))
    rotated_image = resize_input(rotated_image, (img_size, img_size))
    
    np.savez("./%s/%05d.npz" % (split, index), input = np.array(resize_input_img), target = worst_case_map, attention = np.array(rotated_image))
    print("%05d processed" % index)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True) # '/xuanchi_msraimscratch/v-xren/home/nuScenes'
    parser.add_argument('--split', type=str, required=True) # [train, train_val, val]
    parser.add_argument('--img_size', type=int, default=128, help='Size of occupancy map')
    parser.add_argument('--future_length', type=int, default=3, help='Target predict time')
    args = parser.parse_args()
    
    # create output dirs
    os.makedirs(args.split, exist_ok = True) 
    
    DATAROOT = args.data
    nuscenes = NuScenes('v1.0-trainval', dataroot=DATAROOT)
    helper = PredictHelper(nuscenes)

    mini_train = get_prediction_challenge_split("%s" % args.split, dataroot=DATAROOT)
    print(len(mini_train))

    static_layer_rasterizer = StaticLayerRasterizer(helper)
    agent_rasterizer = AgentBoxesWithFadedHistory(helper, seconds_of_history=2)
    mtp_input_representation = InputRepresentation(static_layer_rasterizer, agent_rasterizer, Rasterizer())

    # begin the code
    future_length = args.future_length
    slice_num = 5
    interpolate_length = future_length * slice_num * 2
    color = (255, 255, 255)

    start_time = time.time()
    cores = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=cores)
    pool_list = []
    start_time = time.time()
    for index in range(len(mini_train)):
        pool_list.append(pool.apply_async(my_func, (index, args.split, args.img_size)))
        
    result_list = [xx.get() for xx in pool_list]
    pool.close()
    pool.join()
    print(time.time() - start_time)