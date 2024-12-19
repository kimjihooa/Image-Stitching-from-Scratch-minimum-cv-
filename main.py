import os
import cv2
from PIL import Image, ImageShow
import numpy as np
import math

#----------image load & preprocess----------#
def load_images(path):
    images = []
    for filename in sorted(os.listdir(path)):
        file_path = os.path.join(path, filename)
        if os.path.isfile(file_path) and filename.lower().endswith((".png", ".jpg", ".jpeg")):
            image = cv2.imread(file_path)
            if image is not None:
                images.append(image)
            else:
                print(f"Cannot load file: {file_path}")
    print(len(images), "Images loaded")
    assert len(images) != 0, "No images loaded"
    return images

def convert_to_grayscale(image):
    r = image[:, :, 0] * 0.2989
    g = image[:, :, 1] * 0.5870 
    b = image[:, :, 2] * 0.1140
    return r + g + b

def gaussian_smoothing(image, kernel_size, sigma):
    # Create gaussian kernel
    assert kernel_size, "Kernel size is not odd number"
    mid = kernel_size // 2
    x, y = np.meshgrid(np.arange(-mid, mid+1), np.arange(-mid, mid+1))
    kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    kernel /= np.sum(kernel)

    height, width = image.shape
    padding = kernel_size // 2
    padded_image = np.pad(image, ((padding, padding), (padding, padding)), mode='constant')
    smoothed_image = np.zeros_like(image)

    # Convolution
    for i in range(padding, padding + height):
        for j in range(padding, padding + width):
            selected = padded_image[i - padding:i + padding + 1, j - padding:j + padding + 1] 
            smoothed_image[i - padding, j - padding] = np.sum(selected * kernel)

    return smoothed_image

def save_image(image_array, save_path, filename):
    # Convert to BGR -> RGB
    np.save(os.path.join(save_path, filename), image_array)
    if len(image_array.shape) == 3 and image_array.shape[2] == 3:
        image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image_array)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image.save(os.path.join(save_path, filename))

#----------Feature point extraction----------#
def load_preprocessed_images(path):
    images = []
    images_np = []
    for filename in sorted(os.listdir(path)):
        file_path = os.path.join(path, filename)
        # if os.path.isfile(file_path) and filename.lower().endswith((".png", ".jpg", ".jpeg")):
        #     img = Image.open(file_path).convert("L")
        #     images.append(np.array(img))
        if os.path.isfile(file_path) and filename.lower().endswith((".npy")):
            npy = np.load(file_path)
            #npy = Image.fromarray(npy)
            images_np.append(npy)

    print(len(images_np), "Preprosecced images loaded")
    assert len(images_np) != 0, "No images loaded"
    return images_np

def harris_corner_detector(image, k, threshold, nms_size):
    sobel_x = [[-1, 0, 1],
               [-2, 0, 2],
               [-1, 0 ,1]]
    sobel_y = [[-1, -2, -1],
               [0, 0, 0],
               [1, 2, 1]]
    dx = np.zeros_like(image)
    dy = np.zeros_like(image)
    height, width = image.shape
    # Calculade derative x and derative y & smooth
    padded_image = np.pad(image, ((1, 1), (1, 1)), mode='constant')
    for i in range(1, 1 + height):
        for j in range(1, 1 + width):
            selected = padded_image[i - 1:i + 1 + 1, j - 1:j + 1 + 1] 
            dx[i - 1, j - 1] = np.sum(selected * sobel_x)
            dy[i - 1, j - 1] = np.sum(selected * sobel_y)
    dx2 = dx * dx
    dxy = dx * dy
    dy2 = dy * dy
    dx2 = gaussian_smoothing(dx2, 3, 0.75)
    dxy = gaussian_smoothing(dxy, 3, 0.75)
    dy2 = gaussian_smoothing(dy2, 3, 0.75)

    detM = (dx2 * dy2) - (dxy * dxy)
    trM = (dx2 + dy2) ** 2
    R = detM - (k * trM)

    # Normalize and apply threshold
    R = R / np.max(np.abs(R))
    R[R < threshold] = 0

    result = non_max_suppression(R, nms_size)
    
    return result

def non_max_suppression(R, search_size):
    corners = []
    padding = search_size // 2
    padded_R = np.pad(R, ((padding, padding), (padding, padding)), mode='constant')

    for y in range(padding, R.shape[0] + padding):
        for x in range(padding, R.shape[1] + padding):
            if padded_R[y, x] == 0:
                continue

            local_patch = padded_R[y - padding:y + padding + 1, x - padding:x + padding + 1]
            if padded_R[y, x] == np.max(local_patch):
                corners.append((y - padding, x - padding))
    return corners

def save_harris(image, corners, save_path, filename):
    # Convert to RGB (gray -> RGB)
    np.save(os.path.join(save_path, filename), corners)
    if len(image.shape) == 2:
        image = np.stack([image] * 3, axis=-1)
    else:
        image = image.copy()
    
    # Mark red
    for y, x in corners:
        if 0 <= y < image.shape[0] and 0 <= x < image.shape[1]:
            cv2.circle(image, (int(x), int(y)), 3, (255, 0, 0), -1)

    image = Image.fromarray(image.astype(np.uint8))
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image.save(os.path.join(save_path, filename))

#----------Correspondence matching----------#
def load_corners(path):
    corners = []
    for filename in sorted(os.listdir(path)):
        file_path = os.path.join(path, filename)
        if os.path.isfile(file_path) and filename.lower().endswith((".npy")):
            npy = np.load(file_path)
            corners.append(npy)
    num_corners = 0
    for i in range(len(corners)):
        num_corners += len(corners[i])
    print(num_corners, "Features loaded")
    assert len(corners) != 0, "No images loaded"
    return corners

def get_patch(image, corners, patch_size, normalize = True):
    padding = patch_size // 2
    patches = []
    
    # Normalize
    if normalize:
        mean = np.mean(image, axis = 0)
        std = np.std(image, axis = 0)
        image = (image - mean) / std 
    
    padded_image = np.pad(image, ((padding, padding), (padding, padding)), mode='constant')

    for corner in corners:
        y, x = corner
        patch = padded_image[y:y + patch_size, x:x + patch_size]
        patches.append(patch)
    return patches

def correspondence_match(patch_list, threshold):
    imgage_num = len(patch_list)
    matches = []
    
    # For each images
    for image1_index in range(imgage_num):
        for image2_index in range(image1_index + 1, imgage_num):
            patches1 = patch_list[image1_index]
            patches2 = patch_list[image2_index]
            
            # Calculate SSD
            distances = []
            for patch1_index, patch1 in enumerate(patches1):
                for patch2_index, patch2 in enumerate(patches2):
                    distances.append(np.sum((patch2 - patch1) ** 2))
                sorted_indices = np.argsort(distances)
                best_match_index = sorted_indices[0]
                second_best_match_index = sorted_indices[1]

                if distances[best_match_index] < threshold * distances[second_best_match_index]:
                    # Index of (image1, corner1, image2, corner2)
                    matches.append((image1_index, patch1_index, image2_index, best_match_index))
                distances.clear()

    return matches

def matches_to_images(images, corners, matches, output_path):

    for i in range(len(images)):
        for j in range(i + 1, len(images)):
            image1 = images[i]
            image2 = images[j]
            corners1 = corners[i]
            corners2 = corners[j]
            # Connect two images
            height1, width1 = image1.shape[:2]
            height2, width2 = image2.shape[:2]
            canvas_height = max(height1, height2)
            canvas_width = width1 + width2
            canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

            canvas[:height1, :width1] = image1
            canvas[:height2, width1:width1+width2] = image2

            # Draw
            for corner in corners1:
                y1, x1 = corner
                cv2.circle(canvas, (int(x1), int(y1)), 3, (0, 0, 255), -1)
            for corner in corners2:
                y1, x1 = corner
                x1 += width1
                cv2.circle(canvas, (int(x1), int(y1)), 3, (0, 0, 255), -1)

            for match in matches:
                image1_index, corner1_index, image2_index, corner2_index = match
                if image1_index == i and image2_index == j:
                    y1, x1 = corners1[corner1_index]
                    y2, x2 = corners2[corner2_index]
                    x2 += width1
                    cv2.circle(canvas, (int(x1), int(y1)), 3, (0, 255, 0), -1)
                    cv2.circle(canvas, (int(x2), int(y2)), 3, (255, 0, 0), -1)
                    cv2.line(canvas, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 1)

            cv2.imwrite(output_path + str(i) + "-" + str(j) + ".jpg", canvas)

def save_matching(images, corners, matching, save_path, filename):
    np.save(os.path.join(save_path, filename), matching)
    matches_to_images(images, corners, matching, save_path)

#----------Homography calculation----------#
def load_matches(path):
    for filename in os.listdir(path):
        file_path = os.path.join(path, filename)
        if os.path.isfile(file_path) and filename.lower().endswith((".npy")):
            npy = np.load(file_path)

    print(len(npy), "matches loaded")
    return npy

def get_corners_from_matching(img_indx1, img_indx2, matches, corners):
    # Get matched corners between two images
    corners1 = corners[img_indx1]
    corners2 = corners[img_indx2]
    img1_corners = []
    img2_corners = []
    # Index of (image1, corner1, image2, corner2)
    for match in matches:
        image1, corner1, image2, corner2 = match
        if image1 == img_indx1 and image2 == img_indx2:
            img1_corners.append(corners1[corner1])
            img2_corners.append(corners2[corner2])
    return img1_corners, img2_corners

def compute_homography_ransac(matches, corners, num_images, num_choices, num_iterations, threshold):

    Hs = [[np.eye(3) for _ in range(num_images)] for _ in range(num_images)]
    assert num_choices >= 4, "Number of choices cant be smaller than 4"

    for i in range(num_images):
        for j in range(i + 1, num_images):
            corners1, corners2 = get_corners_from_matching(i, j, matches, corners)

            # If matched corners are not loaded properly
            assert len(corners1) == len(corners2), f"Corner number mismatch for {i} and {j}"
            
            # If matching is not enough
            if len(corners1) < 4 or len(corners2) < 4:
                continue
            # If num_choices is too large
            if len(corners1) < num_choices:
                print("Not enough matches to choice for", i, "-", j, "setting num_choice to 4")
                num_choices = 4
            
            best_homography = None
            max_inliers = []
            num_matches = len(corners1)

            # Select random
            for _ in range(num_iterations):
                rand = np.random.choice(num_matches, num_choices, replace=False)
                selected_corners1 = []
                selected_corners2 = []
                for r in rand:
                    selected_corners1.append(corners1[r])
                    selected_corners2.append(corners2[r])
                # Calculate homography
                A = []
                for (y1, x1), (y2, x2) in zip(selected_corners1, selected_corners2):
                    A.append([x1, y1, 1, 0, 0, 0, -x2 * x1, -x2 * y1, -x2])
                    A.append([0, 0, 0, x1, y1, 1, -y2 * x1, -y2 * y1, -y2])
                A = np.array(A)
                U, S, V = np.linalg.svd(A)
                H = V[-1].reshape(3, 3)
                H = H / H[-1, -1]

                # Calculate SSD for all matched corners
                inliers = []
                for corner1, corner2 in zip(corners1, corners2):
                    y1, x1 = corner1
                    y2, x2 = corner2
                    transformed_point = np.matmul(H, np.array([x1, y1, 1]))
                    transformed_point /= transformed_point[2]
                    #error = np.linalg.norm(transformed_point[:2] - np.array([x2, y2]))

                    error = np.sqrt((transformed_point[0] - x2) ** 2 + (transformed_point[1] - y2) ** 2)
                    if error < threshold:
                        inliers.append((corner1, corner2))
                        
                # This is best!
                if len(inliers) > len(max_inliers):
                    max_inliers = inliers
                    best_homography = H

            if best_homography is not None:
                Hs[i][j] = best_homography
                print("Calculated homography for", i, "-" , j)
                print(Hs[i][j])
    return Hs

def save_homography(homography, save_path, filename):
    homography = np.array(homography)
    np.save(os.path.join(save_path, filename), homography)

#----------Stitching----------#

def load_homography(path):
    for filename in os.listdir(path):
        file_path = os.path.join(path, filename)
        if os.path.isfile(file_path) and filename.lower().endswith((".npy")):
            npy = np.load(file_path)

    print(len(npy), "homographies loaded")
    return npy

def warp_image(image, H, i):
    image_array = np.array(image)
    image_y, image_x = int(image_array.shape[0]), int(image_array.shape[1])

    # Calculate size
    top_left = np.matmul(H, [0, 0, 1])
    top_left_x, top_left_y = int(top_left[0]/top_left[2]), int(top_left[1]/top_left[2])
    top_right = np.matmul(H, [image_x, 0, 1])
    top_right_x, top_right_y = int(top_right[0]/top_right[2]), int(top_right[1]/top_right[2])
    bottom_left = np.matmul(H, [0, image_y, 1])
    bottom_left_x, bottom_left_y = int(bottom_left[0]/bottom_left[2]), int(bottom_left[1]/bottom_left[2])
    bottom_right = np.matmul(H, [image_x, image_y, 1])
    bottom_right_x, bottom_right_y = int(bottom_right[0]/bottom_right[2]), int(bottom_right[1]/bottom_right[2])

    xs = [top_left_x, top_right_x, bottom_left_x, bottom_right_x]
    ys = [top_left_y, top_right_y, bottom_left_y, bottom_right_y]
    min_x = math.floor(min(xs))
    min_y = math.floor(min(ys))
    max_x = math.ceil(max(xs))
    max_y = math.ceil(max(ys))
    size_x = max_x - min_x
    size_y = max_y - min_y

    offset_x = min_x
    offset_y = min_y

    # Backward warping
    H_inv = np.linalg.inv(H)
    output = np.zeros((size_y, size_x, 3))

    for x in range(size_x):
        for y in range(size_y):
            source = np.matmul(H_inv, [x + offset_x, y + offset_y, 1])
            source_y = source[1] / source[2]
            source_x = source[0] / source[2]

            if 0 <= source_x < image_x and 0 <= source_y < image_y:
                nearest_x = int(round(source_x))
                nearest_y = int(round(source_y))

                if 0 <= nearest_x < image_x and 0 <= nearest_y < image_y:
                    output[y, x, :] = image_array[nearest_y, nearest_x, :]
                    
    save_stitching(output, cache_path + "/5. stitching/", str(i))
    print("Warped", i, "image")
    print("Offset:", (offset_x, offset_y))
    return output, offset_x, offset_y

def stitch_images(images, homographies):

    num_images = len(images)
    assert num_images >= 1, "No images loaded for stitching"
    num_images_half = num_images // 2

    # Combine homographies
    global_homographies = [np.eye(3)]
    for i in range(num_images_half + 1, num_images):
        global_homographies.append(np.matmul(global_homographies[-1], np.linalg.inv(homographies[i - 1][i])))
    for i in range(num_images_half - 1, -1, -1):
        global_homographies.insert(0, np.matmul(global_homographies[0], homographies[i][i + 1]))
    #global_homographies = group_adjustment(global_homographies)
    
    # Warp & Make output base
    warped_images = []
    offsets = []
    min_x, min_y = float("inf"), float("inf")
    max_x, max_y = float("-inf"), float("-inf")

    for i, (image, H) in enumerate(zip(images, global_homographies)):
        warped_image, offset_x, offset_y = warp_image(image, H, i)
        warped_images.append(warped_image)
        offsets.append((offset_x, offset_y))
        
        h, w = warped_image.shape[:2]
        min_x = min(min_x, offset_x)
        min_y = min(min_y, offset_y)
        max_x = max(max_x, offset_x + w)
        max_y = max(max_y, offset_y + h)

    # Make output base
    output_width = max_x - min_x
    output_height = max_y - min_y
    output = np.zeros((output_height, output_width, 3))
    print("Output size: " + str((output_width, output_height)))

    for warped_image, (offset_x, offset_y) in zip(warped_images, offsets):
        h, w = warped_image.shape[:2]
        x_start = offset_x - min_x
        y_start = offset_y - min_y
        x_end = x_start + w
        y_end = y_start + h

        for wy in range(h):
            for wx in range(w):
                if warped_image[wy, wx].any():
                    if output[y_start + wy, x_start + wx].any():
                        output[y_start + wy, x_start + wx] = (output[y_start + wy, x_start + wx] + warped_image[wy, wx]) / 2
                    else:
                        output[y_start + wy, x_start + wx] = warped_image[wy, wx]

    return output

def save_stitching(image_array, save_path, filename):
    # Convert to BGR -> RGB
    np.save(os.path.join(save_path, filename), image_array)
    image_array = cv2.cvtColor(image_array.astype(np.uint8), cv2.COLOR_BGR2RGB)
    image_array = Image.fromarray(image_array, 'RGB')
    image_array.save(cache_path + "/5. stitching/" + filename + ".jpg")

def group_adjustment(global_homographies):
    
    # Compute average translation, scale
    translations = []
    scales = []
    for H in global_homographies:
        t = H[:2, 2]  # Translation vector (tx, ty)
        s = np.linalg.norm(H[:2, 0])  # Scaling factor
        translations.append(t)
        scales.append(s)
    avg_translation = np.mean(translations, axis=0)
    avg_scale = np.mean(scales)
    
    # Adjust each homography
    adjusted_homographies = []
    for H in global_homographies:
        adjusted_H = H.copy()
        adjusted_H[:2, 2] -= avg_translation  # Adjust translation
        adjusted_H[:2, :2] /= avg_scale  # Adjust scaling
        adjusted_homographies.append(adjusted_H)
    
    return adjusted_homographies
#~~~~~~~~~~----------#
def remove_cache(path_list):
    for path in path_list:
        full_path = "./cache/" + path + "/"
        for filename in os.listdir(full_path):
            os.remove(full_path + filename)

test_mode = [#"1. preprosess",
             #"2. feature",
             #"3. matching",
             "4. homography",
             "5. stitching"
             ]
remove_cache(test_mode)
cache_path = "./cache"
samples_path = "./samples"
images = load_images(samples_path)
#~~~~~~~~~~run preprosess~~~~~~~~~~#
if "1. preprosess" in test_mode:
    print("*****Starting preprocess*****")

    # Parameters
    kernel_size = 3
    sigma = 0.5

    filename = 0
    preprocessed_images_npy = []
    for image in images:
        gray_image = convert_to_grayscale(image)
        smoothed_image = gaussian_smoothing(gray_image, kernel_size, sigma)
        save_image(smoothed_image, os.path.join(cache_path, "1. preprosess"), str(filename) + ".jpg")
        preprocessed_images_npy.append(smoothed_image)
        print("Preprocessed", filename, "image")
        filename += 1
    print("*****Image preprocessing complete*****\n")

#~~~~~~~~~~Feature point extraction~~~~~~~~~~#
if "1. preprosess" not in test_mode:
    preprocessed_images_npy = load_preprocessed_images("./cache/1. preprosess")

if "2. feature" in test_mode:
    print("*****Starting corner detection*****")

    # Parameters
    k = 0.04
    threshold_detection = 0.001
    nms_size = 15

    filename = 0
    corners = []
    num_corners = 0
    for image in preprocessed_images_npy:
        corners_of_one = harris_corner_detector(image, k, threshold_detection, nms_size)
        save_harris(image, corners_of_one, os.path.join(cache_path, "2. feature"), str(filename) + ".jpg")
        corners.append(corners_of_one)
        print(len(corners[filename]), "corners found for", filename, "image")
        filename += 1
    for corner in corners:
        num_corners += len(corner)
    print(num_corners, "corners found")
    print("*****Corner detection complete*****\n")

#~~~~~~~~~~Correspondence matching~~~~~~~~~~#
if "2. feature" not in test_mode:
    corners = load_corners("./cache/2. feature")

if "3. matching" in test_mode:
    print("*****Starting correspondence matching*****")

    # Parameters
    patch_size = 11
    threshold_matching = 0.4

    patches_for_images = []
    for i in range(len(preprocessed_images_npy)):
        patches_for_images.append(get_patch(preprocessed_images_npy[i], corners[i], patch_size))
    # Index of (image1, corner1, image2, corner2)
    matches = correspondence_match(patches_for_images, threshold_matching)
    save_matching(images, corners, matches, "./cache/3. matching/", "matching result")
    print(len(matches), "matches found")
    print("*****Correspondence matching complete*****\n")

#~~~~~~~~~~Homography calculation~~~~~~~~~~#
if "3. matching" not in test_mode:
    # Index of (image1, corner1, image2, corner2)
    matches = load_matches("./cache/3. matching")

if "4. homography" in test_mode:
    print("*****Starting homography calculation*****")

    # Parameters
    num_choices = 12
    num_iterations = 5000
    threshold_ransec = 1

    homographies = compute_homography_ransac(matches, corners, len(preprocessed_images_npy), num_choices, num_iterations, threshold_ransec)
    save_homography(homographies, "./cache/4. homography", "homography")
    print("homograpies calculated")

    print("*****Homography calculation complete*****\n")

#~~~~~~~~~~Stitching~~~~~~~~~~#
if "4. homography" not in test_mode:
    homographies = load_homography("./cache/4. homography")

print("*****Starting stitching*****")

stitched = stitch_images(images, homographies)
save_stitching(stitched, "./cache/5. stitching", "stitching")

print("*****Stitching complete*****\n")
