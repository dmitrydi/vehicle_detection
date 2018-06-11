# Vehicle Detection
## Overview
This repository contains two classes for vehicle detection and tracking:
1. `FeatureExtractor()`
2. `Tracker()`

## Usage
To use the tracjer one needs:
1. Trained classifier
2. Fitted feature scaler
3. Feature extractor

Example code:
```
tracker = Tracker(classifier, feature_extractor, scaler, max_to_keep = 5, heat_threshold=3)
tracker.set_track_params(step=step,
                         hog_color_transform=cv2.COLOR_RGB2GRAY,
                         color_transform = cv2.COLOR_RGB2BGR,
                         scale_dict=scale_dict, feature_types=ftypes, hist_channels=hist_ch, init_size=init_size, single_threshold=2)
img_with_detections = tracker.track(img)
```

`max_to_keep` - number of frames used to filter out wrong detections
`heat_threshold` - number of detections in the sema region over several videoframe for detection to be considered as true
`step` - sliding window step
`hog_color_transform` - color transform used to get 1-channel image for HOG feature extraction
`color_transform` - color transform used for spatial features and histogram features
`scale_dict` - dictionary in format `{scale: (xmin. ymin, xmax, ymax)}` where `(xmin, ymin, xmax, ymax)` - region of search for a given `scale`
`hist_channels` - channels of image (after color tansform, if any) used for getting histogram features
`init_size` - image size used for training classifier
`single_threshold` - threshold used for filtering each image for different scales detections

## Feature extraction

Class *FeatureExtractor* implements two basic things: generate features and labels of small images used for training a classifier and for getting features from 'big' image for sliding window object detection.

Example code:
```
feature_extractor = FeatureExtractor()

# usage for training a classifier:
features, labels = feature_extractor.return_features_and_labels(class_files, class_label, feature_types=['HOG'],
							                                   hog_color_transform=None, hog_pix_per_cell=8, hog_cell_per_block=2, hog_orient=9,
							                                   hog_feature_vec=True, hog_vis=False,
							                                   color_transform_flag=None, spat_size=(32,32),
							                                   hist_channels=[0], hist_nbins=32, hist_bins_range=(0,256)))

classifier.fit(features, labels)
```

#### Params:
`class_files` - list of paths to image files used for training
`class_label` - label of class 
`feature_types` - list of feature types to be generated, features of different types concatenated to a single feature vector. Can be 'HOG', 'SPAT', 'HIST'
`hog_color_transform` - cv2 color transform flag used to convert images to 1-channel image. I.e.: `cv2.COLOR_BGR2GRAY`
`hog_pix_per_cell` - number of pixels in HOG cell
`hog_cell_per_block` - number of cell per HOG block (in each dimention)
`hog_orient` - number of HOG orientations
`color_transform_flag` - cv2 color transform used for getting spatial features and histogram features, i.e. `cv2.COLOR_BGR2HSL`
`spat_size` - tuple of integers used to resize images before getting spatial features. 
`hist_channels` - color channels after color transformation (if any) used to generate histogram features
`hist_nbins` - number of bins of histogram for each color channel
`hist_bins_range` - parameter used in `np.histogram` as `range`

```
# usage for getting features of image for further detection and tracking:
whole_image_features = feature_extractor.return_sliding_features(img, ymin, ymax, xmin, xmax, feature_types=['HOG', 'SPAT', 'HIST'],
										                           init_size=(64,64), scale=1, step=0.125,
										                           cell_size=8, hog_color_transform=cv2.COLOR_BGR2GRAY,
										                           color_transform=None, spat_size=(32,32),
										                           hist_channels=[0], hist_nbins=32, hist_bins_range=(0,256))
```

#### Params:
`img` - image to get features from
`ymin, ymax, xmin, xmax` - boundaries of region of search
`feature_types` - list of feature types to be generated, features of different types concatenated to a single feature vector. Can be 'HOG', 'SPAT', 'HIST'
`init_size` - size of image used for training classifier 
`scale` - search window size related to `init_size`
`step` - search step (in parts of searching window size)
`hog_color_transform` - cv2 color transform flag used to convert images to 1-channel image. I.e.: `cv2.COLOR_BGR2GRAY`
`hog_pix_per_cell` - number of pixels in HOG cell
`hog_cell_per_block` - number of cell per HOG block (in each dimention)
`hog_orient` - number of HOG orientations
`color_transform_flag` - cv2 color transform used for getting spatial features and histogram features, i.e. `cv2.COLOR_BGR2HSL`
`spat_size` - tuple of integers used to resize images before getting spatial features. 
`hist_channels` - color channels after color transformation (if any) used to generate histogram features
`hist_nbins` - number of bins of histogram for each color channel
`hist_bins_range` - parameter used in `np.histogram` as `range`