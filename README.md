# Vehicle Detection
## Overview
This repository contains two classes for vehicle detection and tracking:
1. `FeatureExtractor()`
2. `Tracker()`

`FeatureExtractor()` class implements feature extraction from images which is needed for classification and object detection tasks. There are several types of features that could be obtained from an image:
* HOG features
* Spatial features that is simply raveled image (possibly resized before raveling)
* Color histogram features
HOG and Color histogram features can be obtained from any number of color channels of an image
In addition, any type and sequence of colorspace transforms can be applied to an image before extracting all types of features
`FeatureExtractor()` inherets from `sklearn.base.BaseEstimator` class and implements `fit` and `transform` methods needed for `sklearn.pipeline.Pipeline` operations, i.e. searching the best parameters of feature extraction.

`Tracker()` class uses sliding window technique for classification of small parts of an image, yielding resultant bounding boxes around detected objects.
Current implementation of `Tracker` can only use binary classification (i.e. object - non-object). For multiple detection elimination Non-Maximum Suppression algorithm is used in combination with heatmap-thresholding technique based on `scipy.ndimage.measurements.label` method.

## Usage
To use the tracker one needs:
1. Binary classifier trained on object - non-object dataset
2. Fitted feature scaler
3. Feature extractor

Example code:
```
scaler = StandardScaler().fit(X_train)
X_scaled = scaler.transform(X_train)
classifier = RandomForestClassifier().fit(X_scaled, y_train)

feature_extractor = FeatureExtractor(color_transform=36,
         feature_types=['HOG', 'SPAT', 'HIST'], hist_bins_range=(0, 256),
         hist_channels=[0, 1, 2], hist_nbins=64, hog_cell_per_block=2,
         hog_method='smart', hog_orient=11, hog_params=[(4, [0])],
         hog_pix_per_cell=8, spat_size=(16, 16))
tracker = Tracker(classifier, feature_extractor, scaler,
                  scale_dict={1:(360,550,300,1280), 1.5:(360,550,300,1280), 2:(360,550,300,1280)}, max_to_keep = 8,
                 heat_threshold=3, single_threshold=1, init_size=(64,64), step=0.125, nms_overlap=0.1)
img_with_detections = tracker.track(img)
```

`scale_dict` - dictionary in format `{scale: (xmin. ymin, xmax, ymax)}` where `(xmin, ymin, xmax, ymax)` - region of search for a given `scale`
`max_to_keep` - number of frames used to filter out wrong detections
`single_threshold` - threshold used for filtering each image for different scales detections
`heat_threshold` - number of detections in the sema region over several videoframe for detection to be considered as true
`step` - sliding window step
`init_size` - image size used for training classifier
`nms_overlap` - non-maximum suppression threshold


## Feature extraction

Class *FeatureExtractor* implements two basic things: generate features and labels of small images used for training a classifier and for getting features from 'big' image for sliding window object detection.

Example code:
```
# usage for training a classifier:
feature_extractor = FeatureExtractor(color_transform=cv2.COLOR_BGR2YCrCb,
         feature_types=['HOG', 'SPAT', 'HIST'], hist_bins_range=(0, 256),
         hist_channels=[0, 1, 2], hist_nbins=64, hog_cell_per_block=2,
         hog_method='smart', hog_orient=11, hog_params=[(cv2.COLOR_BGR2HLS, [0])],
         hog_pix_per_cell=8, spat_size=(16, 16))
features = feature_extractor.transform(array_of_training_images)
scaler = StandardScaler()
scaler.fit(features)
scaled_features = scaler.transform(features)

classifier.fit(scaled_features, labels)
```

#### Params:
`color_transform` - cv2 color transform code used for getting spatial features and histogram features, i.e. `cv2.COLOR_BGR2HSL` or integer code
`feature_types` - list of feature types to be generated, features of different types concatenated to a single feature vector. Can be 'HOG', 'SPAT', 'HIST'
`hist_bins_range` - parameter used in `np.histogram` as `range`
`hist_channels` - color channels after color transformation (if any) used to generate histogram features
`hist_nbins` - number of bins of histogram for each color channel
`hog_cell_per_block` - number of cell per HOG block (in each dimention)
`hog_method` - can be 'smart' or 'simple', use only 'smart'
`hog_orient` - number of HOG orientations
`hog_params` - list of tuples where each tuple has form (cv2_color_transformation_code, list_of_channels_to_generate_features_on). This allows to use any set of color transformations ang getting features.
`hog_pix_per_cell` - number of pixels in HOG cell
`spat_size` - tuple of integers used to resize images before getting spatial features. 


```
# usage for getting features of image for further detection and tracking:
feature_extractor = FeatureExtractor(color_transform=cv2.COLOR_BGR2YCrCb,
         feature_types=['HOG', 'SPAT', 'HIST'], hist_bins_range=(0, 256),
         hist_channels=[0, 1, 2], hist_nbins=64, hog_cell_per_block=2,
         hog_method='smart', hog_orient=11, hog_params=[(cv2.COLOR_BGR2HLS, [0])],
         hog_pix_per_cell=8, spat_size=(16, 16))

whole_image_features = feature_extractor.return_sliding_features(img, ymin, ymax, xmin, xmax,
										                           init_size=(64,64), scale=1, step=0.125)
```

#### Params:
`img` - image to get features from
`ymin, ymax, xmin, xmax` - boundaries of region of search
`init_size` - size of image used for training classifier 
`scale` - search window size related to `init_size`
`step` - search step (in parts of searching window size)