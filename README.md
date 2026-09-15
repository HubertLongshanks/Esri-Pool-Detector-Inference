# Quickstart

```python
from pool_detector import EsriPoolFasterRCNNInference

# Initialize the detector
detector = EsriPoolFasterRCNNInference(
    raster_mosaic_path="path/to/raster.tif",
    model_weights_path="path/to/model.pth",
    detection_confidence_score=0.6,
    batch_size=16
)

# Run inference and output vector bounding boxes
results = detector.perform_inference(
    device="cuda",
    detections_file="detected_pools.geojson"
)

print(f"Detected {len(results.bboxes)} pools.")
```

## What it is
You can run inference using the Esri pool detector model in python over an arbitrary raster mosaic without needed to configure an Esri environment. It can run in a fully open-source and maintainable env, you can spawn multiple instances on multiple devices just like Esri does (but youll need to write some more code lol). 

## Getting Esri Model Weights
Esri keeps their model weights inside a *.dlpk* file, which is really just a zip file, just rename the file extension and unzip to find the model weights *.pth* file. 
```sh
unzip myrenamed_dlpk.zip
```
And then give the class defined here the model weights and off to the races you go!

## Classes & Methods

EsriPoolFasterRCNNInference

Handles tiling, sliding-window Faster R-CNN inference over spatial rasters, global coordinate mapping, and Non-Maximum Suppression (NMS).

__init__(...) Parameters:

raster_mosaic_path (str): Path to the target raster image file.

model_weights_path (str): Path to saved PyTorch .pth model state dict.

detection_confidence_score (float, optional): Prediction threshold between 0.0 and 1.0. Default: 0.6.

expected_model_img_size (int, optional): Pixel dimensions for sliding window chips. Default: 224.

window_stride_fraction (float, optional): Fraction of window size determining tile overlap. Default: 0.5.

batch_size (int, optional): Batch size for model inference. Default: 16.

nodata (int, optional): Pixel value designated as no-data. Default: 0.

perform_inference(...) Parameters:

device (str, optional): Execution target ("cuda" or "cpu"). Default: "cuda".

num_dataworkers (int, optional): CPU worker threads for DataLoader. Default: system auto.

detections_file (str | None, optional): Output path to save spatial detections (.geojson, .shp, .fgb). Default: None.

Returns: PoolDetectorResults

PoolDetectorResults

A dataclass storing detection outputs mapped to the raster's native CRS.

Attributes:

inference_file (str): Source raster path processed.

bboxes (list[Tensor]): Detected bounding box coordinates in spatial/CRS units.

confidences (list[Tensor]): Model confidence score for each detection.
