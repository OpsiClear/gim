# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GIM (Generalizable Image Matcher) is a framework for learning robust image matching from internet videos. It supports multiple state-of-the-art matching networks (RoMa, DKM, LoFTR, LightGlue) unified under a common training and evaluation infrastructure.

## Common Development Commands

### Environment Setup
```bash
# Install with uv (cross-platform compatible)
uv sync
# or
uv pip install -e .
```

### Running Demos
```bash
# Test different matching networks
python demo.py --model gim_roma
python demo.py --model gim_dkm
python demo.py --model gim_loftr
python demo.py --model gim_lightglue
```

### Running Tests (ZEB Benchmark)
```bash
# Run comprehensive tests with specified GPU count
sh TEST_GIM_ROMA.sh 1
sh TEST_GIM_DKM.sh 1
sh TEST_GIM_LOFTR.sh 1
sh TEST_GIM_LIGHTGLUE.sh 1
sh TEST_ROOT_SIFT.sh 1

# Check test results
python check.py

# Analyze results
python analysis.py --dir dump/zeb --wid gim_dkm --version 100h --verbose
```

### 3D Reconstruction
```bash
# Create input directory structure
mkdir -p inputs/<scene_name>/images
# Place images in the images folder

# Run reconstruction
sh reconstruction.sh <scene_name> gim_dkm
# or
sh reconstruction.sh <scene_name> gim_lightglue
```

### Video Processing for Training Data
```bash
# Add video IDs to video_list.txt
chmod +x process_videos.sh
./process_videos.sh video_list.txt
python -m datasets.walk.propagate video_list.txt
python -m datasets.walk.walk video_list.txt
```

## High-Level Architecture

### Network Variants
The framework implements four distinct matching approaches, each in `networks/`:

1. **GIM-RoMa** (`networks/roma/`): CNN+ViT hybrid with Gaussian Process correspondence prediction
   - Input: 672x672, Classification-based with 64x64 discrete space

2. **GIM-DKM** (`networks/dkm/`): Dense Kernel Matching with hierarchical refinement
   - Input: 672x896, Regression-based with multi-scale ConvRefiners

3. **GIM-LoFTR** (`networks/loftr/`): Detector-free transformer matching
   - Coarse-to-fine strategy without explicit keypoint detection

4. **GIM-LightGlue** (`networks/lightglue/`): SuperPoint detector + attention-based matcher
   - Two-stage sparse matching with up to 2048 keypoints

### Core Components

**Data Pipeline** (`datasets/`):
- Handles multiple benchmark datasets (GL3D, KITTI, ETH3D, RobotCar, etc.)
- Video preprocessing extracts training pairs from internet videos
- Semantic segmentation filters dynamic objects during processing

**Training Infrastructure** (`trainer/`):
- PyTorch Lightning-based distributed training
- Unified trainer supports all network variants through conditional loading
- Automatic metric computation for pose estimation and epipolar errors

**3D Reconstruction** (`hloc/`):
- Integrates with COLMAP for Structure-from-Motion
- Dense matching pipeline: feature extraction → matching → triangulation
- Automatic camera parameter estimation and geometric verification

### Key Design Patterns

- **Factory Pattern**: Dataset creation in `datasets/data.py`
- **Strategy Pattern**: Interchangeable matching networks with common interfaces
- **Template Method**: Unified training pipeline in `trainer/lightning.py`

### Model Weights Location
Pre-trained weights should be placed in the `weights/` directory. Models automatically look for:
- `gim_roma_50h.ckpt` or `gim_roma_100h.ckpt`
- `gim_dkm_50h.ckpt` or `gim_dkm_100h.ckpt`
- `gim_loftr_50h.ckpt` or `gim_loftr_100h.ckpt`
- `gim_lightglue_50h.ckpt` or `gim_lightglue_100h.ckpt`

### Dependencies Note
- **Cross-platform Triton**: The project uses `triton` on Linux and `triton-windows` on Windows automatically
- **Semantic Segmentation**: Download `decoder_epoch_20.pth` for video preprocessing and 3D reconstruction
- **COLMAP/pycolmap**: Required for 3D reconstruction. On Windows:
  ```powershell
  # VCPKG is included in ./external/vcpkg
  # Install pycolmap from local external/colmap directory
  uv pip install ./external/colmap `
      --config-settings="cmake.define.CMAKE_TOOLCHAIN_FILE=./external/vcpkg/scripts/buildsystems/vcpkg.cmake" `
      --config-settings="cmake.define.VCPKG_TARGET_TRIPLET=x64-windows"
  ```

## Training on Custom Data

To train on different branches:
```bash
git checkout train-gim-loftr  # or train-gim-roma, train-gim-dkm, train-gim-lightglue
python train.py --num_nodes 1 --gpus $GPUS --max_epochs 10 --maxlen 938240 938240 938240 --lr 0.001 --min_lr 0.00005 --git $GITID --wid $MODELID --resample --img_size 840 --batch_size 1 --valid_batch_size 2
```

Different models require different training parameters - see README.md for specific configurations.