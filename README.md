<div align="center">

# Triplane Gaussian Meets LGM Convert:<br> Fast and Generalizable Single-View 3D Reconstruction into a mesh with Transformers

<p align="center">
<a href="https://huggingface.co/spaces/VAST-AI/TriplaneGaussian"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Gradio%20Demo-Huggingface-orange"></a>
<a href="https://huggingface.co/VAST-AI/TriplaneGaussian"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Model_Card-Huggingface-orange"></a>
</p>

TriGSMesh combines Triplane Gaussian with an optimized version of LGM's GS to mesh conversion method to create a fast and generalizable single-view 3D reconstruction object mesh. This model is able to reconstruct 3D into a GLB format within a few minutes.

Please refer to the original repositories for more details on the reconstruction model and the conversion process:
- [TriplaneGaussian](https://github.com/VAST-AI-Research/TriplaneGaussian)
- [LGM](https://github.com/3DTopia/LGM)

</div>



### Installation
- Python = 3.10

Please use conda to create a new environment and install the required packages:

```sh
conda create -n trigsmesh python=3.10
conda activate trigsmesh
```

### Install pytorch using instructions from here: https://pytorch.org/get-started/locally/

### Example (for CUDA 11.8):
```sh
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia # Recommended usage
```
- Install pointnet2_ops
```sh
cd tgs/models/snowflake/pointnet2_ops_lib && python setup.py install && cd -
```
- Install pytorch_scatter
```sh
pip install git+https://github.com/rusty1s/pytorch_scatter.git
```
- Install nvdiffrast
```sh
pip install git+https://github.com/NVlabs/nvdiffrast
```
- Install nerfacc
```sh
pip install git+https://github.com/nerfstudio-project/nerfacc.git
```
- Install kiui's diff-gaussian-rasterization (Inria's version will not work!)
```sh
git clone --recursive https://github.com/ashawkey/diff-gaussian-rasterization
pip install ./diff-gaussian-rasterization
```
- Install dependencies:
```sh
pip install -r requirements.txt
```
- Install PyTorch3D following its official [installation](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md) instruction.

### Download the Pretrained Model
TriplaneGaussian offers a pretrained checkpoint available for download from [Hugging Face](https://huggingface.co/VAST-AI/TriplaneGaussian); download the checkpoint and place it in the folder `checkpoints`.
```python
from huggingface_hub import hf_hub_download
MODEL_CKPT_PATH = hf_hub_download(repo_id="VAST-AI/TriplaneGaussian", local_dir="./checkpoints", filename="model_lvis_rel.ckpt", repo_type="model")
```
Please note this model is only trained on Objaverse-LVIS dataset (**~45K** 3D models).

### Inference
Use the following command to reconstruct a 3DGS model from a single image. Please update `data.image_list` to some specific list of image paths.
```sh
python infer.py --config config.yaml data.image_list=[path/to/image1,]
# e.g. python infer.py --config config.yaml data.image_list=[example_images/a_pikachu_with_smily_face.webp,]
```

The inference script will save an OBJ file with a seperated texture (not vertex colors).


### Local Gradio Demo
Launch the Gradio demo locally by:
```sh
python gradio_inference.py
```

## Acknowledgements

### TriplaneGaussian
```bibtex
@article{zou2023triplane,
  title={Triplane Meets Gaussian Splatting: Fast and Generalizable Single-View 3D Reconstruction with Transformers},
  author={Zou, Zi-Xin and Yu, Zhipeng and Guo, Yuan-Chen and Li, Yangguang and Liang, Ding and Cao, Yan-Pei and Zhang, Song-Hai},
  journal={arXiv preprint arXiv:2312.09147},
  year={2023}
}
```

### LGM (Convert to mesh only)
```bibtex
@article{tang2024lgm,
  title={LGM: Large Multi-View Gaussian Model for High-Resolution 3D Content Creation},
  author={Tang, Jiaxiang and Chen, Zhaoxi and Chen, Xiaokang and Wang, Tengfei and Zeng, Gang and Liu, Ziwei},
  journal={arXiv preprint arXiv:2402.05054},
  year={2024}
}
```