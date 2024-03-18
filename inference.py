import os.path

from tgs.utils.config import ExperimentConfig
from tgs.infer import TGS, DataLoader
from utils import image_process
from lgm import gs_convert


def generate_triplane_gs(cfg: ExperimentConfig, device, image_path: str):
    from huggingface_hub import hf_hub_download
    model_path = hf_hub_download(
        repo_id="VAST-AI/TriplaneGaussian",
        local_dir="./checkpoints",
        filename="model_lvis_rel.ckpt",
        repo_type="model"
    )
    cfg.system.weights = model_path

    model = TGS(cfg=cfg.system).to(device)
    model.set_save_dir(args.out)
    print("load model ckpt done.")

    # Put image into configuration dataset
    cfg.data.image_list = [image_path]

    # Use cam_distance = 1.9 by default
    cfg.data.cond_camera_distance = 1.9
    cfg.data.eval_camera_distance = 1.9
    dataset = CustomImageOrbitDataset(cfg.data)
    dataloader = DataLoader(dataset,
                            batch_size=cfg.data.eval_batch_size,
                            num_workers=cfg.data.num_workers,
                            shuffle=False,
                            collate_fn=dataset.collate
                            )

    for batch in dataloader:
        batch = todevice(batch)
        model(batch)
    # GS has been created, the PLY format is stored in model.gs_output
    ply_data = model.gs_output
    # Clean up
    del model, dataloader, dataset

    return ply_data


def cli_inference():
    device = get_device()

    cfg: ExperimentConfig = load_config(args.config, cli_args=extras)

    # Pre process image
    image_path = image_process.process(cfg.data.image_list[0])
    # First stage: generate the triplane Gaussian Splatting
    ply_output = generate_triplane_gs(cfg, device, image_path)
    # Determine output path
    output_path = os.path.join(args.out, "model.glb")
    # Verify output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # Convert GS to mesh
    gs_convert.convert_gs_to_mesh(device, ply_output, output_path)


if __name__ == "__main__":
    import argparse
    from tgs.utils.config import ExperimentConfig, load_config
    from tgs.data import CustomImageOrbitDataset
    from tgs.utils.misc import todevice, get_device

    parser = argparse.ArgumentParser("TriGS Mesh Inference")
    parser.add_argument("--config", required=True, help="path to config file")
    parser.add_argument("--out", default="outputs", help="path to output folder")
    args, extras = parser.parse_known_args()

    cli_inference()
