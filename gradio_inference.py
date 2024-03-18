import os
import tempfile
import time

import gradio as gr

from lgm import gs_convert
from tgs.data import CustomImageOrbitDataset
from tgs.infer import TGS, DataLoader
from tgs.utils.config import load_config
from tgs.utils.misc import get_device
from tgs.utils.misc import todevice
from utils import image_process

EXP_ROOT_DIR = os.path.join(os.path.dirname(__file__), "outputs-gradio")


class GradioTGS(object):

    def __init__(self):
        # Initialize CUDA device
        self.device = get_device()
        # Initialize TGS Configuration
        self.cfg = self.init_cfg()
        print("[INFO] Loaded TGS model configuration.")
        # Initialize TGS model
        self.model = self.init_model()
        print("[INFO] Loaded TGS model checkpoint.")

        self.working_dir = None

    def init_model(self):
        from huggingface_hub import hf_hub_download
        model_path = hf_hub_download(
            repo_id="VAST-AI/TriplaneGaussian",
            local_dir="./checkpoints",
            filename="model_lvis_rel.ckpt",
            repo_type="model"
        )
        self.cfg.system.weights = model_path

        return TGS(cfg=self.cfg.system).to(self.device)

    def create_working_dir(self):
        os.makedirs(EXP_ROOT_DIR, exist_ok=True)
        current_dir = tempfile.TemporaryDirectory(dir=EXP_ROOT_DIR).name
        os.makedirs(current_dir, exist_ok=True)
        self.model.set_save_dir(current_dir)
        self.working_dir = current_dir

    @staticmethod
    def init_cfg():
        return load_config('config.yaml', cli_args=[])

    @staticmethod
    def assert_input_image(current_input_image):
        if current_input_image is None:
            raise gr.Error("No image selected or uploaded!")

    def generate_gs(self, image_path: str):
        # Put image into configuration dataset
        self.cfg.data.image_list = [image_path]

        # Use cam_distance = 1.9 by default
        self.cfg.data.cond_camera_distance = 1.9
        self.cfg.data.eval_camera_distance = 1.9
        dataset = CustomImageOrbitDataset(self.cfg.data)
        dataloader = DataLoader(dataset,
                                batch_size=self.cfg.data.eval_batch_size,
                                num_workers=self.cfg.data.num_workers,
                                shuffle=False,
                                collate_fn=dataset.collate
                                )

        for batch in dataloader:
            batch = todevice(batch)
            self.model(batch)

        return self.model.gs_output


def gradio_inference(image):
    device = get_device()

    app.create_working_dir()

    if image is None:
        raise gr.Error("No image selected or uploaded!")

    # Pre process image
    processed_image_path = image_process.process(image)
    yield processed_image_path, None, None, "Generating Gaussian Splatting..."
    # First stage: generate the triplane Gaussian Splatting
    ply_output = app.generate_gs(processed_image_path)
    ply_path = os.path.join(app.working_dir, 'model.ply')
    app.model.save_ply(ply_path)
    yield processed_image_path, ply_path, None, "Converting to 3D mesh..."
    # Determine output path
    output_path = os.path.join(app.working_dir, "{}.glb".format(int(time.time())))
    # Verify output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # Convert GS to mesh
    gs_convert.convert_gs_to_mesh(device, ply_output, output_path)
    yield processed_image_path, ply_path, output_path, "3D mesh generated"


if __name__ == "__main__":
    app = GradioTGS()

    with gr.Blocks() as demo:
        gr.Markdown("# TriGS Mesh Inference")
        gr.Markdown("Upload an image to generate a 3D mesh using TriGS.")

        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(
                    value=None,
                    image_mode="RGB",
                    type="filepath",
                    sources=["upload"],
                    label="Input Image"
                )
                submit_button = gr.Button("Generate 3D Mesh")

            with gr.Column(scale=2):
                with gr.Tab("Processed Image"):
                    output_image = gr.Image(
                        value=None,
                        type="pil",
                        image_mode="RGBA",
                        label="Processed Image",
                        interactive=False
                    )
                with gr.Tab("3D Model"):
                    output_gs = gr.Model3D(
                        label="3D Model"
                    )
                with gr.Tab("Output GLB"):
                    output_3d = gr.Model3D(
                        label="Output GLB"
                    )


        def inference(image):
            for output in gradio_inference(image):
                if output[1] is not None and output_gs.visible is False:
                    output_gs.visible = True
                if output[2] is not None and output_3d.visible is False:
                    output_3d.visible = True
                yield output


        submit_button.click(
            fn=inference,
            inputs=input_image,
            outputs=[output_image, output_gs, output_3d],
        )

    demo.launch()
