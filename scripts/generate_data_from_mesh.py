from pathlib import Path
import numpy as np
import pyrallis
from dataclasses import dataclass
import torch
import torch.nn.functional as F

from src import utils
from src.configs.train_config import GuideConfig, RenderConfig
from src.training.views_dataset import ViewsDataset
from src.utils import tensor2numpy
from PIL import Image


@dataclass
class RunConfig:
    # Path to shape file
    shape_path: str
    # Name of rendered dataset
    render_name: str = "example_render"
    # Path to initial texture
    initial_texture_path: str = None
    # Root dir for rendered dataset
    renders_root: Path = Path('texture_renders/')
    # Size of images to render
    render_size: int = 1024
    # Resolution of texture map
    texture_resolution: int = 1024
    # Number of viewpoints to render
    num_views: int = 6
    # Whether to use augmentations on the mesh
    augmentations: bool = True
    # Whether to crop renders to the bounding box of the mesh
    crop_renders: bool = True
    # Easily rotate the shape
    front_offset: float = 0.0

    @property
    def render_dir(self) -> Path:
        return self.renders_root / self.render_name


@pyrallis.wrap()
def main(cfg: RunConfig):
    # Configure guide and render settings
    guide_cfg = GuideConfig(text='A spot', shape_path=cfg.shape_path, shape_scale=0.6)
    render_cfg = RenderConfig(front_offset=cfg.front_offset)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    utils.seed_everything(18)

    # Initialize TexturedMeshModel
    from src.models.textured_mesh import TexturedMeshModel
    model = TexturedMeshModel(
        guide_cfg,
        device=device,
        render_grid_size=cfg.render_size,
        cache_path=cfg.render_dir,
        initial_texture_path=cfg.initial_texture_path,
        texture_resolution=cfg.texture_resolution,
        augmentations=cfg.augmentations
    ).to(device)
    model.eval()

    # Ensure output directory exists
    cfg.render_dir.mkdir(parents=True, exist_ok=True)

    # Dynamically set viewpoints
    dirs = ['front', 'left', 'back', 'right', 'overhead', 'bottom'][:cfg.num_views]

    count_per_view = {d: 0 for d in dirs}
    global_ind = 0

    # Render views for each direction
    while not all([c >= 256 for c in count_per_view.values()]):
        dataloader = ViewsDataset(render_cfg, device=device, size=10000, random_views=True).dataloader()
        for i, data in enumerate(dataloader):
            theta = data['theta']
            phi = data['phi']
            radius = data['radius']
            dir = dirs[data['dir']]
            if count_per_view[dir] >= 256:
                if all([c >= 256 for c in count_per_view.values()]):
                    break
                continue
            count_per_view[dir] += 1

            # Adjust phi angle
            phi = phi - np.deg2rad(render_cfg.front_offset)
            if phi < 0:
                phi = phi + 2 * np.pi
            phi = float(phi)

            dim = cfg.render_size
            try:
                outputs = model.render(theta=theta, phi=phi, radius=radius, background='random')
                nz_indices = outputs['mask'][0, 0].nonzero()

                # Calculate the bounding box for cropping
                min_h, max_h = nz_indices[:, 0].min(), nz_indices[:, 0].max()
                min_w, max_w = nz_indices[:, 1].min(), nz_indices[:, 1].max()
                size = max(max_h - min_h + 1, max_w - min_w + 1) * 1.1
                if size > dim:
                    continue

                # Adjust for cropping
                h_start = min(min_h, max_h) - (size - (max_h - min_h + 1)) / 2
                w_start = min(min_w, max_w) - (size - (max_w - min_w + 1)) / 2
                min_h = max(0, int(h_start))
                min_w = max(0, int(w_start))
                max_h = min(outputs['mask'].shape[2], int(min_h + size))
                max_w = min(outputs['mask'].shape[3], int(min_w + size))

                pred_rgb = outputs['image']
                pred_depth = outputs['depth']

                if cfg.crop_renders:
                    pred_rgb = pred_rgb[:, :, min_h:max_h, min_w:max_w]
                    pred_rgb = F.interpolate(pred_rgb, (512, 512), mode='bilinear')
                    pred_depth = pred_depth[:, :, min_h:max_h, min_w:max_w]
                    pred_depth = F.interpolate(pred_depth, (512, 512), mode='bicubic')

                # Save outputs
                pred_rgb = pred_rgb.permute(0, 2, 3, 1).contiguous().clamp(0, 1)
                pred_depth_vis = pred_depth[0, 0].contiguous().clamp(0, 1)
                pred = tensor2numpy(pred_rgb[0])
                Image.fromarray(pred).save(cfg.render_dir / f'{dir}_{global_ind}.png')
                torch.save(pred_depth.cpu().detach(), cfg.render_dir / f'{dir}_{global_ind}.pt')

                pred_depth_vis = (pred_depth_vis + 1) * 0.5
                Image.fromarray((255 * pred_depth_vis).detach().cpu().numpy().astype(np.uint8)).convert(
                    'RGB').save(cfg.render_dir / f'{dir}_{global_ind}_depth.jpg')

                global_ind += 1
            except Exception as e:
                print(f"Error rendering {dir}: {e}")

    print(f"Rendering completed. Files saved in {cfg.render_dir}")


if __name__ == '__main__':
    main()
