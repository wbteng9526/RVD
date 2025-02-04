# training config
n_step = 1000000
scheduler_checkpoint_step = 100000
log_checkpoint_step = 4000
gradient_accumulate_every = 1
lr = 5e-5
decay = 0.8
minf = 0.2
ema_decay = 0.99
optimizer = "adam"  # adamw or adam
ema_step = 5
ema_start_step = 2000

# load
load_model = False
load_step = False

# diffusion config
loss_type = "l1"
iteration_step = 1600

context_dim_factor = 1
transform_dim_factor = 1
init_num_of_frame = 4
pred_mode = "noise"
clip_noise = True
transform_mode = "residual"
val_num_of_batch = 1
backbone = "resnet"
aux_loss = False
additional_note = ""

# controlnet config
use_checkpoint = True
image_size = 32 # unused
in_channels = 3
hint_channels = 3
model_channels = 320
attention_resolutions = [ 4, 2, 1 ]
num_res_blocks = 2
channel_mult = [ 1, 2, 4, 4 ]
num_head_channels = 64 # need to fix for flash-attn
use_spatial_transformer = True
use_linear_in_transformer = True
transformer_depth = 1
context_dim = 1024
legacy = False

# data config
data_config = {
    "dataset_name": "carla",
    "data_path": "/home/wteng/data/carla/sequence/",
    "sequence_length": 8,
    "img_size": 256,
    "img_channel": 3,
    "add_noise": False,
    "img_hz_flip": False,
}

if data_config["img_size"] == 64:
    embed_dim = 48
    transform_dim_mults = (1, 2, 2, 4)
    dim_mults = (1, 2, 4, 8)
    batch_size = 2
elif data_config["img_size"] in [128, 256]:
    embed_dim = 64
    transform_dim_mults = (1, 2, 3, 4)
    dim_mults = (1, 1, 2, 2, 4, 4)
    batch_size = 1
else:
    raise NotImplementedError

model_name = f"{backbone}-{optimizer}-{pred_mode}-{loss_type}-{data_config['dataset_name']}-d{embed_dim}-t{iteration_step}-{transform_mode}-al{aux_loss}{additional_note}"

result_root = "res"
tensorboard_root = "vis"
