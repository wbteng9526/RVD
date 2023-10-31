import torch
from torch import nn
import torch.nn.functional as F
from functools import partial
import numpy as np
from tqdm import tqdm
from torch.distributions import Normal
from .utils import exists, cosine_beta_schedule, extract, noise_like, default, extract_tensor
from .denoising_diffusion import GaussianDiffusion
from .network_components import (
    linear,
    conv_nd,
    zero_module,
    timestep_embedding
)

from ldm.modules.attention import SpatialTransformer
from ldm.modules.diffusionmodules.openaimodel import TimestepEmbedSequential, ResBlock, Downsample, AttentionBlock


class ControlGaussianDiffusion(GaussianDiffusion):
    def __init__(self, control_fn, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.control_fn = control_fn
        self.control_scales = [1.0] * 13
    
    def p_losses(self, x_start, context, cond, t, trans_shift_scale):
        noise = torch.randn_like(x_start)
        cur_frame = x_start
        if exists(self.transform_fn):
            self.otherlogs["predict"].append(trans_shift_scale[0].detach())
            if self.transform_fn.context_mode in ["residual"]:
                x_start = (x_start - trans_shift_scale[0]) / trans_shift_scale[1]
            else:
                raise NotImplementedError
            
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        control = self.control_fn(x=x_noisy, hint=torch.cat([cond], dim=1), timesteps=t, context=context)
        control = [c * scale for c, scale in zip(control, self.control_scales)]
        x_recon = self.denoise_fn(x_noisy, t, context=context, control=control)

        if self.pred_mode == "noise":
            if self.loss_type == "l1":
                loss = (noise - x_recon).abs().mean()
            elif self.loss_type == "l2":
                loss = F.mse_loss(noise, x_recon)
            else:
                raise NotImplementedError()
        elif self.pred_mode == "pred_true":
            if self.loss_type == "l1":
                loss = (x_start - x_recon).abs().mean()
            elif self.loss_type == "l2":
                loss = F.mse_loss(x_start, x_recon)
            else:
                raise NotImplementedError()
        
        return loss

    def step_forward(self, x, context, cond, t, trans_shift_scale):
        return self.p_losses(x, context, cond, t, trans_shift_scale)
    
    def forward(self, batch):
        video, control_video = batch[:,:,:3], batch[:,:,3:]
        device = video.device
        T, B, C, H, W = video.shape
        t = torch.randint(0, self.num_timesteps, (B,), device=device).long()
        loss = 0
        state_shape = (B, 1, H, W)
        self.history_fn.init_state(state_shape)
        if exists(self.transform_fn):
            self.transform_fn.init_state(state_shape)
            self.otherlogs["predict"] = []
        
        for i in range(video.shape[0]):
            if i >= 2:
                L = self.step_forward(video[i], context, control_video[i], t, trans_shift_scale)
                loss += L
            if i < video.shape[0] - 1:
                context, trans_shift_scale = self.scan_context(video[i])
        
        if exists(self.transform_fn):
            self.otherlogs["predict"] = torch.stack(self.otherlogs["predict"], 0)
        return loss / (video.shape[0] - 2)
    
    def p_mean_variance(self, x, c, t, context, clip_denoised: bool):
        if self.pred_mode == "noise":
            noise = self.denoise_fn(x, t, context=context, control=c)
            x_recon = self.predict_start_from_noise(x, t=t, noise=noise)
        elif self.pred_mode == "pred_true":
            x_recon = self.denoise_fn(x, t, context=context, control=c)
        if clip_denoised:
            x_recon.clamp_(-2, 2)
        
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t
        )
        return model_mean, posterior_variance, posterior_log_variance
    
    @torch.no_grad()
    def p_sample(self, x, c, t, context, clip_denoised=True, repeat_noise=False):
        b, *_, device = *x.shape, x.device
        model_mean, _,  model_log_variance = self.p_mean_variance(
            x=x, c=c, t=t, context=context, clip_denoised=clip_denoised
        )
        noise = noise_like(x.shape, device, repeat_noise)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, cond, shape, context):
        device = self.betas.device

        b = shape[0]
        img = torch.randn(shape, device=device)
        # res = [img]
        for count, i in enumerate(tqdm(
            reversed(range(0, self.num_timesteps)),
            desc="sampling loop time step",
            total=self.num_timesteps
        )):
            time = torch.full((b,), i, device=device, dtype=torch.long)
            img = self.p_sample(
                img,
                cond,
                time,
                context=context,
                clip_denoised=self.clip_noise
            )
        
        return img
    
    @torch.no_grad()
    def sample_single(self, cond, shape):
        '''
        Pre-train a ControlNet to get the initial frame
        '''
        pass


    @torch.no_grad()
    def sample(self, conds, init_frames, num_of_frames=3):
        video = [frame for frame in init_frames]
        T, B, C, H, W = init_frames.shape
        state_shape = (B, 1, H, W)
        self.history_fn.init_state(state_shape)
        if exists(self.transform_fn):
            self.transform_fn.init_state(state_shape)
        for frame in video:
            context = self.history_fn(frame)
            if exists(self.transform_fn):
                trans_shift_scale = self.transform_fn(frame)
        for i in range(num_of_frames):
            generated_frame = self.p_sample_loop(conds, init_frames[0].shape, context)
            if exists(self.transform_fn) and (
                self.transform_fn.context_mode in ["residual"]
            ):
                generated_frame = generated_frame * trans_shift_scale[1] + trans_shift_scale[0]
            
            context = self.history_fn(generated_frame.clamp(-1, 1))
            if exists(self.transform_fn):
                trans_shift_scale = self.transform_fn(generated_frame.clamp(-1, 1))
            video.append(generated_frame)
        return torch.stack(video, 0)


class ControlNet(nn.Module):
    def __init__(
            self,
            image_size,
            in_channels,
            model_channels,
            hint_channels,
            num_res_blocks,
            attention_resolutions,
            dropout=0,
            channel_mult=(1, 2, 4, 8),
            conv_resample=True,
            dims=2,
            use_checkpoint=False,
            use_fp16=False,
            num_heads=-1,
            num_head_channels=-1,
            num_heads_upsample=-1,
            use_scale_shift_norm=False,
            resblock_updown=False,
            use_new_attention_order=False,
            use_spatial_transformer=False,  # custom transformer support
            transformer_depth=1,  # custom transformer support
            context_dim=None,  # custom transformer support
            n_embed=None,  # custom support for prediction of discrete ids into codebook of first stage vq model
            legacy=True,
            disable_self_attentions=None,
            num_attention_blocks=None,
            disable_middle_self_attn=False,
            use_linear_in_transformer=False,
    ):
        super().__init__()
        if use_spatial_transformer:
            assert context_dim is not None, 'Fool!! You forgot to include the dimension of your cross-attention conditioning...'

        if context_dim is not None:
            assert use_spatial_transformer, 'Fool!! You forgot to use the spatial transformer for your cross-attention conditioning...'
            from omegaconf.listconfig import ListConfig
            if type(context_dim) == ListConfig:
                context_dim = list(context_dim)

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if num_heads == -1:
            assert num_head_channels != -1, 'Either num_heads or num_head_channels has to be set'

        if num_head_channels == -1:
            assert num_heads != -1, 'Either num_heads or num_head_channels has to be set'

        self.dims = dims
        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        if isinstance(num_res_blocks, int):
            self.num_res_blocks = len(channel_mult) * [num_res_blocks]
        else:
            if len(num_res_blocks) != len(channel_mult):
                raise ValueError("provide num_res_blocks either as an int (globally constant) or "
                                 "as a list/tuple (per-level) with the same length as channel_mult")
            self.num_res_blocks = num_res_blocks
        if disable_self_attentions is not None:
            # should be a list of booleans, indicating whether to disable self-attention in TransformerBlocks or not
            assert len(disable_self_attentions) == len(channel_mult)
        if num_attention_blocks is not None:
            assert len(num_attention_blocks) == len(self.num_res_blocks)
            assert all(map(lambda i: self.num_res_blocks[i] >= num_attention_blocks[i], range(len(num_attention_blocks))))
            print(f"Constructor of UNetModel received num_attention_blocks={num_attention_blocks}. "
                  f"This option has LESS priority than attention_resolutions {attention_resolutions}, "
                  f"i.e., in cases where num_attention_blocks[i] > 0 but 2**i not in attention_resolutions, "
                  f"attention will still not be set.")

        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint
        self.dtype = torch.float16 if use_fp16 else torch.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.predict_codebook_ids = n_embed is not None

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels, model_channels, 3, padding=1)
                )
            ]
        )
        self.zero_convs = nn.ModuleList([self.make_zero_conv(model_channels)])

        self.input_hint_block = TimestepEmbedSequential(
            conv_nd(dims, hint_channels, 16, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 16, 16, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 16, 32, 3, padding=1, stride=2),
            nn.SiLU(),
            conv_nd(dims, 32, 32, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 32, 96, 3, padding=1, stride=2),
            nn.SiLU(),
            conv_nd(dims, 96, 96, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 96, 256, 3, padding=1, stride=2),
            nn.SiLU(),
            zero_module(conv_nd(dims, 256, model_channels, 3, padding=1))
        )

        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for nr in range(self.num_res_blocks[level]):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        # num_heads = 1
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    if exists(disable_self_attentions):
                        disabled_sa = disable_self_attentions[level]
                    else:
                        disabled_sa = False

                    if not exists(num_attention_blocks) or nr < num_attention_blocks[level]:
                        layers.append(
                            AttentionBlock(
                                ch,
                                use_checkpoint=use_checkpoint,
                                num_heads=num_heads,
                                num_head_channels=dim_head,
                                use_new_attention_order=use_new_attention_order,
                            ) if not use_spatial_transformer else SpatialTransformer(
                                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                                disable_self_attn=disabled_sa, use_linear=use_linear_in_transformer,
                                use_checkpoint=use_checkpoint
                            )
                        )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self.zero_convs.append(self.make_zero_conv(ch))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                self.zero_convs.append(self.make_zero_conv(ch))
                ds *= 2
                self._feature_size += ch

        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        if legacy:
            # num_heads = 1
            dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=dim_head,
                use_new_attention_order=use_new_attention_order,
            ) if not use_spatial_transformer else SpatialTransformer(  # always uses a self-attn
                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                disable_self_attn=disable_middle_self_attn, use_linear=use_linear_in_transformer,
                use_checkpoint=use_checkpoint
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self.middle_block_out = self.make_zero_conv(ch)
        self._feature_size += ch

    def make_zero_conv(self, channels):
        return TimestepEmbedSequential(zero_module(conv_nd(self.dims, channels, channels, 1, padding=0)))

    def forward(self, x, hint, timesteps, context, **kwargs):
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)

        guided_hint = self.input_hint_block(hint, emb, context)
        print(context.size())

        outs = []

        h = x.type(self.dtype)
        for module, zero_conv in zip(self.input_blocks, self.zero_convs):
            if guided_hint is not None:
                h = module(h, emb, context)
                print(h.size())
                h += guided_hint
                guided_hint = None
            else:
                h = module(h, emb, context)
            outs.append(zero_conv(h, emb, context))

        h = self.middle_block(h, emb, context)
        outs.append(self.middle_block_out(h, emb, context))

        return outs