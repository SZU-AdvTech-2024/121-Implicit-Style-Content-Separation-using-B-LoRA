#!/bin/bash

accelerate launch train_dreambooth_b-lora_sdxl.py \
 --pretrained_model_name_or_path="../modelscope/stable-diffusion-xl-base-1.0" \
 --instance_data_dir="temp_dir/abs7" \
 --output_dir="temp_out/abs7" \
 --instance_prompt="A [tos123]" \
 --resolution=1024 \
 --rank=64 \
 --train_batch_size=1 \
 --learning_rate=5e-5 \
 --lr_scheduler="constant" \
 --lr_warmup_steps=0 \
 --max_train_steps=1000 \
 --checkpointing_steps=200 \
 --seed="48" \
 --gradient_checkpointing \
 --use_8bit_adam \
 --mixed_precision="fp16" \
 --report_to="tensorboard"


# A [ab123]
# A [ab112233]
# ab12414
# tkk123
# tokk111
# rys123
# tos123
# A [ctt123]


# "Twitter bird logo, static image, monochrome background, minimalistic style, black color, vector graphic, high resolution"
# 'photo of a TOK dog', 'in the style of TOK'
# A apple in the style of TTOOKKS
# A aojsdoja book
# An icon of a TOK apple with clouds
# photo of a TOKta dog
# in the style of sksksk(appleIcon with white background)
# "a tok book"
# icon888,an apple,in the style of minimalist, cartoonish illustrations with a pastel color palette
# "a bay,in the style of watercolor painting" \
#a chicken,in the style of icon
#"a girl,in the style of animation" \
#a house,in the style of icon
# icon888,bird,in the style of cute color

# a [v392] twitter
# A [v513] gloves
# A [v303] apple
# A [v025] colorful_bird
# A [vps10] pen_sketch
# A [cct9] cat
# A [v99] iphone logo
# An apple,Monochrome icon
# A house,Monochrome icon
# A bird,minimal icon
# A pig,minimal icon,stickers
# A pig,Monochrome icon,Glyph Neue
# A [v10] chicken in the style of iconography

# A chicken. minimal flat 2d icon. lineal color. on a white background. trending on artstation（line_chicken）
# A bird. minimal flat 2d icon. lineal color. on a white background. trending on artstation（colorful_bird）
# A bird. minimal flat 2d icon. color hand drawn. on a white background. trending on artstation（colorful_bird）
# A chicken. minimal flat 2d icon. Matisse style. on a white background. trending on artstation（chicken_matisse）

