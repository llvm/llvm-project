// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx1100 %s 2>&1 | FileCheck --check-prefixes=NOGFX11 --implicit-check-not=error: %s

image_sample_d v[64:66], [v32, v16, v8, v4, v2, v1], s[4:11], s[100:103] dmask:0x7 dim:SQ_RSRC_IMG_2D
// NOGFX11: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

image_sample_d v[64:66], [v32, v16, v8, v4, v2, v1, v0, v20, v21], s[4:11], s[100:103] dmask:0x7 dim:SQ_RSRC_IMG_3D
// NOGFX11: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

image_sample_d v[64:66], [v32, v16, v8, v4, v2, v1, v5], s[4:11], s[100:103] dmask:0x7 dim:SQ_RSRC_IMG_CUBE
// NOGFX11: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

image_sample_d v[64:66], [v32, v16, v8, v4, v0, v20, v21], s[4:11], s[100:103] dmask:0x7 dim:SQ_RSRC_IMG_2D_ARRAY
// NOGFX11: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

image_sample_d_cl v[64:66], [v32, v16, v8, v4, v2, v1, v0, v20, v21, v48], s[4:11], s[100:103] dmask:0x7 dim:SQ_RSRC_IMG_3D
// NOGFX11: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

image_sample_c_d v[64:66], [v32, v16, v0, v2, v1, v4, v8, v12, v16, v17], s[4:11], s[100:103] dmask:0x7 dim:SQ_RSRC_IMG_3D
// NOGFX11: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

image_sample_c_d_cl v[64:66], [v32, v16, v0, v2, v1, v4, v8, v12, v16, v17, v18], s[4:11], s[100:103] dmask:0x7 dim:SQ_RSRC_IMG_3D
// NOGFX11: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

image_sample_c_b_cl v[64:66], [v32, v16, v0, v2, v1, v5], s[4:11], s[100:103] dmask:0x7 dim:SQ_RSRC_IMG_3D
// NOGFX11: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

image_sample_d_o v[64:66], [v32, v16, v0, v2, v4, v5, v6, v7, v8, v9], s[4:11], s[100:103] dmask:0x7 dim:SQ_RSRC_IMG_3D
// NOGFX11: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

image_sample_d_cl_o v[64:66], [v32, v16, v0, v2, v4, v5, v6, v7, v8, v9, v10], s[4:11], s[100:103] dmask:0x7 dim:SQ_RSRC_IMG_3D
// NOGFX11: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

image_sample_b_cl_o v[64:66], [v32, v16, v0, v2, v1, v4], s[4:11], s[100:103] dmask:0x7 dim:SQ_RSRC_IMG_3D
// NOGFX11: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

image_sample_c_cl_o v[64:66], [v32, v16, v0, v2, v1, v4], s[4:11], s[100:103] dmask:0x7 dim:SQ_RSRC_IMG_3D
// NOGFX11: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

image_sample_c_d_o v[64:66], [v32, v16, v0, v2, v1, v4, v5, v6, v7, v8, v9], s[4:11], s[100:103] dmask:0x7 dim:SQ_RSRC_IMG_3D
// NOGFX11: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

image_sample_c_d_cl_o v[64:66], [v32, v16, v0, v2, v1, v4, v5, v6, v7, v8, v9, v10], s[4:11], s[100:103] dmask:0x7 dim:SQ_RSRC_IMG_3D
// NOGFX11: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

image_sample_c_l_o v[64:66], [v32, v16, v0, v2, v1, v4], s[4:11], s[100:103] dmask:0x7 dim:SQ_RSRC_IMG_3D
// NOGFX11: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

image_sample_c_b_o v[64:66], [v32, v16, v0, v2, v1, v4], s[4:11], s[100:103] dmask:0x7 dim:SQ_RSRC_IMG_3D
// NOGFX11: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

image_sample_c_b_cl_o v[64:66], [v32, v16, v0, v2, v1, v4, v5], s[4:11], s[100:103] dmask:0x7 dim:SQ_RSRC_IMG_3D
// NOGFX11: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

image_gather4_c_b_cl v[64:67], [v32, v0, v4, v5, v6, v7], s[4:11], s[100:103] dmask:0x1 dim:SQ_RSRC_IMG_3D
// NOGFX11: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

image_gather4_cl_o v[64:67], [v32, v0, v4, v5, v6], s[4:11], s[100:103] dmask:0x1 dim:SQ_RSRC_IMG_3D
// NOGFX11: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_gather4_l_o v[64:67], [v32, v0, v4, v5, v6], s[4:11], s[100:103] dmask:0x1 dim:SQ_RSRC_IMG_3D
// NOGFX11: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_gather4_b_o v[64:67], [v32, v0, v4, v5, v6], s[4:11], s[100:103] dmask:0x1 dim:SQ_RSRC_IMG_3D
// NOGFX11: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_gather4_b_cl_o v[64:67], [v32, v0, v4, v5, v6, v7], s[4:11], s[100:103] dmask:0x1 dim:SQ_RSRC_IMG_3D
// NOGFX11: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_gather4_c_o v[64:67], [v32, v0, v4, v5, v6], s[4:11], s[100:103] dmask:0x1 dim:SQ_RSRC_IMG_3D
// NOGFX11: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_gather4_c_cl_o v[64:67], [v32, v0, v4, v5, v6, v7], s[4:11], s[100:103] dmask:0x1 dim:SQ_RSRC_IMG_3D
// NOGFX11: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_gather4_c_l_o v[64:67], [v32, v0, v4, v5, v6, v7], s[4:11], s[100:103] dmask:0x1 dim:SQ_RSRC_IMG_3D
// NOGFX11: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_gather4_c_b_o v[64:67], [v32, v0, v4, v5, v6, v7], s[4:11], s[100:103] dmask:0x1 dim:SQ_RSRC_IMG_3D
// NOGFX11: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_gather4_c_b_cl_o v[64:67], [v32, v0, v4, v5, v6, v7, v8], s[4:11], s[100:103] dmask:0x1 dim:SQ_RSRC_IMG_3D
// NOGFX11: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_sample_cd v[64:66], [v32, v16, v0, v2, v1, v4, v5, v6, v7], s[4:11], s[100:103] dmask:0x7 dim:SQ_RSRC_IMG_3D
// NOGFX11: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_sample_cd_cl v[64:66], [v32, v16, v0, v2, v1, v4, v5, v6, v7, v8], s[4:11], s[100:103] dmask:0x7 dim:SQ_RSRC_IMG_3D
// NOGFX11: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_sample_c_cd v[64:66], [v32, v16, v0, v2, v1, v4, v5, v6, v7, v8], s[4:11], s[100:103] dmask:0x7 dim:SQ_RSRC_IMG_3D
// NOGFX11: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_sample_c_cd_cl v[64:66], [v32, v16, v0, v2, v1, v4, v5, v6, v7, v8, v9], s[4:11], s[100:103] dmask:0x7 dim:SQ_RSRC_IMG_3D
// NOGFX11: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_sample_cd_o v[64:66], [v32, v16, v0, v2, v1, v4, v5, v6, v7, v8], s[4:11], s[100:103] dmask:0x7 dim:SQ_RSRC_IMG_3D
// NOGFX11: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_sample_cd_cl_o v[64:66], [v32, v16, v0, v2, v1, v4, v5, v6, v7, v8, v9], s[4:11], s[100:103] dmask:0x7 dim:SQ_RSRC_IMG_3D
// NOGFX11: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_sample_c_cd_o v[64:66], [v32, v16, v0, v2, v1, v4, v5, v6, v7, v8, v9], s[4:11], s[100:103] dmask:0x7 dim:SQ_RSRC_IMG_3D
// NOGFX11: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_sample_c_cd_cl_o v[64:66], [v32, v16, v0, v2, v1, v4, v5, v6, v7, v8, v9, v10], s[4:11], s[100:103] dmask:0x7 dim:SQ_RSRC_IMG_3D
// NOGFX11: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_atomic_fcmpswap v[4:5], v32, s[96:103] dmask:0x3 dim:SQ_RSRC_IMG_1D glc
// NOGFX11: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_atomic_fmin v4, v32, s[96:103] dmask:0x1 dim:SQ_RSRC_IMG_1D glc
// NOGFX11: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_atomic_fmax v4, v32, s[96:103] dmask:0x1 dim:SQ_RSRC_IMG_1D glc
// NOGFX11: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_msaa_load v[1:4], v[5:7], s[8:15] dmask:0xf dim:SQ_RSRC_IMG_2D_MSAA
// NOGFX11: :[[@LINE-1]]:{{[0-9]+}}: error: invalid image_gather dmask: only one bit must be set

image_msaa_load v5, v[1:3], s[8:15] dmask:0x1 dim:SQ_RSRC_IMG_2D_MSAA d16
// NOGFX11: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

image_msaa_load v14, [v204,v11,v14,v19], s[40:47] dmask:0x1 dim:SQ_RSRC_IMG_2D_MSAA_ARRAY
// NOGFX11: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
