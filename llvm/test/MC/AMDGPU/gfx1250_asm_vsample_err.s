; RUN: not llvm-mc -triple=amdgcn -mcpu=gfx1250 -show-encoding %s 2>&1 | FileCheck --check-prefix=GFX1250-ERR --implicit-check-not=error: --strict-whitespace %s

image_sample v64, v32, s[4:11], s[100:103] dmask:0x1 dim:SQ_RSRC_IMG_1D
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_sample_d v64, [v32, v33, v34], s[4:11], s[4:7] dmask:0x1 dim:SQ_RSRC_IMG_1D
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_sample_l v64, [v32, v33], s[4:11], s[4:7] dmask:0x1 dim:SQ_RSRC_IMG_1D
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_sample_b v64, [v32, v33], s[4:11], s[4:7] dmask:0x1 dim:SQ_RSRC_IMG_1D
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_sample_lz v64, v32, s[4:11], s[4:7] dmask:0x1 dim:SQ_RSRC_IMG_1D
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_sample_c v64, [v32, v33], s[4:11], s[4:7] dmask:0x1 dim:SQ_RSRC_IMG_1D
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_sample_c_d v64, [v32, v33, v34, v35], s[4:11], s[4:7] dmask:0x1 dim:SQ_RSRC_IMG_1D
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_sample_c_l v64, [v32, v33, v34], s[4:11], s[4:7] dmask:0x1 dim:SQ_RSRC_IMG_1D
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_sample_c_b v64, [v32, v33, v34], s[4:11], s[4:7] dmask:0x1 dim:SQ_RSRC_IMG_1D
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_sample_c_lz v64, [v32, v33], s[4:11], s[4:7] dmask:0x1 dim:SQ_RSRC_IMG_1D
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_sample_o v64, [v32, v33], s[4:11], s[4:7] dmask:0x1 dim:SQ_RSRC_IMG_1D
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_sample_d_o v64, [v32, v33, v34, v35], s[4:11], s[4:7] dmask:0x1 dim:SQ_RSRC_IMG_1D
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_sample_l_o v64, [v32, v33, v34], s[4:11], s[4:7] dmask:0x1 dim:SQ_RSRC_IMG_1D
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_sample_b_o v64, [v32, v33, v34], s[4:11], s[4:7] dmask:0x1 dim:SQ_RSRC_IMG_1D
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_sample_lz_o v64, [v32, v33], s[4:11], s[4:7] dmask:0x1 dim:SQ_RSRC_IMG_1D
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_sample_c_o v64, [v32, v33, v34], s[4:11], s[4:7] dmask:0x1 dim:SQ_RSRC_IMG_1D
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_sample_c_d_o v64, [v32, v33, v34, v[35:36]], s[4:11], s[4:7] dmask:0x1 dim:SQ_RSRC_IMG_1D
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_sample_c_l_o v64, [v32, v33, v34, v35], s[4:11], s[4:7] dmask:0x1 dim:SQ_RSRC_IMG_1D
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_sample_c_b_o v64, [v32, v33, v34, v35], s[4:11], s[4:7] dmask:0x1 dim:SQ_RSRC_IMG_1D
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_sample_c_lz_o v64, [v32, v33, v34], s[4:11], s[4:7] dmask:0x1 dim:SQ_RSRC_IMG_1D
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_gather4 v[64:67], [v32, v33], s[4:11], s[4:7] dmask:0x1 dim:SQ_RSRC_IMG_2D
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_gather4_l v[64:67], [v32, v33, v34], s[4:11], s[4:7] dmask:0x1 dim:SQ_RSRC_IMG_2D
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_gather4_b v[64:67], [v32, v33, v34], s[4:11], s[4:7] dmask:0x1 dim:SQ_RSRC_IMG_2D
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_gather4_lz v[64:67], [v32, v33], s[4:11], s[4:7] dmask:0x1 dim:SQ_RSRC_IMG_2D
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_gather4_c v[64:67], [v32, v33, v34], s[4:11], s[4:7] dmask:0x1 dim:SQ_RSRC_IMG_2D
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_gather4_c_lz v[64:67], [v32, v33, v34], s[4:11], s[4:7] dmask:0x1 dim:SQ_RSRC_IMG_2D
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_gather4_o v[64:67], [v32, v33, v34], s[4:11], s[4:7] dmask:0x1 dim:SQ_RSRC_IMG_2D
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_gather4_lz_o v[64:67], [v32, v33, v34], s[4:11], s[4:7] dmask:0x1 dim:SQ_RSRC_IMG_2D
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_gather4_c_lz_o v[64:67], [v32, v33, v34, v35], s[4:11], s[4:7] dmask:0x1 dim:SQ_RSRC_IMG_2D
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_get_lod v64, v32, s[4:11], s[100:103] dmask:0x1 dim:SQ_RSRC_IMG_1D
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_sample_d_g16 v64, [v32, v33, v34], s[4:11], s[4:7] dmask:0x1 dim:SQ_RSRC_IMG_1D
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_sample_c_d_g16 v64, [v32, v33, v34, v35], s[4:11], s[4:7] dmask:0x1 dim:SQ_RSRC_IMG_1D
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_sample_d_o_g16 v64, [v32, v33, v34, v35], s[4:11], s[4:7] dmask:0x1 dim:SQ_RSRC_IMG_1D
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_sample_c_d_o_g16 v64, [v32, v33, v34, v[35:36]], s[4:11], s[4:7] dmask:0x1 dim:SQ_RSRC_IMG_1D
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_sample_cl v64, [v32, v33], s[4:11], s[4:7] dmask:0x1 dim:SQ_RSRC_IMG_1D
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_sample_d_cl v64, [v32, v33, v34, v35], s[4:11], s[4:7] dmask:0x1 dim:SQ_RSRC_IMG_1D
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_sample_b_cl v64, [v32, v33, v34], s[4:11], s[4:7] dmask:0x1 dim:SQ_RSRC_IMG_1D
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_sample_c_cl v64, [v32, v33, v34], s[4:11], s[4:7] dmask:0x1 dim:SQ_RSRC_IMG_1D
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_sample_c_d_cl v64, [v32, v33, v34, v[35:36]], s[4:11], s[4:7] dmask:0x1 dim:SQ_RSRC_IMG_1D
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_sample_c_b_cl v64, [v32, v33, v34, v35], s[4:11], s[4:7] dmask:0x1 dim:SQ_RSRC_IMG_1D
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_sample_cl_o v64, [v32, v33, v34], s[4:11], s[4:7] dmask:0x1 dim:SQ_RSRC_IMG_1D
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_sample_d_cl_o v64, [v32, v33, v34, v[35:36]], s[4:11], s[4:7] dmask:0x1 dim:SQ_RSRC_IMG_1D
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_sample_b_cl_o v64, [v32, v33, v34, v35], s[4:11], s[4:7] dmask:0x1 dim:SQ_RSRC_IMG_1D
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_sample_c_cl_o v64, [v32, v33, v34, v35], s[4:11], s[4:7] dmask:0x1 dim:SQ_RSRC_IMG_1D
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_sample_c_d_cl_o v64, [v32, v33, v34, v[35:37]], s[4:11], s[4:7] dmask:0x1 dim:SQ_RSRC_IMG_1D
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_sample_c_b_cl_o v64, [v32, v33, v34, v[35:36]], s[4:11], s[4:7] dmask:0x1 dim:SQ_RSRC_IMG_1D
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_sample_c_d_cl_g16 v64, [v32, v33, v34, v[35:36]], s[4:11], s[4:7] dmask:0x1 dim:SQ_RSRC_IMG_1D
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_sample_d_cl_o_g16 v64, [v32, v33, v34, v[35:36]], s[4:11], s[4:7] dmask:0x1 dim:SQ_RSRC_IMG_1D
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_sample_c_d_cl_o_g16 v64, [v32, v33, v34, v[35:37]], s[4:11], s[4:7] dmask:0x1 dim:SQ_RSRC_IMG_1D
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_sample_d_cl_g16 v64, [v32, v33, v34, v35], s[4:11], s[4:7] dmask:0x1 dim:SQ_RSRC_IMG_1D
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_gather4_cl v[64:67], [v32, v33, v34], s[4:11], s[4:7] dmask:0x1 dim:SQ_RSRC_IMG_2D
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_gather4_b_cl v[64:67], [v32, v33, v34, v35], s[4:11], s[4:7] dmask:0x1 dim:SQ_RSRC_IMG_2D
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_gather4_c_cl v[64:67], [v32, v33, v34, v35], s[4:11], s[4:7] dmask:0x1 dim:SQ_RSRC_IMG_2D
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_gather4_c_l v[64:67], [v32, v33, v34, v35], s[4:11], s[4:7] dmask:0x1 dim:SQ_RSRC_IMG_2D
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_gather4_c_b v[64:67], [v32, v33, v34, v35], s[4:11], s[4:7] dmask:0x1 dim:SQ_RSRC_IMG_2D
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_gather4_c_b_cl v[64:67], [v32, v33, v34, v[35:36]], s[4:11], s[4:7] dmask:0x1 dim:SQ_RSRC_IMG_2D
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_gather4h v[64:67], [v32, v33], s[4:11], s[4:7] dmask:0x1 dim:SQ_RSRC_IMG_2D
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_msaa_load v[1:4], [v5, v6, v7], s[8:15] dmask:0x1 dim:SQ_RSRC_IMG_2D_MSAA
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU
