; RUN: not llvm-mc -arch=amdgcn -mcpu=gfx1210 -show-encoding %s 2>&1 | FileCheck --check-prefix=GFX1210-ERR --implicit-check-not=error: --strict-whitespace %s

image_load v[0:3], v0, s[0:7] dmask:0xf dim:SQ_RSRC_IMG_1D
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_load_mip v[252:255], [v0, v1], s[0:7] dmask:0xf dim:SQ_RSRC_IMG_1D
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_load_pck v5, v1, s[8:15] dmask:0x1 dim:SQ_RSRC_IMG_1D
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_load_pck_sgn v5, v1, s[8:15] dmask:0x1 dim:SQ_RSRC_IMG_1D
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_load_mip_pck v5, [v0, v1], s[8:15] dmask:0x1 dim:SQ_RSRC_IMG_1D
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_store v[0:3], v0, s[0:7] dmask:0xf dim:SQ_RSRC_IMG_1D
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_store_mip v[252:255], [v0, v1], s[0:7] dmask:0xf dim:SQ_RSRC_IMG_1D
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_store_pck v5, v1, s[8:15] dmask:0x1 dim:SQ_RSRC_IMG_1D
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_store_mip_pck v5, [v0, v1], s[8:15] dmask:0x1 dim:SQ_RSRC_IMG_1D
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_atomic_swap v0, v0, s[0:7] dmask:0x1 dim:SQ_RSRC_IMG_1D
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_atomic_cmpswap v[0:1], v0, s[0:7] dmask:0x3 dim:SQ_RSRC_IMG_1D
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_atomic_add_uint v0, v0, s[0:7] dmask:0x1 dim:SQ_RSRC_IMG_1D
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_atomic_sub_uint v0, v0, s[0:7] dmask:0x1 dim:SQ_RSRC_IMG_1D
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_atomic_min_int v0, v0, s[0:7] dmask:0x1 dim:SQ_RSRC_IMG_1D
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_atomic_min_uint v0, v0, s[0:7] dmask:0x1 dim:SQ_RSRC_IMG_1D
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_atomic_max_int v0, v0, s[0:7] dmask:0x1 dim:SQ_RSRC_IMG_1D
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_atomic_max_uint v0, v0, s[0:7] dmask:0x1 dim:SQ_RSRC_IMG_1D
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_atomic_and v0, v0, s[0:7] dmask:0x1 dim:SQ_RSRC_IMG_1D
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_atomic_or v0, v0, s[0:7] dmask:0x1 dim:SQ_RSRC_IMG_1D
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_atomic_xor v0, v0, s[0:7] dmask:0x1 dim:SQ_RSRC_IMG_1D
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_atomic_inc_uint v0, v0, s[0:7] dmask:0x1 dim:SQ_RSRC_IMG_1D
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_atomic_dec_uint v1, [v2, v3], s[4:11] dmask:0x1 dim:SQ_RSRC_IMG_2D
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_atomic_pk_add_f16 v0, v0, s[0:7] dmask:0x1 dim:SQ_RSRC_IMG_1D
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_atomic_pk_add_bf16 v0, v0, s[0:7] dmask:0x1 dim:SQ_RSRC_IMG_1D
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_atomic_add_flt v0, v0, s[0:7] dmask:0x1 dim:SQ_RSRC_IMG_1D
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_atomic_min_flt v0, v0, s[0:7] dmask:0x1 dim:SQ_RSRC_IMG_1D
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_atomic_max_flt v0, v0, s[0:7] dmask:0x1 dim:SQ_RSRC_IMG_1D
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_bvh_intersect_ray v[4:7], [v9, v10, v[11:13], v[14:16], v[17:19]], s[4:7]
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_bvh64_intersect_ray v[4:7], [v[9:10], v11, v[12:14], v[15:17], v[18:20]], s[4:7]
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_bvh_dual_intersect_ray v[0:9], [v[0:1], v[11:12], v[3:5], v[6:8], v[9:10]], s[0:3]
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_bvh8_intersect_ray v[0:9], [v[0:1], v[11:12], v[3:5], v[6:8], v9], s[0:3]
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_get_resinfo v4, v32, s[96:103] dmask:0x1 dim:SQ_RSRC_IMG_1D
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU
