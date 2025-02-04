// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx1010 %s 2>&1 | FileCheck --check-prefixes=NOGFX10 --implicit-check-not=error: %s

image_atomic_add v5, v1, s[8:15] dmask:0x1 unorm glc
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: missing dim operand

image_atomic_and v5, v1, s[8:15] dmask:0x1 unorm glc
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: missing dim operand

image_atomic_cmpswap v[5:6], v1, s[8:15] dmask:0x3 unorm glc
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: missing dim operand

image_atomic_dec v5, v1, s[8:15] dmask:0x1 unorm glc
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: missing dim operand

image_atomic_fcmpswap v[1:2], v2, s[12:19] dmask:0x3 unorm glc
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: missing dim operand

image_atomic_fmax v4, v32, s[96:103] dmask:0x1 glc
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: missing dim operand

image_atomic_fmin v4, v32, s[96:103] dmask:0x1 glc
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: missing dim operand

image_atomic_inc v5, v1, s[8:15] dmask:0x1 unorm
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: missing dim operand

image_atomic_or v5, v1, s[8:15] dmask:0x1 unorm
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: missing dim operand

image_atomic_smax v5, v1, s[8:15] dmask:0x1 unorm
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: missing dim operand

image_atomic_smin v5, v1, s[8:15] dmask:0x1 unorm
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: missing dim operand

image_atomic_sub v5, v1, s[8:15] dmask:0x1 unorm
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: missing dim operand

image_atomic_swap v5, v1, s[8:15] dmask:0x1 unorm
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: missing dim operand

image_atomic_umax v5, v1, s[8:15] dmask:0x1 unorm
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: missing dim operand

image_atomic_umin v5, v1, s[8:15] dmask:0x1 unorm
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: missing dim operand

image_atomic_xor v5, v1, s[8:15] dmask:0x1 unorm
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: missing dim operand

image_gather4 v[5:8], v[1:2], s[8:15], s[12:15] dmask:0x1 unorm
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: missing dim operand

image_gather4_b v[5:8], v[1:3], s[8:15], s[12:15] dmask:0x1 unorm
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: missing dim operand

image_gather4_b_cl v[5:8], v[1:4], s[8:15], s[12:15] dmask:0x1 unorm
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: missing dim operand

image_gather4_b_cl_o v[5:8], v[1:8], s[8:15], s[12:15] dmask:0x1 unorm
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: missing dim operand

image_gather4_b_o v[5:8], v[1:4], s[8:15], s[12:15] dmask:0x1 unorm
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: missing dim operand

image_gather4_c v[5:8], v[1:3], s[8:15], s[12:15] dmask:0x1 unorm
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: missing dim operand

image_gather4_c_b v[5:8], v[1:4], s[8:15], s[12:15] dmask:0x1 unorm
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: missing dim operand

image_gather4_c_b_cl v[5:8], v[1:8], s[8:15], s[12:15] dmask:0x1 unorm
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: missing dim operand

image_gather4_c_b_cl_o v[5:8], v[1:8], s[8:15], s[12:15] dmask:0x1 unorm
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: missing dim operand

image_gather4_c_b_o v[5:8], v[1:8], s[8:15], s[12:15] dmask:0x1 unorm
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: missing dim operand

image_gather4_c_cl v[5:8], v[1:4], s[8:15], s[12:15] dmask:0x1 unorm
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: missing dim operand

image_gather4_c_cl_o v[5:8], v[1:8], s[8:15], s[12:15] dmask:0x1 unorm
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: missing dim operand

image_gather4_c_l v[5:8], v[1:4], s[8:15], s[12:15] dmask:0x1 unorm
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: missing dim operand

image_gather4_cl v[5:8], v[1:3], s[8:15], s[12:15] dmask:0x1 unorm
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: missing dim operand

image_gather4_c_l_o v[5:8], v[1:8], s[8:15], s[12:15] dmask:0x1 unorm
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: missing dim operand

image_gather4_cl_o v[5:8], v[1:4], s[8:15], s[12:15] dmask:0x1 unorm
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: missing dim operand

image_gather4_c_lz v[5:8], v[1:3], s[8:15], s[12:15] dmask:0x1 unorm
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: missing dim operand

image_gather4_c_lz_o v[5:8], v[1:4], s[8:15], s[12:15] dmask:0x1 unorm
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: missing dim operand

image_gather4_c_o v[5:8], v[1:4], s[8:15], s[12:15] dmask:0x1 unorm
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: missing dim operand

image_gather4h v[254:255], v[254:255], ttmp[8:15], ttmp[12:15] dmask:0x4 unorm
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: missing dim operand

image_gather4_l v[5:8], v[1:3], s[8:15], s[12:15] dmask:0x1 unorm
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: missing dim operand

image_gather4_l_o v[5:8], v[1:4], s[8:15], s[12:15] dmask:0x1 unorm
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: missing dim operand

image_gather4_lz v[5:8], v[1:2], s[8:15], s[12:15] dmask:0x1 unorm
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: missing dim operand

image_gather4_lz_o v[5:8], v[1:3], s[8:15], s[12:15] dmask:0x1 unorm
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: missing dim operand

image_gather4_o v[5:8], v[1:3], s[8:15], s[12:15] dmask:0x1 unorm
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: missing dim operand

image_get_lod v5, v1, s[8:15], s[12:15] dmask:0x1 unorm
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: missing dim operand

image_get_resinfo v5, v1, s[8:15] dmask:0x1 unorm
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: missing dim operand

image_load v[0:3], v0, s[0:7] dmask:0xf unorm
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: missing dim operand

image_load_mip v[5:6], v1, s[8:15] dmask:0x3 a16
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: missing dim operand

image_load_pck v[5:6], v1, s[8:15] dmask:0x1 tfe
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: missing dim operand

image_load_pck_sgn v5, v1, s[8:15] dmask:0x1 unorm
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: missing dim operand

image_load_mip_pck v5, v[1:2], s[8:15] dmask:0x1 unorm
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: missing dim operand

image_load_mip_pck_sgn v5, v[1:2], s[8:15] dmask:0x1 unorm
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: missing dim operand

image_sample v[5:6], v1, s[8:15], s[12:15] dmask:0x1 tfe
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: missing dim operand

image_sample_b v5, v[1:2], s[8:15], s[12:15] dmask:0x1 unorm
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: missing dim operand

image_sample_b_cl v5, v[1:3], s[8:15], s[12:15] dmask:0x1 unorm
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: missing dim operand

image_sample_b_cl_o v5, v[1:4], s[8:15], s[12:15] dmask:0x1 unorm
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: missing dim operand

image_sample_b_o v5, v[1:3], s[8:15], s[12:15] dmask:0x1 unorm
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: missing dim operand

image_sample_c v5, v[1:2], s[8:15], s[12:15] dmask:0x1 unorm
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: missing dim operand

image_sample_c_b v5, v[1:3], s[8:15], s[12:15] dmask:0x1 unorm
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: missing dim operand

image_sample_c_b_cl v5, v[1:4], s[8:15], s[12:15] dmask:0x1 unorm
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: missing dim operand

image_sample_c_b_cl_o v5, v[1:8], s[8:15], s[12:15] dmask:0x1 unorm
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: missing dim operand

image_sample_c_b_o v5, v[1:4], s[8:15], s[12:15] dmask:0x1 unorm
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: missing dim operand

image_sample_c_cd v5, v[1:4], s[8:15], s[12:15] dmask:0x1 unorm
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: missing dim operand

image_sample_c_cd_cl v5, v[1:8], s[8:15], s[12:15] dmask:0x1 unorm
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: missing dim operand

image_sample_c_cd_cl_g16 v[0:3], v[0:4], s[0:7], s[8:11] dmask:0xf
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: missing dim operand

image_sample_c_cd_cl_o v5, v[1:8], s[8:15], s[12:15] dmask:0x1 unorm
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: missing dim operand

image_sample_c_cd_cl_o_g16 v[5:6], v[1:6], s[8:15], s[12:15] dmask:0x3
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: missing dim operand

image_sample_c_cd_g16 v[5:6], v[1:4], s[8:15], s[12:15] dmask:0x3
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: missing dim operand

image_sample_c_cd_o v5, v[1:8], s[8:15], s[12:15] dmask:0x1 unorm
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: missing dim operand

image_sample_c_cd_o_g16 v[5:6], v[1:5], s[8:15], s[12:15] dmask:0x3
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: missing dim operand

image_sample_c_cl v5, v[1:3], s[8:15], s[12:15] dmask:0x1 unorm
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: missing dim operand

image_sample_c_cl_o v5, v[1:4], s[8:15], s[12:15] dmask:0x1 unorm
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: missing dim operand

image_sample_c_d v5, v[1:4], s[8:15], s[12:15] dmask:0x1 unorm
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: missing dim operand

image_sample_cd v5, v[1:3], s[8:15], s[12:15] dmask:0x1 unorm
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: missing dim operand

image_sample_c_d_cl v5, v[1:8], s[8:15], s[12:15] dmask:0x1 unorm
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: missing dim operand

image_sample_cd_cl v5, v[1:4], s[8:15], s[12:15] dmask:0x1 unorm
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: missing dim operand

image_sample_c_d_cl_g16 v[0:3], v[0:4], s[0:7], s[8:11] dmask:0xf
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: missing dim operand

image_sample_cd_cl_g16 v[0:3], v[0:3], s[0:7], s[8:11] dmask:0xf
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: missing dim operand

image_sample_c_d_cl_o v5, v[1:8], s[8:15], s[12:15] dmask:0x1 unorm
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: missing dim operand

image_sample_cd_cl_o v5, v[1:8], s[8:15], s[12:15] dmask:0x1 unorm
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: missing dim operand

image_sample_c_d_cl_o_g16 v[5:6], v[1:6], s[8:15], s[12:15] dmask:0x3
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: missing dim operand

image_sample_cd_cl_o_g16 v[5:6], v[1:5], s[8:15], s[12:15] dmask:0x3
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: missing dim operand

image_sample_c_d_g16 v[0:3], v[0:3], s[0:7], s[8:11] dmask:0xf
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: missing dim operand

image_sample_cd_g16 v[0:3], v[0:2], s[0:7], s[8:11] dmask:0xf
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: missing dim operand

image_sample_c_d_o v5, v[1:8], s[8:15], s[12:15] dmask:0x1 unorm
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: missing dim operand

image_sample_cd_o v5, v[1:4], s[8:15], s[12:15] dmask:0x1 unorm
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: missing dim operand

image_sample_c_d_o_g16 v0, [v0, v1, v2, v4, v6, v7, v8], s[0:7], s[8:11] dmask:0x4
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: missing dim operand

image_sample_cd_o_g16 v[5:6], v[1:4], s[8:15], s[12:15] dmask:0x3
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: missing dim operand

image_sample_c_l v5, v[1:3], s[8:15], s[12:15] dmask:0x1 unorm
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: missing dim operand

image_sample_cl v5, v[1:2], s[8:15], s[12:15] dmask:0x1 unorm
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: missing dim operand

image_sample_c_l_o v5, v[1:4], s[8:15], s[12:15] dmask:0x1 unorm
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: missing dim operand

image_sample_cl_o v5, v[1:3], s[8:15], s[12:15] dmask:0x1 unorm
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: missing dim operand

image_sample_c_lz v5, v[1:2], s[8:15], s[12:15] dmask:0x1 unorm
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: missing dim operand

image_sample_c_lz_o v5, v[1:3], s[8:15], s[12:15] dmask:0x1 unorm
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: missing dim operand

image_sample_c_o v5, v[1:3], s[8:15], s[12:15] dmask:0x1 unorm
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: missing dim operand

image_sample_d v5, v[1:3], s[8:15], s[12:15] dmask:0x1 unorm
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: missing dim operand

image_sample_d_cl v5, v[1:4], s[8:15], s[12:15] dmask:0x1 unorm
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: missing dim operand

image_sample_d_cl_g16 v[0:3], v[0:3], s[0:7], s[8:11] dmask:0xf
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: missing dim operand

image_sample_d_cl_o v5, v[1:8], s[8:15], s[12:15] dmask:0x1 unorm
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: missing dim operand

image_sample_d_cl_o_g16 v[5:6], v[1:5], s[8:15], s[12:15] dmask:0x3
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: missing dim operand

image_sample_d_g16 v[0:3], v[0:2], s[0:7], s[8:11] dmask:0xf
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: missing dim operand

image_sample_d_o v5, v[1:4], s[8:15], s[12:15] dmask:0x1 unorm
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: missing dim operand

image_sample_d_o_g16 v[5:6], v[1:4], s[8:15], s[12:15] dmask:0x3
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: missing dim operand

image_sample_l v5, v[1:2], s[8:15], s[12:15] dmask:0x1 unorm
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: missing dim operand

image_sample_l_o v5, v[1:3], s[8:15], s[12:15] dmask:0x1 unorm
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: missing dim operand

image_sample_lz v5, v1, s[8:15], s[12:15] dmask:0x1 unorm
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: missing dim operand

image_sample_lz_o v5, v[1:2], s[8:15], s[12:15] dmask:0x1 unorm
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: missing dim operand

image_sample_o v5, v[1:2], s[8:15], s[12:15] dmask:0x1 unorm
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: missing dim operand

image_store v1, v2, s[12:19] dmask:0x0 unorm
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: missing dim operand

image_store_mip v1, v[2:3], s[12:19] dmask:0x0 unorm
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: missing dim operand

image_store_pck v1, v[2:3], s[12:19] dmask:0x1 unorm
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: missing dim operand

image_store_mip_pck v1, v[2:3], s[12:19] dmask:0x0 unorm
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: missing dim operand

image_load v[0:3], v0, s[0:7] dmask:0xf dim:SQ_RSRC_IMG_1D da
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

image_load_pck v[0:3], v0, s[0:7] dmask:0xf dim:SQ_RSRC_IMG_1D d16
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

image_load v[0:1], v0, s[0:7] dmask:0xf dim:SQ_RSRC_IMG_1D
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: image data size does not match dmask, d16 and tfe

image_load v[0:3], v[0:1], s[0:7] dmask:0xf dim:SQ_RSRC_IMG_1D
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: image address size does not match dim and a16

image_load_mip v[0:3], v[0:2], s[0:7] dmask:0xf dim:SQ_RSRC_IMG_CUBE
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: image address size does not match dim and a16

image_sample_d v[0:3], [v0, v1, v2, v3, v4], s[0:7], s[8:11] dmask:0xf dim:SQ_RSRC_IMG_1D_ARRAY
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: image address size does not match dim and a16

image_sample_b_cl v[0:3], v[0:3], s[0:7], s[8:11] dmask:0xf dim:SQ_RSRC_IMG_CUBE
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: image address size does not match dim and a16

image_sample_c_d v[0:3], [v0, v1, v2, v3, v4, v5, v6], s[0:7], s[8:11] dmask:0xf dim:SQ_RSRC_IMG_2D_ARRAY
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: image address size does not match dim and a16

image_sample_c_d_cl v[0:3], [v0, v1, v2, v3, v4, v5, v6, v7], s[0:7], s[8:11] dmask:0xf dim:SQ_RSRC_IMG_2D_ARRAY
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: image address size does not match dim and a16

image_sample_c_d_cl_o v[0:3], v[0:7], s[0:7], s[8:11] dmask:0xf dim:SQ_RSRC_IMG_2D
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: image address size does not match dim and a16

image_load v[0:1], v0, s[0:7] dmask:0x9 dim:1 D
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: invalid dim value

// null is not allowed as SRSRC or SSAMP
image_atomic_add v1, v[10:11], null dmask:0x1 dim:SQ_RSRC_IMG_2D
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

image_atomic_and v1, v[10:11], null dmask:0x1 dim:SQ_RSRC_IMG_2D
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

image_atomic_cmpswap v[0:1], v[10:11], null dmask:0x3 dim:SQ_RSRC_IMG_2D
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

image_atomic_dec v1, v[10:11], null dmask:0x1 dim:SQ_RSRC_IMG_2D
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

image_atomic_fcmpswap v[1:2], v[2:3], null dmask:0x3 dim:SQ_RSRC_IMG_2D
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

image_atomic_fmax v1, v[10:11], null dmask:0x1 dim:SQ_RSRC_IMG_2D
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

image_atomic_fmin v1, v[10:11], null dmask:0x1 dim:SQ_RSRC_IMG_2D
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

image_atomic_inc v1, v[10:11], null dmask:0x1 dim:SQ_RSRC_IMG_2D
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

image_atomic_or v1, v[10:11], null dmask:0x1 dim:SQ_RSRC_IMG_2D
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

image_atomic_smax v1, v[10:11], null dmask:0x1 dim:SQ_RSRC_IMG_2D
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

image_atomic_smin v1, v[10:11], null dmask:0x1 dim:SQ_RSRC_IMG_2D
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

image_atomic_sub v1, v[10:11], null dmask:0x1 dim:SQ_RSRC_IMG_2D
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

image_atomic_swap v1, v[10:11], null dmask:0x1 dim:SQ_RSRC_IMG_2D
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

image_atomic_umax v1, v[10:11], null dmask:0x1 dim:SQ_RSRC_IMG_2D
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

image_atomic_umin v1, v[10:11], null dmask:0x1 dim:SQ_RSRC_IMG_2D
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

image_atomic_xor v1, v[10:11], null dmask:0x1 dim:SQ_RSRC_IMG_2D
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

image_gather4 v[64:67], v32, null, s[4:11], dmask:0x1 dim:SQ_RSRC_IMG_1D
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

image_gather4 v[64:67], v32, s[4:11], null dmask:0x1 dim:SQ_RSRC_IMG_1D
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

image_gather4_b v[64:67], v[32:33], null, s[100:103] dmask:0x1 dim:SQ_RSRC_IMG_1D
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

image_gather4_b v[64:67], v[32:33], s[4:11], null dmask:0x1 dim:SQ_RSRC_IMG_1D
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

image_gather4_c v[64:67], v[32:33], null, s[100:103] dmask:0x1 dim:SQ_RSRC_IMG_1D
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

image_gather4_c v[64:67], v[32:33], s[4:11], null dmask:0x1 dim:SQ_RSRC_IMG_1D
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

image_gather4h v[64:67], v32, null, s[100:103] dmask:0x1 dim:SQ_RSRC_IMG_1D
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

image_gather4h v[64:67], v32, s[4:11], null dmask:0x1 dim:SQ_RSRC_IMG_1D
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

image_gather4_l v[64:67], v[32:33], null, s[100:103] dmask:0x1 dim:SQ_RSRC_IMG_1D
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

image_gather4_l v[64:67], v[32:33], s[4:11], null dmask:0x1 dim:SQ_RSRC_IMG_1D
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

image_gather4_o v[64:67], v[32:33], null, s[100:103] dmask:0x1 dim:SQ_RSRC_IMG_1D
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

image_gather4_o v[64:67], v[32:33], s[4:11], null dmask:0x1 dim:SQ_RSRC_IMG_1D
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

image_load v[4:7], v0, null dmask:0xf dim:SQ_RSRC_IMG_1D
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

image_store v[0:3], v[254:255], null dmask:0xf dim:SQ_RSRC_IMG_2D
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

image_sample v[5:6], v1, null, s[12:15] dmask:0x3 dim:SQ_RSRC_IMG_1D
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

image_sample v[5:6], v1, s[8:15], null dmask:0x3 dim:SQ_RSRC_IMG_1D
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

image_sample_b v[5:6], v[1:2], null, s[12:15] dmask:0x3 dim:SQ_RSRC_IMG_1D
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

image_sample_b v[5:6], v[1:2], s[8:15], null dmask:0x3 dim:SQ_RSRC_IMG_1D
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

image_sample_c v[5:6], v[1:2], null, s[12:15] dmask:0x3 dim:SQ_RSRC_IMG_1D
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

image_sample_c v[5:6], v[1:2], s[8:15], null dmask:0x3 dim:SQ_RSRC_IMG_1D
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

image_sample_d v[5:6], v[1:3], null, s[12:15] dmask:0x3 dim:SQ_RSRC_IMG_1D
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

image_sample_d v[5:6], v[1:3], s[8:15], null dmask:0x3 dim:SQ_RSRC_IMG_1D
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

image_sample_l v[5:6], v[1:2], null, s[12:15] dmask:0x3 dim:SQ_RSRC_IMG_1D
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

image_sample_l v[5:6], v[1:2], s[8:15], null dmask:0x3 dim:SQ_RSRC_IMG_1D
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

image_sample_o v[5:6], v[1:2], null, s[12:15] dmask:0x3 dim:SQ_RSRC_IMG_1D
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

image_sample_o v[5:6], v[1:2], s[8:15], null dmask:0x3 dim:SQ_RSRC_IMG_1D
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
