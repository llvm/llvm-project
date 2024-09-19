// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx1100 -mattr=+wavefrontsize32 %s 2>&1 | FileCheck --implicit-check-not=error: %s
// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx1100 -mattr=+wavefrontsize64 %s 2>&1 | FileCheck --implicit-check-not=error: %s

buffer_atomic_add_f64 v[2:3], off, s[12:15], s4 offset:4095
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_atomic_fcmpswap_x2 v[0:3], off, s[0:3], s0 offset:4095
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_atomic_fmax_x2 v[0:1], v0, s[0:3], s0 idxen offset:4095
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_atomic_fmin_x2 v[0:1], off, s[0:3], s0 offset:4095 slc
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_atomic_max_f64 v[2:3], off, s[12:15], s4 offset:4095
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_atomic_min_f64 v[2:3], off, s[12:15], s4 offset:4095
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_atomic_pk_add_f16 v0, v2, s[4:7], 0 idxen glc
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_atomic_pk_add_bf16 v5, off, s[8:11], s3 offset:8388607
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_inv
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_invl2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_lds_dword s[4:7], -1 offset:4095 lds
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_wbinvl1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_wbinvl1_vol
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_wbl2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_add_f64 v1, v[254:255] offset:65535
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_add_rtn_f64 v[10:11], v1, v[4:5] offset:65535
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_add_src2_f32 v0 offset:4 gds
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_add_src2_u32 v1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_add_src2_u64 v1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_and_src2_b32 v1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_and_src2_b64 v1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_dec_src2_u32 v1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_dec_src2_u64 v1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_inc_src2_u32 v1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_inc_src2_u64 v1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_max_src2_f32 v1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_max_src2_f64 v1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_max_src2_i32 v1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_max_src2_i64 v1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_max_src2_u32 v1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_max_src2_u64 v1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_min_src2_f32 v1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_min_src2_f64 v1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_min_src2_i32 v1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_min_src2_i64 v1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_min_src2_u32 v1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_min_src2_u64 v1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_or_src2_b32 v1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_or_src2_b64 v1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_pk_add_bf16 v1, v2 offset:65535
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_pk_add_f16 v1, v2 offset:65535
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_pk_add_rtn_bf16  a3, v2, a1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_pk_add_rtn_f16  a3, v2, a1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_rsub_src2_u32 v1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_rsub_src2_u64 v1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_sub_src2_u32 v1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_sub_src2_u64 v1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_write_src2_b32 v1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_write_src2_b64 v1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_xor_src2_b32 v1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_xor_src2_b64 v1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

flat_atomic_add_f64 v[0:1], v[0:1], v[2:3] glc
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

flat_atomic_fcmpswap_x2 v[0:1], v[1:2], v[2:5] glc
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

flat_atomic_fmax_x2 v[0:1], v[1:2], v[2:3] glc
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

flat_atomic_fmin_x2 v[0:1], v[1:2], v[2:3] glc
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

flat_atomic_max_f64 v[0:1], v[0:1], v[2:3] glc
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

flat_atomic_min_f64 v[0:1], v[0:1], v[2:3] glc
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

flat_atomic_pk_add_bf16 a4, v[2:3], a1 sc0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

flat_atomic_pk_add_bf16 v1, v[2:3], v2 th:TH_ATOMIC_RETURN
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

flat_atomic_pk_add_f16 a4, v[2:3], a1 sc0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

flat_atomic_pk_add_f16 v1, v[2:3], v2 th:TH_ATOMIC_RETURN
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

global_atomic_add_f64 v[0:1], v[0:1], v[2:3], off glc
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

global_atomic_fcmpswap_x2 v[1:2], v[2:5], off offset:-1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

global_atomic_fmax_x2 v[1:2], v[2:3], off dlc
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

global_atomic_fmin_x2 v[1:2], v[2:3], off dlc
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

global_atomic_max_f64 v[0:1], v[0:1], v[2:3], off glc
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

global_atomic_min_f64 v[0:1], v[0:1], v[2:3], off glc
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

global_atomic_pk_add_bf16 a4, v[2:3], a1, off sc0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

global_atomic_pk_add_f16 v0, v[0:1], v2, off glc
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

global_atomic_pk_add_f16 v1, v0, v2, s[0:1] offset:-64 th:TH_ATOMIC_RETURN
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

global_load_lds_dword v2, s[4:5] offset:4
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

global_load_lds_sbyte v[2:3], off
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

global_load_lds_sshort v[2:3], off
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

global_load_lds_ubyte v[2:3], off
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

global_load_lds_ushort v[2:3], off
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

global_load_tr_b128 v[1:4], v5, s[2:3]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

global_load_tr_b64 v[1:2], v[3:4], off
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_atomic_fcmpswap v[1:2], v2, s[12:19] dmask:0x3 dim:SQ_RSRC_IMG_1D unorm
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_atomic_fmax v4, v32, s[96:103] dmask:0x1 dim:SQ_RSRC_IMG_1D glc
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_atomic_fmin v4, v32, s[96:103] dmask:0x1 dim:SQ_RSRC_IMG_1D glc
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_atomic_pk_add_f16 v0, v0, s[0:7] dmask:0x1 dim:SQ_RSRC_IMG_1D
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_atomic_pk_add_bf16 v4, [v4, v5, v6], s[8:15] dmask:0x1 dim:SQ_RSRC_IMG_3D
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_gather4_b_cl_o v[252:255], v[1:8], s[8:15], s[12:15] dmask:0x1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_gather4_b_o v[252:255], v[1:4], s[8:15], s[12:15] dmask:0x1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_gather4_c_b_cl_o v[252:255], v[1:8], s[8:15], s[12:15] dmask:0x1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_gather4_c_b_o v[252:255], v[1:8], s[8:15], s[12:15] dmask:0x1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_gather4_c_cl_o v[252:255], v[1:8], s[8:15], s[12:15] dmask:0x1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_gather4_c_l_o v[252:255], v[1:8], s[8:15], s[12:15] dmask:0x1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_gather4_c_o v[252:255], v[1:4], s[8:15], s[12:15] dmask:0x1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_gather4_cl_o v[252:255], v[1:4], s[8:15], s[12:15] dmask:0x1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_gather4_l_o v[252:255], v[1:4], s[8:15], s[12:15] dmask:0x1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_sample_c_cd v252, v[1:4], s[8:15], s[12:15] dmask:0x1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_sample_c_cd_cl v252, v[1:8], s[8:15], s[12:15] dmask:0x1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_sample_c_cd_cl_g16 v[0:3], [v0, v1, v3, v5, v6, v7], s[0:7], s[8:11] dmask:0xf dim:SQ_RSRC_IMG_2D
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_sample_c_cd_cl_o v252, v[1:8], s[8:15], s[12:15] dmask:0x1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_sample_c_cd_cl_o_g16 v[5:6], v[1:6], s[8:15], s[12:15] dmask:0x3
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_sample_c_cd_g16 v[0:3], [v0, v1, v3, v5, v6], s[0:7], s[8:11] dmask:0xf dim:SQ_RSRC_IMG_2D
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_sample_c_cd_o v252, v[1:8], s[8:15], s[12:15] dmask:0x1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_sample_c_cd_o_g16 v[5:6], v[1:5], s[8:15], s[12:15] dmask:0x3
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_sample_cd v252, v[1:3], s[8:15], s[12:15] dmask:0x1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_sample_cd_cl v252, v[1:4], s[8:15], s[12:15] dmask:0x1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_sample_cd_cl_g16 v[0:3], [v0, v2, v4, v5, v6], s[0:7], s[8:11] dmask:0xf dim:SQ_RSRC_IMG_2D
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_sample_cd_cl_o v252, v[1:8], s[8:15], s[12:15] dmask:0x1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_sample_cd_cl_o_g16 v[5:6], v[1:5], s[8:15], s[12:15] dmask:0x3
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_sample_cd_g16 v[0:3], [v0, v2, v4, v5], s[0:7], s[8:11] dmask:0xf dim:SQ_RSRC_IMG_2D
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_sample_cd_o v252, v[1:4], s[8:15], s[12:15] dmask:0x1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_sample_cd_o_g16 v[5:6], v[1:4], s[8:15], s[12:15] dmask:0x3
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_atomic_add flat_scratch_hi, s[2:3], s0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_atomic_add_x2 flat_scratch, s[2:3], s0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_atomic_and flat_scratch_hi, s[2:3], s0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_atomic_and_x2 flat_scratch, s[2:3], s0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_atomic_cmpswap flat_scratch, s[2:3], s0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_atomic_cmpswap_x2 s[20:23], flat_scratch, s0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_atomic_dec flat_scratch_hi, s[2:3], s0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_atomic_dec_x2 flat_scratch, s[2:3], s0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_atomic_inc flat_scratch_hi, s[2:3], s0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_atomic_inc_x2 flat_scratch, s[2:3], s0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_atomic_or flat_scratch_hi, s[2:3], s0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_atomic_or_x2 flat_scratch, s[2:3], s0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_atomic_smax flat_scratch_hi, s[2:3], s0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_atomic_smax_x2 flat_scratch, s[2:3], s0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_atomic_smin flat_scratch_hi, s[2:3], s0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_atomic_smin_x2 flat_scratch, s[2:3], s0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_atomic_sub flat_scratch_hi, s[2:3], s0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_atomic_sub_x2 flat_scratch, s[2:3], s0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_atomic_swap flat_scratch_hi, s[2:3], s0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_atomic_swap_x2 flat_scratch, s[2:3], s0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_atomic_umax flat_scratch_hi, s[2:3], s0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_atomic_umax_x2 flat_scratch, s[2:3], s0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_atomic_umin flat_scratch_hi, s[2:3], s0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_atomic_umin_x2 flat_scratch, s[2:3], s0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_atomic_xor flat_scratch_hi, s[2:3], s0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_atomic_xor_x2 flat_scratch, s[2:3], s0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_buffer_atomic_add flat_scratch_hi, s[4:7], s0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_buffer_atomic_add_x2 flat_scratch, s[4:7], s0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_buffer_atomic_and flat_scratch_hi, s[4:7], s0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_buffer_atomic_and_x2 flat_scratch, s[4:7], s0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_buffer_atomic_cmpswap flat_scratch, s[4:7], s0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_buffer_atomic_cmpswap_x2 s[20:23], s[4:7], 0x0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_buffer_atomic_dec flat_scratch_hi, s[4:7], s0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_buffer_atomic_dec_x2 flat_scratch, s[4:7], s0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_buffer_atomic_inc flat_scratch_hi, s[4:7], s0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_buffer_atomic_inc_x2 flat_scratch, s[4:7], s0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_buffer_atomic_or flat_scratch_hi, s[4:7], s0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_buffer_atomic_or_x2 flat_scratch, s[4:7], s0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_buffer_atomic_smax flat_scratch_hi, s[4:7], s0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_buffer_atomic_smax_x2 flat_scratch, s[4:7], s0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_buffer_atomic_smin flat_scratch_hi, s[4:7], s0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_buffer_atomic_smin_x2 flat_scratch, s[4:7], s0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_buffer_atomic_sub flat_scratch_hi, s[4:7], s0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_buffer_atomic_sub_x2 flat_scratch, s[4:7], s0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_buffer_atomic_swap flat_scratch_hi, s[4:7], s0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_buffer_atomic_swap_x2 flat_scratch, s[4:7], s0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_buffer_atomic_umax flat_scratch_hi, s[4:7], s0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_buffer_atomic_umax_x2 flat_scratch, s[4:7], s0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_buffer_atomic_umin flat_scratch_hi, s[4:7], s0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_buffer_atomic_umin_x2 flat_scratch, s[4:7], s0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_buffer_atomic_xor flat_scratch_hi, s[4:7], s0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_buffer_atomic_xor_x2 flat_scratch, s[4:7], s0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_buffer_store_dword exec_hi, s[0:3], 0x0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_buffer_store_dwordx2 exec, s[0:3], 0x0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_buffer_store_dwordx4 s[4:7], s[12:15], m0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_cbranch_g_fork -1, s[4:5]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_cbranch_i_fork exec, 12609
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_cbranch_join 1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_dcache_discard flat_scratch, s0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_dcache_discard_x2 flat_scratch, s0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_dcache_inv_vol
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_dcache_wb
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_dcache_wb_vol
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_ff0_i32_b32 exec_hi, s1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_ff0_i32_b64 exec_hi, s[2:3]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_get_waveid_in_workgroup s0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_memrealtime exec
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_memtime exec
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_rfe_restore_b64 -1, s2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_scratch_load_dword flat_scratch_hi, s[2:3], s0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_scratch_load_dwordx2 flat_scratch, s[2:3], s0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_scratch_load_dwordx4 s[20:23], flat_scratch, s0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_scratch_store_dword flat_scratch_hi, s[4:5], s0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_scratch_store_dwordx2 flat_scratch, s[4:5], s0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_scratch_store_dwordx4 s[4:7], flat_scratch, s0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_set_gpr_idx_idx -1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_set_gpr_idx_mode 0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_set_gpr_idx_off
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_set_gpr_idx_on -1, 0x0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_setvskip -1, s2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_store_dword exec_hi, s[2:3], 0x0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_store_dwordx2 exec, s[2:3], 0x0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_store_dwordx4 s[4:7], flat_scratch, m0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

scratch_load_lds_dword off, off
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

scratch_load_lds_sbyte off, off
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

scratch_load_lds_sshort off, off
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

scratch_load_lds_ubyte off, off
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

scratch_load_lds_ushort off, off
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_accvgpr_mov_b32 a1, a2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_accvgpr_read_b32 a0, a0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_accvgpr_write_b32 a0, 65
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_add_u16_dpp v255, v1, v2 quad_perm:[0,1,2,3] row_mask:0x0 bank_mask:0x0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_add_u16_e32 v1, v2, v3
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_add_u16_sdwa v0, scc, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_addc_co_u32 v0, vcc, shared_base, v0, vcc
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_addc_co_u32_dpp v255, vcc, v1, v2, vcc quad_perm:[0,1,2,3] row_mask:0x0 bank_mask:0x0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_addc_co_u32_e32 v3, vcc, 12345, v3, vcc
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_addc_co_u32_e64 v255, s[12:13], v1, v2, s[6:7]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_addc_co_u32_sdwa v1, vcc, v2, v3, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:BYTE_2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_addc_u32 v0, vcc, exec_hi, v0, vcc
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_addc_u32_dpp v255, vcc, v1, v2, vcc quad_perm:[0,1,2,3] row_mask:0x0 bank_mask:0x0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_addc_u32_e32 v1, -1, v2, v3, s0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_addc_u32_e64 v0, s[0:1], s0, s0, s[0:1]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_addc_u32_sdwa v1, vcc, v2, v3, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:BYTE_2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_ashr_i32 v255, v1, v2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_ashr_i32_e32 v1, v2, v3
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_ashr_i32_e64 v255, v1, v2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_ashr_i64 v[254:255], v[1:2], v2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_clrexcp
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_clrexcp_e32
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_clrexcp_e64
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cmp_f_i16 vcc, -1, v2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cmp_f_i16_e64 flat_scratch, v1, v2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cmp_f_i16_sdwa flat_scratch, v1, v2 src0_sel:DWORD src1_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cmp_f_u16 vcc, -1, v2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cmp_f_u16_e64 flat_scratch, v1, v2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cmp_f_u16_sdwa flat_scratch, v1, v2 src0_sel:DWORD src1_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cmp_t_i16 vcc, -1, v2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cmp_t_i16_e64 flat_scratch, v1, v2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cmp_t_i16_sdwa flat_scratch, v1, v2 src0_sel:DWORD src1_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cmp_t_u16 vcc, -1, v2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cmp_t_u16_e64 flat_scratch, v1, v2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cmp_t_u16_sdwa flat_scratch, v1, v2 src0_sel:DWORD src1_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cmps_eq_f32 vcc, -1, v2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cmps_eq_f32_e64 flat_scratch, v1, v2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cmps_eq_f64 vcc, -1, v[2:3]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cmps_eq_f64_e64 flat_scratch, v[1:2], v[2:3]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cmps_f_f32 vcc, -1, v2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cmps_f_f32_e64 flat_scratch, v1, v2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cmps_f_f64 vcc, -1, v[2:3]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cmps_f_f64_e64 flat_scratch, v[1:2], v[2:3]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cmps_ge_f32 vcc, -1, v2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cmps_ge_f32_e64 flat_scratch, v1, v2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cmps_ge_f64 vcc, -1, v[2:3]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cmps_ge_f64_e64 flat_scratch, v[1:2], v[2:3]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cmps_gt_f32 vcc, -1, v2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cmps_gt_f32_e64 flat_scratch, v1, v2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cmps_gt_f64 vcc, -1, v[2:3]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cmps_gt_f64_e64 flat_scratch, v[1:2], v[2:3]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cmps_le_f32 vcc, -1, v2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cmps_le_f32_e64 flat_scratch, v1, v2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cmps_le_f64 vcc, -1, v[2:3]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cmps_le_f64_e64 flat_scratch, v[1:2], v[2:3]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cmps_lg_f32 vcc, -1, v2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cmps_lg_f32_e64 flat_scratch, v1, v2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cmps_lg_f64 vcc, -1, v[2:3]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cmps_lg_f64_e64 flat_scratch, v[1:2], v[2:3]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cmps_lt_f32 vcc, -1, v2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cmps_lt_f32_e64 flat_scratch, v1, v2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cmps_lt_f64 vcc, -1, v[2:3]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cmps_lt_f64_e64 flat_scratch, v[1:2], v[2:3]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cmps_neq_f32 vcc, -1, v2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cmps_neq_f32_e64 flat_scratch, v1, v2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cmps_neq_f64 vcc, -1, v[2:3]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cmps_neq_f64_e64 flat_scratch, v[1:2], v[2:3]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cmps_nge_f32 vcc, -1, v2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cmps_nge_f32_e64 flat_scratch, v1, v2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cmps_nge_f64 vcc, -1, v[2:3]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cmps_nge_f64_e64 flat_scratch, v[1:2], v[2:3]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cmps_ngt_f32 vcc, -1, v2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cmps_ngt_f32_e64 flat_scratch, v1, v2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cmps_ngt_f64 vcc, -1, v[2:3]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cmps_ngt_f64_e64 flat_scratch, v[1:2], v[2:3]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cmps_nle_f32 vcc, -1, v2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cmps_nle_f32_e64 flat_scratch, v1, v2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cmps_nle_f64 vcc, -1, v[2:3]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cmps_nle_f64_e64 flat_scratch, v[1:2], v[2:3]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cmps_nlg_f32 vcc, -1, v2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cmps_nlg_f32_e64 flat_scratch, v1, v2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cmps_nlg_f64 vcc, -1, v[2:3]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cmps_nlg_f64_e64 flat_scratch, v[1:2], v[2:3]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cmps_nlt_f32 vcc, -1, v2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cmps_nlt_f32_e64 flat_scratch, v1, v2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cmps_nlt_f64 vcc, -1, v[2:3]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cmps_nlt_f64_e64 flat_scratch, v[1:2], v[2:3]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cmps_o_f32 vcc, -1, v2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cmps_o_f32_e64 flat_scratch, v1, v2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cmps_o_f64 vcc, -1, v[2:3]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cmps_o_f64_e64 flat_scratch, v[1:2], v[2:3]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cmps_tru_f32 vcc, -1, v2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cmps_tru_f32_e64 flat_scratch, v1, v2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cmps_tru_f64 vcc, -1, v[2:3]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cmps_tru_f64_e64 flat_scratch, v[1:2], v[2:3]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cmps_u_f32 vcc, -1, v2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cmps_u_f32_e64 flat_scratch, v1, v2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cmps_u_f64 vcc, -1, v[2:3]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cmps_u_f64_e64 flat_scratch, v[1:2], v[2:3]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cmpsx_eq_f32 vcc, -1, v2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cmpsx_eq_f32_e64 flat_scratch, v1, v2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cmpsx_eq_f64 vcc, -1, v[2:3]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cmpsx_eq_f64_e64 flat_scratch, v[1:2], v[2:3]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cmpsx_f_f32 vcc, -1, v2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cmpsx_f_f32_e64 flat_scratch, v1, v2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cmpsx_f_f64 vcc, -1, v[2:3]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cmpsx_f_f64_e64 flat_scratch, v[1:2], v[2:3]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cmpsx_ge_f32 vcc, -1, v2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cmpsx_ge_f32_e64 flat_scratch, v1, v2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cmpsx_ge_f64 vcc, -1, v[2:3]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cmpsx_ge_f64_e64 flat_scratch, v[1:2], v[2:3]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cmpsx_gt_f32 vcc, -1, v2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cmpsx_gt_f32_e64 flat_scratch, v1, v2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cmpsx_gt_f64 vcc, -1, v[2:3]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cmpsx_gt_f64_e64 flat_scratch, v[1:2], v[2:3]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cmpsx_le_f32 vcc, -1, v2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cmpsx_le_f32_e64 flat_scratch, v1, v2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cmpsx_le_f64 vcc, -1, v[2:3]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cmpsx_le_f64_e64 flat_scratch, v[1:2], v[2:3]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cmpsx_lg_f32 vcc, -1, v2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cmpsx_lg_f32_e64 flat_scratch, v1, v2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cmpsx_lg_f64 vcc, -1, v[2:3]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cmpsx_lg_f64_e64 flat_scratch, v[1:2], v[2:3]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cmpsx_lt_f32 vcc, -1, v2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cmpsx_lt_f32_e64 flat_scratch, v1, v2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cmpsx_lt_f64 vcc, -1, v[2:3]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cmpsx_lt_f64_e64 flat_scratch, v[1:2], v[2:3]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cmpsx_neq_f32 vcc, -1, v2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cmpsx_neq_f32_e64 flat_scratch, v1, v2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cmpsx_neq_f64 vcc, -1, v[2:3]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cmpsx_neq_f64_e64 flat_scratch, v[1:2], v[2:3]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cmpsx_nge_f32 vcc, -1, v2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cmpsx_nge_f32_e64 flat_scratch, v1, v2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cmpsx_nge_f64 vcc, -1, v[2:3]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cmpsx_nge_f64_e64 flat_scratch, v[1:2], v[2:3]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cmpsx_ngt_f32 vcc, -1, v2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cmpsx_ngt_f32_e64 flat_scratch, v1, v2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cmpsx_ngt_f64 vcc, -1, v[2:3]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cmpsx_ngt_f64_e64 flat_scratch, v[1:2], v[2:3]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cmpsx_nle_f32 vcc, -1, v2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cmpsx_nle_f32_e64 flat_scratch, v1, v2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cmpsx_nle_f64 vcc, -1, v[2:3]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cmpsx_nle_f64_e64 flat_scratch, v[1:2], v[2:3]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cmpsx_nlg_f32 vcc, -1, v2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cmpsx_nlg_f32_e64 flat_scratch, v1, v2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cmpsx_nlg_f64 vcc, -1, v[2:3]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cmpsx_nlg_f64_e64 flat_scratch, v[1:2], v[2:3]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cmpsx_nlt_f32 vcc, -1, v2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cmpsx_nlt_f32_e64 flat_scratch, v1, v2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cmpsx_nlt_f64 vcc, -1, v[2:3]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cmpsx_nlt_f64_e64 flat_scratch, v[1:2], v[2:3]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cmpsx_o_f32 vcc, -1, v2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cmpsx_o_f32_e64 flat_scratch, v1, v2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cmpsx_o_f64 vcc, -1, v[2:3]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cmpsx_o_f64_e64 flat_scratch, v[1:2], v[2:3]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cmpsx_tru_f32 vcc, -1, v2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cmpsx_tru_f32_e64 flat_scratch, v1, v2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cmpsx_tru_f64 vcc, -1, v[2:3]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cmpsx_tru_f64_e64 flat_scratch, v[1:2], v[2:3]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cmpsx_u_f32 vcc, -1, v2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cmpsx_u_f32_e64 flat_scratch, v1, v2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cmpsx_u_f64 vcc, -1, v[2:3]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cmpsx_u_f64_e64 flat_scratch, v[1:2], v[2:3]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cmpx_f_i16 vcc, -1, v2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cmpx_f_i16_e64 exec, v1, v2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cmpx_f_i16_sdwa flat_scratch, v1, v2 src0_sel:DWORD src1_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cmpx_f_u16 vcc, -1, v2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cmpx_f_u16_e64 exec, v1, v2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cmpx_f_u16_sdwa flat_scratch, v1, v2 src0_sel:DWORD src1_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cmpx_t_i16 vcc, -1, v2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cmpx_t_i16_e64 exec, v1, v2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cmpx_t_i16_sdwa flat_scratch, v1, v2 src0_sel:DWORD src1_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cmpx_t_u16 vcc, -1, v2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cmpx_t_u16_e64 exec, v1, v2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cmpx_t_u16_sdwa flat_scratch, v1, v2 src0_sel:DWORD src1_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_f32_bf8 v1, 3
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_f32_bf8_dpp v5, v1 quad_perm:[0,1,2,3] row_mask:0xf bank_mask:0xf
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_f32_bf8_e64 v5, v1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_f32_bf8_sdwa v5, v1 src0_sel:BYTE_0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_f32_fp8 v1, 3
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_f32_fp8_dpp v5, v1 quad_perm:[0,1,2,3] row_mask:0xf bank_mask:0xf
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_f32_fp8_e64 v5, v1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_f32_fp8_sdwa v5, v1 src0_sel:BYTE_0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_pk_bf8_f32 v1, -v2, |v3|
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_pk_f32_bf8 v[0:1], v3 quad_perm:[0,2,1,1] row_mask:0xf bank_mask:0xf
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_pk_f32_bf8_dpp v[10:11], v1 quad_perm:[0,1,2,3] row_mask:0xf bank_mask:0xf
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_pk_f32_bf8_sdwa v[10:11], v1 src0_sel:WORD_0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_pk_f32_fp8 v[0:1], v3 quad_perm:[0,2,1,1] row_mask:0xf bank_mask:0xf
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_pk_f32_fp8_dpp v[10:11], v1 quad_perm:[0,1,2,3] row_mask:0xf bank_mask:0xf
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_pk_f32_fp8_sdwa v[10:11], v1 src0_sel:WORD_0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_pk_fp8_f32 v1, -v2, |v3|
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_pkaccum_u8_f32 v1, v2, v3
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_pkaccum_u8_f32_e64 v255, v1, v2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_sr_bf8_f32 v1, -|s2|, v3
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_sr_fp8_f32 v1, -|s2|, v3
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_div_fixup_legacy_f16 v5, v1, v2, v3
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dot2_i32_i16 v0, -v1, -v2, -v3
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dot2_u32_u16 v0, -v1, -v2, -v3
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dot2c_i32_i16 v0, v1, v2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dot2c_i32_i16_dpp v255, v1, v2 quad_perm:[0,1,2,3] row_mask:0x0 bank_mask:0x0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dot2c_i32_i16_e64 v0, v1, v2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dot4c_i32_i8 v0, v1, v2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dot4c_i32_i8_dpp v255, v1, v2  quad_perm:[0,1,2,3] row_mask:0x0 bank_mask:0x0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dot4c_i32_i8_e32 v255, v1, v2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dot4c_i32_i8_e64 v0, v1, v2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dot8c_i32_i4 v0, v1, v2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dot8c_i32_i4_dpp v255, v1, v2 quad_perm:[0,1,2,3] row_mask:0x0 bank_mask:0x0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dot8c_i32_i4_e64 v0, v1, v2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_exp_legacy_f32 v1, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_exp_legacy_f32_dpp v255, v1 quad_perm:[0,1,2,3] row_mask:0x0 bank_mask:0x0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_exp_legacy_f32_e64 v255, v1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_exp_legacy_f32_sdwa v255, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_fma_legacy_f16 v5, v1, v2, v3
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_fmac_f64 v[0:1], v[2:3], v[4:5] row_newbcast:2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_fmac_f64_dpp v[10:11], v[2:3], v[4:5] row_newbcast:1 row_mask:0xf bank_mask:0xf
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_fmac_f64_e32 v[254:255], v[2:3], v[4:5]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_fmac_f64_e64 v[10:11], v[2:3], v[4:5]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_interp_mov_f32 v0, p10, attr0.x
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_interp_mov_f32_e64 v255, p10, attr0.x
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_interp_p1_f32 v0, v1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_interp_p1_f32_e64 v255, v2, attr0.x
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_interp_p1ll_f16 v5, p0, attr31.x
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_interp_p1lv_f16 v5, v1, attr0.x, v3
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_interp_p2_f16 v5, v1, attr0.x, v3
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_interp_p2_legacy_f16 v5, v1, attr0.x, v3
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_log_clamp_f32 v1, 0.5
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_log_clamp_f32_e64 v255, v1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_log_legacy_f32 v1, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_log_legacy_f32_dpp v255, v1 quad_perm:[0,1,2,3] row_mask:0x0 bank_mask:0x0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_log_legacy_f32_e64 v255, v1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_log_legacy_f32_sdwa v255, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_lshl_add_u64 v[10:11], v[2:3], v2, v[6:7]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_lshl_b32 v255, v1, v2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_lshl_b32_e32 v1, v2, v3
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_lshl_b32_e64 v255, v1, v2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_lshl_b64 v[254:255], v[1:2], v2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_lshr_b32 v255, v1, v2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_lshr_b32_e32 v1, v2, v3
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_lshr_b32_e64 v255, v1, v2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_lshr_b64 v[254:255], v[1:2], v2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mac_f16 v1, v2, v3 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:BYTE_2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mac_f16_dpp v255, v1, v2 quad_perm:[0,1,2,3] row_mask:0x0 bank_mask:0x0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mac_f16_e32 v1, v2, v3
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mac_f16_e64 v255, v1, v2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mac_f16_sdwa v255, v1, v2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mac_f32 v0, v0, v0 quad_perm:[1,3,0,1] row_mask:0xa bound_ctrl:0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mac_f32_dpp v255, v1, v2 quad_perm:[0,1,2,3] row_mask:0x0 bank_mask:0x0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mac_f32_e32 v1, v2, v3
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mac_f32_e64 v255, v1, v2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mac_f32_sdwa v255, v1, v2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mac_legacy_f32 v0, v1, v2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mac_legacy_f32_e32 v255, v1, v2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mac_legacy_f32_e64 v255, v1, v2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mad_f16 v5, v1, v2, v3
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mad_f32 v0, s0, s0, flat_scratch_lo
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mad_legacy_f16 v5, v1, v2, v3
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mad_legacy_f32 v0, v1, v2, v3
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mad_legacy_i16 v5, v1, v2, v3
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mad_legacy_u16 v5, v1, v2, v3
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mad_mix_f32 v0, -abs(v1), v2, v3
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mad_mixhi_f16 v0, -v1, abs(v2), -abs(v3)
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mad_mixlo_f16 v0, abs(v1), -v2, abs(v3)
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_madak_f16 v0, 0xff32, v0, 0x1122
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_madak_f32 v0, 0x11213141, v0, 0x11213141
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_madmk_f16 v0, 0xff32, 0x1122, v0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_madmk_f32 v0, 0x11213141, 0x11213141, v0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_max_legacy_f32 v1, v2, v3
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_max_legacy_f32_e64 v255, v1, v2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_16x16x16_bf16 a[0:3], v[2:3], v[4:5], a[2:5]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_16x16x16_f16 a[0:3], v[0:1], v[2:3], a[2:5]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_16x16x16bf16 a[0:3], v[2:3], v[4:5], a[2:5]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_16x16x16bf16_1k a[0:3], a[0:1], a[2:3], -2.0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_16x16x16f16 a[0:3], a[0:1], a[1:2], -2.0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_16x16x1_4b_f32 a[0:15], v0, v1, a[18:33]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_16x16x1f32 a[0:15], a0, a1, -2.0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_16x16x2bf16 a[0:15], a0, a1, -2.0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_16x16x32_bf8_bf8 a[0:3], v[2:3], v[4:5], a[0:3]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_16x16x32_bf8_fp8 a[0:3], v[2:3], v[4:5], a[0:3]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_16x16x32_fp8_bf8 a[0:3], v[2:3], v[4:5], a[0:3]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_16x16x32_fp8_fp8 a[0:3], v[2:3], v[4:5], a[0:3]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_16x16x4_4b_bf16 a[0:15], v[2:3], v[4:5], a[18:33]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_16x16x4_4b_f16 a[0:15], v[0:1], v[2:3], a[18:33]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_16x16x4_f32 a[0:3], v0, v1, a[2:5]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_16x16x4bf16 a[0:15], v[2:3], v[4:5], a[18:33] blgp:5
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_16x16x4bf16_1k a[0:15], a[0:1], a[2:3], -2.0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_16x16x4f16 a[0:15], a[0:1], a[1:2], -2.0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_16x16x4f32 a[0:3], a0, a1, -2.0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_16x16x8_xf32 a[0:3], v[2:3], v[4:5], a[2:5]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_16x16x8bf16 a[0:3], a0, a1, -2.0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_16x16x8xf32 a[0:3], v[2:3], v[4:5], a[2:5]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_32x32x16_bf8_bf8 a[0:15], v[2:3], v[4:5], a[0:15]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_32x32x16_bf8_fp8 a[0:15], v[2:3], v[4:5], a[0:15]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_32x32x16_fp8_bf8 a[0:15], v[2:3], v[4:5], a[0:15]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_32x32x16_fp8_fp8 a[0:15], v[2:3], v[4:5], a[0:15]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_32x32x1_2b_f32 a[0:31], v0, v1, a[0:31] neg:[1,0,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_32x32x1f32 a[0:31], 1, v1, a[0:31]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_32x32x2_f32 a[0:15], v0, v1, a[18:33]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_32x32x2bf16 a[0:31], a0, a1, -2.0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_32x32x2f32 a[0:15], a0, a1, -2.0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_32x32x4_2b_bf16 a[0:31], v[2:3], v[4:5], a[34:65]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_32x32x4_2b_f16 a[0:31], v[0:1], v[2:3], a[34:65]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_32x32x4_xf32 a[0:15], v[2:3], v[4:5], a[18:33]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_32x32x4bf16 a[0:15], a0, a1, -2.0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_32x32x4bf16_1k a[0:31], a[0:1], a[2:3], -2.0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_32x32x4f16 a[0:31], a[0:1], a[1:2], -2.0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_32x32x4xf32 a[0:15], v[2:3], v[4:5], a[18:33]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_32x32x8_bf16 a[0:15], v[2:3], v[4:5], a[18:33]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_32x32x8_f16 a[0:15], v[0:1], v[2:3], a[18:33]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_32x32x8bf16 a[0:15], v[2:3], v[4:5], a[18:33]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_32x32x8bf16_1k a[0:15], a[0:1], a[2:3], -2.0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_32x32x8f16 a[0:15], a[0:1], a[1:2], -2.0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_4x4x1_16b_f32 a[0:3], v0, v1, a[2:5]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_4x4x1f32 a[0:3], a0, a1, -2.0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_4x4x2bf16 a[0:3], a0, a1, -2.0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_4x4x4_16b_bf16 a[0:3], v[2:3], v[4:5], a[2:5]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_4x4x4_16b_f16 a[0:3], v[0:1], v[2:3], a[2:5]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_4x4x4bf16 a[0:3], v[2:3], v[4:5], a[2:5]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_4x4x4bf16_1k a[0:3], a[0:1], a[2:3], -2.0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_4x4x4f16 a[0:3], a[0:1], a[1:2], -2.0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f64_16x16x4_f64 a[0:7], v[0:1], v[2:3], a[0:7]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f64_16x16x4f64 a[0:7], a[0:1], a[2:3], -2.0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f64_4x4x4_4b_f64 a[0:1], v[0:1], a[2:3], a[2:3]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f64_4x4x4f64 a[0:1], a[0:1], a[2:3], -2.0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_i32_16x16x16i8 a[0:3], a0, a1, 2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_i32_16x16x32_i8 a[0:3], v[2:3], v[4:5], a[0:3]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_i32_16x16x32i8 a[0:3], v[2:3], v[4:5], a[0:3] blgp:5
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_i32_16x16x4_4b_i8 a[0:15], v0, v1, a[18:33]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_i32_16x16x4i8 a[0:15], a0, a1, 2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_i32_32x32x16_i8 a[0:15], v[2:3], v[4:5], a[0:15]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_i32_32x32x16i8 a[0:15], v[2:3], v[4:5], a[0:15] blgp:5
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_i32_32x32x4_2b_i8 a[0:31], v0, v1, a[34:65]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_i32_32x32x4i8 a[0:31], a0, a1, 2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_i32_32x32x8i8 a[0:15], a0, a1, 2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_i32_4x4x4_16b_i8 a[0:3], v0, v1, a[2:5]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_i32_4x4x4i8 a[0:3], a0, a1, 2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_min_legacy_f32 v255, v1, v2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_min_legacy_f32_e32 v1, v2, v3
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_min_legacy_f32_e64 v255, v1, v2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mov_b64 v[10:11], v[2:3]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mov_b64_dpp v[10:11], v[2:3] row_newbcast:1 row_mask:0xf bank_mask:0xf
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mov_b64_e64 v[10:11], v[2:3]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mul_lo_i32 v0, v1, v2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_permlane16_var_b32 v0, v0, v1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_permlanex16_var_b32 v0, v0, v1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_pk_add_f32 v[10:11], v[2:3], v[4:5]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_pk_fma_f32 v[0:1], v[4:5], v[8:9], v[16:17]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_pk_mov_b32 v[0:1], flat_scratch, v[4:5]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_pk_mul_f32 v[10:11], v[2:3], v[4:5]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_rcp_clamp_f32 v255, v1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_rcp_clamp_f32_e64 v255, v1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_rcp_clamp_f64 v[254:255], v[1:2]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_rcp_clamp_f64_e64 v[254:255], v[1:2]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_rcp_legacy_f32 v255, v1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_rcp_legacy_f32_e64 v255, v1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_rsq_clamp_f32 v255, v1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_rsq_clamp_f32_e64 v255, v1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_rsq_clamp_f64 v[254:255], v[1:2]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_rsq_clamp_f64_e64 v[254:255], v[1:2]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_rsq_legacy_f32 v255, v1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_rsq_legacy_f32_e64 v255, v1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_screen_partition_4se_b32 v255, v1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_screen_partition_4se_b32_dpp v255, v1 quad_perm:[0,1,2,3] row_mask:0x0 bank_mask:0x0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_screen_partition_4se_b32_e64 v255, v1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_screen_partition_4se_b32_sdwa v255, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_smfmac_f32_16x16x32_bf16 a[10:13], v[2:3], a[4:7], v5
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_smfmac_f32_16x16x32_f16 a[10:13], v[2:3], a[4:7], v1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_smfmac_f32_16x16x32bf16 v[10:13], a[2:3], v[4:7], v4 cbsz:3 abid:1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_smfmac_f32_16x16x32f16 v[10:13], a[2:3], v[4:7], v0 cbsz:3 abid:1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_smfmac_f32_16x16x64_bf8_bf8 a[0:3], v[2:3], a[4:7], v1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_smfmac_f32_16x16x64_bf8_fp8 a[0:3], v[2:3], a[4:7], v1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_smfmac_f32_16x16x64_fp8_bf8 a[0:3], v[2:3], a[4:7], v1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_smfmac_f32_16x16x64_fp8_fp8 a[0:3], v[2:3], a[4:7], v1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_smfmac_f32_16x16x64bf8bf8 v[0:3], a[2:3], v[4:7], v1 cbsz:3 abid:1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_smfmac_f32_16x16x64bf8fp8 v[0:3], a[2:3], v[4:7], v1 cbsz:3 abid:1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_smfmac_f32_16x16x64fp8bf8 v[0:3], a[2:3], v[4:7], v1 cbsz:3 abid:1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_smfmac_f32_16x16x64fp8fp8 v[0:3], a[2:3], v[4:7], v1 cbsz:3 abid:1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_smfmac_f32_32x32x16_bf16 a[10:25], v[2:3], a[4:7], v7
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_smfmac_f32_32x32x16_f16 a[10:25], v[2:3], a[4:7], v3
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_smfmac_f32_32x32x16bf16 v[10:25], a[2:3], v[4:7], v6 cbsz:3 abid:1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_smfmac_f32_32x32x16f16 v[10:25], a[2:3], v[4:7], v2 cbsz:3 abid:1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_smfmac_f32_32x32x32_bf8_bf8 a[0:15], v[2:3], a[4:7], v1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_smfmac_f32_32x32x32_bf8_fp8 a[0:15], v[2:3], a[4:7], v1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_smfmac_f32_32x32x32_fp8_bf8 a[0:15], v[2:3], a[4:7], v1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_smfmac_f32_32x32x32_fp8_fp8 a[0:15], v[2:3], a[4:7], v1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_smfmac_f32_32x32x32bf8bf8 v[0:15], a[2:3], v[4:7], v1 cbsz:3 abid:1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_smfmac_f32_32x32x32bf8fp8 v[0:15], a[2:3], v[4:7], v1 cbsz:3 abid:1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_smfmac_f32_32x32x32fp8bf8 v[0:15], a[2:3], v[4:7], v1 cbsz:3 abid:1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_smfmac_f32_32x32x32fp8fp8 v[0:15], a[2:3], v[4:7], v1 cbsz:3 abid:1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_smfmac_i32_16x16x64_i8 a[10:13], v[2:3], a[4:7], v9
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_smfmac_i32_16x16x64i8 v[10:13], a[2:3], v[4:7], v8 cbsz:3 abid:1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_smfmac_i32_32x32x32_i8 a[10:25], v[2:3], a[4:7], v11
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_smfmac_i32_32x32x32i8 a[10:25], v[2:3], a[4:7], v11
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_sub_u16_dpp v255, v1, v2 quad_perm:[0,1,2,3] row_mask:0x0 bank_mask:0x0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_sub_u16_e32 v1, v2, v3
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_sub_u16_sdwa v255, v1, v2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_subb_co_u32 v1, vcc, v2, v3, vcc row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_subb_co_u32_dpp v255, vcc, v1, v2, vcc quad_perm:[0,1,2,3] row_mask:0x0 bank_mask:0x0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_subb_co_u32_e64 v255, s[12:13], v1, v2, s[6:7]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_subb_co_u32_sdwa v1, vcc, v2, v3, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:BYTE_2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_subb_u32 v1, s[0:1], v2, v3, vcc
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_subb_u32_dpp v255, vcc, v1, v2, vcc quad_perm:[0,1,2,3] row_mask:0x0 bank_mask:0x0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_subb_u32_e64 v255, s[12:13], v1, v2, s[6:7]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_subb_u32_sdwa v1, vcc, v2, v3, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:BYTE_2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_subbrev_co_u32 v0, vcc, src_lds_direct, v0, vcc
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_subbrev_co_u32_dpp v255, vcc, v1, v2, vcc quad_perm:[0,1,2,3] row_mask:0x0 bank_mask:0x0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_subbrev_co_u32_e64 v255, s[12:13], v1, v2, s[6:7]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_subbrev_co_u32_sdwa v1, vcc, v2, v3, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:BYTE_2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_subbrev_u32 v1, s[0:1], v2, v3, vcc
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_subbrev_u32_dpp v255, vcc, v1, v2, vcc quad_perm:[0,1,2,3] row_mask:0x0 bank_mask:0x0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_subbrev_u32_e64 v255, s[12:13], v1, v2, s[6:7]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_subbrev_u32_sdwa v1, vcc, v2, v3, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:BYTE_2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_subrev_i32 v1, s[0:1], v2, v3
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_subrev_i32_e64 v255, s[12:13], v1, v2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_subrev_u16 v0, src_lds_direct, v0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_subrev_u16_dpp v255, v1, v2 quad_perm:[0,1,2,3] row_mask:0x0 bank_mask:0x0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_subrev_u16_e32 v1, v2, v3
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_subrev_u16_e64 v255, v1, v2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_subrev_u16_sdwa v255, v1, v2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_cvt_f32_i32 s5, s1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_cvt_f32_u32 s5, s1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_cvt_u32_f32 s5, s1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_cvt_i32_f32 s5, s1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_cvt_f16_f32 s5, s1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_cvt_f32_f16 s5, s1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_cvt_hi_f32_f16 s5, s1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_trunc_f32 s5, s1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_ceil_f32 s5, s1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_rndne_f32 s5, s1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_floor_f32 s5, s1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_floor_f16 s5, s1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_ceil_f16 s5, s1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_trunc_f16 s5, s1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_rndne_f16 s5, s1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_add_f32 s5, s1, s2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_sub_f32 s5, s1, s2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_mul_f32 s5, s1, s2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_min_f32 s5, s1, s2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_max_f32 s5, s1, s2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_fmac_f32 s5, s1, s2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_fmamk_f32 s5, s1, 0x11213141, s3
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_fmaak_f32 s5, s1, s2, 0x11213141
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_cvt_pk_rtz_f16_f32 s5, s1, s2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_add_f16 s5, s1, s2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_sub_f16 s5, s1, s2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_mul_f16 s5, s1, s2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_fmac_f16 s5, s1, s2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_max_f16 s5, s1, s2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_min_f16 s5, s1, s2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_cmp_lt_f32 s1, s2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_cmp_eq_f32 s1, s2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_cmp_le_f32 s1, s2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_cmp_gt_f32 s1, s2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_cmp_lg_f32 s1, s2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_cmp_ge_f32 s1, s2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_cmp_o_f32 s1, s2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_cmp_u_f32 s1, s2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_cmp_nge_f32 s1, s2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_cmp_nlg_f32 s1, s2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_cmp_ngt_f32 s1, s2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_cmp_nle_f32 s1, s2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_cmp_neq_f32 s1, s2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_cmp_nlt_f32 s1, s2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_cmp_lt_f16 s1, s2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_cmp_eq_f16 s1, s2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_cmp_le_f16 s1, s2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_cmp_gt_f16 s1, s2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_cmp_lg_f16 s1, s2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_cmp_ge_f16 s1, s2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_cmp_o_f16 s1, s2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_cmp_u_f16 s1, s2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_cmp_nge_f16 s1, s2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_cmp_nlg_f16 s1, s2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_cmp_ngt_f16 s1, s2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_cmp_nle_f16 s1, s2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_cmp_neq_f16 s1, s2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_cmp_nlt_f16 s1, s2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_singleuse_vdst 0x1234
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_atomic_sub_clamp_u32 v5, off, s[8:11], s3 offset:0 glc
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_atomic_max_num_f32 v5, off, s[8:11], s3 offset:4095
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_atomic_min_num_f32 v5, off, s[8:11], s3 offset:4095
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

global_atomic_sub_clamp_u32 v5, v[1:2], v2, off glc
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

flat_atomic_csub_u32 v1, v[0:1], v2 offset:64 th:TH_ATOMIC_RETURN
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

flat_atomic_sub_clamp_u32 v1, v[0:1], v2 offset:64 th:TH_ATOMIC_RETURN
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_cond_sub_rtn_u32 v5, v1, v2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_cond_sub_u32 v1, v2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_sub_clamp_rtn_u32 v5, v1, v2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_sub_clamp_u32 v1, v2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

flat_atomic_cond_sub_u32 v[0:1], v2 offset:64
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

global_atomic_cond_sub_u32 v0, v2, s[0:1] offset:64
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

global_atomic_ordered_add_b64 v0, v[2:3], s[0:1] offset:64
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_subrev_u32 v1, v2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_subrev_rtn_u32 v5, v1, v2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_subrev_u64 v1, v[2:3]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_subrev_rtn_u64 v[5:6], v1, v[2:3]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU
