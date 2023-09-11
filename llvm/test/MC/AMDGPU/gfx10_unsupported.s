// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx1010 -mattr=+wavefrontsize32,-wavefrontsize64 %s 2>&1 | FileCheck --check-prefixes=CHECK,GFX1010 --implicit-check-not=error: %s
// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx1010 -mattr=-wavefrontsize32,+wavefrontsize64 %s 2>&1 | FileCheck --check-prefixes=CHECK,GFX1010 --implicit-check-not=error: %s
// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx1013 -mattr=+wavefrontsize32,-wavefrontsize64 %s 2>&1 | FileCheck --implicit-check-not=error: %s

buffer_atomic_add_f32 v0, v2, s[4:7], 0 idxen glc
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_atomic_add_f64 v[2:3], off, s[12:15], s4 offset:4095
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_atomic_add_u32 v1, off, s[12:15], s4 offset:4095
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_atomic_add_u64 v[1:2], off, s[12:15], s4 offset:4095
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_atomic_and_b32 v1, off, s[12:15], s4 offset:4095
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_atomic_and_b64 v[1:2], off, s[12:15], s4 offset:4095
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_atomic_cmpswap_b32 v[1:2], off, s[12:15], s4 offset:4095
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_atomic_cmpswap_b64 v[1:4], off, s[12:15], s4 offset:4095
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_atomic_cmpswap_f32 v[1:2], off, s[12:15], s4 offset:4095
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_atomic_csub v1, off, s[12:15], s4 offset:4095 glc
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_atomic_csub_u32 v1, off, s[12:15], s4 offset:4095 glc
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_atomic_dec_u32 v1, off, s[12:15], s4 offset:4095
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_atomic_dec_u64 v[1:2], off, s[12:15], s4 offset:4095
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_atomic_inc_u32 v1, off, s[12:15], s4 offset:4095
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_atomic_inc_u64 v[1:2], off, s[12:15], s4 offset:4095
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_atomic_max_f32 v1, off, s[12:15], s4 offset:4095
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_atomic_max_f64 v[2:3], off, s[12:15], s4 offset:4095
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_atomic_max_i32 v1, off, s[12:15], s4 offset:4095
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_atomic_max_i64 v[1:2], off, s[12:15], s4 offset:4095
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_atomic_max_u32 v1, off, s[12:15], s4 offset:4095
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_atomic_max_u64 v[1:2], off, s[12:15], s4 offset:4095
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_atomic_min_f32 v1, off, s[12:15], s4 offset:4095
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_atomic_min_f64 v[2:3], off, s[12:15], s4 offset:4095
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_atomic_min_i32 v1, off, s[12:15], s4 offset:4095
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_atomic_min_i64 v[1:2], off, s[12:15], s4 offset:4095
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_atomic_min_u32 v1, off, s[12:15], s4 offset:4095
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_atomic_min_u64 v[1:2], off, s[12:15], s4 offset:4095
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_atomic_or_b32 v1, off, s[12:15], s4 offset:4095
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_atomic_or_b64 v[1:2], off, s[12:15], s4 offset:4095
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_atomic_pk_add_f16 v0, v2, s[4:7], 0 idxen glc
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_atomic_sub_u32 v1, off, s[12:15], s4 offset:4095
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_atomic_sub_u64 v[1:2], off, s[12:15], s4 offset:4095
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_atomic_swap_b32 v1, off, s[12:15], s4 offset:4095
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_atomic_swap_b64 v[1:2], off, s[12:15], s4 offset:4095
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_atomic_xor_b32 v1, off, s[12:15], s4 offset:4095
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_atomic_xor_b64 v[1:2], off, s[12:15], s4 offset:4095
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_inv
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_invl2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_b128 v[252:255], off, s[8:11], s3 offset:4095
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_b32 v255, off, s[8:11], s3 offset:4095
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_b64 v[254:255], off, s[8:11], s3 offset:4095
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_b96 v[253:255], off, s[8:11], s3 offset:4095
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_d16_b16 v255, off, s[8:11], s3 offset:4095
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_d16_format_x v255, off, s[8:11], s3 offset:4095
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_d16_format_xy v255, off, s[8:11], s3 offset:4095
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_d16_format_xyz v[254:255], off, s[8:11], s3 offset:4095
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_d16_format_xyzw v[254:255], off, s[8:11], s3 offset:4095
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_d16_hi_b16 v255, off, s[8:11], s3 offset:4095
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_d16_hi_format_x v255, off, s[8:11], s3 offset:4095
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_d16_hi_i8 v255, off, s[8:11], s3 offset:4095
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_d16_hi_u8 v255, off, s[8:11], s3 offset:4095
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_d16_i8 v255, off, s[8:11], s3 offset:4095
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_d16_u8 v255, off, s[8:11], s3 offset:4095
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_i16 v255, off, s[8:11], s3 offset:4095
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_i8 v255, off, s[8:11], s3 offset:4095
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_u16 v255, off, s[8:11], s3 offset:4095
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_u8 v255, off, s[8:11], s3 offset:4095
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_b128 v[1:4], off, s[12:15], -1 offset:4095
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_b16 v1, off, s[12:15], -1 offset:4095
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_b32 v1, off, s[12:15], -1 offset:4095
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_b64 v[1:2], off, s[12:15], -1 offset:4095
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_b8 v1, off, s[12:15], -1 offset:4095
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_b96 v[1:3], off, s[12:15], -1 offset:4095
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_d16_format_x v1, off, s[12:15], -1 offset:4095
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_d16_format_xy v1, off, s[12:15], -1 offset:4095
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_d16_format_xyz v[1:2], off, s[12:15], -1 offset:4095
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_d16_format_xyzw v[1:2], off, s[12:15], -1 offset:4095
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_d16_hi_b16 v1, off, s[12:15], -1 offset:4095
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_d16_hi_b8 v1, off, s[12:15], -1 offset:4095
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_d16_hi_format_x v1, off, s[12:15], -1 offset:4095
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_lds_dword s[4:7], -1 offset:4095 lds
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_wbinvl1_vol
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_wbl2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_add_f64 v1, v[254:255] offset:65535
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_add_gs_reg_rtn v[254:255], v255 offset:4 gds
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_add_rtn_f64 v[10:11], v1, v[4:5] offset:65535
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_bvh_stack_rtn_b32 v255, v254, v253, v[249:252]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_cmpstore_b32 v1, v2, v3
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_cmpstore_b64 v1, v[2:3], v[3:4]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_cmpstore_f32 v1, v2, v3
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_cmpstore_f64 v1, v[2:3], v[3:4]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_cmpstore_rtn_b32 v255, v255, v255, v255 offset:4 gds
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_cmpstore_rtn_b64 v[254:255], v255, v[254:255], v[254:255] offset:4 gds
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_cmpstore_rtn_f32 v255, v255, v255, v255 offset:4 gds
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_cmpstore_rtn_f64 v[254:255], v255, v[254:255], v[254:255] offset:4 gds
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_load_2addr_b32 v[254:255], v255 offset0:16 offset1:1 gds
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_load_2addr_b64 v[252:255], v255 offset0:16 offset1:1 gds
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_load_2addr_stride64_b32 v[254:255], v255 offset0:16 offset1:1 gds
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_load_2addr_stride64_b64 v[252:255], v255 offset0:16 offset1:1 gds
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_load_addtid_b32 v255 offset:4
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_load_b128 v[252:255], v255 offset:4 gds
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_load_b32 v255, v255 offset:4 gds
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_load_b64 v[254:255], v255 offset:4 gds
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_load_b96 v[253:255], v255 offset:4 gds
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_load_i16 v255, v255 offset:4 gds
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_load_i8 v255, v255 offset:4 gds
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_load_i8_d16 v255, v255 offset:4 gds
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_load_i8_d16_hi v255, v255 offset:4 gds
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_load_u16 v255, v255 offset:4 gds
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_load_u16_d16 v255, v255 offset:4 gds
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_load_u16_d16_hi v255, v255 offset:4 gds
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_load_u8 v255, v255 offset:4 gds
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_load_u8_d16 v255, v255 offset:4 gds
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_load_u8_d16_hi v255, v255 offset:4 gds
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_pk_add_bf16 v1, v2 offset:65535
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_pk_add_f16 v1, v2 offset:65535
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_pk_add_rtn_bf16  a3, v2, a1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_pk_add_rtn_f16  a3, v2, a1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_store_2addr_b32 v1, v2, v3
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_store_2addr_b64 v1, v[2:3], v[3:4]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_store_2addr_stride64_b32 v1, v2, v3
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_store_2addr_stride64_b64 v1, v[2:3], v[3:4]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_store_addtid_b32 v1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_store_b128 v1, v[2:5]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_store_b16 v1, v2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_store_b16_d16_hi v1, v2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_store_b32 v1, v2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_store_b64 v1, v[2:3]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_store_b8 v1, v2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_store_b8_d16_hi v1, v2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_store_b96 v1, v[2:4]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_storexchg_2addr_rtn_b32 v[254:255], v255, v255, v255 offset0:16 offset1:1 gds
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_storexchg_2addr_rtn_b64 v[252:255], v255, v[254:255], v[254:255] offset0:16 offset1:1 gds
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_storexchg_2addr_stride64_rtn_b32 v[254:255], v255, v255, v255 offset0:16 offset1:1 gds
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_storexchg_2addr_stride64_rtn_b64 v[252:255], v255, v[254:255], v[254:255] offset0:16 offset1:1 gds
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_storexchg_rtn_b32 v255, v255, v255 offset:4 gds
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_storexchg_rtn_b64 v[254:255], v255, v[254:255] offset:4 gds
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_sub_gs_reg_rtn v[254:255], v255 offset:4 gds
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

flat_atomic_add_f32 a4, v[2:3], a1 sc0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

flat_atomic_add_f64 v[0:1], v[0:1], v[2:3] glc
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

flat_atomic_add_u32 v1, v[3:4], v5 offset:8 slc
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

flat_atomic_add_u64 v[1:2], v[2:3]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

flat_atomic_and_b32 v255, v[254:255], v255 offset:7 glc slc dlc
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

flat_atomic_and_b64 v[1:2], v[2:3]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

flat_atomic_cmpswap_b32 v0, v[1:2], v[3:4]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

flat_atomic_cmpswap_b64 v[1:2], v[2:5]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

flat_atomic_cmpswap_f32 v255, v[254:255], v[254:255] offset:7 glc slc dlc
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

flat_atomic_dec_u32 v255, v[254:255], v255 offset:7 glc slc dlc
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

flat_atomic_dec_u64 v[1:2], v[2:3]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

flat_atomic_inc_u32 v255, v[254:255], v255 offset:7 glc slc dlc
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

flat_atomic_inc_u64 v[1:2], v[2:3]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

flat_atomic_max_f32 v255, v[254:255], v255 offset:7 glc slc dlc
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

flat_atomic_max_f64 v[0:1], v[0:1], v[2:3] glc
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

flat_atomic_max_i32 v255, v[254:255], v255 offset:7 glc slc dlc
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

flat_atomic_max_i64 v[1:2], v[2:3]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

flat_atomic_max_u32 v255, v[254:255], v255 offset:7 glc slc dlc
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

flat_atomic_max_u64 v[1:2], v[2:3]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

flat_atomic_min_f32 v255, v[254:255], v255 offset:7 glc slc dlc
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

flat_atomic_min_f64 v[0:1], v[0:1], v[2:3] glc
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

flat_atomic_min_i32 v255, v[254:255], v255 offset:7 glc slc dlc
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

flat_atomic_min_i64 v[1:2], v[2:3]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

flat_atomic_min_u32 v255, v[254:255], v255 offset:7 glc slc dlc
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

flat_atomic_min_u64 v[1:2], v[2:3]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

flat_atomic_or_b32 v255, v[254:255], v255 offset:7 glc slc dlc
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

flat_atomic_or_b64 v[1:2], v[2:3]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

flat_atomic_pk_add_bf16 a4, v[2:3], a1 sc0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

flat_atomic_pk_add_f16 a4, v[2:3], a1 sc0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

flat_atomic_sub_u32 v255, v[254:255], v255 offset:7 glc slc dlc
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

flat_atomic_sub_u64 v[1:2], v[2:3]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

flat_atomic_swap_b32 v0, v[1:2], v3
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

flat_atomic_swap_b64 v[1:2], v[2:3]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

flat_atomic_xor_b32 v255, v[254:255], v255 offset:7 glc slc dlc
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

flat_atomic_xor_b64 v[1:2], v[2:3]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

flat_load_b128 v[1:4], v[5:6]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

flat_load_b32 v1, v[3:4]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

flat_load_b64 v[1:2], v[3:4]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

flat_load_b96 v[1:3], v[5:6]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

flat_load_d16_b16 v1, v[3:4]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

flat_load_d16_hi_b16 v1, v[3:4]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

flat_load_d16_hi_i8 v1, v[3:4]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

flat_load_d16_hi_u8 v1, v[3:4]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

flat_load_d16_i8 v1, v[3:4]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

flat_load_d16_u8 v1, v[3:4]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

flat_load_i16 v1, v[3:4]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

flat_load_i8 v1, v[3:4]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

flat_load_u16 v1, v[3:4]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

flat_load_u8 v1, v[3:4]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

flat_store_b128 v[1:2], v[2:5]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

flat_store_b16 v[1:2], v2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

flat_store_b32 v[1:2], v2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

flat_store_b64 v[1:2], v[2:3]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

flat_store_b8 v[1:2], v2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

flat_store_b96 v[1:2], v[2:4]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

flat_store_d16_hi_b16 v[1:2], v2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

flat_store_d16_hi_b8 v[1:2], v2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

global_atomic_add_f32 v0, v2, s[0:1]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

global_atomic_add_f64 v[0:1], v[0:1], v[2:3], off glc
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

global_atomic_add_u32 v1, v2, s[104:105]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

global_atomic_add_u64 v1, v[2:3], s[104:105]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

global_atomic_and_b32 v1, v2, s[104:105]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

global_atomic_and_b64 v1, v[2:3], s[104:105]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

global_atomic_cmpswap_b32 v0, v1, s[2:3], v3
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

global_atomic_cmpswap_b64 v1, v[2:5], s[104:105]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

global_atomic_cmpswap_f32 v1, v[2:3], s[104:105]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

global_atomic_csub v2, v0, v2, s[2:3] glc
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

global_atomic_csub_u32 v255, v255, v255, ttmp[14:15] offset:-4096 glc slc dlc
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

global_atomic_dec_u32 v1, v2, s[104:105]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

global_atomic_dec_u64 v1, v[2:3], s[104:105]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

global_atomic_inc_u32 v1, v2, s[104:105]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

global_atomic_inc_u64 v1, v[2:3], s[104:105]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

global_atomic_max_f32 v1, v2, s[104:105]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

global_atomic_max_f64 v[0:1], v[0:1], v[2:3], off glc
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

global_atomic_max_i32 v1, v2, s[104:105]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

global_atomic_max_i64 v1, v[2:3], s[104:105]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

global_atomic_max_u32 v1, v2, s[104:105]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

global_atomic_max_u64 v1, v[2:3], s[104:105]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

global_atomic_min_f32 v1, v2, s[104:105]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

global_atomic_min_f64 v[0:1], v[0:1], v[2:3], off glc
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

global_atomic_min_i32 v1, v2, s[104:105]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

global_atomic_min_i64 v1, v[2:3], s[104:105]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

global_atomic_min_u32 v1, v2, s[104:105]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

global_atomic_min_u64 v1, v[2:3], s[104:105]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

global_atomic_or_b32 v1, v2, s[104:105]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

global_atomic_or_b64 v1, v[2:3], s[104:105]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

global_atomic_pk_add_bf16 a4, v[2:3], a1, off sc0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

global_atomic_pk_add_f16 v0, v[0:1], v2, off glc
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

global_atomic_sub_u32 v1, v2, s[104:105]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

global_atomic_sub_u64 v1, v[2:3], s[104:105]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

global_atomic_swap_b32 v0, v1, v3, s[2:3]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

global_atomic_swap_b64 v1, v[2:3], s[104:105]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

global_atomic_xor_b32 v1, v2, s[104:105]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

global_atomic_xor_b64 v1, v[2:3], s[104:105]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

global_load_addtid_b32 v1, off
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

global_load_b128 v[1:4], v5, s[2:3]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

global_load_b32 v1, v3, exec_hi
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

global_load_b64 v[1:2], v3, s[2:3]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

global_load_b96 v[1:3], v5, s[2:3]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

global_load_d16_b16 v1, v3, s[2:3]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

global_load_d16_hi_b16 v1, v3, s[2:3]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

global_load_d16_hi_i8 v1, v3, s[2:3]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

global_load_d16_hi_u8 v1, v3, s[2:3]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

global_load_d16_i8 v1, v3, s[2:3]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

global_load_d16_u8 v1, v3, s[2:3]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

global_load_dword_addtid v1, off offset:16
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

global_load_i16 v1, v3, s[2:3]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

global_load_i8 v1, v3, s[2:3]
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

global_load_u16 v1, v3, s[2:3]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

global_load_u8 v1, v3, s2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

global_store_addtid_b32 v1, null offset:4095
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

global_store_b128 v1, v[2:5], s[104:105]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

global_store_b16 v1, v2, s[104:105]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

global_store_b32 v1, v2, s[104:105]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

global_store_b64 v1, v[2:3], s[104:105]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

global_store_b8 v1, v2, s[104:105]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

global_store_b96 v1, v[2:4], s[104:105]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

global_store_d16_hi_b16 v1, v2, s[104:105]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

global_store_d16_hi_b8 v1, v2, s[104:105]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

global_store_dword_addtid v1, off offset:16 glc slc dlc
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_bvh64_intersect_ray v[252:255], v[247:255], ttmp[12:15] a16
// GFX1010: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_bvh_intersect_ray v[252:255], v[1:11], s[8:11]
// GFX1010: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_msaa_load v14, [v204,v11,v14,v19], s[40:47] dmask:0x1 dim:SQ_RSRC_IMG_2D_MSAA_ARRAY
// GFX1010: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

lds_direct_load v1 wait_vdst:15
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

lds_param_load v1, attr0.x wait_vdst:15
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_and_not0_saveexec_b32 null, 0xaf123456
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_and_not0_saveexec_b64 null, 0xaf123456
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_and_not0_wrexec_b32 null, 0xaf123456
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_and_not0_wrexec_b64 null, 0xaf123456
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_and_not1_b32 exec_hi, src_scc, vcc_lo
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_and_not1_b64 exec, src_scc, exec
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_and_not1_saveexec_b32 null, 0xaf123456
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_and_not1_saveexec_b64 null, 0xaf123456
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_and_not1_wrexec_b32 null, 0xaf123456
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_and_not1_wrexec_b64 null, 0xaf123456
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_buffer_load_b128 s[20:23], s[4:7], 0x0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_buffer_load_b256 s[20:27], s[4:7], 0x0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_buffer_load_b32 s101, s[4:7], s0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_buffer_load_b512 s[20:35], s[4:7], 0x0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_buffer_load_b64 s[100:101], s[4:7], s0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_cbranch_g_fork -1, s[4:5]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_cbranch_i_fork exec, 12609
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_cbranch_join 1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_cls_i32 exec_hi, src_scc
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_cls_i32_i64 exec_hi, src_scc
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_clz_i32_u32 exec_hi, src_scc
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_clz_i32_u64 exec_hi, src_scc
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_ctz_i32_b32 exec_hi, src_scc
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_ctz_i32_b64 exec_hi, src_scc
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_dcache_inv_vol
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_dcache_wb_vol
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_delay_alu
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_load_b128 s[20:23], s[100:101], s0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_load_b256 s[20:27], s[100:101], s0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_load_b32 s101, s[2:3], s0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_load_b512 s[20:35], s[100:101], s0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_load_b64 s[100:101], s[2:3], s0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_or_not0_saveexec_b32 null, 0xaf123456
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_or_not0_saveexec_b64 null, 0xaf123456
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_or_not1_b32 exec_hi, src_scc, vcc_lo
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_or_not1_b64 exec, src_scc, exec
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_or_not1_saveexec_b32 null, 0xaf123456
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_or_not1_saveexec_b64 null, 0xaf123456
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_pack_hl_b32_b16 exec_hi, src_scc, vcc_lo
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_rfe_restore_b64 -1, s2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_sendmsg_rtn_b32 s0, sendmsg(MSG_RTN_GET_DDID)
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_sendmsg_rtn_b64 s[0:1], 0x0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_set_gpr_idx_idx -1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_set_gpr_idx_mode 0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_set_gpr_idx_off
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_set_gpr_idx_on -1, 0x0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_set_inst_prefetch_distance 0x3141
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_setvskip -1, s2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_wait_event 0x3141
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

scratch_load_b128 v[1:4], v2, s1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

scratch_load_b32 v1, off, off
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

scratch_load_b64 v[1:2], v2, s1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

scratch_load_b96 v[1:3], v2, s1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

scratch_load_d16_b16 v1, v2, s1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

scratch_load_d16_hi_b16 v1, v2, s1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

scratch_load_d16_hi_i8 v1, v2, s1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

scratch_load_d16_hi_u8 v1, v2, s1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

scratch_load_d16_i8 v1, v2, s1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

scratch_load_d16_u8 v1, v2, s1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

scratch_load_i16 v1, v2, s1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

scratch_load_i8 v1, v2, s1
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

scratch_load_u16 v1, v2, s1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

scratch_load_u8 v1, v2, s1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

scratch_store_b128 off, v[2:5], null
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

scratch_store_b16 off, v2, null
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

scratch_store_b32 off, v2, null
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

scratch_store_b64 off, v[2:3], null
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

scratch_store_b8 off, v2, null
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

scratch_store_b96 off, v[2:4], null
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

scratch_store_d16_hi_b16 off, v2, null
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

scratch_store_d16_hi_b8 off, v2, null
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_load_d16_format_x v255, off, s[8:11], s3, format:1 offset:4095
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_load_d16_format_xy v255, off, s[8:11], s3, format:6 offset:4095
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_load_d16_format_xyz v[254:255], off, s[8:11], s3, format:11 offset:4095
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_load_d16_format_xyzw v[254:255], off, s[8:11], s3, format:16 offset:4095
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_store_d16_format_x v1, off, s[12:15], s4 offset:4095
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_store_d16_format_xy v1, off, s[12:15], s4 offset:4095
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_store_d16_format_xyz v[1:2], off, s[12:15], s4 offset:4095
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_store_d16_format_xyzw v[1:2], off, s[12:15], s4 offset:4095
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_accvgpr_mov_b32 a1, a2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_accvgpr_read_b32 a0, a0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_accvgpr_write_b32 a0, 65
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_add_i16 v5, v1, v2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_add_i32 lds_direct, v0, v0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_add_i32_e32 v0, vcc, 0.5, v0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_add_i32_e64 v1, s[0:1], v2, v3
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_add_u16 v0, src_shared_base, v0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_add_u16_dpp v255, v1, v2 quad_perm:[0,1,2,3] row_mask:0x0 bank_mask:0x0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_add_u16_e32 v1, v2, v3
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_add_u16_e64 v255, v1, v2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_add_u16_sdwa v0, scc, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_add_u32 v0, execz, v0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_add_u32_dpp v255, v1, v2 quad_perm:[0,1,2,3] row_mask:0x0 bank_mask:0x0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_add_u32_e32 v1, s1, v3
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_add_u32_e64 v0, scc, v0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_add_u32_sdwa v1, vcc, v2, v3 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:BYTE_2
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

v_and_b16 v5, v1, v2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_and_b16_e64_dpp v5, v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_ashr_i32 v255, v1, v2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_ashr_i32_e32 v1, v2, v3
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_ashr_i32_e64 v255, v1, v2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_ashr_i64 v[254:255], v[1:2], v2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cls_i32 v255, 0xaf123456
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cls_i32_dpp v5, v1 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cls_i32_e64 v5, v1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cls_i32_e64_dpp v5, v1 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_clz_i32_u32 v255, 0xaf123456
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_clz_i32_u32_dpp v5, v1 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_clz_i32_u32_e64 v5, v1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_clz_i32_u32_e64_dpp v5, v1 dpp8:[7,6,5,4,3,2,1,0]
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

v_cmp_t_f16 vcc, -1, v2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cmp_t_f16_dpp vcc, v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cmp_t_f16_e32 vcc, v1, v255
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cmp_t_f16_e64 null, -|0xfe0b|, -|vcc_hi| clamp
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cmp_t_f16_e64_dpp null, -|v255|, -|v255| clamp dpp8:[0,0,0,0,0,0,0,0] fi:0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cmp_t_f32 vcc, -1, v2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cmp_t_f32_dpp vcc, v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cmp_t_f32_e64 null, -|0xaf123456|, -|vcc_hi| clamp
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cmp_t_f32_e64_dpp null, -|v255|, -|v255| clamp dpp8:[0,0,0,0,0,0,0,0] fi:0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cmp_t_f64 vcc, -1, v[2:3]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cmp_t_f64_e64 null, 0xaf123456, -|vcc| clamp
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

v_cmpx_t_f16 -1, v2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cmpx_t_f16_dpp v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cmpx_t_f16_e32 v1, v255
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cmpx_t_f16_e64 -1, exec_hi
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cmpx_t_f16_e64_dpp -v1, |v2| dpp8:[7,6,5,4,3,2,1,0] fi:1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cmpx_t_f32 -1, v2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cmpx_t_f32_dpp v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cmpx_t_f32_e64 -1, exec_hi
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cmpx_t_f32_e64_dpp -v1, |v2| dpp8:[7,6,5,4,3,2,1,0] fi:1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cmpx_t_f64 -1, v[2:3]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cmpx_t_f64_e64 -1, -1
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

v_cndmask_b16 v5, v1, v2, s3
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cndmask_b16_e64_dpp v5, v1, v2, s3 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_ctz_i32_b32 v255, 0xaf123456
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_ctz_i32_b32_dpp v5, v1 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_ctz_i32_b32_e64 v5, v1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_ctz_i32_b32_e64_dpp v5, v1 dpp8:[7,6,5,4,3,2,1,0]
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

v_cvt_floor_i32_f32 v255, -|v255| row_xmask:15 row_mask:0x3 bank_mask:0x0 bound_ctrl:0 fi:1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_floor_i32_f32_dpp v5, v1 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_floor_i32_f32_e64 v5, v1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_floor_i32_f32_e64_dpp v5, v1 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_i32_i16 v255, 0xfe0b
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_i32_i16_dpp v5, v1 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_i32_i16_e32 v5, v199
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_i32_i16_e64 v5, v1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_i32_i16_e64_dpp v5, v1 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_nearest_i32_f32 v255, -|v255| row_xmask:15 row_mask:0x3 bank_mask:0x0 bound_ctrl:0 fi:1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_nearest_i32_f32_dpp v5, v1 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_nearest_i32_f32_e64 v5, v1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_nearest_i32_f32_e64_dpp v5, v1 dpp8:[7,6,5,4,3,2,1,0]
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

v_cvt_pk_i16_f32 v5, v1, v2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_pk_i16_f32_e64_dpp v5, v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_pk_norm_i16_f16 v5, v1, v2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_pk_norm_i16_f16_e64_dpp v5, v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_pk_norm_u16_f16 v5, v1, v2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_pk_norm_u16_f16_e64_dpp v5, v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_pk_rtz_f16_f32 v255, -|v255|, -|v255| row_xmask:15 row_mask:0x3 bank_mask:0x0 bound_ctrl:0 fi:1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_pk_rtz_f16_f32_dpp v5, v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_pk_rtz_f16_f32_e64 v5, v1, v2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_pk_rtz_f16_f32_e64_dpp v5, v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_pk_u16_f32 v5, v1, v2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_pk_u16_f32_e64_dpp v5, v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_pkaccum_u8_f32 v1, v2, v3
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_pkaccum_u8_f32_e64 v255, v1, v2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_sr_bf8_f32 v1, -|s2|, v3
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_sr_fp8_f32 v1, -|s2|, v3
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_u32_u16 v255, 0xfe0b
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_u32_u16_dpp v5, v1 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_u32_u16_e32 v5, v199
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_u32_u16_e64 v5, v1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_u32_u16_e64_dpp v5, v1 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_div_fixup_legacy_f16 v5, v1, v2, v3
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dot2_bf16_bf16 v5, v1, v2, v3
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dot2_bf16_bf16_e64_dpp v5, v1, v2, v3 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dot2_f16_f16 v5, v1, v2, v3
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dot2_f16_f16_e64_dpp v5, v1, v2, v3 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dot2_f32_bf16 v0, v1, v2, v3
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dot2_f32_f16 v0, -v1, -v2, -v3
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dot2_f32_f16_e64_dpp v0, v1, v2, v3 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dot2_i32_i16 v0, -v1, -v2, -v3
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dot2_u32_u16 v0, -v1, -v2, -v3
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dot2acc_f32_f16 v255, -|v255|, -|v255| row_xmask:15 row_mask:0x3 bank_mask:0x0 bound_ctrl:0 fi:1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dot2acc_f32_f16_dpp v5, v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dot2c_f32_f16 v0, v1, v2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dot2c_f32_f16_dpp v255, v1, v2  quad_perm:[0,1,2,3] row_mask:0x0 bank_mask:0x0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dot2c_f32_f16_e32 v255, v1, v2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dot2c_f32_f16_e64 v0, v1, v2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dot2c_i32_i16 v0, v1, v2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dot2c_i32_i16_dpp v255, v1, v2 quad_perm:[0,1,2,3] row_mask:0x0 bank_mask:0x0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dot2c_i32_i16_e64 v0, v1, v2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dot4_i32_i8 v0, v1, v2, v3
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dot4_i32_iu8 v255, 0xaf123456, vcc_hi, null neg_lo:[0,0,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dot4_u32_u8 v0, v1, v2, v3
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dot4c_i32_i8 v0, v1, v2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dot4c_i32_i8_dpp v255, v1, v2  quad_perm:[0,1,2,3] row_mask:0x0 bank_mask:0x0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dot4c_i32_i8_e32 v255, v1, v2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dot4c_i32_i8_e64 v0, v1, v2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dot8_i32_i4 v0, v1, v2, v3
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dot8_i32_iu4 v255, 0xaf123456, vcc_hi, null neg_lo:[0,0,0] clamp
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dot8_u32_u4 v0, v1, v2, v3
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dot8c_i32_i4 v0, v1, v2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dot8c_i32_i4_dpp v255, v1, v2 quad_perm:[0,1,2,3] row_mask:0x0 bank_mask:0x0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dot8c_i32_i4_e64 v0, v1, v2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_add_f32      v255, s105, v2               ::  v_dual_cndmask_b32   v6, s105, v3
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_add_f32      v5, 0xaf123456, v2           ::  v_dual_fmaak_f32     v6, v3, v1, 0xaf123456 ;
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_add_f32      v6, 0xfe0b, v5               ::  v_dual_dot2acc_f32_f16  v255, 0xfe0b, v4
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_add_f32 v255, -1, v4 :: v_dual_add_f32 v6, src_scc, v5
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_add_f32 v255, -1, v4 :: v_dual_add_nc_u32 v6, src_scc, v5
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_add_f32 v255, -1, v4 :: v_dual_and_b32 v6, src_scc, v5
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_add_f32 v255, -1, v4 :: v_dual_fmac_f32 v6, src_scc, v5
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_add_f32 v255, -1, v4 :: v_dual_fmamk_f32 v6, src_scc, 0xaf123456, v255
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_add_f32 v255, -1, v4 :: v_dual_lshlrev_b32 v6, src_scc, v5
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_add_f32 v255, -1, v4 :: v_dual_max_f32 v6, src_scc, v5
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_add_f32 v255, -1, v4 :: v_dual_min_f32 v6, src_scc, v5
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_add_f32 v255, -1, v4 :: v_dual_mov_b32 v6, src_scc
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_add_f32 v255, -1, v4 :: v_dual_mul_dx9_zero_f32 v6, src_scc, v5
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_add_f32 v255, -1, v4 :: v_dual_mul_f32 v6, src_scc, v5
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_add_f32 v255, -1, v4 :: v_dual_sub_f32 v6, src_scc, v5
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_add_f32 v255, -1, v4 :: v_dual_subrev_f32 v6, src_scc, v5
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_cndmask_b32  v255, 0xbabe, v2             ::  v_dual_cndmask_b32   v6, 0xbabe, v3
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_cndmask_b32 v255, -1, v4 :: v_dual_add_f32 v6, src_scc, v5
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_cndmask_b32 v255, -1, v4 :: v_dual_add_nc_u32 v6, src_scc, v5
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_cndmask_b32 v255, -1, v4 :: v_dual_and_b32 v6, src_scc, v5
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_cndmask_b32 v255, -1, v4 :: v_dual_dot2acc_f32_f16 v6, src_scc, v5
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_cndmask_b32 v255, -1, v4 :: v_dual_fmaak_f32 v6, 0.5, v5, 0xaf123456
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_cndmask_b32 v255, -1, v4 :: v_dual_fmac_f32 v6, src_scc, v5
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_cndmask_b32 v255, -1, v4 :: v_dual_fmamk_f32 v6, 0.5, 0xaf123456, v255
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_cndmask_b32 v255, -1, v4 :: v_dual_lshlrev_b32 v6, src_scc, v5
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_cndmask_b32 v255, -1, v4 :: v_dual_max_f32 v6, src_scc, v5
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_cndmask_b32 v255, -1, v4 :: v_dual_min_f32 v6, src_scc, v5
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_cndmask_b32 v255, -1, v4 :: v_dual_mov_b32 v6, src_scc
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_cndmask_b32 v255, -1, v4 :: v_dual_mul_dx9_zero_f32 v6, src_scc, v5
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_cndmask_b32 v255, -1, v4 :: v_dual_mul_f32 v6, src_scc, v5
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_cndmask_b32 v255, -1, v4 :: v_dual_sub_f32 v6, src_scc, v5
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_cndmask_b32 v255, -1, v4 :: v_dual_subrev_f32 v6, src_scc, v5
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_dot2acc_f32_f16 v255, -1, v4 :: v_dual_add_f32 v6, src_scc, v5
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_dot2acc_f32_f16 v255, -1, v4 :: v_dual_add_nc_u32 v6, src_scc, v5
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_dot2acc_f32_f16 v255, -1, v4 :: v_dual_and_b32 v6, src_scc, v5
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_dot2acc_f32_f16 v255, -1, v4 :: v_dual_cndmask_b32 v6, src_scc, v5
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_dot2acc_f32_f16 v255, -1, v4 :: v_dual_dot2acc_f32_f16 v6, src_scc, v5
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_dot2acc_f32_f16 v255, -1, v4 :: v_dual_fmaak_f32 v6, src_scc, v5, 0xaf123456
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_dot2acc_f32_f16 v255, -1, v4 :: v_dual_fmac_f32 v6, src_scc, v5
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_dot2acc_f32_f16 v255, -1, v4 :: v_dual_fmamk_f32 v6, src_scc, 0xaf123456, v4
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_dot2acc_f32_f16 v255, -1, v4 :: v_dual_lshlrev_b32 v6, src_scc, v5
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_dot2acc_f32_f16 v255, -1, v4 :: v_dual_max_f32 v6, src_scc, v5
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_dot2acc_f32_f16 v255, -1, v4 :: v_dual_min_f32 v6, src_scc, v5
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_dot2acc_f32_f16 v255, -1, v4 :: v_dual_mov_b32 v6, src_scc
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_dot2acc_f32_f16 v255, -1, v4 :: v_dual_mul_dx9_zero_f32 v6, src_scc, v5
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_dot2acc_f32_f16 v255, -1, v4 :: v_dual_mul_f32 v6, src_scc, v5
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_dot2acc_f32_f16 v255, -1, v4 :: v_dual_sub_f32 v6, src_scc, v5
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_dot2acc_f32_f16 v255, -1, v4 :: v_dual_subrev_f32 v6, src_scc, v5
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_fmaak_f32    v122, s74, v161, 2.741       ::  v_dual_and_b32       v247, s74, v98
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_fmaak_f32    v122, s74, v161, 2.741       ::  v_dual_fmamk_f32     v3, s74, 2.741, v1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_fmaak_f32    v122, v0, v161, 2.741       ::  v_dual_cndmask_b32   v1, 2.741, v3
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_fmaak_f32 v255, -1, v4, 0xaf123456 :: v_dual_add_f32 v6, src_scc, v5
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_fmaak_f32 v255, -1, v4, 0xaf123456 :: v_dual_add_nc_u32 v6, src_scc, v5
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_fmaak_f32 v255, -1, v4, 0xaf123456 :: v_dual_dot2acc_f32_f16 v6, 0.5, v5
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_fmaak_f32 v255, -1, v4, 0xaf123456 :: v_dual_fmaak_f32 v6, src_scc, v5, 0xaf123456
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_fmaak_f32 v255, -1, v4, 0xaf123456 :: v_dual_fmac_f32 v6, src_scc, v5
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_fmaak_f32 v255, -1, v4, 0xaf123456 :: v_dual_lshlrev_b32 v6, src_scc, v5
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_fmaak_f32 v255, -1, v4, 0xaf123456 :: v_dual_max_f32 v6, src_scc, v5
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_fmaak_f32 v255, -1, v4, 0xaf123456 :: v_dual_min_f32 v6, src_scc, v5
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_fmaak_f32 v255, -1, v4, 0xaf123456 :: v_dual_mov_b32 v6, src_scc
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_fmaak_f32 v255, -1, v4, 0xaf123456 :: v_dual_mul_dx9_zero_f32 v6, src_scc, v5
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_fmaak_f32 v255, -1, v4, 0xaf123456 :: v_dual_mul_f32 v6, src_scc, v5
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_fmaak_f32 v255, -1, v4, 0xaf123456 :: v_dual_sub_f32 v6, src_scc, v5
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_fmaak_f32 v255, -1, v4, 0xaf123456 :: v_dual_subrev_f32 v6, src_scc, v5
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_fmac_f32     v6, v1, v2                   :: v_dual_fmamk_f32      v7, v2, 0xaf123456, v7
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_fmac_f32 v255, -1, v4 :: v_dual_add_f32 v6, src_scc, v5
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_fmac_f32 v255, -1, v4 :: v_dual_add_nc_u32 v6, src_scc, v5
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_fmac_f32 v255, -1, v4 :: v_dual_and_b32 v6, src_scc, v5
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_fmac_f32 v255, -1, v4 :: v_dual_cndmask_b32 v6, src_scc, v5
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_fmac_f32 v255, -1, v4 :: v_dual_dot2acc_f32_f16 v6, src_scc, v5
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_fmac_f32 v255, -1, v4 :: v_dual_fmaak_f32 v6, src_scc, v5, 0xaf123456
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_fmac_f32 v255, -1, v4 :: v_dual_fmac_f32 v6, src_scc, v5
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_fmac_f32 v255, -1, v4 :: v_dual_lshlrev_b32 v6, src_scc, v5
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_fmac_f32 v255, -1, v4 :: v_dual_max_f32 v6, src_scc, v5
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_fmac_f32 v255, -1, v4 :: v_dual_min_f32 v6, src_scc, v5
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_fmac_f32 v255, -1, v4 :: v_dual_mov_b32 v6, src_scc
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_fmac_f32 v255, -1, v4 :: v_dual_mul_dx9_zero_f32 v6, src_scc, v5
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_fmac_f32 v255, -1, v4 :: v_dual_mul_f32 v6, src_scc, v5
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_fmac_f32 v255, -1, v4 :: v_dual_sub_f32 v6, src_scc, v5
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_fmac_f32 v255, -1, v4 :: v_dual_subrev_f32 v6, src_scc, v5
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_fmamk_f32    v122, 0xdeadbeef, 0xdeadbeef, v161 ::  v_dual_fmamk_f32  v123, 0xdeadbeef, 0xdeadbeef, v162
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_fmamk_f32    v122, v74, 0xa0172923, v161  ::  v_dual_lshlrev_b32   v247, 0xa0172923, v99
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_fmamk_f32    v122, v74, 0xfe0b, v162      ::  v_dual_dot2acc_f32_f16  v247, 0xfe0b, v99
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_fmamk_f32    v5, v1, 0xaf123456, v5       :: v_dual_fmac_f32       v6, v2, v3
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_fmamk_f32 v255, -1, 0xaf123456, v255 :: v_dual_add_f32 v6, src_scc, v4
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_fmamk_f32 v255, -1, 0xaf123456, v255 :: v_dual_add_nc_u32 v6, src_scc, v4
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_fmamk_f32 v255, -1, 0xaf123456, v255 :: v_dual_and_b32 v6, src_scc, v4
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_fmamk_f32 v255, -1, 0xaf123456, v255 :: v_dual_cndmask_b32 v6, 0.5, v4
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_fmamk_f32 v255, -1, 0xaf123456, v255 :: v_dual_fmaak_f32 v6, src_scc, v4, 0xaf123456
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_fmamk_f32 v255, -1, 0xaf123456, v255 :: v_dual_max_f32 v6, src_scc, v4
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_fmamk_f32 v255, -1, 0xaf123456, v255 :: v_dual_min_f32 v6, src_scc, v4
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_fmamk_f32 v255, -1, 0xaf123456, v255 :: v_dual_mov_b32 v6, src_scc
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_fmamk_f32 v255, -1, 0xaf123456, v255 :: v_dual_mul_dx9_zero_f32 v6, src_scc, v4
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_fmamk_f32 v255, -1, 0xaf123456, v255 :: v_dual_mul_f32 v6, src_scc, v4
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_fmamk_f32 v255, -1, 0xaf123456, v255 :: v_dual_sub_f32 v6, src_scc, v4
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_fmamk_f32 v255, -1, 0xaf123456, v255 :: v_dual_subrev_f32 v6, src_scc, v4
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_max_f32 v255, -1, v4 :: v_dual_add_f32 v6, src_scc, v5
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_max_f32 v255, -1, v4 :: v_dual_add_nc_u32 v6, src_scc, v5
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_max_f32 v255, -1, v4 :: v_dual_and_b32 v6, src_scc, v5
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_max_f32 v255, -1, v4 :: v_dual_cndmask_b32 v6, src_scc, v5
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_max_f32 v255, -1, v4 :: v_dual_dot2acc_f32_f16 v6, src_scc, v5
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_max_f32 v255, -1, v4 :: v_dual_fmaak_f32 v6, src_scc, v5, 0xaf123456
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_max_f32 v255, -1, v4 :: v_dual_fmac_f32 v6, src_scc, v5
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_max_f32 v255, -1, v4 :: v_dual_fmamk_f32 v6, src_scc, 0xaf123456, v255
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_max_f32 v255, -1, v4 :: v_dual_lshlrev_b32 v6, src_scc, v5
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_max_f32 v255, -1, v4 :: v_dual_max_f32 v6, src_scc, v5
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_max_f32 v255, -1, v4 :: v_dual_min_f32 v6, src_scc, v5
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_max_f32 v255, -1, v4 :: v_dual_mov_b32 v6, src_scc
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_max_f32 v255, -1, v4 :: v_dual_mul_dx9_zero_f32 v6, src_scc, v5
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_max_f32 v255, -1, v4 :: v_dual_mul_f32 v6, src_scc, v5
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_max_f32 v255, -1, v4 :: v_dual_sub_f32 v6, src_scc, v5
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_max_f32 v255, -1, v4 :: v_dual_subrev_f32 v6, src_scc, v5
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_min_f32 v255, -1, v4 :: v_dual_add_f32 v6, src_scc, v5
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_min_f32 v255, -1, v4 :: v_dual_add_nc_u32 v6, src_scc, v5
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_min_f32 v255, -1, v4 :: v_dual_and_b32 v6, src_scc, v5
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_min_f32 v255, -1, v4 :: v_dual_cndmask_b32 v6, src_scc, v5
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_min_f32 v255, -1, v4 :: v_dual_dot2acc_f32_f16 v6, src_scc, v5
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_min_f32 v255, -1, v4 :: v_dual_fmaak_f32 v6, src_scc, v5, 0xaf123456
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_min_f32 v255, -1, v4 :: v_dual_fmac_f32 v6, src_scc, v5
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_min_f32 v255, -1, v4 :: v_dual_fmamk_f32 v6, src_scc, 0xaf123456, v255
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_min_f32 v255, -1, v4 :: v_dual_lshlrev_b32 v6, src_scc, v5
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_min_f32 v255, -1, v4 :: v_dual_max_f32 v6, src_scc, v5
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_min_f32 v255, -1, v4 :: v_dual_min_f32 v6, src_scc, v5
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_min_f32 v255, -1, v4 :: v_dual_mov_b32 v6, src_scc
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_min_f32 v255, -1, v4 :: v_dual_mul_dx9_zero_f32 v6, src_scc, v5
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_min_f32 v255, -1, v4 :: v_dual_mul_f32 v6, src_scc, v5
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_min_f32 v255, -1, v4 :: v_dual_sub_f32 v6, src_scc, v5
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_min_f32 v255, -1, v4 :: v_dual_subrev_f32 v6, src_scc, v5
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_mov_b32      v247, v160                   ::  v_dual_fmaak_f32     v122, s74, v161, 2.741
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_mov_b32 v255, -1 :: v_dual_add_f32 v6, src_scc, v4
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_mov_b32 v255, -1 :: v_dual_add_nc_u32 v6, src_scc, v4
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_mov_b32 v255, -1 :: v_dual_and_b32 v6, src_scc, v4
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_mov_b32 v255, -1 :: v_dual_cndmask_b32 v6, src_scc, v4
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_mov_b32 v255, -1 :: v_dual_dot2acc_f32_f16 v6, src_scc, v4
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_mov_b32 v255, -1 :: v_dual_fmac_f32 v6, src_scc, v4
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_mov_b32 v255, -1 :: v_dual_fmamk_f32 v6, src_scc, 0xaf123456, v255
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_mov_b32 v255, -1 :: v_dual_lshlrev_b32 v6, src_scc, v4
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_mov_b32 v255, -1 :: v_dual_max_f32 v6, src_scc, v4
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_mov_b32 v255, -1 :: v_dual_min_f32 v6, src_scc, v4
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_mov_b32 v255, -1 :: v_dual_mov_b32 v6, src_scc
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_mov_b32 v255, -1 :: v_dual_mul_dx9_zero_f32 v6, src_scc, v4
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_mov_b32 v255, -1 :: v_dual_mul_f32 v6, src_scc, v4
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_mov_b32 v255, -1 :: v_dual_sub_f32 v6, src_scc, v4
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_mov_b32 v255, -1 :: v_dual_subrev_f32 v6, src_scc, v4
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_mul_dx9_zero_f32 v255, -1, v4 :: v_dual_add_f32 v6, src_scc, v5
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_mul_dx9_zero_f32 v255, -1, v4 :: v_dual_add_nc_u32 v6, src_scc, v5
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_mul_dx9_zero_f32 v255, -1, v4 :: v_dual_and_b32 v6, src_scc, v5
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_mul_dx9_zero_f32 v255, -1, v4 :: v_dual_cndmask_b32 v6, src_scc, v5
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_mul_dx9_zero_f32 v255, -1, v4 :: v_dual_dot2acc_f32_f16 v6, src_scc, v5
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_mul_dx9_zero_f32 v255, -1, v4 :: v_dual_fmaak_f32 v6, src_scc, v5, 0xaf123456
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_mul_dx9_zero_f32 v255, -1, v4 :: v_dual_fmac_f32 v6, src_scc, v5
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_mul_dx9_zero_f32 v255, -1, v4 :: v_dual_fmamk_f32 v6, src_scc, 0xaf123456, v255
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_mul_dx9_zero_f32 v255, -1, v4 :: v_dual_lshlrev_b32 v6, src_scc, v5
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_mul_dx9_zero_f32 v255, -1, v4 :: v_dual_max_f32 v6, src_scc, v5
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_mul_dx9_zero_f32 v255, -1, v4 :: v_dual_min_f32 v6, src_scc, v5
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_mul_dx9_zero_f32 v255, -1, v4 :: v_dual_mov_b32 v6, src_scc
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_mul_dx9_zero_f32 v255, -1, v4 :: v_dual_mul_dx9_zero_f32 v6, src_scc, v5
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_mul_dx9_zero_f32 v255, -1, v4 :: v_dual_mul_f32 v6, src_scc, v5
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_mul_dx9_zero_f32 v255, -1, v4 :: v_dual_sub_f32 v6, src_scc, v5
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_mul_dx9_zero_f32 v255, -1, v4 :: v_dual_subrev_f32 v6, src_scc, v5
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_mul_f32      v0, s1, v2                   ::  v_dual_mul_f32       v3, s4, v5
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_mul_f32 v255, -1, v4 :: v_dual_add_f32 v6, src_scc, v5
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_mul_f32 v255, -1, v4 :: v_dual_add_nc_u32 v6, src_scc, v5
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_mul_f32 v255, -1, v4 :: v_dual_and_b32 v6, src_scc, v5
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_mul_f32 v255, -1, v4 :: v_dual_cndmask_b32 v6, src_scc, v5
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_mul_f32 v255, -1, v4 :: v_dual_dot2acc_f32_f16 v6, src_scc, v5
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_mul_f32 v255, -1, v4 :: v_dual_fmaak_f32 v6, src_scc, v5, 0xaf123456
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_mul_f32 v255, -1, v4 :: v_dual_fmac_f32 v6, src_scc, v5
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_mul_f32 v255, -1, v4 :: v_dual_fmamk_f32 v6, src_scc, 0xaf123456, v255
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_mul_f32 v255, -1, v4 :: v_dual_lshlrev_b32 v6, src_scc, v5
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_mul_f32 v255, -1, v4 :: v_dual_max_f32 v6, src_scc, v5
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_mul_f32 v255, -1, v4 :: v_dual_min_f32 v6, src_scc, v5
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_mul_f32 v255, -1, v4 :: v_dual_mov_b32 v6, src_scc
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_mul_f32 v255, -1, v4 :: v_dual_mul_dx9_zero_f32 v6, src_scc, v5
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_mul_f32 v255, -1, v4 :: v_dual_sub_f32 v6, src_scc, v5
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_mul_f32 v255, -1, v4 :: v_dual_subrev_f32 v6, src_scc, v5
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_sub_f32 v255, -1, v4 :: v_dual_add_f32 v6, src_scc, v5
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_sub_f32 v255, -1, v4 :: v_dual_add_nc_u32 v6, src_scc, v5
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_sub_f32 v255, -1, v4 :: v_dual_and_b32 v6, src_scc, v5
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_sub_f32 v255, -1, v4 :: v_dual_cndmask_b32 v6, src_scc, v5
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_sub_f32 v255, -1, v4 :: v_dual_dot2acc_f32_f16 v6, src_scc, v5
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_sub_f32 v255, -1, v4 :: v_dual_fmaak_f32 v6, src_scc, v5, 0xaf123456
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_sub_f32 v255, -1, v4 :: v_dual_fmac_f32 v6, src_scc, v5
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_sub_f32 v255, -1, v4 :: v_dual_fmamk_f32 v6, src_scc, 0xaf123456, v255
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_sub_f32 v255, -1, v4 :: v_dual_lshlrev_b32 v6, src_scc, v5
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_sub_f32 v255, -1, v4 :: v_dual_max_f32 v6, src_scc, v5
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_sub_f32 v255, -1, v4 :: v_dual_min_f32 v6, src_scc, v5
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_sub_f32 v255, -1, v4 :: v_dual_mov_b32 v6, src_scc
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_sub_f32 v255, -1, v4 :: v_dual_mul_dx9_zero_f32 v6, src_scc, v5
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_sub_f32 v255, -1, v4 :: v_dual_mul_f32 v6, src_scc, v5
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_sub_f32 v255, -1, v4 :: v_dual_sub_f32 v6, src_scc, v5
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_sub_f32 v255, -1, v4 :: v_dual_subrev_f32 v6, src_scc, v5
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_subrev_f32 v255, -1, v4 :: v_dual_add_f32 v6, src_scc, v5
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_subrev_f32 v255, -1, v4 :: v_dual_add_nc_u32 v6, src_scc, v5
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_subrev_f32 v255, -1, v4 :: v_dual_and_b32 v6, src_scc, v5
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_subrev_f32 v255, -1, v4 :: v_dual_cndmask_b32 v6, src_scc, v5
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_subrev_f32 v255, -1, v4 :: v_dual_dot2acc_f32_f16 v6, src_scc, v5
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_subrev_f32 v255, -1, v4 :: v_dual_fmaak_f32 v6, src_scc, v5, 0xaf123456
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_subrev_f32 v255, -1, v4 :: v_dual_fmac_f32 v6, src_scc, v5
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_subrev_f32 v255, -1, v4 :: v_dual_fmamk_f32 v6, src_scc, 0xaf123456, v255
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_subrev_f32 v255, -1, v4 :: v_dual_lshlrev_b32 v6, src_scc, v5
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_subrev_f32 v255, -1, v4 :: v_dual_max_f32 v6, src_scc, v5
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_subrev_f32 v255, -1, v4 :: v_dual_min_f32 v6, src_scc, v5
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_subrev_f32 v255, -1, v4 :: v_dual_mov_b32 v6, src_scc
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_subrev_f32 v255, -1, v4 :: v_dual_mul_dx9_zero_f32 v6, src_scc, v5
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_subrev_f32 v255, -1, v4 :: v_dual_mul_f32 v6, src_scc, v5
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_subrev_f32 v255, -1, v4 :: v_dual_sub_f32 v6, src_scc, v5
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_subrev_f32 v255, -1, v4 :: v_dual_subrev_f32 v6, src_scc, v5
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_exp_legacy_f32 v1, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_exp_legacy_f32_dpp v255, v1 quad_perm:[0,1,2,3] row_mask:0x0 bank_mask:0x0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_exp_legacy_f32_e64 v255, v1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_exp_legacy_f32_sdwa v255, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_fma_dx9_zero_f32 v5, v1, v2, v3
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_fma_legacy_f16 v5, v1, v2, v3
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_fma_legacy_f32 v0, s1, 2.0, -v3
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_fmac_dx9_zero_f32 v255, 0xaf123456, v255
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_fmac_dx9_zero_f32_e64 v5, v1, v2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_fmac_f64 v[0:1], v[2:3], v[4:5] row_newbcast:2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_fmac_f64_dpp v[10:11], v[2:3], v[4:5] row_newbcast:1 row_mask:0xf bank_mask:0xf
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_fmac_f64_e32 v[254:255], v[2:3], v[4:5]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_fmac_f64_e64 v[10:11], v[2:3], v[4:5]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_fmac_legacy_f32 v0, s1, 2.0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_fmac_legacy_f32_e64 v5, v1, v2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_interp_p10_f16_f32 v0, -v1, -v2, -v3 clamp op_sel:[1,0,0,1] wait_exp:5
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_interp_p10_f32 v0, -v1, v2, v3
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_interp_p10_rtz_f16_f32 v0, -v1, -v2, -v3 clamp op_sel:[1,0,0,1] wait_exp:5
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_interp_p2_f16_f32 v0, -v1, -v2, -v3 clamp op_sel:[1,0,0,1] wait_exp:5
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_interp_p2_legacy_f16 v5, v1, attr0.x, v3
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_interp_p2_rtz_f16_f32 v0, -v1, -v2, -v3 clamp op_sel:[1,0,0,1] wait_exp:5
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

v_mad_f16 v5, v1, v2, v3
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mad_legacy_f16 v5, v1, v2, v3
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

v_madmk_f16 v0, 0xff32, 0x1122, v0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_max_legacy_f32 v1, v2, v3
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_max_legacy_f32_e64 v255, v1, v2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_maxmin_f16 v5, v1, v2, v3
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_maxmin_f16_e64_dpp v5, v1, v2, v3 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_maxmin_f32 v5, v1, v2, v3
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_maxmin_f32_e64_dpp v5, v1, v2, v3 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_maxmin_i32 v5, v1, v2, v3
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_maxmin_i32_e64_dpp v5, v1, v2, v3 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_maxmin_u32 v5, v1, v2, v3
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_maxmin_u32_e64_dpp v5, v1, v2, v3 dpp8:[7,6,5,4,3,2,1,0]
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

v_minmax_f16 v5, v1, v2, v3
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_minmax_f16_e64_dpp v5, v1, v2, v3 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_minmax_f32 v5, v1, v2, v3
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_minmax_f32_e64_dpp v5, v1, v2, v3 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_minmax_i32 v5, v1, v2, v3
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_minmax_i32_e64_dpp v5, v1, v2, v3 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_minmax_u32 v5, v1, v2, v3
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_minmax_u32_e64_dpp v5, v1, v2, v3 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mov_b64 v[10:11], v[2:3]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mov_b64_dpp v[10:11], v[2:3] row_newbcast:1 row_mask:0xf bank_mask:0xf
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mov_b64_e64 v[10:11], v[2:3]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mul_dx9_zero_f32 v255, -|v255|, -|v255| row_xmask:15 row_mask:0x3 bank_mask:0x0 bound_ctrl:0 fi:1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mul_dx9_zero_f32_dpp v5, v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mul_dx9_zero_f32_e64 v5, v1, v2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mul_dx9_zero_f32_e64_dpp v5, v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_not_b16 v127, 0xfe0b
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_not_b16_dpp v5, v1 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_not_b16_e32 v128, 0xfe0b
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_not_b16_e64 v5, v1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_not_b16_e64_dpp v5, v1 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_or_b16 v5, v1, v2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_or_b16_e64_dpp v5, v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_permlane64_b32 v255, v255
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

v_sub_i16 v5, v1, v2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_sub_i32 v1, s[0:1], v2, v3
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_sub_i32_e64 v255, s[12:13], v1, v2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_sub_u16 v1, v2, v3 clamp
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_sub_u16_dpp v255, v1, v2 quad_perm:[0,1,2,3] row_mask:0x0 bank_mask:0x0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_sub_u16_e32 v1, v2, v3
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_sub_u16_e64 v255, v1, v2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_sub_u16_sdwa v255, v1, v2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_sub_u32 v1, 4.0, v2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_sub_u32_dpp v255, v1, v2 quad_perm:[0,1,2,3] row_mask:0x0 bank_mask:0x0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_sub_u32_e32 v1, s1, v3
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_sub_u32_e64 v255, s[12:13], v1, v2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_sub_u32_sdwa v1, vcc, v2, v3 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:BYTE_2
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

v_subrev_u32 v0, src_lds_direct, v0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_subrev_u32_dpp v255, v1, v2 quad_perm:[0,1,2,3] row_mask:0x0 bank_mask:0x0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_subrev_u32_e32 v1, s1, v3
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_subrev_u32_e64 v255, s[12:13], v1, v2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_subrev_u32_sdwa v1, vcc, v2, v3 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:BYTE_2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_wmma_bf16_16x16x16_bf16 v[16:19], 1.0, v[8:15], v[16:19]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_wmma_f16_16x16x16_f16 v[16:19], 1.0, v[8:15], v[16:19]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_wmma_f32_16x16x16_bf16 v[16:19], 1.0, v[8:15], v[16:19]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_wmma_f32_16x16x16_f16 v[16:19], 1.0, v[8:15], v[16:19]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_wmma_i32_16x16x16_iu4 v[16:19], v[0:7], v[8:15], v[16:19] op_sel:[0,0,1]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_wmma_i32_16x16x16_iu8 v[16:19], v[0:7], v[8:15], v[16:19] op_sel:[0,0,1]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_xor_b16 v5, v1, v2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_xor_b16_e64_dpp v5, v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU
