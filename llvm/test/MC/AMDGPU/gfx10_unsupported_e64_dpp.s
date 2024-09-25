// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx1010 -mattr=+wavefrontsize32 %s 2>&1 | FileCheck --implicit-check-not=error: %s
// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx1010 -mattr=+wavefrontsize64 %s 2>&1 | FileCheck --implicit-check-not=error: %s
// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx1013 -mattr=+wavefrontsize32 %s 2>&1 | FileCheck --implicit-check-not=error: %s

v_add3_u32_e64_dpp v5, v1, s1, v0 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_add_co_ci_u32_e64_dpp v5, s6, v1, v2, s3 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_add_co_u32_e64_dpp v5, s6, v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_add_f16_e64_dpp v5, v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_add_f32_e64_dpp v5, v1, v2 div:2 quad_perm:[3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_add_lshl_u32_e64_dpp v5, v1, v2, v3 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_add_nc_i16_e64_dpp v5, v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_add_nc_i32_e64_dpp v5, v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_add_nc_u16_e64_dpp v5, v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_add_nc_u32_e64_dpp v5, v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_alignbit_b32_e64_dpp v5, v1, v2, v3 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_alignbyte_b32_e64_dpp v5, v1, v2, v3 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_and_b32_e64_dpp v5, v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_and_or_b32_e64_dpp v5, v1, v2, v3 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_ashrrev_i16_e64_dpp v5, v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_ashrrev_i32_e64_dpp v5, v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_bcnt_u32_b32_e64_dpp v5, v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_bfe_i32_e64_dpp v5, v1, v2, v3 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_bfe_u32_e64_dpp v5, v1, v2, v3 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_bfi_b32_e64_dpp v5, v1, v2, v3 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_bfm_b32_e64_dpp v5, v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_bfrev_b32_e64_dpp v5, v1 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_ceil_f16_e64_dpp v5, v1 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_ceil_f32_e64_dpp v5, v1 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_cmp_class_f16_e64_dpp null, -|v255|, v255 dpp8:[0,0,0,0,0,0,0,0] fi:0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_cmp_class_f32_e64_dpp null, -|v255|, v255 dpp8:[0,0,0,0,0,0,0,0] fi:0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_cmp_eq_f16_e64_dpp null, -|v255|, -|v255| clamp dpp8:[0,0,0,0,0,0,0,0] fi:0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_cmp_eq_f32_e64_dpp null, -|v255|, -|v255| clamp dpp8:[0,0,0,0,0,0,0,0] fi:0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_cmp_eq_i16_e64_dpp null, v255, v255 dpp8:[0,0,0,0,0,0,0,0] fi:0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_cmp_eq_i32_e64_dpp null, v255, v255 dpp8:[0,0,0,0,0,0,0,0] fi:0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_cmp_eq_u16_e64_dpp null, v255, v255 dpp8:[0,0,0,0,0,0,0,0] fi:0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_cmp_eq_u32_e64_dpp null, v255, v255 dpp8:[0,0,0,0,0,0,0,0] fi:0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_cmp_f_f16_e64_dpp null, -|v255|, -|v255| clamp dpp8:[0,0,0,0,0,0,0,0] fi:0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_cmp_f_f32_e64_dpp null, -|v255|, -|v255| clamp dpp8:[0,0,0,0,0,0,0,0] fi:0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_cmp_f_i32_e64_dpp null, v255, v255 dpp8:[0,0,0,0,0,0,0,0] fi:0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_cmp_f_u32_e64_dpp null, v255, v255 dpp8:[0,0,0,0,0,0,0,0] fi:0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_cmp_ge_f16_e64_dpp null, -|v255|, -|v255| clamp dpp8:[0,0,0,0,0,0,0,0] fi:0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_cmp_ge_f32_e64_dpp null, -|v255|, -|v255| clamp dpp8:[0,0,0,0,0,0,0,0] fi:0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_cmp_ge_i16_e64_dpp null, v255, v255 dpp8:[0,0,0,0,0,0,0,0] fi:0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_cmp_ge_i32_e64_dpp null, v255, v255 dpp8:[0,0,0,0,0,0,0,0] fi:0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_cmp_ge_u16_e64_dpp null, v255, v255 dpp8:[0,0,0,0,0,0,0,0] fi:0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_cmp_ge_u32_e64_dpp null, v255, v255 dpp8:[0,0,0,0,0,0,0,0] fi:0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_cmp_gt_f16_e64_dpp null, -|v255|, -|v255| clamp dpp8:[0,0,0,0,0,0,0,0] fi:0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_cmp_gt_f32_e64_dpp null, -|v255|, -|v255| clamp dpp8:[0,0,0,0,0,0,0,0] fi:0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_cmp_gt_i16_e64_dpp null, v255, v255 dpp8:[0,0,0,0,0,0,0,0] fi:0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_cmp_gt_i32_e64_dpp null, v255, v255 dpp8:[0,0,0,0,0,0,0,0] fi:0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_cmp_gt_u16_e64_dpp null, v255, v255 dpp8:[0,0,0,0,0,0,0,0] fi:0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_cmp_gt_u32_e64_dpp null, v255, v255 dpp8:[0,0,0,0,0,0,0,0] fi:0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_cmp_le_f16_e64_dpp null, -|v255|, -|v255| clamp dpp8:[0,0,0,0,0,0,0,0] fi:0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_cmp_le_f32_e64_dpp null, -|v255|, -|v255| clamp dpp8:[0,0,0,0,0,0,0,0] fi:0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_cmp_le_i16_e64_dpp null, v255, v255 dpp8:[0,0,0,0,0,0,0,0] fi:0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_cmp_le_i32_e64_dpp null, v255, v255 dpp8:[0,0,0,0,0,0,0,0] fi:0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_cmp_le_u16_e64_dpp null, v255, v255 dpp8:[0,0,0,0,0,0,0,0] fi:0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_cmp_le_u32_e64_dpp null, v255, v255 dpp8:[0,0,0,0,0,0,0,0] fi:0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_cmp_lg_f16_e64_dpp null, -|v255|, -|v255| clamp dpp8:[0,0,0,0,0,0,0,0] fi:0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_cmp_lg_f32_e64_dpp null, -|v255|, -|v255| clamp dpp8:[0,0,0,0,0,0,0,0] fi:0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_cmp_lt_f16_e64_dpp null, -|v255|, -|v255| clamp dpp8:[0,0,0,0,0,0,0,0] fi:0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_cmp_lt_f32_e64_dpp null, -|v255|, -|v255| clamp dpp8:[0,0,0,0,0,0,0,0] fi:0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_cmp_lt_i16_e64_dpp null, v255, v255 dpp8:[0,0,0,0,0,0,0,0] fi:0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_cmp_lt_i32_e64_dpp null, v255, v255 dpp8:[0,0,0,0,0,0,0,0] fi:0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_cmp_lt_u16_e64_dpp null, v255, v255 dpp8:[0,0,0,0,0,0,0,0] fi:0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_cmp_lt_u32_e64_dpp null, v255, v255 dpp8:[0,0,0,0,0,0,0,0] fi:0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_cmp_ne_i16_e64_dpp null, v255, v255 dpp8:[0,0,0,0,0,0,0,0] fi:0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_cmp_ne_i32_e64_dpp null, v255, v255 dpp8:[0,0,0,0,0,0,0,0] fi:0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_cmp_ne_u16_e64_dpp null, v255, v255 dpp8:[0,0,0,0,0,0,0,0] fi:0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_cmp_ne_u32_e64_dpp null, v255, v255 dpp8:[0,0,0,0,0,0,0,0] fi:0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_cmp_neq_f16_e64_dpp null, -|v255|, -|v255| clamp dpp8:[0,0,0,0,0,0,0,0] fi:0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_cmp_neq_f32_e64_dpp null, -|v255|, -|v255| clamp dpp8:[0,0,0,0,0,0,0,0] fi:0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_cmp_nge_f16_e64_dpp null, -|v255|, -|v255| clamp dpp8:[0,0,0,0,0,0,0,0] fi:0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_cmp_nge_f32_e64_dpp null, -|v255|, -|v255| clamp dpp8:[0,0,0,0,0,0,0,0] fi:0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_cmp_ngt_f16_e64_dpp null, -|v255|, -|v255| clamp dpp8:[0,0,0,0,0,0,0,0] fi:0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_cmp_ngt_f32_e64_dpp null, -|v255|, -|v255| clamp dpp8:[0,0,0,0,0,0,0,0] fi:0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_cmp_nle_f16_e64_dpp null, -|v255|, -|v255| clamp dpp8:[0,0,0,0,0,0,0,0] fi:0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_cmp_nle_f32_e64_dpp null, -|v255|, -|v255| clamp dpp8:[0,0,0,0,0,0,0,0] fi:0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_cmp_nlg_f16_e64_dpp null, -|v255|, -|v255| clamp dpp8:[0,0,0,0,0,0,0,0] fi:0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_cmp_nlg_f32_e64_dpp null, -|v255|, -|v255| clamp dpp8:[0,0,0,0,0,0,0,0] fi:0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_cmp_nlt_f16_e64_dpp null, -|v255|, -|v255| clamp dpp8:[0,0,0,0,0,0,0,0] fi:0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_cmp_nlt_f32_e64_dpp null, -|v255|, -|v255| clamp dpp8:[0,0,0,0,0,0,0,0] fi:0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_cmp_o_f16_e64_dpp null, -|v255|, -|v255| clamp dpp8:[0,0,0,0,0,0,0,0] fi:0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_cmp_o_f32_e64_dpp null, -|v255|, -|v255| clamp dpp8:[0,0,0,0,0,0,0,0] fi:0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_cmp_t_i32_e64_dpp null, v255, v255 dpp8:[0,0,0,0,0,0,0,0] fi:0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_cmp_t_u32_e64_dpp null, v255, v255 dpp8:[0,0,0,0,0,0,0,0] fi:0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_cmp_tru_f16_e64_dpp null, -|v255|, -|v255| clamp dpp8:[0,0,0,0,0,0,0,0] fi:0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_cmp_tru_f32_e64_dpp null, -|v255|, -|v255| clamp dpp8:[0,0,0,0,0,0,0,0] fi:0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_cmp_u_f16_e64_dpp null, -|v255|, -|v255| clamp dpp8:[0,0,0,0,0,0,0,0] fi:0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_cmp_u_f32_e64_dpp null, -|v255|, -|v255| clamp dpp8:[0,0,0,0,0,0,0,0] fi:0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_cmpx_class_f16_e64_dpp -|v255|, v255 dpp8:[0,0,0,0,0,0,0,0] fi:0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_cmpx_class_f32_e64_dpp -v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_cmpx_eq_f16_e64_dpp -v1, |v2| dpp8:[7,6,5,4,3,2,1,0] fi:1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_cmpx_eq_f32_e64_dpp -v1, |v2| dpp8:[7,6,5,4,3,2,1,0] fi:1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_cmpx_eq_i16_e64_dpp v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_cmpx_eq_i32_e64_dpp v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_cmpx_eq_u16_e64_dpp v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_cmpx_eq_u32_e64_dpp v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_cmpx_f_f16_e64_dpp -v1, |v2| dpp8:[7,6,5,4,3,2,1,0] fi:1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_cmpx_f_f32_e64_dpp -v1, |v2| dpp8:[7,6,5,4,3,2,1,0] fi:1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_cmpx_f_i32_e64_dpp v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_cmpx_f_u32_e64_dpp v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_cmpx_ge_f16_e64_dpp -v1, |v2| dpp8:[7,6,5,4,3,2,1,0] fi:1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_cmpx_ge_f32_e64_dpp -v1, |v2| dpp8:[7,6,5,4,3,2,1,0] fi:1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_cmpx_ge_i16_e64_dpp v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_cmpx_ge_i32_e64_dpp v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_cmpx_ge_u16_e64_dpp v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_cmpx_ge_u32_e64_dpp v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_cmpx_gt_f16_e64_dpp -v1, |v2| dpp8:[7,6,5,4,3,2,1,0] fi:1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_cmpx_gt_f32_e64_dpp -v1, |v2| dpp8:[7,6,5,4,3,2,1,0] fi:1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_cmpx_gt_i16_e64_dpp v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_cmpx_gt_i32_e64_dpp v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_cmpx_gt_u16_e64_dpp v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_cmpx_gt_u32_e64_dpp v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_cmpx_le_f16_e64_dpp -v1, |v2| dpp8:[7,6,5,4,3,2,1,0] fi:1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_cmpx_le_f32_e64_dpp -v1, |v2| dpp8:[7,6,5,4,3,2,1,0] fi:1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_cmpx_le_i16_e64_dpp v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_cmpx_le_i32_e64_dpp v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_cmpx_le_u16_e64_dpp v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_cmpx_le_u32_e64_dpp v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_cmpx_lg_f16_e64_dpp -v1, |v2| dpp8:[7,6,5,4,3,2,1,0] fi:1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_cmpx_lg_f32_e64_dpp -v1, |v2| dpp8:[7,6,5,4,3,2,1,0] fi:1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_cmpx_lt_f16_e64_dpp -v1, |v2| dpp8:[7,6,5,4,3,2,1,0] fi:1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_cmpx_lt_f32_e64_dpp -v1, |v2| dpp8:[7,6,5,4,3,2,1,0] fi:1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_cmpx_lt_i16_e64_dpp v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_cmpx_lt_i32_e64_dpp v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_cmpx_lt_u16_e64_dpp v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_cmpx_lt_u32_e64_dpp v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_cmpx_ne_i16_e64_dpp v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_cmpx_ne_i32_e64_dpp v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_cmpx_ne_u16_e64_dpp v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_cmpx_ne_u32_e64_dpp v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_cmpx_neq_f16_e64_dpp -v1, |v2| dpp8:[7,6,5,4,3,2,1,0] fi:1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_cmpx_neq_f32_e64_dpp -v1, |v2| dpp8:[7,6,5,4,3,2,1,0] fi:1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_cmpx_nge_f16_e64_dpp -v1, |v2| dpp8:[7,6,5,4,3,2,1,0] fi:1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_cmpx_nge_f32_e64_dpp -v1, |v2| dpp8:[7,6,5,4,3,2,1,0] fi:1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_cmpx_ngt_f16_e64_dpp -v1, |v2| dpp8:[7,6,5,4,3,2,1,0] fi:1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_cmpx_ngt_f32_e64_dpp -v1, |v2| dpp8:[7,6,5,4,3,2,1,0] fi:1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_cmpx_nle_f16_e64_dpp -v1, |v2| dpp8:[7,6,5,4,3,2,1,0] fi:1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_cmpx_nle_f32_e64_dpp -v1, |v2| dpp8:[7,6,5,4,3,2,1,0] fi:1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_cmpx_nlg_f16_e64_dpp -v1, |v2| dpp8:[7,6,5,4,3,2,1,0] fi:1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_cmpx_nlg_f32_e64_dpp -v1, |v2| dpp8:[7,6,5,4,3,2,1,0] fi:1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_cmpx_nlt_f16_e64_dpp -v1, |v2| dpp8:[7,6,5,4,3,2,1,0] fi:1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_cmpx_nlt_f32_e64_dpp -v1, |v2| dpp8:[7,6,5,4,3,2,1,0] fi:1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_cmpx_o_f16_e64_dpp -v1, |v2| dpp8:[7,6,5,4,3,2,1,0] fi:1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_cmpx_o_f32_e64_dpp -v1, |v2| dpp8:[7,6,5,4,3,2,1,0] fi:1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_cmpx_t_i32_e64_dpp v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_cmpx_t_u32_e64_dpp v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_cmpx_tru_f16_e64_dpp -v1, |v2| dpp8:[7,6,5,4,3,2,1,0] fi:1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_cmpx_tru_f32_e64_dpp -v1, |v2| dpp8:[7,6,5,4,3,2,1,0] fi:1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_cmpx_u_f16_e64_dpp -v1, |v2| dpp8:[7,6,5,4,3,2,1,0] fi:1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_cmpx_u_f32_e64_dpp -v1, |v2| dpp8:[7,6,5,4,3,2,1,0] fi:1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_cndmask_b32_e64_dpp v5, v1, v2, s3 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_cos_f16_e64_dpp v5, v1 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_cos_f32_e64_dpp v5, v1 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_cubeid_f32_e64_dpp v5, v1, v2, v3 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_cubema_f32_e64_dpp v5, v1, v2, v3 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_cubesc_f32_e64_dpp v5, v1, v2, 12345678 row_shr:4 row_mask:0xf bank_mask:0xf
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_cubetc_f32_e64_dpp v5, v1, v2, v3 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_cvt_f16_f32_e64_dpp v5, v1 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_cvt_f16_i16_e64_dpp v5, v1 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_cvt_f16_u16_e64_dpp v5, s1 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_cvt_f32_f16_e64_dpp v5, v1 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_cvt_f32_i32_e64_dpp v5, s1 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_cvt_f32_u32_e64_dpp v5, v1 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_cvt_f32_ubyte0_e64_dpp v5, v1 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_cvt_f32_ubyte1_e64_dpp v5, v1 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_cvt_f32_ubyte2_e64_dpp v5, v1 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_cvt_f32_ubyte3_e64_dpp v5, v1 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_cvt_flr_i32_f32_e64_dpp v5, v1 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_cvt_i16_f16_e64_dpp v5, v1 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_cvt_i32_f32_e64_dpp v5, v1 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_cvt_norm_i16_f16_e64_dpp v5, v1 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_cvt_norm_u16_f16_e64_dpp v5, v1 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_cvt_off_f32_i4_e64_dpp v5, v1 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_cvt_pk_i16_i32_e64_dpp v5, v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_cvt_pk_u16_u32_e64_dpp v5, v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_cvt_pk_u8_f32_e64_dpp v5, v1, v2, v3 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_cvt_pknorm_i16_f16_e64_dpp v5, v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_cvt_pknorm_i16_f32_e64_dpp v5, v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_cvt_pknorm_u16_f16_e64_dpp v5, v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_cvt_pknorm_u16_f32_e64_dpp v5, v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_cvt_pkrtz_f16_f32_e64_dpp v5, v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_cvt_rpi_i32_f32_e64_dpp v5, v1 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_cvt_u16_f16_e64_dpp v5, v1 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_cvt_u32_f32_e64_dpp v5, v1 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_div_fixup_f16_e64_dpp v5, v1, v2, v3 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_exp_f16_e64_dpp v5, v1 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_exp_f32_e64_dpp v5, v1 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_ffbh_i32_e64_dpp v5, v1 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_ffbh_u32_e64_dpp v5, v1 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_ffbl_b32_e64_dpp v5, v1 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_floor_f16_e64_dpp v5, v1 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_floor_f32_e64_dpp v5, v1 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_fma_f16_e64_dpp v5, v1, v2, v3 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_fma_f32_e64_dpp v5, v1, v2, v3 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_fma_mix_f32_e64_dpp v5, s1, v3, v4 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_fma_mixhi_f16_e64_dpp v5, v1, 0, v4 quad_perm:[3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_fma_mixlo_f16_e64_dpp v5, v1, 1, v4 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_fmac_f16_e64_dpp v5, v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_fmac_f32_e64_dpp v5, v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_fract_f16_e64_dpp v5, v1 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_fract_f32_e64_dpp v5, v1 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_frexp_exp_i16_f16_e64_dpp v5, v1 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_frexp_exp_i32_f32_e64_dpp v5, v1 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_frexp_mant_f16_e64_dpp v5, v1 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_frexp_mant_f32_e64_dpp v5, v1 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_ldexp_f16_e64_dpp v5, v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_ldexp_f32_e64_dpp v5, v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_lerp_u8_e64_dpp v5, v1, v2, v3 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_log_f16_e64_dpp v5, v1 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_log_f32_e64_dpp v5, v1 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_lshl_add_u32_e64_dpp v5, v1, v2, v3 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_lshl_or_b32_e64_dpp v5, v1, v2, v3 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_lshlrev_b16_e64_dpp v5, v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_lshlrev_b32_e64_dpp v5, v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_lshrrev_b16_e64_dpp v5, v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_lshrrev_b32_e64_dpp v5, v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_mad_i16_e64_dpp v5, v1, v2, v3 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_mad_i32_i16_e64_dpp v5, v1, v2, v3 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_mad_i32_i24_e64_dpp v5, v1, v2, v3 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_mad_u16_e64_dpp v5, v1, v2, v3 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_mad_u32_u16_e64_dpp v5, v1, v2, v3 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_mad_u32_u24_e64_dpp v5, v1, v2, v3 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_max3_f16_e64_dpp v5, v1, v2, v3 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_max3_f32_e64_dpp v5, v1, v2, v3 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_max3_i16_e64_dpp v5, v1, v2, v3 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_max3_i32_e64_dpp v5, v1, v2, v3 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_max3_u16_e64_dpp v5, v1, v2, v3 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_max3_u32_e64_dpp v5, v1, v2, v3 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_max_f16_e64_dpp v5, v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_max_f32_e64_dpp v5, v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_max_i16_e64_dpp v5, v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_max_i32_e64_dpp v5, v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_max_u16_e64_dpp v5, v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_max_u32_e64_dpp v5, v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_mbcnt_hi_u32_b32_e64_dpp v5, v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_mbcnt_lo_u32_b32_e64_dpp v5, v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_med3_f16_e64_dpp v5, v1, v2, v3 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_med3_f32_e64_dpp v5, v1, v2, v3 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_med3_i16_e64_dpp v5, v1, v2, v3 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_med3_i32_e64_dpp v5, v1, v2, v3 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_med3_u16_e64_dpp v5, v1, v2, v3 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_med3_u32_e64_dpp v5, v1, v2, v3 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_min3_f16_e64_dpp v5, v1, v2, v3 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_min3_f32_e64_dpp v5, v1, v2, v3 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_min3_i16_e64_dpp v5, v1, v2, v3 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_min3_i32_e64_dpp v5, v1, v2, v3 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_min3_u16_e64_dpp v5, v1, v2, v3 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_min3_u32_e64_dpp v5, v1, v2, v3 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_min_f16_e64_dpp v5, v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_min_f32_e64_dpp v5, v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_min_i16_e64_dpp v5, v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_min_i32_e64_dpp v5, v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_min_u16_e64_dpp v5, v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_min_u32_e64_dpp v5, v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_mov_b32_e64_dpp v5, v1 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_movreld_b32_e64_dpp v5, v1 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_movrels_b32_e64_dpp v5, v1 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_movrelsd_2_b32_e64_dpp v5, v1 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_movrelsd_b32_e64_dpp v5, v1 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_msad_u8_e64_dpp v5, v1, v2, v3 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_mul_f16_e64_dpp v5, v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_mul_f32_e64_dpp v5, v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_mul_hi_i32_i24_e64_dpp v5, v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_mul_hi_u32_u24_e64_dpp v5, v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_mul_i32_i24_e64_dpp v5, v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_mul_legacy_f32_e64_dpp v5, v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_mul_lo_u16_e64_dpp v5, v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_mul_u32_u24_e64_dpp v5, v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_mullit_f32_e64_dpp v5, v1, v2, v3 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_not_b32_e64_dpp v5, v1 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_or3_b32_e64_dpp v5, v1, v2, v3 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_or_b32_e64_dpp v5, v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_pack_b32_f16_e64_dpp v5, v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_perm_b32_e64_dpp v5, v1, v2, v3 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_rcp_f16_e64_dpp v5, v1 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_rcp_f32_e64_dpp v5, v1 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_rcp_iflag_f32_e64_dpp v5, v1 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_rndne_f16_e64_dpp v5, v1 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_rndne_f32_e64_dpp v5, v1 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_rsq_f16_e64_dpp v5, v1 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_rsq_f32_e64_dpp v5, v1 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_sad_hi_u8_e64_dpp v5, v1, v2, v3 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_sad_u16_e64_dpp v5, v1, v2, v3 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_sad_u32_e64_dpp v5, v1, v2, v3 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_sad_u8_e64_dpp v5, v1, v2, v3 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_sat_pk_u8_i16_e64_dpp v5, v1 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_sin_f16_e64_dpp v5, v1 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_sin_f32_e64_dpp v5, v1 div:2 dpp8:[0,0,0,0,0,0,0,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_sqrt_f16_e64_dpp v5, v1 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_sqrt_f32_e64_dpp v5, v1 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_sub_co_ci_u32_e64_dpp v5, s6, v1, v2, s3 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_sub_co_u32_e64_dpp v5, s6, v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_sub_f16_e64_dpp v5, v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_sub_f32_e64_dpp v5, v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_sub_nc_i16_e64_dpp v5, v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_sub_nc_i32_e64_dpp v5, v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_sub_nc_u16_e64_dpp v5, v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_sub_nc_u32_e64_dpp v5, v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_subrev_co_ci_u32_e64_dpp v5, s6, v1, v2, s3 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_subrev_co_u32_e64_dpp v5, s6, v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_subrev_f16_e64_dpp v5, v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_subrev_f32_e64_dpp v5, v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_subrev_nc_u32_e64_dpp v5, v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_trunc_f16_e64_dpp v5, v1 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_trunc_f32_e64_dpp v5, v1 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_xad_u32_e64_dpp v5, v1, v2, v3 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_xnor_b32_e64_dpp v5, v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_xor3_b32_e64_dpp v5, v1, v2, v3 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_xor_b32_e64_dpp v5, v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported
