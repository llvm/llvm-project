// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx1010 -mattr=+wavefrontsize32,-wavefrontsize64 %s 2>&1 | FileCheck --implicit-check-not=error: %s
// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx1010 -mattr=-wavefrontsize32,+wavefrontsize64 %s 2>&1 | FileCheck --implicit-check-not=error: %s
// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx1013 -mattr=+wavefrontsize32,-wavefrontsize64 %s 2>&1 | FileCheck --implicit-check-not=error: %s

v_add_co_u32_dpp v255, vcc, v1, v2 quad_perm:[0,1,2,3] row_mask:0x0 bank_mask:0x0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported

v_ashrrev_i16_dpp v255, v1, v2 quad_perm:[0,1,2,3] row_mask:0x0 bank_mask:0x0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported

v_ceil_f64_dpp v[0:1], v[2:3] quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0xf
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported

v_cmp_class_f16_dpp vcc, v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported

v_cmp_class_f32_dpp vcc, v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported

v_cmp_eq_f16_dpp vcc, v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported

v_cmp_eq_f32_dpp vcc, v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported

v_cmp_eq_i16_dpp vcc, v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported

v_cmp_eq_i32_dpp vcc, v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported

v_cmp_eq_u16_dpp vcc, v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported

v_cmp_eq_u32_dpp vcc, v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported

v_cmp_f_f16_dpp vcc, v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported

v_cmp_f_f32_dpp vcc, v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported

v_cmp_f_i32_dpp vcc, v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported

v_cmp_f_u32_dpp vcc, v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported

v_cmp_ge_f16_dpp vcc, v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported

v_cmp_ge_f32_dpp vcc, v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported

v_cmp_ge_i16_dpp vcc, v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported

v_cmp_ge_i32_dpp vcc, v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported

v_cmp_ge_u16_dpp vcc, v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported

v_cmp_ge_u32_dpp vcc, v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported

v_cmp_gt_f16_dpp vcc, v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported

v_cmp_gt_f32_dpp vcc, v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported

v_cmp_gt_i16_dpp vcc, v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported

v_cmp_gt_i32_dpp vcc, v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported

v_cmp_gt_u16_dpp v1, v2 row_shl:0x7 row_mask:0x0 bank_mask:0x0 fi:1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported

v_cmp_gt_u32_dpp vcc, v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported

v_cmp_le_f16_dpp vcc, v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported

v_cmp_le_f32_dpp vcc, v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported

v_cmp_le_i16_dpp vcc, v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported

v_cmp_le_i32_dpp vcc, v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported

v_cmp_le_u16_dpp v1, v2 dpp8:[7,7,7,3,4,4,6,7] fi:1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported

v_cmp_le_u32_dpp vcc, v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported

v_cmp_lg_f16_dpp vcc, v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported

v_cmp_lg_f32_dpp vcc, v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported

v_cmp_lt_f16_dpp vcc, v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported

v_cmp_lt_f32_dpp vcc, v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported

v_cmp_lt_i16_dpp vcc, v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported

v_cmp_lt_i32_dpp vcc, v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported

v_cmp_lt_u16_dpp vcc, v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported

v_cmp_lt_u32_dpp vcc, v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported

v_cmp_ne_i16_dpp vcc, v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported

v_cmp_ne_i32_dpp vcc, v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported

v_cmp_ne_u16_dpp vcc, v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported

v_cmp_ne_u32_dpp vcc, v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported

v_cmp_neq_f16_dpp vcc, v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported

v_cmp_neq_f32_dpp vcc, v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported

v_cmp_nge_f16_dpp vcc, v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported

v_cmp_nge_f32_dpp vcc, v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported

v_cmp_ngt_f16_dpp vcc, v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported

v_cmp_ngt_f32_dpp vcc, v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported

v_cmp_nle_f16_dpp vcc, v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported

v_cmp_nle_f32_dpp vcc, v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported

v_cmp_nlg_f16_dpp vcc, v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported

v_cmp_nlg_f32_dpp vcc, v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported

v_cmp_nlt_f16_dpp vcc, v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported

v_cmp_nlt_f32_dpp vcc, v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported

v_cmp_o_f16_dpp vcc, v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported

v_cmp_o_f32_dpp vcc, v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported

v_cmp_t_i32_dpp vcc, v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported

v_cmp_t_u32_dpp vcc, v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported

v_cmp_tru_f16_dpp vcc, v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported

v_cmp_tru_f32_dpp vcc, v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported

v_cmp_u_f16_dpp vcc, v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported

v_cmp_u_f32_dpp vcc, v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported

v_cmpx_class_f16_dpp v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported

v_cmpx_class_f32_dpp v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported

v_cmpx_eq_f16_dpp v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported

v_cmpx_eq_f32_dpp v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported

v_cmpx_eq_i16_dpp v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported

v_cmpx_eq_i32_dpp v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported

v_cmpx_eq_u16_dpp v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported

v_cmpx_eq_u32_dpp v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported

v_cmpx_f_f16_dpp v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported

v_cmpx_f_f32_dpp v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported

v_cmpx_f_i32_dpp v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported

v_cmpx_f_u32_dpp v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported

v_cmpx_ge_f16_dpp v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported

v_cmpx_ge_f32_dpp v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported

v_cmpx_ge_i16_dpp v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported

v_cmpx_ge_i32_dpp v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported

v_cmpx_ge_u16_dpp v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported

v_cmpx_ge_u32_dpp v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported

v_cmpx_gt_f16_dpp v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported

v_cmpx_gt_f32_dpp v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported

v_cmpx_gt_i16_dpp v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported

v_cmpx_gt_i32_dpp v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported

v_cmpx_gt_u16_dpp v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported

v_cmpx_gt_u32_dpp v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported

v_cmpx_le_f16_dpp v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported

v_cmpx_le_f32_dpp v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported

v_cmpx_le_i16_dpp v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported

v_cmpx_le_i32_dpp v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported

v_cmpx_le_u16_dpp v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported

v_cmpx_le_u32_dpp v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported

v_cmpx_lg_f16_dpp v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported

v_cmpx_lg_f32_dpp v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported

v_cmpx_lt_f16_dpp v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported

v_cmpx_lt_f32_dpp v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported

v_cmpx_lt_i16_dpp v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported

v_cmpx_lt_i32_dpp v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported

v_cmpx_lt_u16_dpp v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported

v_cmpx_lt_u32_dpp v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported

v_cmpx_ne_i16_dpp v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported

v_cmpx_ne_i32_dpp v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported

v_cmpx_ne_u16_dpp v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported

v_cmpx_ne_u32_dpp v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported

v_cmpx_neq_f16_dpp v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported

v_cmpx_neq_f32_dpp v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported

v_cmpx_nge_f16_dpp v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported

v_cmpx_nge_f32_dpp v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported

v_cmpx_ngt_f16_dpp v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported

v_cmpx_ngt_f32_dpp v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported

v_cmpx_nle_f16_dpp v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported

v_cmpx_nle_f32_dpp v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported

v_cmpx_nlg_f16_dpp v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported

v_cmpx_nlg_f32_dpp v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported

v_cmpx_nlt_f16_dpp v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported

v_cmpx_nlt_f32_dpp v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported

v_cmpx_o_f16_dpp v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported

v_cmpx_o_f32_dpp v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported

v_cmpx_t_i32_dpp v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported

v_cmpx_t_u32_dpp v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported

v_cmpx_tru_f16_dpp v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported

v_cmpx_tru_f32_dpp v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported

v_cmpx_u_f16_dpp v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported

v_cmpx_u_f32_dpp v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported

v_cvt_f32_f64_dpp v5, v[2:3] row_newbcast:1 row_mask:0xf bank_mask:0xf
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported

v_cvt_i32_f64_dpp v5, v[2:3] row_newbcast:1 row_mask:0xf bank_mask:0xf
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported

v_cvt_u32_f64_dpp v5, v[2:3] row_newbcast:1 row_mask:0xf bank_mask:0xf
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported

v_floor_f64_dpp v[10:11], v[2:3] row_newbcast:1 row_mask:0xf bank_mask:0xf
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported

v_fract_f64_dpp v[10:11], v[2:3] row_newbcast:1 row_mask:0xf bank_mask:0xf
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported

v_frexp_exp_i32_f64_dpp v5, v[2:3] row_newbcast:1 row_mask:0xf bank_mask:0xf
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported

v_frexp_mant_f64_dpp v[10:11], v[2:3] row_newbcast:1 row_mask:0xf bank_mask:0xf
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported

v_lshlrev_b16_dpp v255, v1, v2 quad_perm:[0,1,2,3] row_mask:0x0 bank_mask:0x0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported

v_lshrrev_b16_dpp v255, v1, v2 quad_perm:[0,1,2,3] row_mask:0x0 bank_mask:0x0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported

v_max_i16_dpp v255, v1, v2 quad_perm:[0,1,2,3] row_mask:0x0 bank_mask:0x0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported

v_max_u16_dpp v255, v1, v2 quad_perm:[0,1,2,3] row_mask:0x0 bank_mask:0x0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported

v_min_i16_dpp v255, v1, v2 quad_perm:[0,1,2,3] row_mask:0x0 bank_mask:0x0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported

v_min_u16_dpp v255, v1, v2 quad_perm:[0,1,2,3] row_mask:0x0 bank_mask:0x0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported

v_mul_lo_u16_dpp v255, v1, v2 quad_perm:[0,1,2,3] row_mask:0x0 bank_mask:0x0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported

v_sub_co_u32_dpp v255, vcc, v1, v2 quad_perm:[0,1,2,3] row_mask:0x0 bank_mask:0x0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported

v_subrev_co_u32_dpp v255, vcc, v1, v2 quad_perm:[0,1,2,3] row_mask:0x0 bank_mask:0x0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported

v_trunc_f64_dpp v[10:11], v[2:3] row_newbcast:1 row_mask:0xf bank_mask:0xf
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported
