// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx1100 -mattr=+wavefrontsize32,-wavefrontsize64 %s 2>&1 | FileCheck --implicit-check-not=error: %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx1100 -mattr=-wavefrontsize32,+wavefrontsize64 %s 2>&1 | FileCheck --implicit-check-not=error: %s

v_add_co_u32_dpp v255, vcc, v1, v2 quad_perm:[0,1,2,3] row_mask:0x0 bank_mask:0x0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported

v_ashrrev_i16_dpp v255, v1, v2 quad_perm:[0,1,2,3] row_mask:0x0 bank_mask:0x0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported

v_ceil_f64_dpp v[0:1], v[2:3] quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0xf
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported

v_cvt_f32_f64_dpp v5, v[2:3] row_newbcast:1 row_mask:0xf bank_mask:0xf
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported

v_cvt_i32_f64_dpp v5, v[2:3] row_newbcast:1 row_mask:0xf bank_mask:0xf
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported

v_cvt_u32_f64_dpp v5, v[2:3] row_newbcast:1 row_mask:0xf bank_mask:0xf
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported

v_floor_f64_dpp v[10:11], v[2:3] row_newbcast:1 row_mask:0xf bank_mask:0xf
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported

v_fmac_legacy_f32_dpp v255, v1, v2 quad_perm:[0,1,2,3] row_mask:0x0 bank_mask:0x0
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
