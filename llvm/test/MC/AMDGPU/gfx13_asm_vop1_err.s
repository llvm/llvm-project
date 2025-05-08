// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx1300 -show-encoding %s 2>&1 | FileCheck --check-prefix=GFX13-ERR --implicit-check-not=error: --strict-whitespace %s

v_cvt_i32_f64 v2, v[4:5] dpp8:[7,6,5,4,3,2,1,0]
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX13-ERR-NEXT:{{^}}v_cvt_i32_f64 v2, v[4:5] dpp8:[7,6,5,4,3,2,1,0]
// GFX13-ERR-NEXT:{{^}}                         ^

v_cvt_i32_f64 v2, v[4:5] quad_perm:[3,2,1,0]
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX13-ERR-NEXT:{{^}}v_cvt_i32_f64 v2, v[4:5] quad_perm:[3,2,1,0]
// GFX13-ERR-NEXT:{{^}}                         ^

v_cvt_f64_i32 v[4:5], v2 dpp8:[7,6,5,4,3,2,1,0]
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX13-ERR-NEXT:{{^}}v_cvt_f64_i32 v[4:5], v2 dpp8:[7,6,5,4,3,2,1,0]
// GFX13-ERR-NEXT:{{^}}                         ^

v_cvt_f64_i32 v[4:5], v2 quad_perm:[3,2,1,0]
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX13-ERR-NEXT:{{^}}v_cvt_f64_i32 v[4:5], v2 quad_perm:[3,2,1,0]
// GFX13-ERR-NEXT:{{^}}                         ^

v_cvt_f32_f64 v2, v[4:5] dpp8:[7,6,5,4,3,2,1,0]
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX13-ERR-NEXT:{{^}}v_cvt_f32_f64 v2, v[4:5] dpp8:[7,6,5,4,3,2,1,0]
// GFX13-ERR-NEXT:{{^}}                         ^

v_cvt_f32_f64 v2, v[4:5] quad_perm:[3,2,1,0]
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX13-ERR-NEXT:{{^}}v_cvt_f32_f64 v2, v[4:5] quad_perm:[3,2,1,0]
// GFX13-ERR-NEXT:{{^}}                         ^

v_cvt_f64_f32 v[4:5], v2 dpp8:[7,6,5,4,3,2,1,0]
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX13-ERR-NEXT:{{^}}v_cvt_f64_f32 v[4:5], v2 dpp8:[7,6,5,4,3,2,1,0]
// GFX13-ERR-NEXT:{{^}}                         ^

v_cvt_f64_f32 v[4:5], v2 quad_perm:[3,2,1,0]
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX13-ERR-NEXT:{{^}}v_cvt_f64_f32 v[4:5], v2 quad_perm:[3,2,1,0]
// GFX13-ERR-NEXT:{{^}}                         ^

v_cvt_u32_f64 v2, v[4:5] dpp8:[7,6,5,4,3,2,1,0]
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX13-ERR-NEXT:{{^}}v_cvt_u32_f64 v2, v[4:5] dpp8:[7,6,5,4,3,2,1,0]
// GFX13-ERR-NEXT:{{^}}                         ^

v_cvt_u32_f64 v2, v[4:5] quad_perm:[3,2,1,0]
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX13-ERR-NEXT:{{^}}v_cvt_u32_f64 v2, v[4:5] quad_perm:[3,2,1,0]
// GFX13-ERR-NEXT:{{^}}                         ^

v_cvt_f64_u32 v[4:5], v2 dpp8:[7,6,5,4,3,2,1,0]
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX13-ERR-NEXT:{{^}}v_cvt_f64_u32 v[4:5], v2 dpp8:[7,6,5,4,3,2,1,0]
// GFX13-ERR-NEXT:{{^}}                         ^

v_cvt_f64_u32 v[4:5], v2 quad_perm:[3,2,1,0]
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX13-ERR-NEXT:{{^}}v_cvt_f64_u32 v[4:5], v2 quad_perm:[3,2,1,0]
// GFX13-ERR-NEXT:{{^}}                         ^

v_trunc_f64 v[2:3], v[4:5] dpp8:[7,6,5,4,3,2,1,0]
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX13-ERR-NEXT:{{^}}v_trunc_f64 v[2:3], v[4:5] dpp8:[7,6,5,4,3,2,1,0]
// GFX13-ERR-NEXT:{{^}}                           ^

v_trunc_f64 v[2:3], v[4:5] quad_perm:[3,2,1,0]
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX13-ERR-NEXT:{{^}}v_trunc_f64 v[2:3], v[4:5] quad_perm:[3,2,1,0]
// GFX13-ERR-NEXT:{{^}}                           ^

v_ceil_f64 v[2:3], v[4:5] dpp8:[7,6,5,4,3,2,1,0]
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX13-ERR-NEXT:{{^}}v_ceil_f64 v[2:3], v[4:5] dpp8:[7,6,5,4,3,2,1,0]
// GFX13-ERR-NEXT:{{^}}                          ^

v_ceil_f64 v[2:3], v[4:5] quad_perm:[3,2,1,0]
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX13-ERR-NEXT:{{^}}v_ceil_f64 v[2:3], v[4:5] quad_perm:[3,2,1,0]
// GFX13-ERR-NEXT:{{^}}                          ^

v_rndne_f64 v[2:3], v[4:5] dpp8:[7,6,5,4,3,2,1,0]
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX13-ERR-NEXT:{{^}}v_rndne_f64 v[2:3], v[4:5] dpp8:[7,6,5,4,3,2,1,0]
// GFX13-ERR-NEXT:{{^}}                           ^

v_rndne_f64 v[2:3], v[4:5] quad_perm:[3,2,1,0]
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX13-ERR-NEXT:{{^}}v_rndne_f64 v[2:3], v[4:5] quad_perm:[3,2,1,0]
// GFX13-ERR-NEXT:{{^}}                           ^

v_floor_f64 v[2:3], v[4:5] dpp8:[7,6,5,4,3,2,1,0]
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX13-ERR-NEXT:{{^}}v_floor_f64 v[2:3], v[4:5] dpp8:[7,6,5,4,3,2,1,0]
// GFX13-ERR-NEXT:{{^}}                           ^

v_floor_f64 v[2:3], v[4:5] quad_perm:[3,2,1,0]
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX13-ERR-NEXT:{{^}}v_floor_f64 v[2:3], v[4:5] quad_perm:[3,2,1,0]
// GFX13-ERR-NEXT:{{^}}                           ^

v_frexp_exp_i32_f64 v2, v[4:5] dpp8:[7,6,5,4,3,2,1,0]
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX13-ERR-NEXT:{{^}}v_frexp_exp_i32_f64 v2, v[4:5] dpp8:[7,6,5,4,3,2,1,0]
// GFX13-ERR-NEXT:{{^}}                               ^

v_frexp_exp_i32_f64 v2, v[4:5] quad_perm:[3,2,1,0]
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX13-ERR-NEXT:{{^}}v_frexp_exp_i32_f64 v2, v[4:5] quad_perm:[3,2,1,0]
// GFX13-ERR-NEXT:{{^}}                               ^

v_frexp_mant_f64 v[2:3], v[4:5] dpp8:[7,6,5,4,3,2,1,0]
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX13-ERR-NEXT:{{^}}v_frexp_mant_f64 v[2:3], v[4:5] dpp8:[7,6,5,4,3,2,1,0]
// GFX13-ERR-NEXT:{{^}}                                ^

v_frexp_mant_f64 v[2:3], v[4:5] quad_perm:[3,2,1,0]
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX13-ERR-NEXT:{{^}}v_frexp_mant_f64 v[2:3], v[4:5] quad_perm:[3,2,1,0]
// GFX13-ERR-NEXT:{{^}}                                ^

v_fract_f64 v[2:3], v[4:5] dpp8:[7,6,5,4,3,2,1,0]
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX13-ERR-NEXT:{{^}}v_fract_f64 v[2:3], v[4:5] dpp8:[7,6,5,4,3,2,1,0]
// GFX13-ERR-NEXT:{{^}}                           ^

v_fract_f64 v[2:3], v[4:5] quad_perm:[3,2,1,0]
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX13-ERR-NEXT:{{^}}v_fract_f64 v[2:3], v[4:5] quad_perm:[3,2,1,0]
// GFX13-ERR-NEXT:{{^}}                           ^

v_rcp_f64 v[4:5], v[2:3] row_share:1
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX13-ERR-NEXT:{{^}}v_rcp_f64 v[4:5], v[2:3] row_share:1
// GFX13-ERR-NEXT:{{^}}                         ^

v_rsq_f64 v[4:5], v[2:3] row_share:1
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX13-ERR-NEXT:{{^}}v_rsq_f64 v[4:5], v[2:3] row_share:1
// GFX13-ERR-NEXT:{{^}}                         ^

v_sqrt_f64 v[4:5], v[2:3] row_share:1
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX13-ERR-NEXT:{{^}}v_sqrt_f64 v[4:5], v[2:3] row_share:1
// GFX13-ERR-NEXT:{{^}}                          ^

v_cvt_pk_f16_bf8 v1, v2 clamp
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// GFX13-ERR-NEXT:{{^}}v_cvt_pk_f16_bf8 v1, v2 clamp
// GFX13-ERR-NEXT:{{^}}                        ^

v_cvt_pk_f16_bf8 v1, v2 mul:2
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX13-ERR-NEXT:{{^}}v_cvt_pk_f16_bf8 v1, v2 mul:2
// GFX13-ERR-NEXT:{{^}}                        ^

v_cvt_pk_f16_bf8 v1, v2 row_share:1
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX13-ERR-NEXT:{{^}}v_cvt_pk_f16_bf8 v1, v2 row_share:1
// GFX13-ERR-NEXT:{{^}}                        ^

v_cvt_pk_f16_fp8 v1, v2 clamp
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// GFX13-ERR-NEXT:{{^}}v_cvt_pk_f16_fp8 v1, v2 clamp
// GFX13-ERR-NEXT:{{^}}                        ^

v_cvt_pk_f16_fp8 v1, v2 mul:2
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX13-ERR-NEXT:{{^}}v_cvt_pk_f16_fp8 v1, v2 mul:2
// GFX13-ERR-NEXT:{{^}}                        ^

v_cvt_pk_f16_fp8 v1, v2 row_share:1
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX13-ERR-NEXT:{{^}}v_cvt_pk_f16_fp8 v1, v2 row_share:1
// GFX13-ERR-NEXT:{{^}}                        ^

v_cvt_i32_f64 v2, v[4:5] row_share:1
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX13-ERR-NEXT:{{^}}v_cvt_i32_f64 v2, v[4:5] row_share:1
// GFX13-ERR-NEXT:{{^}}                         ^

v_cvt_f64_i32 v[4:5], v2 row_share:1
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX13-ERR-NEXT:{{^}}v_cvt_f64_i32 v[4:5], v2 row_share:1
// GFX13-ERR-NEXT:{{^}}                         ^

v_cvt_f32_f64 v2, v[4:5] row_share:1
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX13-ERR-NEXT:{{^}}v_cvt_f32_f64 v2, v[4:5] row_share:1
// GFX13-ERR-NEXT:{{^}}                         ^

v_cvt_f64_f32 v[4:5], v2 row_share:1
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX13-ERR-NEXT:{{^}}v_cvt_f64_f32 v[4:5], v2 row_share:1
// GFX13-ERR-NEXT:{{^}}                         ^

v_cvt_u32_f64 v2, v[4:5] row_share:1
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX13-ERR-NEXT:{{^}}v_cvt_u32_f64 v2, v[4:5] row_share:1
// GFX13-ERR-NEXT:{{^}}                         ^

v_cvt_f64_u32 v[4:5], v2 row_share:1
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX13-ERR-NEXT:{{^}}v_cvt_f64_u32 v[4:5], v2 row_share:1
// GFX13-ERR-NEXT:{{^}}                         ^

v_trunc_f64 v[2:3], v[4:5] row_share:1
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX13-ERR-NEXT:{{^}}v_trunc_f64 v[2:3], v[4:5] row_share:1
// GFX13-ERR-NEXT:{{^}}                           ^

v_ceil_f64 v[2:3], v[4:5] row_share:1
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX13-ERR-NEXT:{{^}}v_ceil_f64 v[2:3], v[4:5] row_share:1
// GFX13-ERR-NEXT:{{^}}                          ^

v_rndne_f64 v[2:3], v[4:5] row_share:1
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX13-ERR-NEXT:{{^}}v_rndne_f64 v[2:3], v[4:5] row_share:1
// GFX13-ERR-NEXT:{{^}}                           ^

v_floor_f64 v[2:3], v[4:5] row_share:1
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX13-ERR-NEXT:{{^}}v_floor_f64 v[2:3], v[4:5] row_share:1
// GFX13-ERR-NEXT:{{^}}                           ^

v_frexp_exp_i32_f64 v2, v[4:5] row_share:1
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX13-ERR-NEXT:{{^}}v_frexp_exp_i32_f64 v2, v[4:5] row_share:1
// GFX13-ERR-NEXT:{{^}}                               ^

v_frexp_mant_f64 v[2:3], v[4:5] row_share:1
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX13-ERR-NEXT:{{^}}v_frexp_mant_f64 v[2:3], v[4:5] row_share:1
// GFX13-ERR-NEXT:{{^}}                                ^

v_fract_f64 v[2:3], v[4:5] row_share:1
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX13-ERR-NEXT:{{^}}v_fract_f64 v[2:3], v[4:5] row_share:1
// GFX13-ERR-NEXT:{{^}}                           ^
