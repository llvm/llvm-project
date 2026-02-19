// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx1251 -filetype=null %s 2>&1 | FileCheck --check-prefix=GFX1251-ERR --implicit-check-not=error: --strict-whitespace %s

v_mov_b64 v[4:5], v[2:3] dpp8:[7,6,5,4,3,2,1,0]
// GFX1251-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX1251-ERR-NEXT:{{^}}v_mov_b64 v[4:5], v[2:3] dpp8:[7,6,5,4,3,2,1,0]
// GFX1251-ERR-NEXT:{{^}}                         ^

v_cvt_i32_f64 v2, v[4:5] dpp8:[7,6,5,4,3,2,1,0]
// GFX1251-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX1251-ERR-NEXT:{{^}}v_cvt_i32_f64 v2, v[4:5] dpp8:[7,6,5,4,3,2,1,0]
// GFX1251-ERR-NEXT:{{^}}                         ^

v_cvt_f64_i32 v[4:5], v2 dpp8:[7,6,5,4,3,2,1,0]
// GFX1251-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX1251-ERR-NEXT:{{^}}v_cvt_f64_i32 v[4:5], v2 dpp8:[7,6,5,4,3,2,1,0]
// GFX1251-ERR-NEXT:{{^}}                         ^

v_cvt_f32_f64 v2, v[4:5] dpp8:[7,6,5,4,3,2,1,0]
// GFX1251-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX1251-ERR-NEXT:{{^}}v_cvt_f32_f64 v2, v[4:5] dpp8:[7,6,5,4,3,2,1,0]
// GFX1251-ERR-NEXT:{{^}}                         ^

v_cvt_f64_f32 v[4:5], v2 dpp8:[7,6,5,4,3,2,1,0]
// GFX1251-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX1251-ERR-NEXT:{{^}}v_cvt_f64_f32 v[4:5], v2 dpp8:[7,6,5,4,3,2,1,0]
// GFX1251-ERR-NEXT:{{^}}                         ^

v_cvt_u32_f64 v2, v[4:5] dpp8:[7,6,5,4,3,2,1,0]
// GFX1251-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX1251-ERR-NEXT:{{^}}v_cvt_u32_f64 v2, v[4:5] dpp8:[7,6,5,4,3,2,1,0]
// GFX1251-ERR-NEXT:{{^}}                         ^

v_cvt_f64_u32 v[4:5], v2 dpp8:[7,6,5,4,3,2,1,0]
// GFX1251-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX1251-ERR-NEXT:{{^}}v_cvt_f64_u32 v[4:5], v2 dpp8:[7,6,5,4,3,2,1,0]
// GFX1251-ERR-NEXT:{{^}}                         ^

v_trunc_f64 v[2:3], v[4:5] dpp8:[7,6,5,4,3,2,1,0]
// GFX1251-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX1251-ERR-NEXT:{{^}}v_trunc_f64 v[2:3], v[4:5] dpp8:[7,6,5,4,3,2,1,0]
// GFX1251-ERR-NEXT:{{^}}                           ^

v_ceil_f64 v[2:3], v[4:5] dpp8:[7,6,5,4,3,2,1,0]
// GFX1251-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX1251-ERR-NEXT:{{^}}v_ceil_f64 v[2:3], v[4:5] dpp8:[7,6,5,4,3,2,1,0]
// GFX1251-ERR-NEXT:{{^}}                          ^

v_rndne_f64 v[2:3], v[4:5] dpp8:[7,6,5,4,3,2,1,0]
// GFX1251-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX1251-ERR-NEXT:{{^}}v_rndne_f64 v[2:3], v[4:5] dpp8:[7,6,5,4,3,2,1,0]
// GFX1251-ERR-NEXT:{{^}}                           ^

v_floor_f64 v[2:3], v[4:5] dpp8:[7,6,5,4,3,2,1,0]
// GFX1251-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX1251-ERR-NEXT:{{^}}v_floor_f64 v[2:3], v[4:5] dpp8:[7,6,5,4,3,2,1,0]
// GFX1251-ERR-NEXT:{{^}}                           ^

v_frexp_exp_i32_f64 v2, v[4:5] dpp8:[7,6,5,4,3,2,1,0]
// GFX1251-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX1251-ERR-NEXT:{{^}}v_frexp_exp_i32_f64 v2, v[4:5] dpp8:[7,6,5,4,3,2,1,0]
// GFX1251-ERR-NEXT:{{^}}                               ^

v_frexp_mant_f64 v[2:3], v[4:5] dpp8:[7,6,5,4,3,2,1,0]
// GFX1251-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX1251-ERR-NEXT:{{^}}v_frexp_mant_f64 v[2:3], v[4:5] dpp8:[7,6,5,4,3,2,1,0]
// GFX1251-ERR-NEXT:{{^}}                                ^

v_fract_f64 v[2:3], v[4:5] dpp8:[7,6,5,4,3,2,1,0]
// GFX1251-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX1251-ERR-NEXT:{{^}}v_fract_f64 v[2:3], v[4:5] dpp8:[7,6,5,4,3,2,1,0]
// GFX1251-ERR-NEXT:{{^}}                           ^

v_mov_b64 v[4:5], v[2:3] quad_perm:[3,2,1,0]
// GFX1251-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: DP ALU dpp only supports row_share
// GFX1251-ERR-NEXT:{{^}}v_mov_b64 v[4:5], v[2:3] quad_perm:[3,2,1,0]
// GFX1251-ERR-NEXT:{{^}}                         ^

v_cvt_i32_f64 v2, v[4:5] quad_perm:[3,2,1,0]
// GFX1251-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: DP ALU dpp only supports row_share
// GFX1251-ERR-NEXT:{{^}}v_cvt_i32_f64 v2, v[4:5] quad_perm:[3,2,1,0]
// GFX1251-ERR-NEXT:{{^}}                         ^

v_cvt_f64_i32 v[4:5], v2 quad_perm:[3,2,1,0]
// GFX1251-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: DP ALU dpp only supports row_share
// GFX1251-ERR-NEXT:{{^}}v_cvt_f64_i32 v[4:5], v2 quad_perm:[3,2,1,0]
// GFX1251-ERR-NEXT:{{^}}                         ^

v_cvt_f32_f64 v2, v[4:5] quad_perm:[3,2,1,0]
// GFX1251-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: DP ALU dpp only supports row_share
// GFX1251-ERR-NEXT:{{^}}v_cvt_f32_f64 v2, v[4:5] quad_perm:[3,2,1,0]
// GFX1251-ERR-NEXT:{{^}}                         ^

v_cvt_f64_f32 v[4:5], v2 quad_perm:[3,2,1,0]
// GFX1251-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: DP ALU dpp only supports row_share
// GFX1251-ERR-NEXT:{{^}}v_cvt_f64_f32 v[4:5], v2 quad_perm:[3,2,1,0]
// GFX1251-ERR-NEXT:{{^}}                         ^

v_cvt_u32_f64 v2, v[4:5] quad_perm:[3,2,1,0]
// GFX1251-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: DP ALU dpp only supports row_share
// GFX1251-ERR-NEXT:{{^}}v_cvt_u32_f64 v2, v[4:5] quad_perm:[3,2,1,0]
// GFX1251-ERR-NEXT:{{^}}                         ^

v_cvt_f64_u32 v[4:5], v2 quad_perm:[3,2,1,0]
// GFX1251-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: DP ALU dpp only supports row_share
// GFX1251-ERR-NEXT:{{^}}v_cvt_f64_u32 v[4:5], v2 quad_perm:[3,2,1,0]
// GFX1251-ERR-NEXT:{{^}}                         ^

v_trunc_f64 v[2:3], v[4:5] quad_perm:[3,2,1,0]
// GFX1251-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: DP ALU dpp only supports row_share
// GFX1251-ERR-NEXT:{{^}}v_trunc_f64 v[2:3], v[4:5] quad_perm:[3,2,1,0]
// GFX1251-ERR-NEXT:{{^}}                           ^

v_ceil_f64 v[2:3], v[4:5] quad_perm:[3,2,1,0]
// GFX1251-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: DP ALU dpp only supports row_share
// GFX1251-ERR-NEXT:{{^}}v_ceil_f64 v[2:3], v[4:5] quad_perm:[3,2,1,0]
// GFX1251-ERR-NEXT:{{^}}                          ^

v_rndne_f64 v[2:3], v[4:5] quad_perm:[3,2,1,0]
// GFX1251-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: DP ALU dpp only supports row_share
// GFX1251-ERR-NEXT:{{^}}v_rndne_f64 v[2:3], v[4:5] quad_perm:[3,2,1,0]
// GFX1251-ERR-NEXT:{{^}}                           ^

v_floor_f64 v[2:3], v[4:5] quad_perm:[3,2,1,0]
// GFX1251-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: DP ALU dpp only supports row_share
// GFX1251-ERR-NEXT:{{^}}v_floor_f64 v[2:3], v[4:5] quad_perm:[3,2,1,0]
// GFX1251-ERR-NEXT:{{^}}                           ^

v_frexp_exp_i32_f64 v2, v[4:5] quad_perm:[3,2,1,0]
// GFX1251-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: DP ALU dpp only supports row_share
// GFX1251-ERR-NEXT:{{^}}v_frexp_exp_i32_f64 v2, v[4:5] quad_perm:[3,2,1,0]
// GFX1251-ERR-NEXT:{{^}}                               ^

v_frexp_mant_f64 v[2:3], v[4:5] quad_perm:[3,2,1,0]
// GFX1251-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: DP ALU dpp only supports row_share
// GFX1251-ERR-NEXT:{{^}}v_frexp_mant_f64 v[2:3], v[4:5] quad_perm:[3,2,1,0]
// GFX1251-ERR-NEXT:{{^}}                                ^

v_fract_f64 v[2:3], v[4:5] quad_perm:[3,2,1,0]
// GFX1251-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: DP ALU dpp only supports row_share
// GFX1251-ERR-NEXT:{{^}}v_fract_f64 v[2:3], v[4:5] quad_perm:[3,2,1,0]
// GFX1251-ERR-NEXT:{{^}}                           ^

v_rcp_f64 v[4:5], v[2:3] row_share:1
// GFX1251-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX1251-ERR-NEXT:{{^}}v_rcp_f64 v[4:5], v[2:3] row_share:1
// GFX1251-ERR-NEXT:{{^}}                         ^

v_rsq_f64 v[4:5], v[2:3] row_share:1
// GFX1251-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX1251-ERR-NEXT:{{^}}v_rsq_f64 v[4:5], v[2:3] row_share:1
// GFX1251-ERR-NEXT:{{^}}                         ^

v_sqrt_f64 v[4:5], v[2:3] row_share:1
// GFX1251-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX1251-ERR-NEXT:{{^}}v_sqrt_f64 v[4:5], v[2:3] row_share:1
// GFX1251-ERR-NEXT:{{^}}                          ^
