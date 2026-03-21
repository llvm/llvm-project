// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx1251 -filetype=null %s 2>&1 | FileCheck --check-prefix=GFX1251-ERR --implicit-check-not=error: --strict-whitespace %s

v_add_nc_u64_e64_dpp v[2:3], v[2:3], v[4:5] dpp8:[7,6,5,4,3,2,1,0]
// GFX1251-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX1251-ERR-NEXT:{{^}}v_add_nc_u64_e64_dpp v[2:3], v[2:3], v[4:5] dpp8:[7,6,5,4,3,2,1,0]
// GFX1251-ERR-NEXT:{{^}}                                            ^

v_sub_nc_u64_e64_dpp v[2:3], v[2:3], v[4:5] dpp8:[7,6,5,4,3,2,1,0]
// GFX1251-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX1251-ERR-NEXT:{{^}}v_sub_nc_u64_e64_dpp v[2:3], v[2:3], v[4:5] dpp8:[7,6,5,4,3,2,1,0]
// GFX1251-ERR-NEXT:{{^}}                                            ^

v_fmac_f64_e64_dpp v[4:5], v[2:3], v[4:5] dpp8:[7,6,5,4,3,2,1,0]
// GFX1251-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX1251-ERR-NEXT:{{^}}v_fmac_f64_e64_dpp v[4:5], v[2:3], v[4:5] dpp8:[7,6,5,4,3,2,1,0]
// GFX1251-ERR-NEXT:{{^}}                                          ^

v_add_f64_e64_dpp v[4:5], v[2:3], v[4:5] dpp8:[7,6,5,4,3,2,1,0]
// GFX1251-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX1251-ERR-NEXT:{{^}}v_add_f64_e64_dpp v[4:5], v[2:3], v[4:5] dpp8:[7,6,5,4,3,2,1,0]
// GFX1251-ERR-NEXT:{{^}}                                         ^

v_mul_f64_e64_dpp v[4:5], v[2:3], v[4:5] dpp8:[7,6,5,4,3,2,1,0]
// GFX1251-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX1251-ERR-NEXT:{{^}}v_mul_f64_e64_dpp v[4:5], v[2:3], v[4:5] dpp8:[7,6,5,4,3,2,1,0]
// GFX1251-ERR-NEXT:{{^}}                                         ^

v_max_num_f64_e64_dpp v[4:5], v[2:3], v[4:5] dpp8:[7,6,5,4,3,2,1,0]
// GFX1251-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX1251-ERR-NEXT:{{^}}v_max_num_f64_e64_dpp v[4:5], v[2:3], v[4:5] dpp8:[7,6,5,4,3,2,1,0]
// GFX1251-ERR-NEXT:{{^}}                                             ^

v_min_num_f64_e64_dpp v[4:5], v[2:3], v[4:5] dpp8:[7,6,5,4,3,2,1,0]
// GFX1251-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX1251-ERR-NEXT:{{^}}v_min_num_f64_e64_dpp v[4:5], v[2:3], v[4:5] dpp8:[7,6,5,4,3,2,1,0]
// GFX1251-ERR-NEXT:{{^}}                                             ^

v_lshlrev_b64_e64_dpp v[4:5], v2, v[4:5] dpp8:[7,6,5,4,3,2,1,0]
// GFX1251-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX1251-ERR-NEXT:{{^}}v_lshlrev_b64_e64_dpp v[4:5], v2, v[4:5] dpp8:[7,6,5,4,3,2,1,0]
// GFX1251-ERR-NEXT:{{^}}                                         ^

v_add_nc_u64_e64_dpp v[4:5], v[2:3], v[4:5] quad_perm:[3,2,1,0]
// GFX1251-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: DP ALU dpp only supports row_share
// GFX1251-ERR-NEXT:{{^}}v_add_nc_u64_e64_dpp v[4:5], v[2:3], v[4:5] quad_perm:[3,2,1,0]
// GFX1251-ERR-NEXT:{{^}}                                            ^

v_sub_nc_u64_e64_dpp v[4:5], v[2:3], v[4:5] quad_perm:[3,2,1,0]
// GFX1251-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: DP ALU dpp only supports row_share
// GFX1251-ERR-NEXT:{{^}}v_sub_nc_u64_e64_dpp v[4:5], v[2:3], v[4:5] quad_perm:[3,2,1,0]
// GFX1251-ERR-NEXT:{{^}}                                            ^

v_fmac_f64_e64_dpp v[4:5], v[2:3], v[4:5] quad_perm:[3,2,1,0]
// GFX1251-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: DP ALU dpp only supports row_share
// GFX1251-ERR-NEXT:{{^}}v_fmac_f64_e64_dpp v[4:5], v[2:3], v[4:5] quad_perm:[3,2,1,0]
// GFX1251-ERR-NEXT:{{^}}                                          ^

v_add_f64_e64_dpp v[4:5], v[2:3], v[4:5] quad_perm:[3,2,1,0]
// GFX1251-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: DP ALU dpp only supports row_share
// GFX1251-ERR-NEXT:{{^}}v_add_f64_e64_dpp v[4:5], v[2:3], v[4:5] quad_perm:[3,2,1,0]
// GFX1251-ERR-NEXT:{{^}}                                         ^

v_mul_f64_e64_dpp v[4:5], v[2:3], v[4:5] quad_perm:[3,2,1,0]
// GFX1251-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: DP ALU dpp only supports row_share
// GFX1251-ERR-NEXT:{{^}}v_mul_f64_e64_dpp v[4:5], v[2:3], v[4:5] quad_perm:[3,2,1,0]
// GFX1251-ERR-NEXT:{{^}}                                         ^

v_max_num_f64_e64_dpp v[4:5], v[2:3], v[4:5] quad_perm:[3,2,1,0]
// GFX1251-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: DP ALU dpp only supports row_share
// GFX1251-ERR-NEXT:{{^}}v_max_num_f64_e64_dpp v[4:5], v[2:3], v[4:5] quad_perm:[3,2,1,0]
// GFX1251-ERR-NEXT:{{^}}                                             ^

v_min_num_f64_e64_dpp v[4:5], v[2:3], v[4:5] quad_perm:[3,2,1,0]
// GFX1251-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: DP ALU dpp only supports row_share
// GFX1251-ERR-NEXT:{{^}}v_min_num_f64_e64_dpp v[4:5], v[2:3], v[4:5] quad_perm:[3,2,1,0]
// GFX1251-ERR-NEXT:{{^}}                                             ^

v_lshlrev_b64_e64_dpp v[4:5], v2, v[4:5] quad_perm:[3,2,1,0]
// GFX1251-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: DP ALU dpp only supports row_share
// GFX1251-ERR-NEXT:{{^}}v_lshlrev_b64_e64_dpp v[4:5], v2, v[4:5] quad_perm:[3,2,1,0]
// GFX1251-ERR-NEXT:{{^}}                                         ^

v_fmaak_f32_e64_dpp v4, v2, v6, 3 row_share:1
// GFX1251-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_fmamk_f32_e64_dpp v4, v2, 3, v6 row_share:1
// GFX1251-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_fmaak_f16_e64_dpp v4, v2, v6, 3 row_share:1
// GFX1251-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_fmamk_f16_e64_dpp v4, v2, 3, v6 row_share:1
// GFX1251-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported
