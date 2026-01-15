// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx1251 -filetype=null %s 2>&1 | FileCheck --check-prefix=GFX1251-ERR --implicit-check-not=error: --strict-whitespace %s

v_add_nc_u64 v[2:3], v[2:3], v[4:5] dpp8:[7,6,5,4,3,2,1,0]
// GFX1251-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX1251-ERR-NEXT:{{^}}v_add_nc_u64 v[2:3], v[2:3], v[4:5] dpp8:[7,6,5,4,3,2,1,0]
// GFX1251-ERR-NEXT:{{^}}                                    ^

v_sub_nc_u64 v[2:3], v[2:3], v[4:5] dpp8:[7,6,5,4,3,2,1,0]
// GFX1251-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX1251-ERR-NEXT:{{^}}v_sub_nc_u64 v[2:3], v[2:3], v[4:5] dpp8:[7,6,5,4,3,2,1,0]
// GFX1251-ERR-NEXT:{{^}}                                    ^

v_fmac_f64 v[4:5], v[2:3], v[4:5] dpp8:[7,6,5,4,3,2,1,0]
// GFX1251-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX1251-ERR-NEXT:{{^}}v_fmac_f64 v[4:5], v[2:3], v[4:5] dpp8:[7,6,5,4,3,2,1,0]
// GFX1251-ERR-NEXT:{{^}}                                  ^

v_add_f64 v[4:5], v[2:3], v[4:5] dpp8:[7,6,5,4,3,2,1,0]
// GFX1251-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX1251-ERR-NEXT:{{^}}v_add_f64 v[4:5], v[2:3], v[4:5] dpp8:[7,6,5,4,3,2,1,0]
// GFX1251-ERR-NEXT:{{^}}                                 ^

v_mul_f64 v[4:5], v[2:3], v[4:5] dpp8:[7,6,5,4,3,2,1,0]
// GFX1251-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX1251-ERR-NEXT:{{^}}v_mul_f64 v[4:5], v[2:3], v[4:5] dpp8:[7,6,5,4,3,2,1,0]
// GFX1251-ERR-NEXT:{{^}}                                 ^

v_max_num_f64 v[4:5], v[2:3], v[4:5] dpp8:[7,6,5,4,3,2,1,0]
// GFX1251-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX1251-ERR-NEXT:{{^}}v_max_num_f64 v[4:5], v[2:3], v[4:5] dpp8:[7,6,5,4,3,2,1,0]
// GFX1251-ERR-NEXT:{{^}}                                     ^

v_min_num_f64 v[4:5], v[2:3], v[4:5] dpp8:[7,6,5,4,3,2,1,0]
// GFX1251-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX1251-ERR-NEXT:{{^}}v_min_num_f64 v[4:5], v[2:3], v[4:5] dpp8:[7,6,5,4,3,2,1,0]
// GFX1251-ERR-NEXT:{{^}}                                     ^

v_lshlrev_b64 v[4:5], v2, v[4:5] dpp8:[7,6,5,4,3,2,1,0]
// GFX1251-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX1251-ERR-NEXT:{{^}}v_lshlrev_b64 v[4:5], v2, v[4:5] dpp8:[7,6,5,4,3,2,1,0]
// GFX1251-ERR-NEXT:{{^}}                                 ^

v_fmamk_f64 v[4:5], v[2:3], 123.0, v[6:7] dpp8:[7,6,5,4,3,2,1,0]
// GFX1251-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX1251-ERR-NEXT:{{^}}v_fmamk_f64 v[4:5], v[2:3], 123.0, v[6:7] dpp8:[7,6,5,4,3,2,1,0]
// GFX1251-ERR-NEXT:{{^}}                                          ^

v_fmaak_f64 v[4:5], v[2:3], v[6:7], 123.0 dpp8:[7,6,5,4,3,2,1,0]
// GFX1251-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX1251-ERR-NEXT:{{^}}v_fmaak_f64 v[4:5], v[2:3], v[6:7], 123.0 dpp8:[7,6,5,4,3,2,1,0]
// GFX1251-ERR-NEXT:{{^}}                                          ^

v_add_nc_u64 v[4:5], v[2:3], v[4:5] quad_perm:[3,2,1,0]
// GFX1251-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: DP ALU dpp only supports row_share
// GFX1251-ERR-NEXT:{{^}}v_add_nc_u64 v[4:5], v[2:3], v[4:5] quad_perm:[3,2,1,0]
// GFX1251-ERR-NEXT:{{^}}                                    ^

v_sub_nc_u64 v[4:5], v[2:3], v[4:5] quad_perm:[3,2,1,0]
// GFX1251-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: DP ALU dpp only supports row_share
// GFX1251-ERR-NEXT:{{^}}v_sub_nc_u64 v[4:5], v[2:3], v[4:5] quad_perm:[3,2,1,0]
// GFX1251-ERR-NEXT:{{^}}                                    ^

v_fmac_f64 v[4:5], v[2:3], v[4:5] quad_perm:[3,2,1,0]
// GFX1251-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: DP ALU dpp only supports row_share
// GFX1251-ERR-NEXT:{{^}}v_fmac_f64 v[4:5], v[2:3], v[4:5] quad_perm:[3,2,1,0]
// GFX1251-ERR-NEXT:{{^}}                                  ^

v_add_f64 v[4:5], v[2:3], v[4:5] quad_perm:[3,2,1,0]
// GFX1251-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: DP ALU dpp only supports row_share
// GFX1251-ERR-NEXT:{{^}}v_add_f64 v[4:5], v[2:3], v[4:5] quad_perm:[3,2,1,0]
// GFX1251-ERR-NEXT:{{^}}                                 ^

v_mul_f64 v[4:5], v[2:3], v[4:5] quad_perm:[3,2,1,0]
// GFX1251-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: DP ALU dpp only supports row_share
// GFX1251-ERR-NEXT:{{^}}v_mul_f64 v[4:5], v[2:3], v[4:5] quad_perm:[3,2,1,0]
// GFX1251-ERR-NEXT:{{^}}                                 ^

v_max_num_f64 v[4:5], v[2:3], v[4:5] quad_perm:[3,2,1,0]
// GFX1251-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: DP ALU dpp only supports row_share
// GFX1251-ERR-NEXT:{{^}}v_max_num_f64 v[4:5], v[2:3], v[4:5] quad_perm:[3,2,1,0]
// GFX1251-ERR-NEXT:{{^}}                                     ^

v_min_num_f64 v[4:5], v[2:3], v[4:5] quad_perm:[3,2,1,0]
// GFX1251-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: DP ALU dpp only supports row_share
// GFX1251-ERR-NEXT:{{^}}v_min_num_f64 v[4:5], v[2:3], v[4:5] quad_perm:[3,2,1,0]
// GFX1251-ERR-NEXT:{{^}}                                     ^

v_lshlrev_b64 v[4:5], v2, v[4:5] quad_perm:[3,2,1,0]
// GFX1251-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: DP ALU dpp only supports row_share
// GFX1251-ERR-NEXT:{{^}}v_lshlrev_b64 v[4:5], v2, v[4:5] quad_perm:[3,2,1,0]
// GFX1251-ERR-NEXT:{{^}}                                 ^

v_mul_u64 v[2:3], v[4:5], v[6:7] row_share:1
// GFX1251-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX1251-ERR-NEXT:{{^}}v_mul_u64 v[2:3], v[4:5], v[6:7] row_share:1
// GFX1251-ERR-NEXT:{{^}}                                 ^

v_fmamk_f64 v[4:5], v[2:3], 123.0, v[6:7] row_share:1
// GFX1251-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX1251-ERR-NEXT:{{^}}v_fmamk_f64 v[4:5], v[2:3], 123.0, v[6:7] row_share:1
// GFX1251-ERR-NEXT:{{^}}                                          ^

v_fmaak_f64 v[4:5], v[2:3], v[6:7], 123.0 row_share:1
// GFX1251-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX1251-ERR-NEXT:{{^}}v_fmaak_f64 v[4:5], v[2:3], v[6:7], 123.0 row_share:1
// GFX1251-ERR-NEXT:{{^}}                                          ^
