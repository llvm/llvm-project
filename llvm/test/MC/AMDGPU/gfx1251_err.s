// RUN: not llvm-mc -triple=amdgpu12.51 -filetype=null %s 2>&1 | FileCheck --check-prefixes=GFX1251-ERR --implicit-check-not=error: -strict-whitespace %s

v_mov_b64 v[4:5], v[2:3] quad_perm:[1,1,1,1]
// GFX1251-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: DP ALU dpp only supports row_share
// GFX1251-ERR: v_mov_b64 v[4:5], v[2:3] quad_perm:[1,1,1,1]
// GFX1251-ERR:                          ^

v_pk_add_f64 v[4:7], |v[8:11]|, v[12:15]
// GFX1251-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX1251-ERR: v_pk_add_f64 v[4:7], |v[8:11]|, v[12:15]
// GFX1251-ERR:                      ^

v_pk_add_f64 v[4:7], v[8:11], v[12:15] mul:2
// GFX1251-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX1251-ERR: v_pk_add_f64 v[4:7], v[8:11], v[12:15] mul:2
// GFX1251-ERR:                                        ^

v_pk_add_f64 v[4:7], v[8:11], v[12:15] row_share:2
// GFX1251-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX1251-ERR: v_pk_add_f64 v[4:7], v[8:11], v[12:15] row_share:2
// GFX1251-ERR:                                        ^

v_pk_add_f64 v[4:7], lit64(0x12345678a), v[8:11]
// GFX1251-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// GFX1251-ERR: v_pk_add_f64 v[4:7], lit64(0x12345678a), v[8:11]
// GFX1251-ERR:                            ^

v_pk_add_f64 v[4:7], v[8:11], lit64(0x12345678a)
// GFX1251-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// GFX1251-ERR: v_pk_add_f64 v[4:7], v[8:11], lit64(0x12345678a)
// GFX1251-ERR:                                     ^

v_pk_add_f64 v[4:7], v[8:11], v[12:15] op_sel:[1,0]
// GFX1251-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX1251-ERR: v_pk_add_f64 v[4:7], v[8:11], v[12:15] op_sel:[1,0]
// GFX1251-ERR:                                        ^

v_pk_add_f64 v[4:7], v[8:11], v[12:15] op_sel_hi:[1,0]
// GFX1251-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX1251-ERR: v_pk_add_f64 v[4:7], v[8:11], v[12:15] op_sel_hi:[1,0]
// GFX1251-ERR:                                        ^

v_pk_add_f64 v[4:7], v[5:8], null
// GFX1251-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid register class: vgpr tuples must be 64 bit aligned
// GFX1251-ERR: v_pk_add_f64 v[4:7], v[5:8], null

v_pk_mul_f64 v[4:7], |v[8:11]|, v[12:15]
// GFX1251-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX1251-ERR: v_pk_mul_f64 v[4:7], |v[8:11]|, v[12:15]
// GFX1251-ERR:                      ^

v_pk_mul_f64 v[4:7], v[8:11], v[12:15] mul:2
// GFX1251-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX1251-ERR: v_pk_mul_f64 v[4:7], v[8:11], v[12:15] mul:2
// GFX1251-ERR:                                        ^

v_pk_mul_f64 v[4:7], v[8:11], v[12:15] row_share:2
// GFX1251-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX1251-ERR: v_pk_mul_f64 v[4:7], v[8:11], v[12:15] row_share:2
// GFX1251-ERR:                                        ^

v_pk_mul_f64 v[4:7], lit64(0x12345678a), v[8:11]
// GFX1251-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// GFX1251-ERR: v_pk_mul_f64 v[4:7], lit64(0x12345678a), v[8:11]
// GFX1251-ERR:                            ^

v_pk_mul_f64 v[4:7], v[8:11], lit64(0x12345678a)
// GFX1251-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// GFX1251-ERR: v_pk_mul_f64 v[4:7], v[8:11], lit64(0x12345678a)
// GFX1251-ERR:                                     ^

v_pk_mul_f64 v[4:7], v[8:11], v[12:15] op_sel:[1,0]
// GFX1251-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX1251-ERR: v_pk_mul_f64 v[4:7], v[8:11], v[12:15] op_sel:[1,0]
// GFX1251-ERR:                                        ^

v_pk_mul_f64 v[4:7], v[8:11], v[12:15] op_sel_hi:[1,0]
// GFX1251-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX1251-ERR: v_pk_mul_f64 v[4:7], v[8:11], v[12:15] op_sel_hi:[1,0]
// GFX1251-ERR:                                        ^

v_pk_mul_f64 v[4:7], v[5:8], null
// GFX1251-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid register class: vgpr tuples must be 64 bit aligned
// GFX1251-ERR: v_pk_mul_f64 v[4:7], v[5:8], null

v_pk_fma_f64 v[4:7], v[8:11], v[12:15], |v[16:19]|
// GFX1251-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX1251-ERR: v_pk_fma_f64 v[4:7], v[8:11], v[12:15], |v[16:19]|
// GFX1251-ERR:                                         ^

v_pk_fma_f64 v[4:7], v[8:11], v[12:15], v[16:19] mul:2
// GFX1251-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX1251-ERR: v_pk_fma_f64 v[4:7], v[8:11], v[12:15], v[16:19] mul:2
// GFX1251-ERR:                                                  ^

v_pk_fma_f64 v[4:7], v[8:11], v[12:15], v[16:19] row_share:2
// GFX1251-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX1251-ERR: v_pk_fma_f64 v[4:7], v[8:11], v[12:15], v[16:19] row_share:2
// GFX1251-ERR:                                                  ^

v_pk_fma_f64 v[4:7], v[8:11], v[16:19], lit64(0x12345678a)
// GFX1251-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// GFX1251-ERR: v_pk_fma_f64 v[4:7], v[8:11], v[16:19], lit64(0x12345678a)
// GFX1251-ERR:                                         ^

v_pk_fma_f64 v[4:7], v[8:11], v[12:15], v[16:19] op_sel:[1,0,0]
// GFX1251-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX1251-ERR: v_pk_fma_f64 v[4:7], v[8:11], v[12:15], v[16:19] op_sel:[1,0,0]
// GFX1251-ERR:                                                  ^

v_pk_fma_f64 v[4:7], v[8:11], v[12:15], v[16:19] op_sel_hi:[1,0,0]
// GFX1251-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX1251-ERR: v_pk_fma_f64 v[4:7], v[8:11], v[12:15], v[16:19] op_sel_hi:[1,0,0]
// GFX1251-ERR:                                                  ^

v_pk_fma_f64 v[4:7], v[16:19], v[5:8], null
// GFX1251-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid register class: vgpr tuples must be 64 bit aligned
// GFX1251-ERR: v_pk_fma_f64 v[4:7], v[16:19], v[5:8], null

v_pk_max_num_f64 v[4:7], |v[8:11]|, v[12:15]
// GFX1251-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX1251-ERR: v_pk_max_num_f64 v[4:7], |v[8:11]|, v[12:15]
// GFX1251-ERR:                          ^

v_pk_max_num_f64 v[4:7], v[8:11], v[12:15] mul:2
// GFX1251-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX1251-ERR: v_pk_max_num_f64 v[4:7], v[8:11], v[12:15] mul:2
// GFX1251-ERR:                                            ^

v_pk_max_num_f64 v[4:7], v[8:11], v[12:15] row_share:2
// GFX1251-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX1251-ERR: v_pk_max_num_f64 v[4:7], v[8:11], v[12:15] row_share:2
// GFX1251-ERR:                                            ^

v_pk_max_num_f64 v[4:7], lit64(0x12345678a), v[8:11]
// GFX1251-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// GFX1251-ERR: v_pk_max_num_f64 v[4:7], lit64(0x12345678a), v[8:11]
// GFX1251-ERR:                          ^

v_pk_max_num_f64 v[4:7], v[8:11], lit64(0x12345678a)
// GFX1251-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// GFX1251-ERR: v_pk_max_num_f64 v[4:7], v[8:11], lit64(0x12345678a)
// GFX1251-ERR:                                   ^

v_pk_max_num_f64 v[4:7], v[8:11], v[12:15] op_sel:[1,0]
// GFX1251-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX1251-ERR: v_pk_max_num_f64 v[4:7], v[8:11], v[12:15] op_sel:[1,0]
// GFX1251-ERR:                                            ^

v_pk_max_num_f64 v[4:7], v[8:11], v[12:15] op_sel_hi:[1,0]
// GFX1251-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX1251-ERR: v_pk_max_num_f64 v[4:7], v[8:11], v[12:15] op_sel_hi:[1,0]
// GFX1251-ERR:                                            ^

v_pk_max_num_f64 v[4:7], v[5:8], null
// GFX1251-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid register class: vgpr tuples must be 64 bit aligned
// GFX1251-ERR: v_pk_max_num_f64 v[4:7], v[5:8], null

v_pk_min_num_f64 v[4:7], |v[8:11]|, v[12:15]
// GFX1251-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX1251-ERR: v_pk_min_num_f64 v[4:7], |v[8:11]|, v[12:15]
// GFX1251-ERR:                          ^

v_pk_min_num_f64 v[4:7], v[8:11], v[12:15] mul:2
// GFX1251-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX1251-ERR: v_pk_min_num_f64 v[4:7], v[8:11], v[12:15] mul:2
// GFX1251-ERR:                                            ^

v_pk_min_num_f64 v[4:7], v[8:11], v[12:15] row_share:2
// GFX1251-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX1251-ERR: v_pk_min_num_f64 v[4:7], v[8:11], v[12:15] row_share:2
// GFX1251-ERR:                                            ^

v_pk_min_num_f64 v[4:7], lit64(0x12345678a), v[8:11]
// GFX1251-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// GFX1251-ERR: v_pk_min_num_f64 v[4:7], lit64(0x12345678a), v[8:11]
// GFX1251-ERR:                          ^

v_pk_min_num_f64 v[4:7], v[8:11], lit64(0x12345678a)
// GFX1251-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// GFX1251-ERR: v_pk_min_num_f64 v[4:7], v[8:11], lit64(0x12345678a)
// GFX1251-ERR:                                   ^

v_pk_min_num_f64 v[4:7], v[8:11], v[12:15] op_sel:[1,0]
// GFX1251-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX1251-ERR: v_pk_min_num_f64 v[4:7], v[8:11], v[12:15] op_sel:[1,0]
// GFX1251-ERR:                                            ^

v_pk_min_num_f64 v[4:7], v[8:11], v[12:15] op_sel_hi:[1,0]
// GFX1251-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX1251-ERR: v_pk_min_num_f64 v[4:7], v[8:11], v[12:15] op_sel_hi:[1,0]
// GFX1251-ERR:                                            ^

v_pk_min_num_f64 v[4:7], v[5:8], null
// GFX1251-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid register class: vgpr tuples must be 64 bit aligned
// GFX1251-ERR: v_pk_min_num_f64 v[4:7], v[5:8], null

v_pk_add_nc_u64 v[4:7], v[8:11], v[12:15] row_share:2
// GFX1251-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX1251-ERR: v_pk_add_nc_u64 v[4:7], v[8:11], v[12:15] row_share:2
// GFX1251-ERR:                                           ^

v_pk_add_nc_u64 v[4:7], lit64(0x12345678a), v[8:11]
// GFX1251-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// GFX1251-ERR: v_pk_add_nc_u64 v[4:7], lit64(0x12345678a), v[8:11]
// GFX1251-ERR:                         ^

v_pk_add_nc_u64 v[4:7], v[8:11], lit64(0x12345678a)
// GFX1251-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// GFX1251-ERR: v_pk_add_nc_u64 v[4:7], v[8:11], lit64(0x12345678a)
// GFX1251-ERR:                                  ^

v_pk_add_nc_u64 v[4:7], v[8:11], v[12:15] op_sel:[1,0]
// GFX1251-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX1251-ERR: v_pk_add_nc_u64 v[4:7], v[8:11], v[12:15] op_sel:[1,0]
// GFX1251-ERR:                                           ^

v_pk_add_nc_u64 v[4:7], v[8:11], v[12:15] op_sel_hi:[1,0]
// GFX1251-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX1251-ERR: v_pk_add_nc_u64 v[4:7], v[8:11], v[12:15] op_sel_hi:[1,0]
// GFX1251-ERR:                                           ^

v_pk_add_nc_u64 v[4:7], v[5:8], null
// GFX1251-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid register class: vgpr tuples must be 64 bit aligned
// GFX1251-ERR: v_pk_add_nc_u64 v[4:7], v[5:8], null

v_pk_sub_nc_u64 v[4:7], v[8:11], v[12:15] row_share:2
// GFX1251-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX1251-ERR: v_pk_sub_nc_u64 v[4:7], v[8:11], v[12:15] row_share:2
// GFX1251-ERR:                                           ^

v_pk_sub_nc_u64 v[4:7], lit64(0x12345678a), v[8:11]
// GFX1251-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// GFX1251-ERR: v_pk_sub_nc_u64 v[4:7], lit64(0x12345678a), v[8:11]
// GFX1251-ERR:                         ^

v_pk_sub_nc_u64 v[4:7], v[8:11], lit64(0x12345678a)
// GFX1251-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// GFX1251-ERR: v_pk_sub_nc_u64 v[4:7], v[8:11], lit64(0x12345678a)
// GFX1251-ERR:                                  ^

v_pk_sub_nc_u64 v[4:7], v[8:11], v[12:15] op_sel:[1,0]
// GFX1251-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX1251-ERR: v_pk_sub_nc_u64 v[4:7], v[8:11], v[12:15] op_sel:[1,0]
// GFX1251-ERR:                                           ^

v_pk_sub_nc_u64 v[4:7], v[8:11], v[12:15] op_sel_hi:[1,0]
// GFX1251-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX1251-ERR: v_pk_sub_nc_u64 v[4:7], v[8:11], v[12:15] op_sel_hi:[1,0]
// GFX1251-ERR:                                           ^

v_pk_sub_nc_u64 v[4:7], v[5:8], null
// GFX1251-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid register class: vgpr tuples must be 64 bit aligned
// GFX1251-ERR: v_pk_sub_nc_u64 v[4:7], v[5:8], null

v_pk_lshl_add_u64 v[4:7], v[8:11], v[12:13], v[16:19] row_share:2
// GFX1251-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX1251-ERR: v_pk_lshl_add_u64 v[4:7], v[8:11], v[12:13], v[16:19] row_share:2
// GFX1251-ERR:                                                       ^

v_pk_lshl_add_u64 v[4:7], v[8:11], v[16:17], lit64(0x12345678a)
// GFX1251-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// GFX1251-ERR: v_pk_lshl_add_u64 v[4:7], v[8:11], v[16:17], lit64(0x12345678a)
// GFX1251-ERR:                                              ^

v_pk_lshl_add_u64 v[4:7], v[8:11], v[12:13], v[16:19] op_sel:[1,0,0]
// GFX1251-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX1251-ERR: v_pk_lshl_add_u64 v[4:7], v[8:11], v[12:13], v[16:19] op_sel:[1,0,0]
// GFX1251-ERR:                                                       ^

v_pk_lshl_add_u64 v[4:7], v[8:11], v[12:13], v[16:19] op_sel_hi:[1,0,0]
// GFX1251-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX1251-ERR: v_pk_lshl_add_u64 v[4:7], v[8:11], v[12:13], v[16:19] op_sel_hi:[1,0,0]
// GFX1251-ERR:                                                       ^

v_pk_lshl_add_u64 v[4:7], v[8:11], v[12:13], v[16:19] clamp
// GFX1251-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// GFX1251-ERR: v_pk_lshl_add_u64 v[4:7], v[8:11], v[12:13], v[16:19] clamp
// GFX1251-ERR:                                                       ^

v_pk_lshl_add_u64 v[4:7], v[8:11], v[12:13], v[16:19] neg_lo:[1,0,0]
// GFX1251-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX1251-ERR: v_pk_lshl_add_u64 v[4:7], v[8:11], v[12:13], v[16:19] neg_lo:[1,0,0]
// GFX1251-ERR:                                                       ^

v_pk_lshl_add_u64 v[4:7], v[17:20], v[6:7], null
// GFX1251-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid register class: vgpr tuples must be 64 bit aligned
// GFX1251-ERR: v_pk_lshl_add_u64 v[4:7], v[17:20], v[6:7], null
