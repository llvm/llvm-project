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
// GFX1251-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// GFX1251-ERR: v_pk_add_f64 v[4:7], v[5:8], null

v_pk_add_f64 v[4:7], v[8:11], v[13:16]
// GFX1251-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// GFX1251-ERR: v_pk_add_f64 v[4:7], v[8:11], v[13:16]

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
// GFX1251-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
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
// GFX1251-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// GFX1251-ERR: v_pk_fma_f64 v[4:7], v[16:19], v[5:8], null

v_pk_fma_f64 v[4:7], v[8:11], v[12:15], v[17:20]
// GFX1251-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// GFX1251-ERR: v_pk_fma_f64 v[4:7], v[8:11], v[12:15], v[17:20]

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
// GFX1251-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
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
// GFX1251-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
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
// GFX1251-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
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
// GFX1251-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
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
// GFX1251-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// GFX1251-ERR: v_pk_lshl_add_u64 v[4:7], v[17:20], v[6:7], null

v_pk_lshl_add_u64 v[4:7], v[8:11], v[12:13], v[17:20]
// GFX1251-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// GFX1251-ERR: v_pk_lshl_add_u64 v[4:7], v[8:11], v[12:13], v[17:20]

v_swmmac_f32_16x16x32_f16 v[12:19], v[0:3], v[4:11], v20
// GFX1251-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU (gfx1251): v_swmmac_f32_16x16x32_f16
// GFX1251-ERR: v_swmmac_f32_16x16x32_f16 v[12:19], v[0:3], v[4:11], v20
// GFX1251-ERR: ^

v_swmmac_f32_16x16x32_bf16 v[12:19], v[0:3], v[4:11], v20
// GFX1251-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU (gfx1251): v_swmmac_f32_16x16x32_bf16
// GFX1251-ERR: v_swmmac_f32_16x16x32_bf16 v[12:19], v[0:3], v[4:11], v20
// GFX1251-ERR: ^

v_swmmac_f16_16x16x32_f16 v[12:15], v[0:3], v[4:11], v16
// GFX1251-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU (gfx1251): v_swmmac_f16_16x16x32_f16
// GFX1251-ERR: v_swmmac_f16_16x16x32_f16 v[12:15], v[0:3], v[4:11], v16
// GFX1251-ERR: ^

v_swmmac_bf16_16x16x32_bf16 v[12:15], v[0:3], v[4:11], v16
// GFX1251-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU (gfx1251): v_swmmac_bf16_16x16x32_bf16
// GFX1251-ERR: v_swmmac_bf16_16x16x32_bf16 v[12:15], v[0:3], v[4:11], v16
// GFX1251-ERR: ^

v_swmmac_i32_16x16x32_iu8 v[6:13], v[0:1], v[2:5], v14
// GFX1251-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU (gfx1251): v_swmmac_i32_16x16x32_iu8
// GFX1251-ERR: v_swmmac_i32_16x16x32_iu8 v[6:13], v[0:1], v[2:5], v14
// GFX1251-ERR: ^

v_swmmac_i32_16x16x32_iu4 v[3:10], v0, v[1:2], v11
// GFX1251-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU (gfx1251): v_swmmac_i32_16x16x32_iu4
// GFX1251-ERR: v_swmmac_i32_16x16x32_iu4 v[3:10], v0, v[1:2], v11
// GFX1251-ERR: ^

v_swmmac_i32_16x16x64_iu4 v[6:13], v[0:1], v[2:5], v14
// GFX1251-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU (gfx1251): v_swmmac_i32_16x16x64_iu4
// GFX1251-ERR: v_swmmac_i32_16x16x64_iu4 v[6:13], v[0:1], v[2:5], v14
// GFX1251-ERR: ^

v_swmmac_f32_16x16x32_fp8_fp8 v[6:13], v[0:1], v[2:5], v14
// GFX1251-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU (gfx1251): v_swmmac_f32_16x16x32_fp8_fp8
// GFX1251-ERR: v_swmmac_f32_16x16x32_fp8_fp8 v[6:13], v[0:1], v[2:5], v14
// GFX1251-ERR: ^

v_swmmac_f32_16x16x32_fp8_bf8 v[6:13], v[0:1], v[2:5], v14
// GFX1251-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU (gfx1251): v_swmmac_f32_16x16x32_fp8_bf8
// GFX1251-ERR: v_swmmac_f32_16x16x32_fp8_bf8 v[6:13], v[0:1], v[2:5], v14
// GFX1251-ERR: ^

v_swmmac_f32_16x16x32_bf8_fp8 v[6:13], v[0:1], v[2:5], v14
// GFX1251-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU (gfx1251): v_swmmac_f32_16x16x32_bf8_fp8
// GFX1251-ERR: v_swmmac_f32_16x16x32_bf8_fp8 v[6:13], v[0:1], v[2:5], v14
// GFX1251-ERR: ^

v_swmmac_f32_16x16x32_bf8_bf8 v[6:13], v[0:1], v[2:5], v14
// GFX1251-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU (gfx1251): v_swmmac_f32_16x16x32_bf8_bf8
// GFX1251-ERR: v_swmmac_f32_16x16x32_bf8_bf8 v[6:13], v[0:1], v[2:5], v14
// GFX1251-ERR: ^
