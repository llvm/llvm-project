// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx1300 -mattr=+wavefrontsize32 -show-encoding %s 2>&1 | FileCheck --check-prefixes=GFX13,W32 --strict-whitespace --implicit-check-not=error %s
// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx1300 -mattr=+wavefrontsize64 -show-encoding %s 2>&1 | FileCheck --check-prefixes=GFX13,W64 --strict-whitespace --implicit-check-not=error %s

v_ashr_pk_i8_i32 v1, v2, v3, v4 clamp
// GFX13: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// GFX13-NEXT:{{^}}v_ashr_pk_i8_i32 v1, v2, v3, v4 clamp
// GFX13-NEXT:{{^}}                                ^

v_ashr_pk_u8_i32 v1, v2, v3, v4 clamp
// GFX13: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// GFX13-NEXT:{{^}}v_ashr_pk_u8_i32 v1, v2, v3, v4 clamp
// GFX13-NEXT:{{^}}                                ^

v_ashrrev_i64 v[4:5], v2, v[6:7] dpp8:[7,6,5,4,3,2,1,0]
// GFX13: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX13-NEXT:{{^}}v_ashrrev_i64 v[4:5], v2, v[6:7] dpp8:[7,6,5,4,3,2,1,0]
// GFX13-NEXT:{{^}}                                 ^

v_ashrrev_i64 v[4:5], v2, v[6:7] quad_perm:[3,2,1,0]
// GFX13: :[[@LINE-1]]:{{[0-9]+}}: error: DP ALU dpp only supports row_share
// GFX13-NEXT:{{^}}v_ashrrev_i64 v[4:5], v2, v[6:7] quad_perm:[3,2,1,0]
// GFX13-NEXT:{{^}}                                 ^

v_cvt_sr_bf8_f16 v1, v2, v3 clamp
// GFX13: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// GFX13-NEXT:{{^}}v_cvt_sr_bf8_f16 v1, v2, v3 clamp
// GFX13-NEXT:{{^}}                            ^

v_cvt_sr_bf8_f16 v1, v2, v3 mul:2
// GFX13: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX13-NEXT:{{^}}v_cvt_sr_bf8_f16 v1, v2, v3 mul:2
// GFX13-NEXT:{{^}}                            ^

v_cvt_sr_fp8_f16 v1, v2, v3 clamp
// GFX13: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// GFX13-NEXT:{{^}}v_cvt_sr_fp8_f16 v1, v2, v3 clamp
// GFX13-NEXT:{{^}}                            ^

v_cvt_sr_fp8_f16 v1, v2, v3 mul:2
// GFX13: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX13-NEXT:{{^}}v_cvt_sr_fp8_f16 v1, v2, v3 mul:2
// GFX13-NEXT:{{^}}                            ^

v_div_fixup_f64 v[4:5], v[2:3], v[6:7], v[8:9] dpp8:[7,6,5,4,3,2,1,0]
// GFX13: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX13-NEXT:{{^}}v_div_fixup_f64 v[4:5], v[2:3], v[6:7], v[8:9] dpp8:[7,6,5,4,3,2,1,0]
// GFX13-NEXT:{{^}}                                               ^

v_div_fixup_f64 v[4:5], v[2:3], v[6:7], v[8:9] quad_perm:[3,2,1,0]
// GFX13: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX13-NEXT:{{^}}v_div_fixup_f64 v[4:5], v[2:3], v[6:7], v[8:9] quad_perm:[3,2,1,0]
// GFX13-NEXT:{{^}}                                               ^

v_div_fixup_f64 v[4:5], v[2:3], v[6:7], v[8:9] row_share:1
// GFX13: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX13-NEXT:{{^}}v_div_fixup_f64 v[4:5], v[2:3], v[6:7], v[8:9] row_share:1
// GFX13-NEXT:{{^}}                                               ^

v_div_fmas_f64 v[4:5], v[2:3], v[6:7], v[8:9] dpp8:[7,6,5,4,3,2,1,0]
// GFX13: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX13-NEXT:{{^}}v_div_fmas_f64 v[4:5], v[2:3], v[6:7], v[8:9] dpp8:[7,6,5,4,3,2,1,0]
// GFX13-NEXT:{{^}}                                              ^

v_div_fmas_f64 v[4:5], v[2:3], v[6:7], v[8:9] quad_perm:[3,2,1,0]
// GFX13: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX13-NEXT:{{^}}v_div_fmas_f64 v[4:5], v[2:3], v[6:7], v[8:9] quad_perm:[3,2,1,0]
// GFX13-NEXT:{{^}}                                              ^

v_div_fmas_f64 v[4:5], v[2:3], v[6:7], v[8:9] row_share:1
// GFX13: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX13-NEXT:{{^}}v_div_fmas_f64 v[4:5], v[2:3], v[6:7], v[8:9] row_share:1
// GFX13-NEXT:{{^}}                                              ^

v_div_scale_f64 v[4:5], s2, v[2:3], v[6:7], v[8:9] dpp8:[7,6,5,4,3,2,1,0]
// GFX13: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX13-NEXT:{{^}}v_div_scale_f64 v[4:5], s2, v[2:3], v[6:7], v[8:9] dpp8:[7,6,5,4,3,2,1,0]
// GFX13-NEXT:{{^}}                                                   ^

v_div_scale_f64 v[4:5], s2, v[2:3], v[6:7], v[8:9] quad_perm:[3,2,1,0]
// GFX13: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX13-NEXT:{{^}}v_div_scale_f64 v[4:5], s2, v[2:3], v[6:7], v[8:9] quad_perm:[3,2,1,0]
// GFX13-NEXT:{{^}}                                                   ^

v_dot2_bf16_bf16 v5, v1, v2, s3
// GFX13: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dot2_bf16_bf16_e64_dpp v0, s1, v2, v3 quad_perm:[0,1,2,3] row_mask:0x0 bank_mask:0x0
// GFX13: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dot2_bf16_bf16_e64_dpp v0, v1, v2, v3 dpp8:[0,1,2,3,4,4,4,4]
// GFX13: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dot2_f16_f16 v5, v1, v2, s3
// GFX13: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dot2_f16_f16_e64_dpp v0, v1, v2, v3 quad_perm:[0,1,2,3] row_mask:0x0 bank_mask:0x0 fi:1
// GFX13: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dot2_f16_f16_e64_dpp v0, v1, v2, v3 dpp8:[0,1,2,3,4,4,4,4]
// GFX13: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_fma_f64 v[4:5], v[2:3], v[6:7], v[8:9] dpp8:[7,6,5,4,3,2,1,0]
// GFX13: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX13-NEXT:{{^}}v_fma_f64 v[4:5], v[2:3], v[6:7], v[8:9] dpp8:[7,6,5,4,3,2,1,0]
// GFX13-NEXT:{{^}}                                         ^

v_fma_f64 v[4:5], v[2:3], v[6:7], v[8:9] quad_perm:[3,2,1,0]
// GFX13: :[[@LINE-1]]:{{[0-9]+}}: error: DP ALU dpp only supports row_share
// GFX13-NEXT:{{^}}v_fma_f64 v[4:5], v[2:3], v[6:7], v[8:9] quad_perm:[3,2,1,0]
// GFX13-NEXT:{{^}}                                         ^

v_ldexp_f64 v[4:5], v[2:3], v6 dpp8:[7,6,5,4,3,2,1,0]
// GFX13: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX13-NEXT:{{^}}v_ldexp_f64 v[4:5], v[2:3], v6 dpp8:[7,6,5,4,3,2,1,0]
// GFX13-NEXT:{{^}}                               ^

v_ldexp_f64 v[4:5], v[2:3], v6 quad_perm:[3,2,1,0]
// GFX13: :[[@LINE-1]]:{{[0-9]+}}: error: DP ALU dpp only supports row_share
// GFX13-NEXT:{{^}}v_ldexp_f64 v[4:5], v[2:3], v6 quad_perm:[3,2,1,0]
// GFX13-NEXT:{{^}}                               ^

v_lshrrev_b64 v[4:5], v2, v[6:7] dpp8:[7,6,5,4,3,2,1,0]
// GFX13: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX13-NEXT:{{^}}v_lshrrev_b64 v[4:5], v2, v[6:7] dpp8:[7,6,5,4,3,2,1,0]
// GFX13-NEXT:{{^}}                                 ^

v_lshrrev_b64 v[4:5], v2, v[6:7] quad_perm:[3,2,1,0]
// GFX13: :[[@LINE-1]]:{{[0-9]+}}: error: DP ALU dpp only supports row_share
// GFX13-NEXT:{{^}}v_lshrrev_b64 v[4:5], v2, v[6:7] quad_perm:[3,2,1,0]
// GFX13-NEXT:{{^}}                                 ^

v_mad_co_i64_i32 v[4:5], s2, v2, v6, v[8:9] dpp8:[7,6,5,4,3,2,1,0]
// GFX13: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX13-NEXT:{{^}}v_mad_co_i64_i32 v[4:5], s2, v2, v6, v[8:9] dpp8:[7,6,5,4,3,2,1,0]
// GFX13-NEXT:{{^}}                                            ^

v_mad_co_i64_i32 v[4:5], s2, v2, v6, v[8:9] quad_perm:[3,2,1,0]
// GFX13: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX13-NEXT:{{^}}v_mad_co_i64_i32 v[4:5], s2, v2, v6, v[8:9] quad_perm:[3,2,1,0]
// GFX13-NEXT:{{^}}                                            ^

v_mad_co_u64_u32 v[4:5], s2, v2, v6, v[8:9] dpp8:[7,6,5,4,3,2,1,0]
// GFX13: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX13-NEXT:{{^}}v_mad_co_u64_u32 v[4:5], s2, v2, v6, v[8:9] dpp8:[7,6,5,4,3,2,1,0]
// GFX13-NEXT:{{^}}                                            ^

v_mad_co_u64_u32 v[4:5], s2, v2, v6, v[8:9] quad_perm:[3,2,1,0]
// GFX13: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX13-NEXT:{{^}}v_mad_co_u64_u32 v[4:5], s2, v2, v6, v[8:9] quad_perm:[3,2,1,0]
// GFX13-NEXT:{{^}}                                            ^

v_mad_u32 v2, v4, v7, v8 dpp8:[7,6,5,4,3,2,1,0]
// GFX13: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX13-NEXT:{{^}}v_mad_u32 v2, v4, v7, v8 dpp8:[7,6,5,4,3,2,1,0]
// GFX13-NEXT:{{^}}                         ^

v_maximum_f64 v[4:5], v[2:3], v[6:7] dpp8:[7,6,5,4,3,2,1,0]
// GFX13: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX13-NEXT:{{^}}v_maximum_f64 v[4:5], v[2:3], v[6:7] dpp8:[7,6,5,4,3,2,1,0]
// GFX13-NEXT:{{^}}                                     ^

v_maximum_f64 v[4:5], v[2:3], v[6:7] quad_perm:[3,2,1,0]
// GFX13: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX13-NEXT:{{^}}v_maximum_f64 v[4:5], v[2:3], v[6:7] quad_perm:[3,2,1,0]
// GFX13-NEXT:{{^}}                                     ^

v_maximum_f64 v[4:5], v[2:3], v[6:7] row_share:1
// GFX13: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX13-NEXT:{{^}}v_maximum_f64 v[4:5], v[2:3], v[6:7] row_share:1
// GFX13-NEXT:{{^}}                                     ^

v_minimum_f64 v[4:5], v[2:3], v[6:7] dpp8:[7,6,5,4,3,2,1,0]
// GFX13: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX13-NEXT:{{^}}v_minimum_f64 v[4:5], v[2:3], v[6:7] dpp8:[7,6,5,4,3,2,1,0]
// GFX13-NEXT:{{^}}                                     ^

v_minimum_f64 v[4:5], v[2:3], v[6:7] quad_perm:[3,2,1,0]
// GFX13: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX13-NEXT:{{^}}v_minimum_f64 v[4:5], v[2:3], v[6:7] quad_perm:[3,2,1,0]
// GFX13-NEXT:{{^}}                                     ^

v_minimum_f64 v[4:5], v[2:3], v[6:7] row_share:1
// GFX13: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX13-NEXT:{{^}}v_minimum_f64 v[4:5], v[2:3], v[6:7] row_share:1
// GFX13-NEXT:{{^}}                                     ^

v_mul_hi_i32 v4, v2, v6 dpp8:[7,6,5,4,3,2,1,0]
// GFX13: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX13-NEXT:{{^}}v_mul_hi_i32 v4, v2, v6 dpp8:[7,6,5,4,3,2,1,0]
// GFX13-NEXT:{{^}}                        ^

v_mul_hi_i32 v4, v2, v6 quad_perm:[3,2,1,0]
// GFX13: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX13-NEXT:{{^}}v_mul_hi_i32 v4, v2, v6 quad_perm:[3,2,1,0]
// GFX13-NEXT:{{^}}                        ^

v_mul_hi_i32 v4, v2, v6 row_share:1
// GFX13: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX13-NEXT:{{^}}v_mul_hi_i32 v4, v2, v6 row_share:1
// GFX13-NEXT:{{^}}                        ^

v_mul_hi_u32 v4, v2, v6 dpp8:[7,6,5,4,3,2,1,0]
// GFX13: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX13-NEXT:{{^}}v_mul_hi_u32 v4, v2, v6 dpp8:[7,6,5,4,3,2,1,0]
// GFX13-NEXT:{{^}}                        ^

v_mul_hi_u32 v4, v2, v6 quad_perm:[3,2,1,0]
// GFX13: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX13-NEXT:{{^}}v_mul_hi_u32 v4, v2, v6 quad_perm:[3,2,1,0]
// GFX13-NEXT:{{^}}                        ^

v_mul_hi_u32 v4, v2, v6 row_share:1
// GFX13: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX13-NEXT:{{^}}v_mul_hi_u32 v4, v2, v6 row_share:1
// GFX13-NEXT:{{^}}                        ^

v_mul_lo_u32 v4, v2, v6 dpp8:[7,6,5,4,3,2,1,0]
// GFX13: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX13-NEXT:{{^}}v_mul_lo_u32 v4, v2, v6 dpp8:[7,6,5,4,3,2,1,0]
// GFX13-NEXT:{{^}}                        ^

v_mul_lo_u32 v4, v2, v6 quad_perm:[3,2,1,0]
// GFX13: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX13-NEXT:{{^}}v_mul_lo_u32 v4, v2, v6 quad_perm:[3,2,1,0]
// GFX13-NEXT:{{^}}                        ^

v_mul_lo_u32 v4, v2, v6 row_share:1
// GFX13: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX13-NEXT:{{^}}v_mul_lo_u32 v4, v2, v6 row_share:1
// GFX13-NEXT:{{^}}                        ^

v_permlane16_b32 v5, v1, s2, s3 op_sel:[0, 0, 0, 1]
// GFX13: :[[@LINE-1]]:{{[0-9]+}}: error: invalid op_sel operand

v_permlanex16_b32 v5, v1, s2, s3 op_sel:[0, 0, 1, 0]
// GFX13: :[[@LINE-1]]:{{[0-9]+}}: error: invalid op_sel operand

v_permlane16_var_b32 v5, v1, v2 clamp
// GFX13: error: invalid operand for instruction
// GFX13-NEXT:{{^}}v_permlane16_var_b32 v5, v1, v2 clamp
// GFX13-NEXT:{{^}}                                ^

v_permlane16_var_b32 v5, v1, v2 div:2
// GFX13: error: not a valid operand
// GFX13-NEXT:{{^}}v_permlane16_var_b32 v5, v1, v2 div:2
// GFX13-NEXT:{{^}}                                ^

v_permlane16_var_b32 v5, v1, v2 mul:1
// GFX13: error: not a valid operand
// GFX13-NEXT:{{^}}v_permlane16_var_b32 v5, v1, v2 mul:1
// GFX13-NEXT:{{^}}                                ^

v_permlane16_var_b32 -v5, v1, v2 op_sel:[0, 1]
// GFX13: error: not a valid operand
// GFX13-NEXT:{{^}}v_permlane16_var_b32 -v5, v1, v2 op_sel:[0, 1]
// GFX13-NEXT:{{^}}                     ^

v_permlane16_var_b32 v5, -v1, v2 op_sel:[0, 1]
// GFX13: error: not a valid operand
// GFX13-NEXT:{{^}}v_permlane16_var_b32 v5, -v1, v2 op_sel:[0, 1]
// GFX13-NEXT:{{^}}                         ^

v_permlane16_var_b32 v5, v1, -v2 op_sel:[0, 1]
// GFX13: error: not a valid operand
// GFX13-NEXT:{{^}}v_permlane16_var_b32 v5, v1, -v2 op_sel:[0, 1]
// GFX13-NEXT:{{^}}                             ^

v_permlane16_var_b32 -|v5|, v1, v2 op_sel:[0, 1]
// GFX13: error: not a valid operand
// GFX13-NEXT:{{^}}v_permlane16_var_b32 -|v5|, v1, v2 op_sel:[0, 1]
// GFX13-NEXT:{{^}}                     ^

v_permlane16_var_b32 v5, -v1, |v2| op_sel:[0, 1]
// GFX13: error: not a valid operand
// GFX13-NEXT:{{^}}v_permlane16_var_b32 v5, -v1, |v2| op_sel:[0, 1]
// GFX13-NEXT:{{^}}                         ^

v_permlane16_var_b32 v5, v1, -|v2| op_sel:[0, 1]
// GFX13: error: not a valid operand
// GFX13-NEXT:{{^}}v_permlane16_var_b32 v5, v1, -|v2| op_sel:[0, 1]
// GFX13-NEXT:{{^}}                             ^

v_permlane16_var_b32 |v5|, v1, v2 op_sel:[0, 1]
// GFX13: error: not a valid operand
// GFX13-NEXT:{{^}}v_permlane16_var_b32 |v5|, v1, v2 op_sel:[0, 1]
// GFX13-NEXT:{{^}}                     ^

v_permlane16_var_b32 v5, |v1|, v2 op_sel:[0, 1]
// GFX13: error: not a valid operand
// GFX13-NEXT:{{^}}v_permlane16_var_b32 v5, |v1|, v2 op_sel:[0, 1]
// GFX13-NEXT:{{^}}                         ^

v_permlane16_var_b32 v5, v1, |v2| op_sel:[0, 1]
// GFX13: error: not a valid operand
// GFX13-NEXT:{{^}}v_permlane16_var_b32 v5, v1, |v2| op_sel:[0, 1]
// GFX13-NEXT:{{^}}                             ^

v_permlane16_var_b32 v5, v1, v2 op_sel:[-1, 0]
// GFX13: error: invalid op_sel value
// GFX13-NEXT:{{^}}v_permlane16_var_b32 v5, v1, v2 op_sel:[-1, 0]
// GFX13-NEXT:{{^}}                                        ^

v_permlane16_var_b32 v5, v1, v2 op_sel:[1, -1]
// GFX13: error: invalid op_sel value
// GFX13-NEXT:{{^}}v_permlane16_var_b32 v5, v1, v2 op_sel:[1, -1]
// GFX13-NEXT:{{^}}                                           ^

v_permlane16_var_b32 v5, v1, v2 op_sel:[0, 0, 0, 1]
// GFX13: error: invalid op_sel operand
// GFX13-NEXT:{{^}}v_permlane16_var_b32 v5, v1, v2 op_sel:[0, 0, 0, 1]
// GFX13-NEXT:{{^}}                                ^

v_permlane16_var_b32 v5, v1, v2 op_sel:[0, 0, 0, -1]
// GFX13: error: invalid op_sel value
// GFX13-NEXT:{{^}}v_permlane16_var_b32 v5, v1, v2 op_sel:[0, 0, 0, -1]
// GFX13-NEXT:{{^}}                                                 ^

v_permlane16_var_b32 v5, v1, v2 op_sel:[0, 0, 1, 0]
// GFX13: error: invalid op_sel operand
// GFX13-NEXT:{{^}}v_permlane16_var_b32 v5, v1, v2 op_sel:[0, 0, 1, 0]
// GFX13-NEXT:{{^}}                                ^

v_permlane16_var_b32 v5, v1, v2 op_sel:[0, 0, -1, 0]
// GFX13: error: invalid op_sel value
// GFX13-NEXT:{{^}}v_permlane16_var_b32 v5, v1, v2 op_sel:[0, 0, -1, 0]
// GFX13-NEXT:{{^}}                                              ^

v_permlane16_var_b32 v5, v1, v2 op_sel:[0, 0, 1]
// GFX13: error: invalid op_sel operand
// GFX13-NEXT:{{^}}v_permlane16_var_b32 v5, v1, v2 op_sel:[0, 0, 1]
// GFX13-NEXT:{{^}}                                ^

v_trig_preop_f64 v[4:5], v[8:9], v2 row_share:1
// GFX13: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX13-NEXT:{{^}}v_trig_preop_f64 v[4:5], v[8:9], v2 row_share:1
// GFX13-NEXT:{{^}}                                    ^

v_wave_match_b32 v5, v3, |v1|
// GFX13: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX13-NEXT:{{^}}v_wave_match_b32 v5, v3, |v1|
// GFX13-NEXT:{{^}}                         ^

v_wave_match_b32 v5, v3, -v1
// GFX13: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX13-NEXT:{{^}}v_wave_match_b32 v5, v3, -v1
// GFX13-NEXT:{{^}}                         ^

v_wave_match_b32 v5, v3, v1 clamp
// GFX13: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// GFX13-NEXT:{{^}}v_wave_match_b32 v5, v3, v1 clamp
// GFX13-NEXT:{{^}}                            ^

v_wave_match_b32 v5, v3, v1 mul:2
// GFX13: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX13-NEXT:{{^}}v_wave_match_b32 v5, v3, v1 mul:2
// GFX13-NEXT:{{^}}                            ^

v_wave_match_b32_dpp v5, v3, v1 row_share:1
// GFX13: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported

v_wave_match_b32_dpp v5, v3, v1 dpp8:[7,6,5,4,3,2,1,0]
// GFX13: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported

v_wave_match_b32_e64_dpp v5, v3, v1 row_share:1
// GFX13: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_pdep_b32 v5, v3, -v1
// GFX13: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX13-NEXT:{{^}}v_pdep_b32 v5, v3, -v1
// GFX13-NEXT:{{^}}                   ^

v_pdep_b32 v5, v3, v1 clamp
// GFX13: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// GFX13-NEXT:{{^}}v_pdep_b32 v5, v3, v1 clamp
// GFX13-NEXT:{{^}}                      ^

v_pdep_b32 v5, v3, v1 mul:2
// GFX13: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX13-NEXT:{{^}}v_pdep_b32 v5, v3, v1 mul:2
// GFX13-NEXT:{{^}}                      ^

v_pdep_b32 v5, v3, |v1|
// GFX13: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX13-NEXT:{{^}}v_pdep_b32 v5, v3, |v1|
// GFX13-NEXT:{{^}}                   ^

v_pdep_b32_dpp v5, v3, v1 dpp8:[7,6,5,4,3,2,1,0]
// GFX13: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported

v_pdep_b32_dpp v5, v3, v1 row_share:1
// GFX13: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported

v_pdep_b32_e64_dpp v5, v3, v1 row_share:1
// GFX13: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_pext_b32 v5, v3, -v1
// GFX13: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX13-NEXT:{{^}}v_pext_b32 v5, v3, -v1
// GFX13-NEXT:{{^}}                   ^

v_pext_b32 v5, v3, v1 clamp
// GFX13: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// GFX13-NEXT:{{^}}v_pext_b32 v5, v3, v1 clamp
// GFX13-NEXT:{{^}}                      ^

v_pext_b32 v5, v3, v1 mul:2
// GFX13: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX13-NEXT:{{^}}v_pext_b32 v5, v3, v1 mul:2
// GFX13-NEXT:{{^}}                      ^

v_pext_b32 v5, v3, |v1|
// GFX13: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX13-NEXT:{{^}}v_pext_b32 v5, v3, |v1|
// GFX13-NEXT:{{^}}                   ^

v_pext_b32_dpp v5, v3, v1 dpp8:[7,6,5,4,3,2,1,0]
// GFX13: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported

v_pext_b32_dpp v5, v3, v1 row_share:1
// GFX13: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported

v_pext_b32_e64_dpp v5, v3, v1 row_share:1
// GFX13: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_cvt_scalef32_pk8_fp4_f32 v10, v[20:27], v8 clamp
// W32: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// W32-NEXT:{{^}}v_cvt_scalef32_pk8_fp4_f32 v10, v[20:27], v8 clamp
// W32-NEXT:{{^}}                                             ^
// W64: :[[@LINE-4]]:{{[0-9]+}}: error: instruction requires wavesize=32

v_cvt_scalef32_pk8_fp4_f32 v10, v[20:27], v8 mul:2
// W32: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// W32-NEXT:{{^}}v_cvt_scalef32_pk8_fp4_f32 v10, v[20:27], v8 mul:2
// W32-NEXT:{{^}}                                             ^
// W64: :[[@LINE-4]]:{{[0-9]+}}: error: instruction requires wavesize=32

v_cvt_scalef32_pk8_fp4_f32_dpp v5, v3, v1 row_share:1
// W32: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported
// W64: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_scalef32_pk8_fp4_f32_dpp v5, v3, v1 dpp8:[7,6,5,4,3,2,1,0]
// W32: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported
// W64: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_scalef32_pk8_fp4_f32_e64_dpp v5, v3, v1 row_share:1
// W32: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported
// W64: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_scalef32_pk8_fp8_f32 v[10:11], v[20:27], v8 clamp
// W32: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// W32-NEXT:{{^}}v_cvt_scalef32_pk8_fp8_f32 v[10:11], v[20:27], v8 clamp
// W32-NEXT:{{^}}                                                  ^
// W64: :[[@LINE-4]]:{{[0-9]+}}: error: instruction requires wavesize=32

v_cvt_scalef32_pk8_fp8_f32 v[10:11], v[20:27], v8 mul:2
// W32: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// W32-NEXT:{{^}}v_cvt_scalef32_pk8_fp8_f32 v[10:11], v[20:27], v8 mul:2
// W32-NEXT:{{^}}                                                  ^
// W64: :[[@LINE-4]]:{{[0-9]+}}: error: instruction requires wavesize=32

v_cvt_scalef32_pk8_fp8_f32_dpp v[10:11], v[20:27], v8 row_share:1
// W32: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported
// W64: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_scalef32_pk8_fp8_f32_dpp v[10:11], v[20:27], v8 dpp8:[7,6,5,4,3,2,1,0]
// W32: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported
// W64: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_scalef32_pk8_fp8_f32_e64_dpp v[10:11], v[20:27], v8 row_share:1
// W32: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported
// W64: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_scalef32_pk8_fp8_bf16 v[10:11], v[20:23], v8 clamp
// W32: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// W32-NEXT:{{^}}v_cvt_scalef32_pk8_fp8_bf16 v[10:11], v[20:23], v8 clamp
// W32-NEXT:{{^}}                                                   ^
// W64: :[[@LINE-4]]:{{[0-9]+}}: error: instruction requires wavesize=32

v_cvt_scalef32_pk8_fp8_bf16 v[10:11], v[20:23], v8 mul:2
// W32: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// W32-NEXT:{{^}}v_cvt_scalef32_pk8_fp8_bf16 v[10:11], v[20:23], v8 mul:2
// W32-NEXT:{{^}}                                                   ^
// W64: :[[@LINE-4]]:{{[0-9]+}}: error: instruction requires wavesize=32

v_cvt_scalef32_pk8_fp8_bf16_dpp v[10:11], v[20:23], v8 row_share:1
// W32: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported
// W64: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_scalef32_pk8_fp8_bf16_dpp v[10:11], v[20:23], v8 dpp8:[7,6,5,4,3,2,1,0]
// W32: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported
// W64: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_scalef32_pk8_fp8_bf16_e64_dpp v[10:11], v[20:23], v8 row_share:1
// W32: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported
// W64: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_scalef32_sr_pk8_fp4_f32 v10, v[20:27], v4, v8 clamp
// W32: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// W32-NEXT:{{^}}v_cvt_scalef32_sr_pk8_fp4_f32 v10, v[20:27], v4, v8 clamp
// W32-NEXT:{{^}}                                                    ^
// W64: :[[@LINE-4]]:{{[0-9]+}}: error: instruction requires wavesize=32

v_cvt_scalef32_sr_pk8_fp4_f32 v10, v[20:27], v4, v8 mul:2
// W32: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// W32-NEXT:{{^}}v_cvt_scalef32_sr_pk8_fp4_f32 v10, v[20:27], v4, v8 mul:2
// W32-NEXT:{{^}}                                                    ^
// W64: :[[@LINE-4]]:{{[0-9]+}}: error: instruction requires wavesize=32

v_cvt_scalef32_sr_pk8_fp4_f32_dpp v10, v[20:27], v4, v8 row_share:1
// W32: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported
// W64: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_scalef32_sr_pk8_fp4_f32_dpp v10, v[20:27], v4, v8 dpp8:[7,6,5,4,3,2,1,0]
// W32: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported
// W64: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_scalef32_sr_pk8_fp4_f32_e64_dpp v10, v[20:27], v4, v8 row_share:1
// W32: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported
// W64: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_scalef32_sr_pk8_fp8_f32 v[10:11], v[20:27], v4, v8 clamp
// W32: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// W32-NEXT:{{^}}v_cvt_scalef32_sr_pk8_fp8_f32 v[10:11], v[20:27], v4, v8 clamp
// W32-NEXT:{{^}}                                                         ^
// W64: :[[@LINE-4]]:{{[0-9]+}}: error: instruction requires wavesize=32

v_cvt_scalef32_sr_pk8_fp8_f32 v[10:11], v[20:27], v4, v8 mul:2
// W32: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// W32-NEXT:{{^}}v_cvt_scalef32_sr_pk8_fp8_f32 v[10:11], v[20:27], v4, v8 mul:2
// W32-NEXT:{{^}}                                                         ^
// W64: :[[@LINE-4]]:{{[0-9]+}}: error: instruction requires wavesize=32

v_cvt_scalef32_sr_pk8_fp8_f32_dpp v[10:11], v[20:27], v4, v8 row_share:1
// W32: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported
// W64: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_scalef32_sr_pk8_fp8_f32_dpp v[10:11], v[20:27], v4, v8 dpp8:[7,6,5,4,3,2,1,0]
// W32: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported
// W64: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_scalef32_sr_pk8_fp8_f32_e64_dpp v[10:11], v[20:27], v4, v8 row_share:1
// W32: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported
// W64: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_scalef32_sr_pk8_bf8_f32 v[10:11], v[20:27], v4, v8 clamp
// W32: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// W32-NEXT:{{^}}v_cvt_scalef32_sr_pk8_bf8_f32 v[10:11], v[20:27], v4, v8 clamp
// W32-NEXT:{{^}}                                                         ^
// W64: :[[@LINE-4]]:{{[0-9]+}}: error: instruction requires wavesize=32

v_cvt_scalef32_sr_pk8_bf8_f32 v[10:11], v[20:27], v4, v8 mul:2
// W32: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// W32-NEXT:{{^}}v_cvt_scalef32_sr_pk8_bf8_f32 v[10:11], v[20:27], v4, v8 mul:2
// W32-NEXT:{{^}}                                                         ^
// W64: :[[@LINE-4]]:{{[0-9]+}}: error: instruction requires wavesize=32

v_cvt_scalef32_sr_pk8_bf8_f32_dpp v[10:11], v[20:27], v4, v8 row_share:1
// W32: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported
// W64: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_scalef32_sr_pk8_bf8_f32_dpp v[10:11], v[20:27], v4, v8 dpp8:[7,6,5,4,3,2,1,0]
// W32: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported
// W64: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_scalef32_sr_pk8_bf8_f32_e64_dpp v[10:11], v[20:27], v4, v8 row_share:1
// W32: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported
// W64: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_scalef32_pk8_fp4_f16 v10, v[20:23], v8 clamp
// W32: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// W32-NEXT:{{^}}v_cvt_scalef32_pk8_fp4_f16 v10, v[20:23], v8 clamp
// W32-NEXT:{{^}}                                             ^
// W64: :[[@LINE-4]]:{{[0-9]+}}: error: instruction requires wavesize=32

v_cvt_scalef32_pk8_fp4_f16 v10, v[20:23], v8 mul:2
// W32: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// W32-NEXT:{{^}}v_cvt_scalef32_pk8_fp4_f16 v10, v[20:23], v8 mul:2
// W32-NEXT:{{^}}                                             ^
// W64: :[[@LINE-4]]:{{[0-9]+}}: error: instruction requires wavesize=32

v_cvt_scalef32_pk8_fp4_f16_dpp v10, v[20:23], v8 row_share:1
// W32: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported
// W64: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_scalef32_pk8_fp4_f16_dpp v10, v[20:23], v8 dpp8:[7,6,5,4,3,2,1,0]
// W32: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported
// W64: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_scalef32_pk8_fp4_f16_e64_dpp v10, v[20:23], v8 row_share:1
// W32: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported
// W64: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_scalef32_pk8_bf8_bf16 v[10:11], v[20:23], v8 clamp
// W32: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// W32-NEXT:{{^}}v_cvt_scalef32_pk8_bf8_bf16 v[10:11], v[20:23], v8 clamp
// W32-NEXT:{{^}}                                                   ^
// W64: :[[@LINE-4]]:{{[0-9]+}}: error: instruction requires wavesize=32

v_cvt_scalef32_pk8_bf8_bf16 v[10:11], v[20:23], v8 mul:2
// W32: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// W32-NEXT:{{^}}v_cvt_scalef32_pk8_bf8_bf16 v[10:11], v[20:23], v8 mul:2
// W32-NEXT:{{^}}                                                   ^
// W64: :[[@LINE-4]]:{{[0-9]+}}: error: instruction requires wavesize=32

v_cvt_scalef32_pk8_bf8_bf16_dpp v[10:11], v[20:23], v8 row_share:1
// W32: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported
// W64: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_scalef32_pk8_bf8_bf16_dpp v[10:11], v[20:23], v8 dpp8:[7,6,5,4,3,2,1,0]
// W32: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported
// W64: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_scalef32_pk8_bf8_bf16_e64_dpp v[10:11], v[20:23], v8 row_share:1
// W32: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported
// W64: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_scalef32_pk8_fp4_bf16 v10, v[20:23], v8 clamp
// W32: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// W32-NEXT:{{^}}v_cvt_scalef32_pk8_fp4_bf16 v10, v[20:23], v8 clamp
// W32-NEXT:{{^}}                                              ^
// W64: :[[@LINE-4]]:{{[0-9]+}}: error: instruction requires wavesize=32

v_cvt_scalef32_pk8_fp4_bf16 v10, v[20:23], v8 mul:2
// W32: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// W32-NEXT:{{^}}v_cvt_scalef32_pk8_fp4_bf16 v10, v[20:23], v8 mul:2
// W32-NEXT:{{^}}                                              ^
// W64: :[[@LINE-4]]:{{[0-9]+}}: error: instruction requires wavesize=32

v_cvt_scalef32_pk8_fp4_bf16_dpp v10, v[20:23], v8 row_share:1
// W32: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported
// W64: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_scalef32_pk8_fp4_bf16_dpp v10, v[20:23], v8 dpp8:[7,6,5,4,3,2,1,0]
// W32: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported
// W64: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_scalef32_pk8_fp4_bf16_e64_dpp v10, v[20:23], v8 row_share:1
// W32: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported
// W64: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_scalef32_sr_pk8_fp4_f16 v10, v[20:23], v4, v8 clamp
// W32: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// W32-NEXT:{{^}}v_cvt_scalef32_sr_pk8_fp4_f16 v10, v[20:23], v4, v8 clamp
// W32-NEXT:{{^}}                                                    ^
// W64: :[[@LINE-4]]:{{[0-9]+}}: error: instruction requires wavesize=32

v_cvt_scalef32_sr_pk8_fp4_f16 v10, v[20:23], v4, v8 mul:2
// W32: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// W32-NEXT:{{^}}v_cvt_scalef32_sr_pk8_fp4_f16 v10, v[20:23], v4, v8 mul:2
// W32-NEXT:{{^}}                                                    ^
// W64: :[[@LINE-4]]:{{[0-9]+}}: error: instruction requires wavesize=32

v_cvt_scalef32_sr_pk8_fp4_f16_dpp v10, v[20:23], v4, v8 row_share:1
// W32: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported
// W64: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_scalef32_sr_pk8_fp4_f16_dpp v10, v[20:23], v4, v8 dpp8:[7,6,5,4,3,2,1,0]
// W32: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported
// W64: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_scalef32_sr_pk8_fp4_f16_e64_dpp v10, v[20:23], v4, v8 row_share:1
// W32: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported
// W64: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_scalef32_sr_pk8_fp4_bf16 v10, v[20:23], v4, v8 clamp
// W32: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// W32-NEXT:{{^}}v_cvt_scalef32_sr_pk8_fp4_bf16 v10, v[20:23], v4, v8 clamp
// W32-NEXT:{{^}}                                                     ^
// W64: :[[@LINE-4]]:{{[0-9]+}}: error: instruction requires wavesize=32

v_cvt_scalef32_sr_pk8_fp4_bf16 v10, v[20:23], v4, v8 mul:2
// W32: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// W32-NEXT:{{^}}v_cvt_scalef32_sr_pk8_fp4_bf16 v10, v[20:23], v4, v8 mul:2
// W32-NEXT:{{^}}                                                     ^
// W64: :[[@LINE-4]]:{{[0-9]+}}: error: instruction requires wavesize=32

v_cvt_scalef32_sr_pk8_fp4_bf16_dpp v10, v[20:23], v4, v8 row_share:1
// W32: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported
// W64: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_scalef32_sr_pk8_fp4_bf16_dpp v10, v[20:23], v4, v8 dpp8:[7,6,5,4,3,2,1,0]
// W32: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported
// W64: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_scalef32_sr_pk8_fp4_bf16_e64_dpp v10, v[20:23], v4, v8 row_share:1
// W32: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported
// W64: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_scalef32_sr_pk8_fp8_f16 v[10:11], v[20:23], v4, v8 clamp
// W32: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// W32-NEXT:{{^}}v_cvt_scalef32_sr_pk8_fp8_f16 v[10:11], v[20:23], v4, v8 clamp
// W32-NEXT:{{^}}                                                         ^
// W64: :[[@LINE-4]]:{{[0-9]+}}: error: instruction requires wavesize=32

v_cvt_scalef32_sr_pk8_fp8_f16 v[10:11], v[20:23], v4, v8 mul:2
// W32: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// W32-NEXT:{{^}}v_cvt_scalef32_sr_pk8_fp8_f16 v[10:11], v[20:23], v4, v8 mul:2
// W32-NEXT:{{^}}                                                         ^
// W64: :[[@LINE-4]]:{{[0-9]+}}: error: instruction requires wavesize=32

v_cvt_scalef32_sr_pk8_fp8_f16_dpp v[10:11], v[20:23], v4, v8 row_share:1
// W32: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported
// W64: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_scalef32_sr_pk8_fp8_f16_dpp v[10:11], v[20:23], v4, v8 dpp8:[7,6,5,4,3,2,1,0]
// W32: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported
// W64: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_scalef32_sr_pk8_fp8_f16_e64_dpp v[10:11], v[20:23], v4, v8 row_share:1
// W32: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported
// W64: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_scalef32_sr_pk8_fp8_bf16 v[10:11], v[20:23], v4, v8 clamp
// W32: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// W32-NEXT:{{^}}v_cvt_scalef32_sr_pk8_fp8_bf16 v[10:11], v[20:23], v4, v8 clamp
// W32-NEXT:{{^}}                                                          ^
// W64: :[[@LINE-4]]:{{[0-9]+}}: error: instruction requires wavesize=32

v_cvt_scalef32_sr_pk8_fp8_bf16 v[10:11], v[20:23], v4, v8 mul:2
// W32: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// W32-NEXT:{{^}}v_cvt_scalef32_sr_pk8_fp8_bf16 v[10:11], v[20:23], v4, v8 mul:2
// W32-NEXT:{{^}}                                                          ^
// W64: :[[@LINE-4]]:{{[0-9]+}}: error: instruction requires wavesize=32

v_cvt_scalef32_sr_pk8_fp8_bf16_dpp v[10:11], v[20:23], v4, v8 row_share:1
// W32: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported
// W64: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_scalef32_sr_pk8_fp8_bf16_dpp v[10:11], v[20:23], v4, v8 dpp8:[7,6,5,4,3,2,1,0]
// W32: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported
// W64: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_scalef32_sr_pk8_fp8_bf16_e64_dpp v[10:11], v[20:23], v4, v8 row_share:1
// W32: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported
// W64: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_scalef32_sr_pk8_bf8_f16 v[10:11], v[20:23], v4, v8 clamp
// W32: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// W32-NEXT:{{^}}v_cvt_scalef32_sr_pk8_bf8_f16 v[10:11], v[20:23], v4, v8 clamp
// W32-NEXT:{{^}}                                                         ^
// W64: :[[@LINE-4]]:{{[0-9]+}}: error: instruction requires wavesize=32

v_cvt_scalef32_sr_pk8_bf8_f16 v[10:11], v[20:23], v4, v8 mul:2
// W32: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// W32-NEXT:{{^}}v_cvt_scalef32_sr_pk8_bf8_f16 v[10:11], v[20:23], v4, v8 mul:2
// W32-NEXT:{{^}}                                                         ^
// W64: :[[@LINE-4]]:{{[0-9]+}}: error: instruction requires wavesize=32

v_cvt_scalef32_sr_pk8_bf8_f16_dpp v[10:11], v[20:23], v4, v8 row_share:1
// W32: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported
// W64: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_scalef32_sr_pk8_bf8_f16_dpp v[10:11], v[20:23], v4, v8 dpp8:[7,6,5,4,3,2,1,0]
// W32: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported
// W64: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_scalef32_sr_pk8_bf8_f16_e64_dpp v[10:11], v[20:23], v4, v8 row_share:1
// W32: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported
// W64: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_scalef32_sr_pk8_bf8_bf16 v[10:11], v[20:23], v4, v8 clamp
// W32: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// W32-NEXT:{{^}}v_cvt_scalef32_sr_pk8_bf8_bf16 v[10:11], v[20:23], v4, v8 clamp
// W32-NEXT:{{^}}                                                          ^
// W64: :[[@LINE-4]]:{{[0-9]+}}: error: instruction requires wavesize=32

v_cvt_scalef32_sr_pk8_bf8_bf16 v[10:11], v[20:23], v4, v8 mul:2
// W32: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// W32-NEXT:{{^}}v_cvt_scalef32_sr_pk8_bf8_bf16 v[10:11], v[20:23], v4, v8 mul:2
// W32-NEXT:{{^}}                                                          ^
// W64: :[[@LINE-4]]:{{[0-9]+}}: error: instruction requires wavesize=32

v_cvt_scalef32_sr_pk8_bf8_bf16_dpp v[10:11], v[20:23], v4, v8 row_share:1
// W32: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported
// W64: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_scalef32_sr_pk8_bf8_bf16_dpp v[10:11], v[20:23], v4, v8 dpp8:[7,6,5,4,3,2,1,0]
// W32: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported
// W64: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_scalef32_sr_pk8_bf8_bf16_e64_dpp v[10:11], v[20:23], v4, v8 row_share:1
// W32: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported
// W64: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_scalef32_pk8_fp8_f16 v[10:11], v[20:23], v8 clamp
// W32: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// W32-NEXT:{{^}}v_cvt_scalef32_pk8_fp8_f16 v[10:11], v[20:23], v8 clamp
// W32-NEXT:{{^}}                                                  ^
// W64: :[[@LINE-4]]:{{[0-9]+}}: error: instruction requires wavesize=32

v_cvt_scalef32_pk8_fp8_f16 v[10:11], v[20:23], v8 mul:2
// W32: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// W32-NEXT:{{^}}v_cvt_scalef32_pk8_fp8_f16 v[10:11], v[20:23], v8 mul:2
// W32-NEXT:{{^}}                                                  ^
// W64: :[[@LINE-4]]:{{[0-9]+}}: error: instruction requires wavesize=32

v_cvt_scalef32_pk8_fp8_f16_dpp v[10:11], v[20:23], v8 row_share:1
// W32: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported
// W64: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_scalef32_pk8_fp8_f16_dpp v[10:11], v[20:23], v8 dpp8:[7,6,5,4,3,2,1,0]
// W32: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported
// W64: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_scalef32_pk8_fp8_f16_e64_dpp v[10:11], v[20:23], v8 row_share:1
// W32: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported
// W64: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_scalef32_pk8_bf8_f32 v[10:11], v[20:27], v8 clamp
// W32: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// W32-NEXT:{{^}}v_cvt_scalef32_pk8_bf8_f32 v[10:11], v[20:27], v8 clamp
// W32-NEXT:{{^}}                                                  ^
// W64: :[[@LINE-4]]:{{[0-9]+}}: error: instruction requires wavesize=32

v_cvt_scalef32_pk8_bf8_f32 v[10:11], v[20:27], v8 mul:2
// W32: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// W32-NEXT:{{^}}v_cvt_scalef32_pk8_bf8_f32 v[10:11], v[20:27], v8 mul:2
// W32-NEXT:{{^}}                                                  ^
// W64: :[[@LINE-4]]:{{[0-9]+}}: error: instruction requires wavesize=32

v_cvt_scalef32_pk8_bf8_f32_dpp v[10:11], v[20:27], v8 row_share:1
// W32: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported
// W64: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_scalef32_pk8_bf8_f32_dpp v[10:11], v[20:27], v8 dpp8:[7,6,5,4,3,2,1,0]
// W32: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported
// W64: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_scalef32_pk8_bf8_f32_e64_dpp v[10:11], v[20:27], v8 row_share:1
// W32: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported
// W64: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_scalef32_pk8_bf8_f16 v[10:11], v[20:23], v8 clamp
// W32: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// W32-NEXT:{{^}}v_cvt_scalef32_pk8_bf8_f16 v[10:11], v[20:23], v8 clamp
// W32-NEXT:{{^}}                                                  ^
// W64: :[[@LINE-4]]:{{[0-9]+}}: error: instruction requires wavesize=32

v_cvt_scalef32_pk8_bf8_f16 v[10:11], v[20:23], v8 mul:2
// W32: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// W32-NEXT:{{^}}v_cvt_scalef32_pk8_bf8_f16 v[10:11], v[20:23], v8 mul:2
// W32-NEXT:{{^}}                                                  ^
// W64: :[[@LINE-4]]:{{[0-9]+}}: error: instruction requires wavesize=32

v_cvt_scalef32_pk8_bf8_f16_dpp v[10:11], v[20:23], v8 row_share:1
// W32: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported
// W64: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_scalef32_pk8_bf8_f16_dpp v[10:11], v[20:23], v8 dpp8:[7,6,5,4,3,2,1,0]
// W32: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported
// W64: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_scalef32_pk8_bf8_f16_e64_dpp v[10:11], v[20:23], v8 row_share:1
// W32: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported
// W64: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_scalef32_pk32_bf6_bf16 v[20:25], v[10:25], v8 clamp
// W32: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// W32-NEXT:{{^}}v_cvt_scalef32_pk32_bf6_bf16 v[20:25], v[10:25], v8 clamp
// W32-NEXT:{{^}}                                                    ^
// W64: :[[@LINE-4]]:{{[0-9]+}}: error: instruction requires wavesize=32

v_cvt_scalef32_pk32_bf6_bf16 v[20:25], v[10:25], v8 mul:2
// W32: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// W32-NEXT:{{^}}v_cvt_scalef32_pk32_bf6_bf16 v[20:25], v[10:25], v8 mul:2
// W32-NEXT:{{^}}                                                    ^
// W64: :[[@LINE-4]]:{{[0-9]+}}: error: instruction requires wavesize=32

v_cvt_scalef32_pk32_bf6_bf16_dpp v[20:25], v[10:25], v8 row_share:1
// W32: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported
// W64: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_scalef32_pk32_bf6_bf16_dpp v[20:25], v[10:25], v8 dpp8:[7,6,5,4,3,2,1,0]
// W32: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported
// W64: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_scalef32_pk32_bf6_bf16_e64_dpp v[20:25], v[10:25], v8 row_share:1
// W32: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported
// W64: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_scalef32_pk32_bf6_f16 v[20:25], v[10:25], v8 clamp
// W32: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// W32-NEXT:{{^}}v_cvt_scalef32_pk32_bf6_f16 v[20:25], v[10:25], v8 clamp
// W32-NEXT:{{^}}                                                   ^
// W64: :[[@LINE-4]]:{{[0-9]+}}: error: instruction requires wavesize=32

v_cvt_scalef32_pk32_bf6_f16 v[20:25], v[10:25], v8 mul:2
// W32: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// W32-NEXT:{{^}}v_cvt_scalef32_pk32_bf6_f16 v[20:25], v[10:25], v8 mul:2
// W32-NEXT:{{^}}                                                   ^
// W64: :[[@LINE-4]]:{{[0-9]+}}: error: instruction requires wavesize=32

v_cvt_scalef32_pk32_bf6_f16_dpp v[20:25], v[10:25], v8 row_share:1
// W32: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported
// W64: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_scalef32_pk32_bf6_f16_dpp v[20:25], v[10:25], v8 dpp8:[7,6,5,4,3,2,1,0]
// W32: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported
// W64: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_scalef32_pk32_bf6_f16_e64_dpp v[20:25], v[10:25], v8 row_share:1
// W32: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported
// W64: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_scalef32_pk32_bf6_f32 v[20:25], v[6:37], v8 clamp
// W32: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// W32-NEXT:{{^}}v_cvt_scalef32_pk32_bf6_f32 v[20:25], v[6:37], v8 clamp
// W32-NEXT:{{^}}                                                  ^
// W64: :[[@LINE-4]]:{{[0-9]+}}: error: instruction requires wavesize=32

v_cvt_scalef32_pk32_bf6_f32 v[20:25], v[6:37], v8 mul:2
// W32: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// W32-NEXT:{{^}}v_cvt_scalef32_pk32_bf6_f32 v[20:25], v[6:37], v8 mul:2
// W32-NEXT:{{^}}                                                  ^
// W64: :[[@LINE-4]]:{{[0-9]+}}: error: instruction requires wavesize=32

v_cvt_scalef32_pk32_bf6_f32_dpp v[20:25], v[6:37], v8 row_share:1
// W32: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported
// W64: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_scalef32_pk32_bf6_f32_dpp v[20:25], v[6:37], v8 dpp8:[7,6,5,4,3,2,1,0]
// W32: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported
// W64: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_scalef32_pk32_bf6_f32_e64_dpp v[20:25], v[6:37], v8 row_share:1
// W32: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported
// W64: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_scalef32_pk32_fp6_bf16 v[20:25], v[10:25], v8 clamp
// W32: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// W32-NEXT:{{^}}v_cvt_scalef32_pk32_fp6_bf16 v[20:25], v[10:25], v8 clamp
// W32-NEXT:{{^}}                                                    ^
// W64: :[[@LINE-4]]:{{[0-9]+}}: error: instruction requires wavesize=32

v_cvt_scalef32_pk32_fp6_bf16 v[20:25], v[10:25], v8 mul:2
// W32: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// W32-NEXT:{{^}}v_cvt_scalef32_pk32_fp6_bf16 v[20:25], v[10:25], v8 mul:2
// W32-NEXT:{{^}}                                                    ^
// W64: :[[@LINE-4]]:{{[0-9]+}}: error: instruction requires wavesize=32

v_cvt_scalef32_pk32_fp6_bf16_dpp v[20:25], v[10:25], v8 row_share:1
// W32: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported
// W64: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_scalef32_pk32_fp6_bf16_dpp v[20:25], v[10:25], v8 dpp8:[7,6,5,4,3,2,1,0]
// W32: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported
// W64: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_scalef32_pk32_fp6_bf16_e64_dpp v[20:25], v[10:25], v8 row_share:1
// W32: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported
// W64: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_scalef32_pk32_fp6_f16 v[20:25], v[10:25], v8 clamp
// W32: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// W32-NEXT:{{^}}v_cvt_scalef32_pk32_fp6_f16 v[20:25], v[10:25], v8 clamp
// W32-NEXT:{{^}}                                                   ^
// W64: :[[@LINE-4]]:{{[0-9]+}}: error: instruction requires wavesize=32

v_cvt_scalef32_pk32_fp6_f16 v[20:25], v[10:25], v8 mul:2
// W32: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// W32-NEXT:{{^}}v_cvt_scalef32_pk32_fp6_f16 v[20:25], v[10:25], v8 mul:2
// W32-NEXT:{{^}}                                                   ^
// W64: :[[@LINE-4]]:{{[0-9]+}}: error: instruction requires wavesize=32

v_cvt_scalef32_pk32_fp6_f16_dpp v[20:25], v[10:25], v8 row_share:1
// W32: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported
// W64: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_scalef32_pk32_fp6_f16_dpp v[20:25], v[10:25], v8 dpp8:[7,6,5,4,3,2,1,0]
// W32: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported
// W64: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_scalef32_pk32_fp6_f16_e64_dpp v[20:25], v[10:25], v8 row_share:1
// W32: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported
// W64: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_scalef32_pk32_fp6_f32 v[20:25], v[6:37], v8 clamp
// W32: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// W32-NEXT:{{^}}v_cvt_scalef32_pk32_fp6_f32 v[20:25], v[6:37], v8 clamp
// W32-NEXT:{{^}}                                                  ^
// W64: :[[@LINE-4]]:{{[0-9]+}}: error: instruction requires wavesize=32

v_cvt_scalef32_pk32_fp6_f32 v[20:25], v[6:37], v8 mul:2
// W32: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// W32-NEXT:{{^}}v_cvt_scalef32_pk32_fp6_f32 v[20:25], v[6:37], v8 mul:2
// W32-NEXT:{{^}}                                                  ^
// W64: :[[@LINE-4]]:{{[0-9]+}}: error: instruction requires wavesize=32

v_cvt_scalef32_pk32_fp6_f32_dpp v[20:25], v[6:37], v8 row_share:1
// W32: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported
// W64: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_scalef32_pk32_fp6_f32_dpp v[20:25], v[6:37], v8 dpp8:[7,6,5,4,3,2,1,0]
// W32: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported
// W64: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_scalef32_pk32_fp6_f32_e64_dpp v[20:25], v[6:37], v8 row_share:1
// W32: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported
// W64: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_scalef32_sr_pk32_bf6_bf16 v[0:5], v[6:21], v22, v23 clamp
// W32: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// W32-NEXT:{{^}}v_cvt_scalef32_sr_pk32_bf6_bf16 v[0:5], v[6:21], v22, v23 clamp
// W32-NEXT:{{^}}                                                          ^
// W64: :[[@LINE-4]]:{{[0-9]+}}: error: instruction requires wavesize=32

v_cvt_scalef32_sr_pk32_bf6_bf16 v[0:5], v[6:21], v22, v23 mul:2
// W32: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// W32-NEXT:{{^}}v_cvt_scalef32_sr_pk32_bf6_bf16 v[0:5], v[6:21], v22, v23 mul:2
// W32-NEXT:{{^}}                                                          ^
// W64: :[[@LINE-4]]:{{[0-9]+}}: error: instruction requires wavesize=32

v_cvt_scalef32_sr_pk32_bf6_bf16_dpp v[0:5], v[6:21], v22, v23 row_share:1
// W32: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported
// W64: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_scalef32_sr_pk32_bf6_bf16_dpp v[0:5], v[6:21], v22, v23 dpp8:[7,6,5,4,3,2,1,0]
// W32: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported
// W64: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_scalef32_sr_pk32_bf6_bf16_e64_dpp v[0:5], v[6:21], v22, v23 row_share:1
// W32: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported
// W64: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_scalef32_sr_pk32_bf6_f16 v[0:5], v[6:21], v22, v23 clamp
// W32: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// W32-NEXT:{{^}}v_cvt_scalef32_sr_pk32_bf6_f16 v[0:5], v[6:21], v22, v23 clamp
// W32-NEXT:{{^}}                                                         ^
// W64: :[[@LINE-4]]:{{[0-9]+}}: error: instruction requires wavesize=32

v_cvt_scalef32_sr_pk32_bf6_f16 v[0:5], v[6:21], v22, v23 mul:2
// W32: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// W32-NEXT:{{^}}v_cvt_scalef32_sr_pk32_bf6_f16 v[0:5], v[6:21], v22, v23 mul:2
// W32-NEXT:{{^}}                                                         ^
// W64: :[[@LINE-4]]:{{[0-9]+}}: error: instruction requires wavesize=32

v_cvt_scalef32_sr_pk32_bf6_f16_dpp v[0:5], v[6:21], v22, v23 row_share:1
// W32: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported
// W64: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_scalef32_sr_pk32_bf6_f16_dpp v[0:5], v[6:21], v22, v23 dpp8:[7,6,5,4,3,2,1,0]
// W32: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported
// W64: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_scalef32_sr_pk32_bf6_f16_e64_dpp v[0:5], v[6:21], v22, v23 row_share:1
// W32: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported
// W64: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_scalef32_sr_pk32_bf6_f32 v[0:5], v[6:37], v38, v39 clamp
// W32: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// W32-NEXT:{{^}}v_cvt_scalef32_sr_pk32_bf6_f32 v[0:5], v[6:37], v38, v39 clamp
// W32-NEXT:{{^}}                                                         ^
// W64: :[[@LINE-4]]:{{[0-9]+}}: error: instruction requires wavesize=32

v_cvt_scalef32_sr_pk32_bf6_f32 v[0:5], v[6:37], v38, v39 mul:2
// W32: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// W32-NEXT:{{^}}v_cvt_scalef32_sr_pk32_bf6_f32 v[0:5], v[6:37], v38, v39 mul:2
// W32-NEXT:{{^}}                                                         ^
// W64: :[[@LINE-4]]:{{[0-9]+}}: error: instruction requires wavesize=32

v_cvt_scalef32_sr_pk32_bf6_f32_dpp v[0:5], v[6:37], v38, v39 row_share:1
// W32: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported
// W64: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_scalef32_sr_pk32_bf6_f32_dpp v[0:5], v[6:37], v38, v39 dpp8:[7,6,5,4,3,2,1,0]
// W32: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported
// W64: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_scalef32_sr_pk32_bf6_f32_e64_dpp v[0:5], v[6:37], v38, v39 row_share:1
// W32: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported
// W64: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_scalef32_sr_pk32_fp6_bf16 v[0:5], v[6:21], v22, v23 clamp
// W32: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// W32-NEXT:{{^}}v_cvt_scalef32_sr_pk32_fp6_bf16 v[0:5], v[6:21], v22, v23 clamp
// W32-NEXT:{{^}}                                                          ^
// W64: :[[@LINE-4]]:{{[0-9]+}}: error: instruction requires wavesize=32

v_cvt_scalef32_sr_pk32_fp6_bf16 v[0:5], v[6:21], v22, v23 mul:2
// W32: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// W32-NEXT:{{^}}v_cvt_scalef32_sr_pk32_fp6_bf16 v[0:5], v[6:21], v22, v23 mul:2
// W32-NEXT:{{^}}                                                          ^
// W64: :[[@LINE-4]]:{{[0-9]+}}: error: instruction requires wavesize=32

v_cvt_scalef32_sr_pk32_fp6_bf16_dpp v[0:5], v[6:21], v22, v23 row_share:1
// W32: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported
// W64: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_scalef32_sr_pk32_fp6_bf16_dpp v[0:5], v[6:21], v22, v23 dpp8:[7,6,5,4,3,2,1,0]
// W32: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported
// W64: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_scalef32_sr_pk32_fp6_bf16_e64_dpp v[0:5], v[6:21], v22, v23 row_share:1
// W32: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported
// W64: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_scalef32_sr_pk32_fp6_f16 v[0:5], v[6:21], v22, v23 clamp
// W32: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// W32-NEXT:{{^}}v_cvt_scalef32_sr_pk32_fp6_f16 v[0:5], v[6:21], v22, v23 clamp
// W32-NEXT:{{^}}                                                         ^
// W64: :[[@LINE-4]]:{{[0-9]+}}: error: instruction requires wavesize=32

v_cvt_scalef32_sr_pk32_fp6_f16 v[0:5], v[6:21], v22, v23 mul:2
// W32: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// W32-NEXT:{{^}}v_cvt_scalef32_sr_pk32_fp6_f16 v[0:5], v[6:21], v22, v23 mul:2
// W32-NEXT:{{^}}                                                         ^
// W64: :[[@LINE-4]]:{{[0-9]+}}: error: instruction requires wavesize=32

v_cvt_scalef32_sr_pk32_fp6_f16_dpp v[0:5], v[6:21], v22, v23 row_share:1
// W32: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported
// W64: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_scalef32_sr_pk32_fp6_f16_dpp v[0:5], v[6:21], v22, v23 dpp8:[7,6,5,4,3,2,1,0]
// W32: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported
// W64: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_scalef32_sr_pk32_fp6_f16_e64_dpp v[0:5], v[6:21], v22, v23 row_share:1
// W32: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported
// W64: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_scalef32_sr_pk32_fp6_f32 v[0:5], v[6:37], v38, v39 clamp
// W32: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// W32-NEXT:{{^}}v_cvt_scalef32_sr_pk32_fp6_f32 v[0:5], v[6:37], v38, v39 clamp
// W32-NEXT:{{^}}                                                         ^
// W64: :[[@LINE-4]]:{{[0-9]+}}: error: instruction requires wavesize=32

v_cvt_scalef32_sr_pk32_fp6_f32 v[0:5], v[6:37], v38, v39 mul:2
// W32: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// W32-NEXT:{{^}}v_cvt_scalef32_sr_pk32_fp6_f32 v[0:5], v[6:37], v38, v39 mul:2
// W32-NEXT:{{^}}                                                         ^
// W64: :[[@LINE-4]]:{{[0-9]+}}: error: instruction requires wavesize=32

v_cvt_scalef32_sr_pk32_fp6_f32_dpp v[0:5], v[6:37], v38, v39 row_share:1
// W32: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported
// W64: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_scalef32_sr_pk32_fp6_f32_dpp v[0:5], v[6:37], v38, v39 dpp8:[7,6,5,4,3,2,1,0]
// W32: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported
// W64: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_scalef32_sr_pk32_fp6_f32_e64_dpp v[0:5], v[6:37], v38, v39 row_share:1
// W32: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported
// W64: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU
