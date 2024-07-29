// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx1300 -mattr=+wavefrontsize32 -show-encoding %s 2>&1 | FileCheck --check-prefix=GFX13 --strict-whitespace --implicit-check-not=error %s
// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx1300 -mattr=+wavefrontsize64 -show-encoding %s 2>&1 | FileCheck --check-prefix=GFX13 --strict-whitespace --implicit-check-not=error %s

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

v_mad_nc_i64_i32 v[2:3], v4, v5, 1 row_share:0 row_mask:0xf bank_mask:0xf
// GFX13: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX13-NEXT:{{^}}v_mad_nc_i64_i32 v[2:3], v4, v5, 1 row_share:0 row_mask:0xf bank_mask:0xf
// GFX13-NEXT:{{^}}                                   ^

v_mad_nc_i64_i32 v[2:3], v4, v7, v[8:9] row_share:3 fi:1
// GFX13: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX13-NEXT:{{^}}v_mad_nc_i64_i32 v[2:3], v4, v7, v[8:9] row_share:3 fi:1
// GFX13-NEXT:{{^}}                                        ^

v_mad_nc_i64_i32 v[4:5], v2, v5, v[6:7] dpp8:[7,6,5,4,3,2,1,0]
// GFX13: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX13-NEXT:{{^}}v_mad_nc_i64_i32 v[4:5], v2, v5, v[6:7] dpp8:[7,6,5,4,3,2,1,0]
// GFX13-NEXT:{{^}}                                        ^

v_mad_nc_i64_i32 v[4:5], v2, v5, v[6:7] quad_perm:[3,2,1,0]
// GFX13: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX13-NEXT:{{^}}v_mad_nc_i64_i32 v[4:5], v2, v5, v[6:7] quad_perm:[3,2,1,0]
// GFX13-NEXT:{{^}}                                        ^

v_mad_nc_u64_u32 v[2:3], v4, v5, 1 row_share:0 row_mask:0xf bank_mask:0xf
// GFX13: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX13-NEXT:{{^}}v_mad_nc_u64_u32 v[2:3], v4, v5, 1 row_share:0 row_mask:0xf bank_mask:0xf
// GFX13-NEXT:{{^}}                                   ^

v_mad_nc_u64_u32 v[2:3], v4, v7, v[8:9] row_share:3 fi:1
// GFX13: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX13-NEXT:{{^}}v_mad_nc_u64_u32 v[2:3], v4, v7, v[8:9] row_share:3 fi:1
// GFX13-NEXT:{{^}}                                        ^

v_mad_nc_u64_u32 v[4:5], v2, v5, v[6:7] dpp8:[7,6,5,4,3,2,1,0]
// GFX13: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX13-NEXT:{{^}}v_mad_nc_u64_u32 v[4:5], v2, v5, v[6:7] dpp8:[7,6,5,4,3,2,1,0]
// GFX13-NEXT:{{^}}                                        ^

v_mad_nc_u64_u32 v[4:5], v2, v5, v[6:7] quad_perm:[3,2,1,0]
// GFX13: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX13-NEXT:{{^}}v_mad_nc_u64_u32 v[4:5], v2, v5, v[6:7] quad_perm:[3,2,1,0]
// GFX13-NEXT:{{^}}                                        ^

v_mad_u32 v2, v4, v7, v8 dpp8:[7,6,5,4,3,2,1,0]
// GFX13: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX13-NEXT:{{^}}v_mad_u32 v2, v4, v7, v8 dpp8:[7,6,5,4,3,2,1,0]
// GFX13-NEXT:{{^}}                         ^

v_max_i64 v[2:3], v[4:5], v[6:7] dpp8:[7,6,5,4,3,2,1,0]
// GFX13: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX13-NEXT:{{^}}v_max_i64 v[2:3], v[4:5], v[6:7] dpp8:[7,6,5,4,3,2,1,0]
// GFX13-NEXT:{{^}}                                 ^

v_max_u64 v[2:3], v[4:5], v[6:7] dpp8:[7,6,5,4,3,2,1,0]
// GFX13: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX13-NEXT:{{^}}v_max_u64 v[2:3], v[4:5], v[6:7] dpp8:[7,6,5,4,3,2,1,0]
// GFX13-NEXT:{{^}}                                 ^

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

v_min_i64 v[2:3], v[4:5], v[6:7] dpp8:[7,6,5,4,3,2,1,0]
// GFX13: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX13-NEXT:{{^}}v_min_i64 v[2:3], v[4:5], v[6:7] dpp8:[7,6,5,4,3,2,1,0]
// GFX13-NEXT:{{^}}                                 ^

v_min_u64 v[2:3], v[4:5], v[6:7] dpp8:[7,6,5,4,3,2,1,0]
// GFX13: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX13-NEXT:{{^}}v_min_u64 v[2:3], v[4:5], v[6:7] dpp8:[7,6,5,4,3,2,1,0]
// GFX13-NEXT:{{^}}                                 ^

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
