// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx1210 -show-encoding %s 2>&1 | FileCheck --check-prefix=GFX1210-ERR --implicit-check-not=error: --strict-whitespace %s

v_lshl_add_u64 v[2:3], v[4:5], v7, v[8:9] dpp8:[7,6,5,4,3,2,1,0]
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX1210-ERR-NEXT:{{^}}v_lshl_add_u64 v[2:3], v[4:5], v7, v[8:9] dpp8:[7,6,5,4,3,2,1,0]
// GFX1210-ERR-NEXT:{{^}}                                          ^

v_fma_f64 v[4:5], v[2:3], v[6:7], v[8:9] dpp8:[7,6,5,4,3,2,1,0]
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX1210-ERR-NEXT:{{^}}v_fma_f64 v[4:5], v[2:3], v[6:7], v[8:9] dpp8:[7,6,5,4,3,2,1,0]
// GFX1210-ERR-NEXT:{{^}}                                         ^

v_div_fixup_f64 v[4:5], v[2:3], v[6:7], v[8:9] dpp8:[7,6,5,4,3,2,1,0]
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX1210-ERR-NEXT:{{^}}v_div_fixup_f64 v[4:5], v[2:3], v[6:7], v[8:9] dpp8:[7,6,5,4,3,2,1,0]
// GFX1210-ERR-NEXT:{{^}}                                               ^

v_div_fmas_f64 v[4:5], v[2:3], v[6:7], v[8:9] dpp8:[7,6,5,4,3,2,1,0]
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX1210-ERR-NEXT:{{^}}v_div_fmas_f64 v[4:5], v[2:3], v[6:7], v[8:9] dpp8:[7,6,5,4,3,2,1,0]
// GFX1210-ERR-NEXT:{{^}}                                              ^

v_div_scale_f64 v[4:5], s2, v[2:3], v[6:7], v[8:9] dpp8:[7,6,5,4,3,2,1,0]
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX1210-ERR-NEXT:{{^}}v_div_scale_f64 v[4:5], s2, v[2:3], v[6:7], v[8:9] dpp8:[7,6,5,4,3,2,1,0]
// GFX1210-ERR-NEXT:{{^}}                                                   ^

v_mad_co_u64_u32 v[4:5], s2, v2, v6, v[8:9] dpp8:[7,6,5,4,3,2,1,0]
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX1210-ERR-NEXT:{{^}}v_mad_co_u64_u32 v[4:5], s2, v2, v6, v[8:9] dpp8:[7,6,5,4,3,2,1,0]
// GFX1210-ERR-NEXT:{{^}}                                            ^

v_mad_co_i64_i32 v[4:5], s2, v2, v6, v[8:9] dpp8:[7,6,5,4,3,2,1,0]
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX1210-ERR-NEXT:{{^}}v_mad_co_i64_i32 v[4:5], s2, v2, v6, v[8:9] dpp8:[7,6,5,4,3,2,1,0]
// GFX1210-ERR-NEXT:{{^}}                                            ^

v_minimum_f64 v[4:5], v[2:3], v[6:7] dpp8:[7,6,5,4,3,2,1,0]
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX1210-ERR-NEXT:{{^}}v_minimum_f64 v[4:5], v[2:3], v[6:7] dpp8:[7,6,5,4,3,2,1,0]
// GFX1210-ERR-NEXT:{{^}}                                     ^

v_maximum_f64 v[4:5], v[2:3], v[6:7] dpp8:[7,6,5,4,3,2,1,0]
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX1210-ERR-NEXT:{{^}}v_maximum_f64 v[4:5], v[2:3], v[6:7] dpp8:[7,6,5,4,3,2,1,0]
// GFX1210-ERR-NEXT:{{^}}                                     ^

v_ldexp_f64 v[4:5], v[2:3], v6 dpp8:[7,6,5,4,3,2,1,0]
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX1210-ERR-NEXT:{{^}}v_ldexp_f64 v[4:5], v[2:3], v6 dpp8:[7,6,5,4,3,2,1,0]
// GFX1210-ERR-NEXT:{{^}}                               ^

v_mul_lo_u32 v4, v2, v6 dpp8:[7,6,5,4,3,2,1,0]
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX1210-ERR-NEXT:{{^}}v_mul_lo_u32 v4, v2, v6 dpp8:[7,6,5,4,3,2,1,0]
// GFX1210-ERR-NEXT:{{^}}                        ^

v_mul_hi_u32 v4, v2, v6 dpp8:[7,6,5,4,3,2,1,0]
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX1210-ERR-NEXT:{{^}}v_mul_hi_u32 v4, v2, v6 dpp8:[7,6,5,4,3,2,1,0]
// GFX1210-ERR-NEXT:{{^}}                        ^

v_mul_hi_i32 v4, v2, v6 dpp8:[7,6,5,4,3,2,1,0]
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX1210-ERR-NEXT:{{^}}v_mul_hi_i32 v4, v2, v6 dpp8:[7,6,5,4,3,2,1,0]
// GFX1210-ERR-NEXT:{{^}}                        ^

v_lshrrev_b64 v[4:5], v2, v[6:7] dpp8:[7,6,5,4,3,2,1,0]
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX1210-ERR-NEXT:{{^}}v_lshrrev_b64 v[4:5], v2, v[6:7] dpp8:[7,6,5,4,3,2,1,0]
// GFX1210-ERR-NEXT:{{^}}                                 ^

v_ashrrev_i64 v[4:5], v2, v[6:7] dpp8:[7,6,5,4,3,2,1,0]
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX1210-ERR-NEXT:{{^}}v_ashrrev_i64 v[4:5], v2, v[6:7] dpp8:[7,6,5,4,3,2,1,0]
// GFX1210-ERR-NEXT:{{^}}                                 ^

v_mad_u32 v2, v4, v7, v8 dpp8:[7,6,5,4,3,2,1,0]
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX1210-ERR-NEXT:{{^}}v_mad_u32 v2, v4, v7, v8 dpp8:[7,6,5,4,3,2,1,0]
// GFX1210-ERR-NEXT:{{^}}                         ^

v_max_i64 v[2:3], v[4:5], v[6:7] dpp8:[7,6,5,4,3,2,1,0]
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX1210-ERR-NEXT:{{^}}v_max_i64 v[2:3], v[4:5], v[6:7] dpp8:[7,6,5,4,3,2,1,0]
// GFX1210-ERR-NEXT:{{^}}                                 ^

v_max_u64 v[2:3], v[4:5], v[6:7] dpp8:[7,6,5,4,3,2,1,0]
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX1210-ERR-NEXT:{{^}}v_max_u64 v[2:3], v[4:5], v[6:7] dpp8:[7,6,5,4,3,2,1,0]
// GFX1210-ERR-NEXT:{{^}}                                 ^

v_min_i64 v[2:3], v[4:5], v[6:7] dpp8:[7,6,5,4,3,2,1,0]
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX1210-ERR-NEXT:{{^}}v_min_i64 v[2:3], v[4:5], v[6:7] dpp8:[7,6,5,4,3,2,1,0]
// GFX1210-ERR-NEXT:{{^}}                                 ^

v_min_u64 v[2:3], v[4:5], v[6:7] dpp8:[7,6,5,4,3,2,1,0]
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX1210-ERR-NEXT:{{^}}v_min_u64 v[2:3], v[4:5], v[6:7] dpp8:[7,6,5,4,3,2,1,0]
// GFX1210-ERR-NEXT:{{^}}                                 ^

v_mad_nc_u64_u32 v[4:5], v2, v5, v[6:7] dpp8:[7,6,5,4,3,2,1,0]
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX1210-ERR-NEXT:{{^}}v_mad_nc_u64_u32 v[4:5], v2, v5, v[6:7] dpp8:[7,6,5,4,3,2,1,0]
// GFX1210-ERR-NEXT:{{^}}                                        ^

v_mad_nc_i64_i32 v[4:5], v2, v5, v[6:7] dpp8:[7,6,5,4,3,2,1,0]
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX1210-ERR-NEXT:{{^}}v_mad_nc_i64_i32 v[4:5], v2, v5, v[6:7] dpp8:[7,6,5,4,3,2,1,0]
// GFX1210-ERR-NEXT:{{^}}                                        ^

v_lshl_add_u64 v[2:3], v[4:5], v7, v[8:9] quad_perm:[3,2,1,0]
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: DP ALU dpp only supports row_share
// GFX1210-ERR-NEXT:{{^}}v_lshl_add_u64 v[2:3], v[4:5], v7, v[8:9] quad_perm:[3,2,1,0]
// GFX1210-ERR-NEXT:{{^}}                                          ^

v_fma_f64 v[4:5], v[2:3], v[6:7], v[8:9] quad_perm:[3,2,1,0]
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: DP ALU dpp only supports row_share
// GFX1210-ERR-NEXT:{{^}}v_fma_f64 v[4:5], v[2:3], v[6:7], v[8:9] quad_perm:[3,2,1,0]
// GFX1210-ERR-NEXT:{{^}}                                         ^

v_div_fixup_f64 v[4:5], v[2:3], v[6:7], v[8:9] quad_perm:[3,2,1,0]
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: DP ALU dpp only supports row_share
// GFX1210-ERR-NEXT:{{^}}v_div_fixup_f64 v[4:5], v[2:3], v[6:7], v[8:9] quad_perm:[3,2,1,0]
// GFX1210-ERR-NEXT:{{^}}                                               ^

v_div_fmas_f64 v[4:5], v[2:3], v[6:7], v[8:9] quad_perm:[3,2,1,0]
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: DP ALU dpp only supports row_share
// GFX1210-ERR-NEXT:{{^}}v_div_fmas_f64 v[4:5], v[2:3], v[6:7], v[8:9] quad_perm:[3,2,1,0]
// GFX1210-ERR-NEXT:{{^}}                                              ^

v_div_scale_f64 v[4:5], s2, v[2:3], v[6:7], v[8:9] quad_perm:[3,2,1,0]
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: DP ALU dpp only supports row_share
// GFX1210-ERR-NEXT:{{^}}v_div_scale_f64 v[4:5], s2, v[2:3], v[6:7], v[8:9] quad_perm:[3,2,1,0]
// GFX1210-ERR-NEXT:{{^}}                                                   ^

v_mad_co_u64_u32 v[4:5], s2, v2, v6, v[8:9] quad_perm:[3,2,1,0]
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: DP ALU dpp only supports row_share
// GFX1210-ERR-NEXT:{{^}}v_mad_co_u64_u32 v[4:5], s2, v2, v6, v[8:9] quad_perm:[3,2,1,0]
// GFX1210-ERR-NEXT:{{^}}                                            ^

v_mad_co_i64_i32 v[4:5], s2, v2, v6, v[8:9] quad_perm:[3,2,1,0]
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: DP ALU dpp only supports row_share
// GFX1210-ERR-NEXT:{{^}}v_mad_co_i64_i32 v[4:5], s2, v2, v6, v[8:9] quad_perm:[3,2,1,0]
// GFX1210-ERR-NEXT:{{^}}                                            ^

v_minimum_f64 v[4:5], v[2:3], v[6:7] quad_perm:[3,2,1,0]
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: DP ALU dpp only supports row_share
// GFX1210-ERR-NEXT:{{^}}v_minimum_f64 v[4:5], v[2:3], v[6:7] quad_perm:[3,2,1,0]
// GFX1210-ERR-NEXT:{{^}}                                     ^

v_maximum_f64 v[4:5], v[2:3], v[6:7] quad_perm:[3,2,1,0]
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: DP ALU dpp only supports row_share
// GFX1210-ERR-NEXT:{{^}}v_maximum_f64 v[4:5], v[2:3], v[6:7] quad_perm:[3,2,1,0]
// GFX1210-ERR-NEXT:{{^}}                                     ^

v_ldexp_f64 v[4:5], v[2:3], v6 quad_perm:[3,2,1,0]
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: DP ALU dpp only supports row_share
// GFX1210-ERR-NEXT:{{^}}v_ldexp_f64 v[4:5], v[2:3], v6 quad_perm:[3,2,1,0]
// GFX1210-ERR-NEXT:{{^}}                               ^

v_mul_lo_u32 v4, v2, v6 quad_perm:[3,2,1,0]
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: DP ALU dpp only supports row_share
// GFX1210-ERR-NEXT:{{^}}v_mul_lo_u32 v4, v2, v6 quad_perm:[3,2,1,0]
// GFX1210-ERR-NEXT:{{^}}                        ^

v_mul_hi_u32 v4, v2, v6 quad_perm:[3,2,1,0]
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: DP ALU dpp only supports row_share
// GFX1210-ERR-NEXT:{{^}}v_mul_hi_u32 v4, v2, v6 quad_perm:[3,2,1,0]
// GFX1210-ERR-NEXT:{{^}}                        ^

v_mul_hi_i32 v4, v2, v6 quad_perm:[3,2,1,0]
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: DP ALU dpp only supports row_share
// GFX1210-ERR-NEXT:{{^}}v_mul_hi_i32 v4, v2, v6 quad_perm:[3,2,1,0]
// GFX1210-ERR-NEXT:{{^}}                        ^

v_lshrrev_b64 v[4:5], v2, v[6:7] quad_perm:[3,2,1,0]
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: DP ALU dpp only supports row_share
// GFX1210-ERR-NEXT:{{^}}v_lshrrev_b64 v[4:5], v2, v[6:7] quad_perm:[3,2,1,0]
// GFX1210-ERR-NEXT:{{^}}                                 ^

v_ashrrev_i64 v[4:5], v2, v[6:7] quad_perm:[3,2,1,0]
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: DP ALU dpp only supports row_share
// GFX1210-ERR-NEXT:{{^}}v_ashrrev_i64 v[4:5], v2, v[6:7] quad_perm:[3,2,1,0]
// GFX1210-ERR-NEXT:{{^}}                                 ^

v_mad_u32 v2, v4, v7, v8 quad_perm:[3,2,1,0]
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: DP ALU dpp only supports row_share
// GFX1210-ERR-NEXT:{{^}}v_mad_u32 v2, v4, v7, v8 quad_perm:[3,2,1,0]
// GFX1210-ERR-NEXT:{{^}}                         ^

v_max_i64 v[2:3], v[4:5], v[6:7] quad_perm:[3,2,1,0]
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: DP ALU dpp only supports row_share
// GFX1210-ERR-NEXT:{{^}}v_max_i64 v[2:3], v[4:5], v[6:7] quad_perm:[3,2,1,0]
// GFX1210-ERR-NEXT:{{^}}                                 ^

v_max_u64 v[2:3], v[4:5], v[6:7] quad_perm:[3,2,1,0]
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: DP ALU dpp only supports row_share
// GFX1210-ERR-NEXT:{{^}}v_max_u64 v[2:3], v[4:5], v[6:7] quad_perm:[3,2,1,0]
// GFX1210-ERR-NEXT:{{^}}                                 ^

v_min_i64 v[2:3], v[4:5], v[6:7] quad_perm:[3,2,1,0]
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: DP ALU dpp only supports row_share
// GFX1210-ERR-NEXT:{{^}}v_min_i64 v[2:3], v[4:5], v[6:7] quad_perm:[3,2,1,0]
// GFX1210-ERR-NEXT:{{^}}                                 ^

v_min_u64 v[2:3], v[4:5], v[6:7] quad_perm:[3,2,1,0]
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: DP ALU dpp only supports row_share
// GFX1210-ERR-NEXT:{{^}}v_min_u64 v[2:3], v[4:5], v[6:7] quad_perm:[3,2,1,0]
// GFX1210-ERR-NEXT:{{^}}                                 ^

v_mad_nc_u64_u32 v[4:5], v2, v5, v[6:7] quad_perm:[3,2,1,0]
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: DP ALU dpp only supports row_share
// GFX1210-ERR-NEXT:{{^}}v_mad_nc_u64_u32 v[4:5], v2, v5, v[6:7] quad_perm:[3,2,1,0]
// GFX1210-ERR-NEXT:{{^}}                                        ^

v_mad_nc_i64_i32 v[4:5], v2, v5, v[6:7] quad_perm:[3,2,1,0]
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: DP ALU dpp only supports row_share
// GFX1210-ERR-NEXT:{{^}}v_mad_nc_i64_i32 v[4:5], v2, v5, v[6:7] quad_perm:[3,2,1,0]
// GFX1210-ERR-NEXT:{{^}}                                        ^

v_trig_preop_f64 v[4:5], v[8:9], v2 row_share:1
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX1210-ERR-NEXT:{{^}}v_trig_preop_f64 v[4:5], v[8:9], v2 row_share:1
// GFX1210-ERR-NEXT:{{^}}                                    ^

v_ashr_pk_i8_i32 v1, v2, v3, v4 clamp
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// GFX1210-ERR-NEXT:{{^}}v_ashr_pk_i8_i32 v1, v2, v3, v4 clamp
// GFX1210-ERR-NEXT:{{^}}                                ^

v_ashr_pk_u8_i32 v1, v2, v3, v4 clamp
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// GFX1210-ERR-NEXT:{{^}}v_ashr_pk_u8_i32 v1, v2, v3, v4 clamp
// GFX1210-ERR-NEXT:{{^}}                                ^

v_cvt_sr_bf8_f16 v1, v2, v3 clamp
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// GFX1210-ERR-NEXT:{{^}}v_cvt_sr_bf8_f16 v1, v2, v3 clamp
// GFX1210-ERR-NEXT:{{^}}                            ^

v_cvt_sr_bf8_f16 v1, v2, v3 mul:2
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX1210-ERR-NEXT:{{^}}v_cvt_sr_bf8_f16 v1, v2, v3 mul:2
// GFX1210-ERR-NEXT:{{^}}                            ^

v_cvt_sr_fp8_f16 v1, v2, v3 clamp
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// GFX1210-ERR-NEXT:{{^}}v_cvt_sr_fp8_f16 v1, v2, v3 clamp
// GFX1210-ERR-NEXT:{{^}}                            ^

v_cvt_sr_fp8_f16 v1, v2, v3 mul:2
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX1210-ERR-NEXT:{{^}}v_cvt_sr_fp8_f16 v1, v2, v3 mul:2
// GFX1210-ERR-NEXT:{{^}}                            ^

v_cvt_scale_pk8_f32_fp8 v[10:17], v[20:21], v8 scale_sel:8
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// GFX1210-ERR-NEXT:{{^}}v_cvt_scale_pk8_f32_fp8 v[10:17], v[20:21], v8 scale_sel:8
// GFX1210-ERR-NEXT:{{^}}                                               ^

v_cvt_sr_bf8_f16 v1, v2, v3 byte_sel:4
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid byte_sel value.
// GFX1210-ERR-NEXT:{{^}}v_cvt_sr_bf8_f16 v1, v2, v3 byte_sel:4
// GFX1210-ERR-NEXT:{{^}}                            ^
