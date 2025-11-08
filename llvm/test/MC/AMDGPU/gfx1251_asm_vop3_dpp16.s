// RUN: llvm-mc -triple=amdgcn -mcpu=gfx1251 -show-encoding < %s | FileCheck --check-prefix=GFX1251 %s
// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx1250 -show-encoding %s 2>&1 | FileCheck --check-prefix=GFX1250-ERR --implicit-check-not=error: --strict-whitespace %s

v_lshl_add_u64 v[2:3], v[4:5], v7, v[8:9] row_share:3
// GFX1251: v_lshl_add_u64_e64_dpp v[2:3], v[4:5], v7, v[8:9] row_share:3 row_mask:0xf bank_mask:0xf ; encoding: [0x02,0x00,0x52,0xd6,0xfa,0x0e,0x22,0x04,0x04,0x53,0x01,0xff]
// GFX1250-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: not a valid operand.

v_lshl_add_u64 v[2:3], v[4:5], v4, v[2:3] row_share:0 row_mask:0xf bank_mask:0xf
// GFX1251: v_lshl_add_u64_e64_dpp v[2:3], v[4:5], v4, v[2:3] row_share:0 row_mask:0xf bank_mask:0xf ; encoding: [0x02,0x00,0x52,0xd6,0xfa,0x08,0x0a,0x04,0x04,0x50,0x01,0xff]
// GFX1250-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: not a valid operand.

v_fma_f64 v[4:5], v[2:3], v[6:7], v[8:9] row_share:1
// GFX1251: v_fma_f64_e64_dpp v[4:5], v[2:3], v[6:7], v[8:9] row_share:1 row_mask:0xf bank_mask:0xf ; encoding: [0x04,0x00,0x14,0xd6,0xfa,0x0c,0x22,0x04,0x02,0x51,0x01,0xff]
// GFX1250-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: not a valid operand.
// GFX1250-ERR-NEXT:{{^}}v_fma_f64 v[4:5], v[2:3], v[6:7], v[8:9] row_share:1
// GFX1250-ERR-NEXT:{{^}}                                         ^

v_div_fixup_f64 v[4:5], v[2:3], v[6:7], v[8:9] row_share:1
// GFX1251: v_div_fixup_f64_e64_dpp v[4:5], v[2:3], v[6:7], v[8:9] row_share:1 row_mask:0xf bank_mask:0xf ; encoding: [0x04,0x00,0x28,0xd6,0xfa,0x0c,0x22,0x04,0x02,0x51,0x01,0xff]
// GFX1250-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: not a valid operand.
// GFX1250-ERR-NEXT:{{^}}v_div_fixup_f64 v[4:5], v[2:3], v[6:7], v[8:9] row_share:1
// GFX1250-ERR-NEXT:{{^}}                                               ^

v_div_fmas_f64 v[4:5], v[2:3], v[6:7], v[8:9] row_share:1
// GFX1251: v_div_fmas_f64_e64_dpp v[4:5], v[2:3], v[6:7], v[8:9] row_share:1 row_mask:0xf bank_mask:0xf ; encoding: [0x04,0x00,0x38,0xd6,0xfa,0x0c,0x22,0x04,0x02,0x51,0x01,0xff]
// GFX1250-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: not a valid operand.
// GFX1250-ERR-NEXT:{{^}}v_div_fmas_f64 v[4:5], v[2:3], v[6:7], v[8:9] row_share:1
// GFX1250-ERR-NEXT:{{^}}                                              ^

v_div_scale_f64 v[4:5], s2, v[2:3], v[6:7], v[8:9] row_share:1
// GFX1251: v_div_scale_f64_e64_dpp v[4:5], s2, v[2:3], v[6:7], v[8:9] row_share:1 row_mask:0xf bank_mask:0xf ; encoding: [0x04,0x02,0xfd,0xd6,0xfa,0x0c,0x22,0x04,0x02,0x51,0x01,0xff]
// GFX1250-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: not a valid operand.
// GFX1250-ERR-NEXT:{{^}}v_div_scale_f64 v[4:5], s2, v[2:3], v[6:7], v[8:9] row_share:1
// GFX1250-ERR-NEXT:{{^}}                                                   ^

v_mad_co_u64_u32 v[4:5], s2, v2, v6, v[8:9] row_share:1
// GFX1251: v_mad_co_u64_u32_e64_dpp v[4:5], s2, v2, v6, v[8:9] row_share:1 row_mask:0xf bank_mask:0xf ; encoding: [0x04,0x02,0xfe,0xd6,0xfa,0x0c,0x22,0x04,0x02,0x51,0x01,0xff]
// GFX1250-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: not a valid operand.
// GFX1250-ERR-NEXT:{{^}}v_mad_co_u64_u32 v[4:5], s2, v2, v6, v[8:9] row_share:1
// GFX1250-ERR-NEXT:{{^}}                                            ^

v_mad_co_i64_i32 v[4:5], s2, v2, v6, v[8:9] row_share:1
// GFX1251: v_mad_co_i64_i32_e64_dpp v[4:5], s2, v2, v6, v[8:9] row_share:1 row_mask:0xf bank_mask:0xf ; encoding: [0x04,0x02,0xff,0xd6,0xfa,0x0c,0x22,0x04,0x02,0x51,0x01,0xff]
// GFX1250-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: not a valid operand.
// GFX1250-ERR-NEXT:{{^}}v_mad_co_i64_i32 v[4:5], s2, v2, v6, v[8:9] row_share:1
// GFX1250-ERR-NEXT:{{^}}                                            ^

v_minimum_f64 v[4:5], v[2:3], v[6:7] row_share:1
// GFX1251: v_minimum_f64_e64_dpp v[4:5], v[2:3], v[6:7] row_share:1 row_mask:0xf bank_mask:0xf ; encoding: [0x04,0x00,0x41,0xd7,0xfa,0x0c,0x02,0x00,0x02,0x51,0x01,0xff]
// GFX1250-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: not a valid operand.
// GFX1250-ERR-NEXT:{{^}}v_minimum_f64 v[4:5], v[2:3], v[6:7] row_share:1
// GFX1250-ERR-NEXT:{{^}}                                     ^

v_maximum_f64 v[4:5], v[2:3], v[6:7] row_share:1
// GFX1251: v_maximum_f64_e64_dpp v[4:5], v[2:3], v[6:7] row_share:1 row_mask:0xf bank_mask:0xf ; encoding: [0x04,0x00,0x42,0xd7,0xfa,0x0c,0x02,0x00,0x02,0x51,0x01,0xff]
// GFX1250-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: not a valid operand.
// GFX1250-ERR-NEXT:{{^}}v_maximum_f64 v[4:5], v[2:3], v[6:7] row_share:1
// GFX1250-ERR-NEXT:{{^}}                                     ^

v_ldexp_f64 v[4:5], v[2:3], v6 row_share:1
// GFX1251: v_ldexp_f64_e64_dpp v[4:5], v[2:3], v6 row_share:1 row_mask:0xf bank_mask:0xf ; encoding: [0x04,0x00,0x2b,0xd7,0xfa,0x0c,0x02,0x00,0x02,0x51,0x01,0xff]
// GFX1250-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: not a valid operand.
// GFX1250-ERR-NEXT:{{^}}v_ldexp_f64 v[4:5], v[2:3], v6 row_share:1
// GFX1250-ERR-NEXT:{{^}}                               ^

v_mul_lo_u32 v4, v2, v6 row_share:1
// GFX1251: v_mul_lo_u32_e64_dpp v4, v2, v6 row_share:1 row_mask:0xf bank_mask:0xf ; encoding: [0x04,0x00,0x2c,0xd7,0xfa,0x0c,0x02,0x00,0x02,0x51,0x01,0xff]
// GFX1250-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: not a valid operand.
// GFX1250-ERR-NEXT:{{^}}v_mul_lo_u32 v4, v2, v6 row_share:1
// GFX1250-ERR-NEXT:{{^}}                        ^

v_mul_hi_u32 v4, v2, v6 row_share:1
// GFX1251: v_mul_hi_u32_e64_dpp v4, v2, v6 row_share:1 row_mask:0xf bank_mask:0xf ; encoding: [0x04,0x00,0x2d,0xd7,0xfa,0x0c,0x02,0x00,0x02,0x51,0x01,0xff]
// GFX1250-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: not a valid operand.
// GFX1250-ERR-NEXT:{{^}}v_mul_hi_u32 v4, v2, v6 row_share:1
// GFX1250-ERR-NEXT:{{^}}                        ^

v_mul_hi_i32 v4, v2, v6 row_share:1
// GFX1251: v_mul_hi_i32_e64_dpp v4, v2, v6 row_share:1 row_mask:0xf bank_mask:0xf ; encoding: [0x04,0x00,0x2e,0xd7,0xfa,0x0c,0x02,0x00,0x02,0x51,0x01,0xff]
// GFX1250-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: not a valid operand.
// GFX1250-ERR-NEXT:{{^}}v_mul_hi_i32 v4, v2, v6 row_share:1
// GFX1250-ERR-NEXT:{{^}}                        ^

v_lshrrev_b64 v[4:5], v2, v[6:7] row_share:1
// GFX1251: v_lshrrev_b64_e64_dpp v[4:5], v2, v[6:7] row_share:1 row_mask:0xf bank_mask:0xf ; encoding: [0x04,0x00,0x3d,0xd7,0xfa,0x0c,0x02,0x00,0x02,0x51,0x01,0xff]
// GFX1250-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: not a valid operand.
// GFX1250-ERR-NEXT:{{^}}v_lshrrev_b64 v[4:5], v2, v[6:7] row_share:1
// GFX1250-ERR-NEXT:{{^}}                                 ^

v_ashrrev_i64 v[4:5], v2, v[6:7] row_share:1
// GFX1251: v_ashrrev_i64_e64_dpp v[4:5], v2, v[6:7] row_share:1 row_mask:0xf bank_mask:0xf ; encoding: [0x04,0x00,0x3e,0xd7,0xfa,0x0c,0x02,0x00,0x02,0x51,0x01,0xff]
// GFX1250-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: not a valid operand.
// GFX1250-ERR-NEXT:{{^}}v_ashrrev_i64 v[4:5], v2, v[6:7] row_share:1
// GFX1250-ERR-NEXT:{{^}}                                 ^

v_mad_u32 v2, v4, v7, v8 row_share:3 fi:1
// GFX1251: v_mad_u32_e64_dpp v2, v4, v7, v8 row_share:3 row_mask:0xf bank_mask:0xf fi:1 ; encoding: [0x02,0x00,0x35,0xd6,0xfa,0x0e,0x22,0x04,0x04,0x53,0x05,0xff]
// GFX1250-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: not a valid operand.

v_mad_u32 v2, v4, v7, 1 row_share:0 row_mask:0xf bank_mask:0xf
// GFX1251: v_mad_u32_e64_dpp v2, v4, v7, 1 row_share:0 row_mask:0xf bank_mask:0xf ; encoding: [0x02,0x00,0x35,0xd6,0xfa,0x0e,0x06,0x02,0x04,0x50,0x01,0xff]
// GFX1250-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: not a valid operand.

v_max_i64 v[2:3], v[4:5], v[6:7] row_share:3 fi:1
// GFX1251: v_max_i64_e64_dpp v[2:3], v[4:5], v[6:7] row_share:3 row_mask:0xf bank_mask:0xf fi:1 ; encoding: [0x02,0x00,0x1b,0xd7,0xfa,0x0c,0x02,0x00,0x04,0x53,0x05,0xff]
// GFX1250-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: not a valid operand.

v_max_i64 v[2:3], v[4:5], v[6:7] row_share:0 row_mask:0xf bank_mask:0xf
// GFX1251: v_max_i64_e64_dpp v[2:3], v[4:5], v[6:7] row_share:0 row_mask:0xf bank_mask:0xf ; encoding: [0x02,0x00,0x1b,0xd7,0xfa,0x0c,0x02,0x00,0x04,0x50,0x01,0xff]
// GFX1250-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: not a valid operand.

v_max_u64 v[2:3], v[4:5], v[6:7] row_share:3 fi:1
// GFX1251: v_max_u64_e64_dpp v[2:3], v[4:5], v[6:7] row_share:3 row_mask:0xf bank_mask:0xf fi:1 ; encoding: [0x02,0x00,0x19,0xd7,0xfa,0x0c,0x02,0x00,0x04,0x53,0x05,0xff]
// GFX1250-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: not a valid operand.

v_max_u64 v[2:3], v[4:5], v[6:7] row_share:0 row_mask:0xf bank_mask:0xf
// GFX1251: v_max_u64_e64_dpp v[2:3], v[4:5], v[6:7] row_share:0 row_mask:0xf bank_mask:0xf ; encoding: [0x02,0x00,0x19,0xd7,0xfa,0x0c,0x02,0x00,0x04,0x50,0x01,0xff]
// GFX1250-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: not a valid operand.

v_min_i64 v[2:3], v[4:5], v[6:7] row_share:3 fi:1
// GFX1251: v_min_i64_e64_dpp v[2:3], v[4:5], v[6:7] row_share:3 row_mask:0xf bank_mask:0xf fi:1 ; encoding: [0x02,0x00,0x1a,0xd7,0xfa,0x0c,0x02,0x00,0x04,0x53,0x05,0xff]
// GFX1250-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: not a valid operand.

v_min_i64 v[2:3], v[4:5], v[6:7] row_share:0 row_mask:0xf bank_mask:0xf
// GFX1251: v_min_i64_e64_dpp v[2:3], v[4:5], v[6:7] row_share:0 row_mask:0xf bank_mask:0xf ; encoding: [0x02,0x00,0x1a,0xd7,0xfa,0x0c,0x02,0x00,0x04,0x50,0x01,0xff]
// GFX1250-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: not a valid operand.

v_min_u64 v[2:3], v[4:5], v[6:7] row_share:3 fi:1
// GFX1251: v_min_u64_e64_dpp v[2:3], v[4:5], v[6:7] row_share:3 row_mask:0xf bank_mask:0xf fi:1 ; encoding: [0x02,0x00,0x18,0xd7,0xfa,0x0c,0x02,0x00,0x04,0x53,0x05,0xff]
// GFX1250-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: not a valid operand.

v_min_u64 v[2:3], v[4:5], v[6:7] row_share:0 row_mask:0xf bank_mask:0xf
// GFX1251: v_min_u64_e64_dpp v[2:3], v[4:5], v[6:7] row_share:0 row_mask:0xf bank_mask:0xf ; encoding: [0x02,0x00,0x18,0xd7,0xfa,0x0c,0x02,0x00,0x04,0x50,0x01,0xff]
// GFX1250-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: not a valid operand.

v_mad_nc_u64_u32 v[2:3], v4, v7, v[8:9] row_share:3 fi:1
// GFX1251: v_mad_nc_u64_u32_e64_dpp v[2:3], v4, v7, v[8:9] row_share:3 row_mask:0xf bank_mask:0xf fi:1 ; encoding: [0x02,0x00,0xfa,0xd6,0xfa,0x0e,0x22,0x04,0x04,0x53,0x05,0xff]
// GFX1250-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: not a valid operand.

v_mad_nc_u64_u32 v[2:3], v4, v5, 1 row_share:0 row_mask:0xf bank_mask:0xf
// GFX1251: v_mad_nc_u64_u32_e64_dpp v[2:3], v4, v5, 1 row_share:0 row_mask:0xf bank_mask:0xf ; encoding: [0x02,0x00,0xfa,0xd6,0xfa,0x0a,0x06,0x02,0x04,0x50,0x01,0xff]
// GFX1250-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: not a valid operand.

v_mad_nc_i64_i32 v[2:3], v4, v7, v[8:9] row_share:3 fi:1
// GFX1251: v_mad_nc_i64_i32_e64_dpp v[2:3], v4, v7, v[8:9] row_share:3 row_mask:0xf bank_mask:0xf fi:1 ; encoding: [0x02,0x00,0xfb,0xd6,0xfa,0x0e,0x22,0x04,0x04,0x53,0x05,0xff]
// GFX1250-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: not a valid operand.

v_mad_nc_i64_i32 v[2:3], v4, v5, 1 row_share:0 row_mask:0xf bank_mask:0xf
// GFX1251: v_mad_nc_i64_i32_e64_dpp v[2:3], v4, v5, 1 row_share:0 row_mask:0xf bank_mask:0xf ; encoding: [0x02,0x00,0xfb,0xd6,0xfa,0x0a,0x06,0x02,0x04,0x50,0x01,0xff]
// GFX1250-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: not a valid operand.
