// RUN: llvm-mc -triple=amdgcn -mcpu=gfx1251 -show-encoding %s | FileCheck --check-prefixes=GFX1251 %s
// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx1250 -show-encoding %s 2>&1 | FileCheck --check-prefix=GFX1250-ERR --implicit-check-not=error: --strict-whitespace %s

v_mov_b64 v[4:5], v[2:3] row_share:0 row_mask:0xf bank_mask:0xf
// GFX1251: v_mov_b64_dpp v[4:5], v[2:3] row_share:0 row_mask:0xf bank_mask:0xf ; encoding: [0xfa,0x3a,0x08,0x7e,0x02,0x50,0x01,0xff]
// GFX1250-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: not a valid operand.
// GFX1250-ERR-NEXT:{{^}}v_mov_b64 v[4:5], v[2:3] row_share:0 row_mask:0xf bank_mask:0xf
// GFX1250-ERR-NEXT:{{^}}                         ^

v_mov_b64 v[4:5], v[2:3] row_share:15 row_mask:0x0 bank_mask:0x1
// GFX1251: v_mov_b64_dpp v[4:5], v[2:3] row_share:15 row_mask:0x0 bank_mask:0x1 ; encoding: [0xfa,0x3a,0x08,0x7e,0x02,0x5f,0x01,0x01]
// GFX1250-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: not a valid operand.
// GFX1250-ERR-NEXT:{{^}}v_mov_b64 v[4:5], v[2:3] row_share:15 row_mask:0x0 bank_mask:0x1
// GFX1250-ERR-NEXT:{{^}}                         ^

v_mov_b64 v[254:255], v[254:255] row_share:3 row_mask:0x3 bank_mask:0x0 bound_ctrl:0 fi:1
// GFX1251: v_mov_b64_dpp v[254:255], v[254:255] row_share:3 row_mask:0x3 bank_mask:0x0 fi:1 ; encoding: [0xfa,0x3a,0xfc,0x7f,0xfe,0x53,0x05,0x30]
// GFX1250-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: not a valid operand.
// GFX1250-ERR-NEXT:{{^}}v_mov_b64 v[254:255], v[254:255] row_share:3 row_mask:0x3 bank_mask:0x0 bound_ctrl:0 fi:1
// GFX1250-ERR-NEXT:{{^}}                                 ^

v_cvt_i32_f64 v2, v[4:5] row_share:1
// GFX1251: v_cvt_i32_f64_dpp v2, v[4:5] row_share:1 row_mask:0xf bank_mask:0xf ; encoding: [0xfa,0x06,0x04,0x7e,0x04,0x51,0x01,0xff]
// GFX1250-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: not a valid operand.
// GFX1250-ERR-NEXT:{{^}}v_cvt_i32_f64 v2, v[4:5] row_share:1
// GFX1250-ERR-NEXT:{{^}}                         ^

v_cvt_f64_i32 v[4:5], v2 row_share:1
// GFX1251: v_cvt_f64_i32_dpp v[4:5], v2 row_share:1 row_mask:0xf bank_mask:0xf ; encoding: [0xfa,0x08,0x08,0x7e,0x02,0x51,0x01,0xff]
// GFX1250-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: not a valid operand.
// GFX1250-ERR-NEXT:{{^}}v_cvt_f64_i32 v[4:5], v2 row_share:1
// GFX1250-ERR-NEXT:{{^}}                         ^

v_cvt_f32_f64 v2, v[4:5] row_share:1
// GFX1251: v_cvt_f32_f64_dpp v2, v[4:5] row_share:1 row_mask:0xf bank_mask:0xf ; encoding: [0xfa,0x1e,0x04,0x7e,0x04,0x51,0x01,0xff]
// GFX1250-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: not a valid operand.
// GFX1250-ERR-NEXT:{{^}}v_cvt_f32_f64 v2, v[4:5] row_share:1
// GFX1250-ERR-NEXT:{{^}}                         ^

v_cvt_f64_f32 v[4:5], v2 row_share:1
// GFX1251: v_cvt_f64_f32_dpp v[4:5], v2 row_share:1 row_mask:0xf bank_mask:0xf ; encoding: [0xfa,0x20,0x08,0x7e,0x02,0x51,0x01,0xff]
// GFX1250-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: not a valid operand.
// GFX1250-ERR-NEXT:{{^}}v_cvt_f64_f32 v[4:5], v2 row_share:1
// GFX1250-ERR-NEXT:{{^}}                         ^

v_cvt_u32_f64 v2, v[4:5] row_share:1
// GFX1251: v_cvt_u32_f64_dpp v2, v[4:5] row_share:1 row_mask:0xf bank_mask:0xf ; encoding: [0xfa,0x2a,0x04,0x7e,0x04,0x51,0x01,0xff]
// GFX1250-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: not a valid operand.
// GFX1250-ERR-NEXT:{{^}}v_cvt_u32_f64 v2, v[4:5] row_share:1
// GFX1250-ERR-NEXT:{{^}}                         ^

v_cvt_f64_u32 v[4:5], v2 row_share:1
// GFX1251: v_cvt_f64_u32_dpp v[4:5], v2 row_share:1 row_mask:0xf bank_mask:0xf ; encoding: [0xfa,0x2c,0x08,0x7e,0x02,0x51,0x01,0xff]
// GFX1250-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: not a valid operand.
// GFX1250-ERR-NEXT:{{^}}v_cvt_f64_u32 v[4:5], v2 row_share:1
// GFX1250-ERR-NEXT:{{^}}                         ^

v_trunc_f64 v[2:3], v[4:5] row_share:1
// GFX1251: v_trunc_f64_dpp v[2:3], v[4:5] row_share:1 row_mask:0xf bank_mask:0xf ; encoding: [0xfa,0x2e,0x04,0x7e,0x04,0x51,0x01,0xff]
// GFX1250-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: not a valid operand.
// GFX1250-ERR-NEXT:{{^}}v_trunc_f64 v[2:3], v[4:5] row_share:1
// GFX1250-ERR-NEXT:{{^}}                           ^

v_ceil_f64 v[2:3], v[4:5] row_share:1
// GFX1251: v_ceil_f64_dpp v[2:3], v[4:5] row_share:1 row_mask:0xf bank_mask:0xf ; encoding: [0xfa,0x30,0x04,0x7e,0x04,0x51,0x01,0xff]
// GFX1250-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: not a valid operand.
// GFX1250-ERR-NEXT:{{^}}v_ceil_f64 v[2:3], v[4:5] row_share:1
// GFX1250-ERR-NEXT:{{^}}                          ^

v_rndne_f64 v[2:3], v[4:5] row_share:1
// GFX1251: v_rndne_f64_dpp v[2:3], v[4:5] row_share:1 row_mask:0xf bank_mask:0xf ; encoding: [0xfa,0x32,0x04,0x7e,0x04,0x51,0x01,0xff]
// GFX1250-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: not a valid operand.
// GFX1250-ERR-NEXT:{{^}}v_rndne_f64 v[2:3], v[4:5] row_share:1
// GFX1250-ERR-NEXT:{{^}}                           ^

v_floor_f64 v[2:3], v[4:5] row_share:1
// GFX1251: v_floor_f64_dpp v[2:3], v[4:5] row_share:1 row_mask:0xf bank_mask:0xf ; encoding: [0xfa,0x34,0x04,0x7e,0x04,0x51,0x01,0xff]
// GFX1250-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: not a valid operand.
// GFX1250-ERR-NEXT:{{^}}v_floor_f64 v[2:3], v[4:5] row_share:1
// GFX1250-ERR-NEXT:{{^}}                           ^

v_frexp_exp_i32_f64 v2, v[4:5] row_share:1
// GFX1251: v_frexp_exp_i32_f64_dpp v2, v[4:5] row_share:1 row_mask:0xf bank_mask:0xf ; encoding: [0xfa,0x78,0x04,0x7e,0x04,0x51,0x01,0xff]
// GFX1250-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: not a valid operand.
// GFX1250-ERR-NEXT:{{^}}v_frexp_exp_i32_f64 v2, v[4:5] row_share:1
// GFX1250-ERR-NEXT:{{^}}                               ^

v_frexp_mant_f64 v[2:3], v[4:5] row_share:1
// GFX1251: v_frexp_mant_f64_dpp v[2:3], v[4:5] row_share:1 row_mask:0xf bank_mask:0xf ; encoding: [0xfa,0x7a,0x04,0x7e,0x04,0x51,0x01,0xff]
// GFX1250-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: not a valid operand.
// GFX1250-ERR-NEXT:{{^}}v_frexp_mant_f64 v[2:3], v[4:5] row_share:1
// GFX1250-ERR-NEXT:{{^}}                                ^

v_fract_f64 v[2:3], v[4:5] row_share:1
// GFX1251: v_fract_f64_dpp v[2:3], v[4:5] row_share:1 row_mask:0xf bank_mask:0xf ; encoding: [0xfa,0x7c,0x04,0x7e,0x04,0x51,0x01,0xff]
// GFX1250-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: not a valid operand.
// GFX1250-ERR-NEXT:{{^}}v_fract_f64 v[2:3], v[4:5] row_share:1
// GFX1250-ERR-NEXT:{{^}}                           ^
