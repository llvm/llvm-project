// RUN: llvm-mc -triple=amdgcn -mcpu=gfx1251 -show-encoding < %s | FileCheck --check-prefix=GFX1251 %s
// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx1250 -show-encoding %s 2>&1 | FileCheck --check-prefix=GFX1250-ERR --implicit-check-not=error: --strict-whitespace %s

v_mov_b64_e64_dpp v[4:5], v[2:3] row_share:1
// GFX1251: v_mov_b64_e64_dpp v[4:5], v[2:3] row_share:1 row_mask:0xf bank_mask:0xf ; encoding: [0x04,0x00,0x9d,0xd5,0xfa,0x00,0x00,0x00,0x02,0x51,0x01,0xff]
// GFX1250-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_cvt_i32_f64_e64_dpp v2, v[4:5] row_share:1
// GFX1251: v_cvt_i32_f64_e64_dpp v2, v[4:5] row_share:1 row_mask:0xf bank_mask:0xf ; encoding: [0x02,0x00,0x83,0xd5,0xfa,0x00,0x00,0x00,0x04,0x51,0x01,0xff]
// GFX1250-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_cvt_f64_i32_e64_dpp v[4:5], v2 row_share:1
// GFX1251: v_cvt_f64_i32_e64_dpp v[4:5], v2 row_share:1 row_mask:0xf bank_mask:0xf ; encoding: [0x04,0x00,0x84,0xd5,0xfa,0x00,0x00,0x00,0x02,0x51,0x01,0xff]
// GFX1250-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_cvt_f32_f64_e64_dpp v2, v[4:5] row_share:1
// GFX1251: v_cvt_f32_f64_e64_dpp v2, v[4:5] row_share:1 row_mask:0xf bank_mask:0xf ; encoding: [0x02,0x00,0x8f,0xd5,0xfa,0x00,0x00,0x00,0x04,0x51,0x01,0xff]
// GFX1250-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_cvt_f64_f32_e64_dpp v[4:5], v2 row_share:1
// GFX1251: v_cvt_f64_f32_e64_dpp v[4:5], v2 row_share:1 row_mask:0xf bank_mask:0xf ; encoding: [0x04,0x00,0x90,0xd5,0xfa,0x00,0x00,0x00,0x02,0x51,0x01,0xff]
// GFX1250-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_cvt_u32_f64_e64_dpp v2, v[4:5] row_share:1
// GFX1251: v_cvt_u32_f64_e64_dpp v2, v[4:5] row_share:1 row_mask:0xf bank_mask:0xf ; encoding: [0x02,0x00,0x95,0xd5,0xfa,0x00,0x00,0x00,0x04,0x51,0x01,0xff]
// GFX1250-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_cvt_f64_u32_e64_dpp v[4:5], v2 row_share:1
// GFX1251: v_cvt_f64_u32_e64_dpp v[4:5], v2 row_share:1 row_mask:0xf bank_mask:0xf ; encoding: [0x04,0x00,0x96,0xd5,0xfa,0x00,0x00,0x00,0x02,0x51,0x01,0xff]
// GFX1250-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_trunc_f64_e64_dpp v[2:3], v[4:5] row_share:1
// GFX1251: v_trunc_f64_e64_dpp v[2:3], v[4:5] row_share:1 row_mask:0xf bank_mask:0xf ; encoding: [0x02,0x00,0x97,0xd5,0xfa,0x00,0x00,0x00,0x04,0x51,0x01,0xff]
// GFX1250-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_ceil_f64_e64_dpp v[2:3], v[4:5] row_share:1
// GFX1251: v_ceil_f64_e64_dpp v[2:3], v[4:5] row_share:1 row_mask:0xf bank_mask:0xf ; encoding: [0x02,0x00,0x98,0xd5,0xfa,0x00,0x00,0x00,0x04,0x51,0x01,0xff]
// GFX1250-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_rndne_f64_e64_dpp v[2:3], v[4:5] row_share:1
// GFX1251: v_rndne_f64_e64_dpp v[2:3], v[4:5] row_share:1 row_mask:0xf bank_mask:0xf ; encoding: [0x02,0x00,0x99,0xd5,0xfa,0x00,0x00,0x00,0x04,0x51,0x01,0xff]
// GFX1250-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_floor_f64_e64_dpp v[2:3], v[4:5] row_share:1
// GFX1251: v_floor_f64_e64_dpp v[2:3], v[4:5] row_share:1 row_mask:0xf bank_mask:0xf ; encoding: [0x02,0x00,0x9a,0xd5,0xfa,0x00,0x00,0x00,0x04,0x51,0x01,0xff]
// GFX1250-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_frexp_exp_i32_f64_e64_dpp v2, v[4:5] row_share:1
// GFX1251: v_frexp_exp_i32_f64_e64_dpp v2, v[4:5] row_share:1 row_mask:0xf bank_mask:0xf ; encoding: [0x02,0x00,0xbc,0xd5,0xfa,0x00,0x00,0x00,0x04,0x51,0x01,0xff]
// GFX1250-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_frexp_mant_f64_e64_dpp v[2:3], v[4:5] row_share:1
// GFX1251: v_frexp_mant_f64_e64_dpp v[2:3], v[4:5] row_share:1 row_mask:0xf bank_mask:0xf ; encoding: [0x02,0x00,0xbd,0xd5,0xfa,0x00,0x00,0x00,0x04,0x51,0x01,0xff]
// GFX1250-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_fract_f64_e64_dpp v[2:3], v[4:5] row_share:1
// GFX1251: v_fract_f64_e64_dpp v[2:3], v[4:5] row_share:1 row_mask:0xf bank_mask:0xf ; encoding: [0x02,0x00,0xbe,0xd5,0xfa,0x00,0x00,0x00,0x04,0x51,0x01,0xff]
// GFX1250-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported
