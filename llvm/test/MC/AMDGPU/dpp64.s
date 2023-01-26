// RUN: llvm-mc -arch=amdgcn -mcpu=gfx90a -show-encoding %s | FileCheck %s --check-prefix=GFX90A
// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx900 %s 2>&1 | FileCheck %s --check-prefix=GFX900 --implicit-check-not=error:

// GFX90A: v_ceil_f64_dpp v[0:1], v[2:3]  row_newbcast:1 row_mask:0xf bank_mask:0xf ; encoding: [0xfa,0x30,0x00,0x7e,0x02,0x51,0x01,0xff]
// GFX900: :[[@LINE+1]]:{{[0-9]+}}: error: not a valid operand.
v_ceil_f64 v[0:1], v[2:3] row_newbcast:1

// GFX90A: v_fmac_f64_dpp v[0:1], v[2:3], v[4:5]  row_newbcast:2 row_mask:0xf bank_mask:0xf ; encoding: [0xfa,0x08,0x00,0x08,0x02,0x52,0x01,0xff]
// GFX900: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_fmac_f64 v[0:1], v[2:3], v[4:5] row_newbcast:2

// GFX90A: v_cvt_f32_f64_dpp v5, v[0:1]  row_newbcast:15 row_mask:0xf bank_mask:0xf ; encoding: [0xfa,0x1e,0x0a,0x7e,0x00,0x5f,0x01,0xff]
// GFX900: :[[@LINE+1]]:{{[0-9]+}}: error: not a valid operand.
v_cvt_f32_f64 v5, v[0:1] row_newbcast:15

// GFX90A: v_cvt_i32_f64_dpp v5, v[0:1]  row_newbcast:1 row_mask:0xf bank_mask:0xf ; encoding: [0xfa,0x06,0x0a,0x7e,0x00,0x51,0x01,0xff]
// GFX900: :[[@LINE+1]]:{{[0-9]+}}: error: not a valid operand.
v_cvt_i32_f64 v5, v[0:1] row_newbcast:1

// GFX90A: v_cvt_u32_f64_dpp v5, v[0:1]  row_newbcast:2 row_mask:0xf bank_mask:0xf ; encoding: [0xfa,0x2a,0x0a,0x7e,0x00,0x52,0x01,0xff]
// GFX900: :[[@LINE+1]]:{{[0-9]+}}: error: not a valid operand.
v_cvt_u32_f64 v5, v[0:1] row_newbcast:2

// GFX90A: v_floor_f64_dpp v[4:5], v[0:1]  row_newbcast:15 row_mask:0xf bank_mask:0xf ; encoding: [0xfa,0x34,0x08,0x7e,0x00,0x5f,0x01,0xff]
// GFX900: :[[@LINE+1]]:{{[0-9]+}}: error: not a valid operand.
v_floor_f64 v[4:5], v[0:1] row_newbcast:15

// GFX90A: v_fract_f64_dpp v[4:5], v[0:1]  row_newbcast:1 row_mask:0xf bank_mask:0xf ; encoding: [0xfa,0x64,0x08,0x7e,0x00,0x51,0x01,0xff]
// GFX900: :[[@LINE+1]]:{{[0-9]+}}: error: not a valid operand.
v_fract_f64 v[4:5], v[0:1] row_newbcast:1

// GFX90A: v_frexp_exp_i32_f64_dpp v5, v[0:1]  row_newbcast:1 row_mask:0xf bank_mask:0xf ; encoding: [0xfa,0x60,0x0a,0x7e,0x00,0x51,0x01,0xff]
// GFX900: :[[@LINE+1]]:{{[0-9]+}}: error: not a valid operand.
v_frexp_exp_i32_f64 v5, v[0:1] row_newbcast:1

// GFX90A: v_frexp_mant_f64_dpp v[4:5], v[0:1]  row_newbcast:1 row_mask:0xf bank_mask:0xf ; encoding: [0xfa,0x62,0x08,0x7e,0x00,0x51,0x01,0xff]
// GFX900: :[[@LINE+1]]:{{[0-9]+}}: error: not a valid operand.
v_frexp_mant_f64 v[4:5], v[0:1] row_newbcast:1

// GFX90A: v_rcp_f64_dpp v[4:5], v[0:1]  row_newbcast:1 row_mask:0xf bank_mask:0xf ; encoding: [0xfa,0x4a,0x08,0x7e,0x00,0x51,0x01,0xff]
// GFX900: :[[@LINE+1]]:{{[0-9]+}}: error: not a valid operand.
v_rcp_f64 v[4:5], v[0:1] row_newbcast:1

// GFX90A: v_rndne_f64_dpp v[4:5], v[0:1]  row_newbcast:1 row_mask:0xf bank_mask:0xf ; encoding: [0xfa,0x32,0x08,0x7e,0x00,0x51,0x01,0xff]
// GFX900: :[[@LINE+1]]:{{[0-9]+}}: error: not a valid operand.
v_rndne_f64 v[4:5], v[0:1] row_newbcast:1

// GFX90A: v_rsq_f64_dpp v[4:5], v[0:1]  row_newbcast:1 row_mask:0xf bank_mask:0xf ; encoding: [0xfa,0x4c,0x08,0x7e,0x00,0x51,0x01,0xff]
// GFX900: :[[@LINE+1]]:{{[0-9]+}}: error: not a valid operand.
v_rsq_f64 v[4:5], v[0:1] row_newbcast:1

// GFX90A: v_sqrt_f64_dpp v[4:5], v[0:1]  row_newbcast:1 row_mask:0xf bank_mask:0xf ; encoding: [0xfa,0x50,0x08,0x7e,0x00,0x51,0x01,0xff]
// GFX900: :[[@LINE+1]]:{{[0-9]+}}: error: not a valid operand.
v_sqrt_f64 v[4:5], v[0:1] row_newbcast:1

// GFX90A: v_trunc_f64_dpp v[4:5], v[0:1]  row_newbcast:1 row_mask:0xf bank_mask:0xf ; encoding: [0xfa,0x2e,0x08,0x7e,0x00,0x51,0x01,0xff]
// GFX900: :[[@LINE+1]]:{{[0-9]+}}: error: not a valid operand.
v_trunc_f64 v[4:5], v[0:1] row_newbcast:1
