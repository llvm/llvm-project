// RUN: llvm-mc -arch=amdgcn -mcpu=gfx1210 -show-encoding < %s | FileCheck --check-prefix=GFX1210 %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx1200 -show-encoding %s 2>&1 | FileCheck --check-prefix=GFX12-ERR --implicit-check-not=error: --strict-whitespace %s

v_mov_b64_e64_dpp v[4:5], v[2:3] row_share:1
// GFX1210: v_mov_b64_e64_dpp v[4:5], v[2:3] row_share:1 row_mask:0xf bank_mask:0xf ; encoding: [0x04,0x00,0x9d,0xd5,0xfa,0x00,0x00,0x00,0x02,0x51,0x01,0xff]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_i32_f64_e64_dpp v2, v[4:5] row_share:1
// GFX1210: v_cvt_i32_f64_e64_dpp v2, v[4:5] row_share:1 row_mask:0xf bank_mask:0xf ; encoding: [0x02,0x00,0x83,0xd5,0xfa,0x00,0x00,0x00,0x04,0x51,0x01,0xff]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_cvt_f64_i32_e64_dpp v[4:5], v2 row_share:1
// GFX1210: v_cvt_f64_i32_e64_dpp v[4:5], v2 row_share:1 row_mask:0xf bank_mask:0xf ; encoding: [0x04,0x00,0x84,0xd5,0xfa,0x00,0x00,0x00,0x02,0x51,0x01,0xff]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_cvt_f32_f64_e64_dpp v2, v[4:5] row_share:1
// GFX1210: v_cvt_f32_f64_e64_dpp v2, v[4:5] row_share:1 row_mask:0xf bank_mask:0xf ; encoding: [0x02,0x00,0x8f,0xd5,0xfa,0x00,0x00,0x00,0x04,0x51,0x01,0xff]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_cvt_f64_f32_e64_dpp v[4:5], v2 row_share:1
// GFX1210: v_cvt_f64_f32_e64_dpp v[4:5], v2 row_share:1 row_mask:0xf bank_mask:0xf ; encoding: [0x04,0x00,0x90,0xd5,0xfa,0x00,0x00,0x00,0x02,0x51,0x01,0xff]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_cvt_u32_f64_e64_dpp v2, v[4:5] row_share:1
// GFX1210: v_cvt_u32_f64_e64_dpp v2, v[4:5] row_share:1 row_mask:0xf bank_mask:0xf ; encoding: [0x02,0x00,0x95,0xd5,0xfa,0x00,0x00,0x00,0x04,0x51,0x01,0xff]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_cvt_f64_u32_e64_dpp v[4:5], v2 row_share:1
// GFX1210: v_cvt_f64_u32_e64_dpp v[4:5], v2 row_share:1 row_mask:0xf bank_mask:0xf ; encoding: [0x04,0x00,0x96,0xd5,0xfa,0x00,0x00,0x00,0x02,0x51,0x01,0xff]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_trunc_f64_e64_dpp v[2:3], v[4:5] row_share:1
// GFX1210: v_trunc_f64_e64_dpp v[2:3], v[4:5] row_share:1 row_mask:0xf bank_mask:0xf ; encoding: [0x02,0x00,0x97,0xd5,0xfa,0x00,0x00,0x00,0x04,0x51,0x01,0xff]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_ceil_f64_e64_dpp v[2:3], v[4:5] row_share:1
// GFX1210: v_ceil_f64_e64_dpp v[2:3], v[4:5] row_share:1 row_mask:0xf bank_mask:0xf ; encoding: [0x02,0x00,0x98,0xd5,0xfa,0x00,0x00,0x00,0x04,0x51,0x01,0xff]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_rndne_f64_e64_dpp v[2:3], v[4:5] row_share:1
// GFX1210: v_rndne_f64_e64_dpp v[2:3], v[4:5] row_share:1 row_mask:0xf bank_mask:0xf ; encoding: [0x02,0x00,0x99,0xd5,0xfa,0x00,0x00,0x00,0x04,0x51,0x01,0xff]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_floor_f64_e64_dpp v[2:3], v[4:5] row_share:1
// GFX1210: v_floor_f64_e64_dpp v[2:3], v[4:5] row_share:1 row_mask:0xf bank_mask:0xf ; encoding: [0x02,0x00,0x9a,0xd5,0xfa,0x00,0x00,0x00,0x04,0x51,0x01,0xff]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_frexp_exp_i32_f64_e64_dpp v2, v[4:5] row_share:1
// GFX1210: v_frexp_exp_i32_f64_e64_dpp v2, v[4:5] row_share:1 row_mask:0xf bank_mask:0xf ; encoding: [0x02,0x00,0xbc,0xd5,0xfa,0x00,0x00,0x00,0x04,0x51,0x01,0xff]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_frexp_mant_f64_e64_dpp v[2:3], v[4:5] row_share:1
// GFX1210: v_frexp_mant_f64_e64_dpp v[2:3], v[4:5] row_share:1 row_mask:0xf bank_mask:0xf ; encoding: [0x02,0x00,0xbd,0xd5,0xfa,0x00,0x00,0x00,0x04,0x51,0x01,0xff]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_fract_f64_e64_dpp v[2:3], v[4:5] row_share:1
// GFX1210: v_fract_f64_e64_dpp v[2:3], v[4:5] row_share:1 row_mask:0xf bank_mask:0xf ; encoding: [0x02,0x00,0xbe,0xd5,0xfa,0x00,0x00,0x00,0x04,0x51,0x01,0xff]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_tanh_f32_e64_dpp v5, v1 quad_perm:[3,2,1,0]
// GFX1210: v_tanh_f32_e64_dpp v5, v1 quad_perm:[3,2,1,0] row_mask:0xf bank_mask:0xf ; encoding: [0x05,0x00,0x9e,0xd5,0xfa,0x00,0x00,0x00,0x01,0x1b,0x00,0xff]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_tanh_f32_e64_dpp v5, v1 quad_perm:[0,1,2,3]
// GFX1210: v_tanh_f32_e64_dpp v5, v1 quad_perm:[0,1,2,3] row_mask:0xf bank_mask:0xf ; encoding: [0x05,0x00,0x9e,0xd5,0xfa,0x00,0x00,0x00,0x01,0xe4,0x00,0xff]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_tanh_f32_e64_dpp v5, v1 row_mirror
// GFX1210: v_tanh_f32_e64_dpp v5, v1 row_mirror row_mask:0xf bank_mask:0xf ; encoding: [0x05,0x00,0x9e,0xd5,0xfa,0x00,0x00,0x00,0x01,0x40,0x01,0xff]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_tanh_f32_e64_dpp v5, v1 row_half_mirror
// GFX1210: v_tanh_f32_e64_dpp v5, v1 row_half_mirror row_mask:0xf bank_mask:0xf ; encoding: [0x05,0x00,0x9e,0xd5,0xfa,0x00,0x00,0x00,0x01,0x41,0x01,0xff]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_tanh_f32_e64_dpp v5, v1 row_shl:1
// GFX1210: v_tanh_f32_e64_dpp v5, v1 row_shl:1 row_mask:0xf bank_mask:0xf ; encoding: [0x05,0x00,0x9e,0xd5,0xfa,0x00,0x00,0x00,0x01,0x01,0x01,0xff]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_tanh_f32_e64_dpp v5, v1 row_shl:15
// GFX1210: v_tanh_f32_e64_dpp v5, v1 row_shl:15 row_mask:0xf bank_mask:0xf ; encoding: [0x05,0x00,0x9e,0xd5,0xfa,0x00,0x00,0x00,0x01,0x0f,0x01,0xff]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_tanh_f32_e64_dpp v5, v1 row_shr:1
// GFX1210: v_tanh_f32_e64_dpp v5, v1 row_shr:1 row_mask:0xf bank_mask:0xf ; encoding: [0x05,0x00,0x9e,0xd5,0xfa,0x00,0x00,0x00,0x01,0x11,0x01,0xff]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_tanh_f32_e64_dpp v5, v1 row_shr:15
// GFX1210: v_tanh_f32_e64_dpp v5, v1 row_shr:15 row_mask:0xf bank_mask:0xf ; encoding: [0x05,0x00,0x9e,0xd5,0xfa,0x00,0x00,0x00,0x01,0x1f,0x01,0xff]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_tanh_f32_e64_dpp v5, v1 row_ror:1
// GFX1210: v_tanh_f32_e64_dpp v5, v1 row_ror:1 row_mask:0xf bank_mask:0xf ; encoding: [0x05,0x00,0x9e,0xd5,0xfa,0x00,0x00,0x00,0x01,0x21,0x01,0xff]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_tanh_f32_e64_dpp v5, v1 row_ror:15
// GFX1210: v_tanh_f32_e64_dpp v5, v1 row_ror:15 row_mask:0xf bank_mask:0xf ; encoding: [0x05,0x00,0x9e,0xd5,0xfa,0x00,0x00,0x00,0x01,0x2f,0x01,0xff]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_tanh_f32_e64_dpp v5, v1 row_share:0 row_mask:0xf bank_mask:0xf
// GFX1210: v_tanh_f32_e64_dpp v5, v1 row_share:0 row_mask:0xf bank_mask:0xf ; encoding: [0x05,0x00,0x9e,0xd5,0xfa,0x00,0x00,0x00,0x01,0x50,0x01,0xff]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_tanh_f32_e64_dpp v5, v1 mul:2 row_share:15 row_mask:0x0 bank_mask:0x1
// GFX1210: v_tanh_f32_e64_dpp v5, v1 mul:2 row_share:15 row_mask:0x0 bank_mask:0x1 ; encoding: [0x05,0x00,0x9e,0xd5,0xfa,0x00,0x00,0x08,0x01,0x5f,0x01,0x01]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_tanh_f32_e64_dpp v5, v1 mul:4 row_xmask:0 row_mask:0x1 bank_mask:0x3 bound_ctrl:1 fi:0
// GFX1210: v_tanh_f32_e64_dpp v5, v1 mul:4 row_xmask:0 row_mask:0x1 bank_mask:0x3 bound_ctrl:1 ; encoding: [0x05,0x00,0x9e,0xd5,0xfa,0x00,0x00,0x10,0x01,0x60,0x09,0x13]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_tanh_f32_e64_dpp v255, -|v255| clamp div:2 row_xmask:15 row_mask:0x3 bank_mask:0x0 bound_ctrl:0 fi:1
// GFX1210: v_tanh_f32_e64_dpp v255, -|v255| clamp div:2 row_xmask:15 row_mask:0x3 bank_mask:0x0 fi:1 ; encoding: [0xff,0x81,0x9e,0xd5,0xfa,0x00,0x00,0x38,0xff,0x6f,0x05,0x30]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_tanh_f16_e64_dpp v5, v1 quad_perm:[3,2,1,0]
// GFX1210: v_tanh_f16_e64_dpp v5, v1 quad_perm:[3,2,1,0] row_mask:0xf bank_mask:0xf ; encoding: [0x05,0x00,0x9f,0xd5,0xfa,0x00,0x00,0x00,0x01,0x1b,0x00,0xff]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_tanh_f16_e64_dpp v5, v1 quad_perm:[0,1,2,3]
// GFX1210: v_tanh_f16_e64_dpp v5, v1 quad_perm:[0,1,2,3] row_mask:0xf bank_mask:0xf ; encoding: [0x05,0x00,0x9f,0xd5,0xfa,0x00,0x00,0x00,0x01,0xe4,0x00,0xff]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_tanh_f16_e64_dpp v5, v1 row_mirror
// GFX1210: v_tanh_f16_e64_dpp v5, v1 row_mirror row_mask:0xf bank_mask:0xf ; encoding: [0x05,0x00,0x9f,0xd5,0xfa,0x00,0x00,0x00,0x01,0x40,0x01,0xff]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_tanh_f16_e64_dpp v5, v1 row_half_mirror
// GFX1210: v_tanh_f16_e64_dpp v5, v1 row_half_mirror row_mask:0xf bank_mask:0xf ; encoding: [0x05,0x00,0x9f,0xd5,0xfa,0x00,0x00,0x00,0x01,0x41,0x01,0xff]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_tanh_f16_e64_dpp v5, v1 row_shl:1
// GFX1210: v_tanh_f16_e64_dpp v5, v1 row_shl:1 row_mask:0xf bank_mask:0xf ; encoding: [0x05,0x00,0x9f,0xd5,0xfa,0x00,0x00,0x00,0x01,0x01,0x01,0xff]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_tanh_f16_e64_dpp v5, v1 row_shl:15
// GFX1210: v_tanh_f16_e64_dpp v5, v1 row_shl:15 row_mask:0xf bank_mask:0xf ; encoding: [0x05,0x00,0x9f,0xd5,0xfa,0x00,0x00,0x00,0x01,0x0f,0x01,0xff]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_tanh_f16_e64_dpp v5, v1 row_shr:1
// GFX1210: v_tanh_f16_e64_dpp v5, v1 row_shr:1 row_mask:0xf bank_mask:0xf ; encoding: [0x05,0x00,0x9f,0xd5,0xfa,0x00,0x00,0x00,0x01,0x11,0x01,0xff]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_tanh_f16_e64_dpp v5, v1 row_shr:15
// GFX1210: v_tanh_f16_e64_dpp v5, v1 row_shr:15 row_mask:0xf bank_mask:0xf ; encoding: [0x05,0x00,0x9f,0xd5,0xfa,0x00,0x00,0x00,0x01,0x1f,0x01,0xff]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_tanh_f16_e64_dpp v5, v1 row_ror:1
// GFX1210: v_tanh_f16_e64_dpp v5, v1 row_ror:1 row_mask:0xf bank_mask:0xf ; encoding: [0x05,0x00,0x9f,0xd5,0xfa,0x00,0x00,0x00,0x01,0x21,0x01,0xff]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_tanh_f16_e64_dpp v5, v1 row_ror:15
// GFX1210: v_tanh_f16_e64_dpp v5, v1 row_ror:15 row_mask:0xf bank_mask:0xf ; encoding: [0x05,0x00,0x9f,0xd5,0xfa,0x00,0x00,0x00,0x01,0x2f,0x01,0xff]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_tanh_f16_e64_dpp v5, v1 row_share:0 row_mask:0xf bank_mask:0xf
// GFX1210: v_tanh_f16_e64_dpp v5, v1 row_share:0 row_mask:0xf bank_mask:0xf ; encoding: [0x05,0x00,0x9f,0xd5,0xfa,0x00,0x00,0x00,0x01,0x50,0x01,0xff]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_tanh_f16_e64_dpp v5, v1 mul:2 row_share:15 row_mask:0x0 bank_mask:0x1
// GFX1210: v_tanh_f16_e64_dpp v5, v1 mul:2 row_share:15 row_mask:0x0 bank_mask:0x1 ; encoding: [0x05,0x00,0x9f,0xd5,0xfa,0x00,0x00,0x08,0x01,0x5f,0x01,0x01]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_tanh_f16_e64_dpp v5, v1 mul:4 row_xmask:0 row_mask:0x1 bank_mask:0x3 bound_ctrl:1 fi:0
// GFX1210: v_tanh_f16_e64_dpp v5, v1 mul:4 row_xmask:0 row_mask:0x1 bank_mask:0x3 bound_ctrl:1 ; encoding: [0x05,0x00,0x9f,0xd5,0xfa,0x00,0x00,0x10,0x01,0x60,0x09,0x13]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_tanh_f16_e64_dpp v255, -|v255| clamp div:2 row_xmask:15 row_mask:0x3 bank_mask:0x0 bound_ctrl:0 fi:1
// GFX1210: v_tanh_f16_e64_dpp v255, -|v255| clamp div:2 row_xmask:15 row_mask:0x3 bank_mask:0x0 fi:1 ; encoding: [0xff,0x81,0x9f,0xd5,0xfa,0x00,0x00,0x38,0xff,0x6f,0x05,0x30]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_tanh_bf16_e64_dpp v5, v1 quad_perm:[3,2,1,0]
// GFX1210: v_tanh_bf16_e64_dpp v5, v1 quad_perm:[3,2,1,0] row_mask:0xf bank_mask:0xf ; encoding: [0x05,0x00,0xca,0xd5,0xfa,0x00,0x00,0x00,0x01,0x1b,0x00,0xff]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_tanh_bf16_e64_dpp v5, v1 quad_perm:[0,1,2,3]
// GFX1210: v_tanh_bf16_e64_dpp v5, v1 quad_perm:[0,1,2,3] row_mask:0xf bank_mask:0xf ; encoding: [0x05,0x00,0xca,0xd5,0xfa,0x00,0x00,0x00,0x01,0xe4,0x00,0xff]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_tanh_bf16_e64_dpp v5, v1 row_mirror
// GFX1210: v_tanh_bf16_e64_dpp v5, v1 row_mirror row_mask:0xf bank_mask:0xf ; encoding: [0x05,0x00,0xca,0xd5,0xfa,0x00,0x00,0x00,0x01,0x40,0x01,0xff]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_tanh_bf16_e64_dpp v5, v1 row_half_mirror
// GFX1210: v_tanh_bf16_e64_dpp v5, v1 row_half_mirror row_mask:0xf bank_mask:0xf ; encoding: [0x05,0x00,0xca,0xd5,0xfa,0x00,0x00,0x00,0x01,0x41,0x01,0xff]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_tanh_bf16_e64_dpp v5, v1 row_shl:1
// GFX1210: v_tanh_bf16_e64_dpp v5, v1 row_shl:1 row_mask:0xf bank_mask:0xf ; encoding: [0x05,0x00,0xca,0xd5,0xfa,0x00,0x00,0x00,0x01,0x01,0x01,0xff]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_tanh_bf16_e64_dpp v5, v1 row_shl:15
// GFX1210: v_tanh_bf16_e64_dpp v5, v1 row_shl:15 row_mask:0xf bank_mask:0xf ; encoding: [0x05,0x00,0xca,0xd5,0xfa,0x00,0x00,0x00,0x01,0x0f,0x01,0xff]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_tanh_bf16_e64_dpp v5, v1 row_shr:1
// GFX1210: v_tanh_bf16_e64_dpp v5, v1 row_shr:1 row_mask:0xf bank_mask:0xf ; encoding: [0x05,0x00,0xca,0xd5,0xfa,0x00,0x00,0x00,0x01,0x11,0x01,0xff]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_tanh_bf16_e64_dpp v5, v1 row_shr:15
// GFX1210: v_tanh_bf16_e64_dpp v5, v1 row_shr:15 row_mask:0xf bank_mask:0xf ; encoding: [0x05,0x00,0xca,0xd5,0xfa,0x00,0x00,0x00,0x01,0x1f,0x01,0xff]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_tanh_bf16_e64_dpp v5, v1 row_ror:1
// GFX1210: v_tanh_bf16_e64_dpp v5, v1 row_ror:1 row_mask:0xf bank_mask:0xf ; encoding: [0x05,0x00,0xca,0xd5,0xfa,0x00,0x00,0x00,0x01,0x21,0x01,0xff]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_tanh_bf16_e64_dpp v5, v1 row_ror:15
// GFX1210: v_tanh_bf16_e64_dpp v5, v1 row_ror:15 row_mask:0xf bank_mask:0xf ; encoding: [0x05,0x00,0xca,0xd5,0xfa,0x00,0x00,0x00,0x01,0x2f,0x01,0xff]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_tanh_bf16_e64_dpp v5, v1 row_share:0 row_mask:0xf bank_mask:0xf
// GFX1210: v_tanh_bf16_e64_dpp v5, v1 row_share:0 row_mask:0xf bank_mask:0xf ; encoding: [0x05,0x00,0xca,0xd5,0xfa,0x00,0x00,0x00,0x01,0x50,0x01,0xff]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_tanh_bf16_e64_dpp v5, v1 mul:2 row_share:15 row_mask:0x0 bank_mask:0x1
// GFX1210: v_tanh_bf16_e64_dpp v5, v1 mul:2 row_share:15 row_mask:0x0 bank_mask:0x1 ; encoding: [0x05,0x00,0xca,0xd5,0xfa,0x00,0x00,0x08,0x01,0x5f,0x01,0x01]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_tanh_bf16_e64_dpp v5, v1 mul:4 row_xmask:0 row_mask:0x1 bank_mask:0x3 bound_ctrl:1 fi:0
// GFX1210: v_tanh_bf16_e64_dpp v5, v1 mul:4 row_xmask:0 row_mask:0x1 bank_mask:0x3 bound_ctrl:1 ; encoding: [0x05,0x00,0xca,0xd5,0xfa,0x00,0x00,0x10,0x01,0x60,0x09,0x13]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_tanh_bf16_e64_dpp v255, -|v255| clamp div:2 row_xmask:15 row_mask:0x3 bank_mask:0x0 bound_ctrl:0 fi:1
// GFX1210: v_tanh_bf16_e64_dpp v255, -|v255| clamp div:2 row_xmask:15 row_mask:0x3 bank_mask:0x0 fi:1 ; encoding: [0xff,0x81,0xca,0xd5,0xfa,0x00,0x00,0x38,0xff,0x6f,0x05,0x30]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU
