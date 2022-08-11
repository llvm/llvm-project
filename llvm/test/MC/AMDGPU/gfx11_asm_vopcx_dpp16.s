// RUN: llvm-mc -arch=amdgcn -mcpu=gfx1100 -mattr=+wavefrontsize32,-wavefrontsize64 -show-encoding %s | FileCheck --check-prefixes=GFX11 %s
// RUN: llvm-mc -arch=amdgcn -mcpu=gfx1100 -mattr=-wavefrontsize32,+wavefrontsize64 -show-encoding %s | FileCheck --check-prefixes=GFX11 %s

v_cmpx_t_i32_dpp v0, v2 row_shr:0xe row_mask:0x3 bank_mask:0xa bound_ctrl:0
// GFX11: encoding: [0xfa,0x04,0x8e,0x7d,0x00,0x1e,0x09,0x3a]

v_cmpx_f_f32_dpp v255, v2 quad_perm:[2,3,0,0]
// GFX11: encoding: [0xfa,0x04,0x20,0x7d,0xff,0x0e,0x00,0xff]

v_cmpx_class_f16_dpp v12, v101 quad_perm:[2,3,0,0] fi:1
// GFX11: encoding: [0xfa,0xca,0xfa,0x7d,0x0c,0x0e,0x04,0xff]

v_cmpx_class_f16_dpp abs(v12), v101 quad_perm:[2,3,0,0]
// GFX11: encoding: [0xfa,0xca,0xfa,0x7d,0x0c,0x0e,0x20,0xff]
