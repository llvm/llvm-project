// RUN: llvm-mc -arch=amdgcn -mcpu=gfx1100 -mattr=+wavefrontsize32,-wavefrontsize64 -show-encoding %s | FileCheck --check-prefixes=GFX11 %s
// RUN: llvm-mc -arch=amdgcn -mcpu=gfx1100 -mattr=-wavefrontsize32,+wavefrontsize64 -show-encoding %s | FileCheck --check-prefixes=GFX11 %s

v_dot2_f32_f16 v0, v1, v2, v3 neg_lo:[0,0,0] neg_hi:[0,0,0] quad_perm:[2,2,3,1] bound_ctrl:0 fi:1
// GFX11: v_dot2_f32_f16_e64_dpp v0, v1, v2, v3 quad_perm:[2,2,3,1] row_mask:0xf bank_mask:0xf bound_ctrl:1 fi:1 ; encoding: [0x00,0x00,0x13,0xcc,0xfa,0x04,0x0e,0x04,0x01,0x7a,0x0c,0xff]

v_dot2_f32_f16 v0, v1, v2, v3 neg_lo:[1,1,0] neg_hi:[1,0,1] quad_perm:[3,2,1,0] bank_mask:0xe
// GFX11: v_dot2_f32_f16_e64_dpp v0, v1, v2, v3 neg_lo:[1,1,0] neg_hi:[1,0,1] quad_perm:[3,2,1,0] row_mask:0xf bank_mask:0xe ; encoding: [0x00,0x05,0x13,0xcc,0xfa,0x04,0x0e,0x64,0x01,0x1b,0x00,0xfe]

v_fma_mix_f32 v0, v1, v2, v3 op_sel:[0,0,0] row_ror:7 bank_mask:0x1 bound_ctrl:0
// GFX11: v_fma_mix_f32_e64_dpp v0, v1, v2, v3 row_ror:7 row_mask:0xf bank_mask:0x1 bound_ctrl:1 ; encoding: [0x00,0x00,0x20,0xcc,0xfa,0x04,0x0e,0x04,0x01,0x27,0x09,0xf1]

v_fma_mixhi_f16 v0, v1, v2, v3 op_sel_hi:[1,1,1] clamp quad_perm:[0,2,3,1] row_mask:0x0
// GFX11: v_fma_mixhi_f16_e64_dpp v0, v1, v2, v3 op_sel_hi:[1,1,1] clamp quad_perm:[0,2,3,1] row_mask:0x0 bank_mask:0xf ; encoding: [0x00,0xc0,0x22,0xcc,0xfa,0x04,0x0e,0x1c,0x01,0x78,0x00,0x0f]
