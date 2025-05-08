// RUN: llvm-mc -triple=amdgcn -mcpu=gfx1300 -mattr=+wavefrontsize32 -show-encoding %s | FileCheck --check-prefixes=GFX13 %s
// RUN: llvm-mc -triple=amdgcn -mcpu=gfx1300 -mattr=+wavefrontsize64 -show-encoding %s | FileCheck --check-prefixes=GFX13 %s

v_fma_mix_f32 v0, v1, v2, v3 op_sel:[0,0,0] row_ror:7 bank_mask:0x1 bound_ctrl:0
// GFX13: v_fma_mix_f32_e64_dpp v0, v1, v2, v3 row_ror:7 row_mask:0xf bank_mask:0x1 ; encoding: [0x00,0x00,0x20,0xcc,0xfa,0x04,0x0e,0x04,0x01,0x27,0x01,0xf1]

v_fma_mix_f32_bf16 v0, v1, v2, v3 op_sel:[0,0,0] row_ror:7 bank_mask:0x1 bound_ctrl:0
// GFX13: v_fma_mix_f32_bf16_e64_dpp v0, v1, v2, v3 row_ror:7 row_mask:0xf bank_mask:0x1 ; encoding: [0x00,0x00,0x3d,0xcc,0xfa,0x04,0x0e,0x04,0x01,0x27,0x01,0xf1]

v_fma_mixhi_bf16 v0, v1, v2, v3 op_sel_hi:[1,1,1] clamp quad_perm:[0,2,3,1] row_mask:0x0
// GFX13: v_fma_mixhi_bf16_e64_dpp v0, v1, v2, v3 op_sel_hi:[1,1,1] clamp quad_perm:[0,2,3,1] row_mask:0x0 bank_mask:0xf ; encoding: [0x00,0xc0,0x3f,0xcc,0xfa,0x04,0x0e,0x1c,0x01,0x78,0x00,0x0f]

v_fma_mixhi_f16 v0, v1, v2, v3 op_sel_hi:[1,1,1] clamp quad_perm:[0,2,3,1] row_mask:0x0
// GFX13: v_fma_mixhi_f16_e64_dpp v0, v1, v2, v3 op_sel_hi:[1,1,1] clamp quad_perm:[0,2,3,1] row_mask:0x0 bank_mask:0xf ; encoding: [0x00,0xc0,0x22,0xcc,0xfa,0x04,0x0e,0x1c,0x01,0x78,0x00,0x0f]

v_fma_mixlo_bf16 v0, v1, v2, v3 op_sel_hi:[1,1,1] clamp quad_perm:[0,2,3,1] row_mask:0x0
// GFX13: v_fma_mixlo_bf16_e64_dpp v0, v1, v2, v3 op_sel_hi:[1,1,1] clamp quad_perm:[0,2,3,1] row_mask:0x0 bank_mask:0xf ; encoding: [0x00,0xc0,0x3e,0xcc,0xfa,0x04,0x0e,0x1c,0x01,0x78,0x00,0x0f]
