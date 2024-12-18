// RUN: llvm-mc -triple=amdgcn -mcpu=gfx1250 -show-encoding < %s | FileCheck --check-prefix=GFX1250 %s
// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx1200 -show-encoding %s 2>&1 | FileCheck --check-prefix=GFX12-ERR --implicit-check-not=error: --strict-whitespace %s

v_fma_mix_f32_bf16 v0, v1, v2, v3 dpp8:[2,2,2,2,4,4,4,4]
// GFX1250: v_fma_mix_f32_bf16_e64_dpp v0, v1, v2, v3 dpp8:[2,2,2,2,4,4,4,4] ; encoding: [0x00,0x00,0x3d,0xcc,0xe9,0x04,0x0e,0x04,0x01,0x92,0x44,0x92]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_fma_mix_f32_bf16 v0, v1, v2, v3 clamp dpp8:[2,2,2,2,4,4,4,4] fi:1
// GFX1250: v_fma_mix_f32_bf16_e64_dpp v0, v1, v2, v3 clamp dpp8:[2,2,2,2,4,4,4,4] fi:1 ; encoding: [0x00,0x80,0x3d,0xcc,0xea,0x04,0x0e,0x04,0x01,0x92,0x44,0x92]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_fma_mixlo_bf16 v0, abs(v1), -v2, abs(v3) dpp8:[2,2,2,2,4,4,4,4]
// GFX1250: v_fma_mixlo_bf16_e64_dpp v0, |v1|, -v2, |v3| dpp8:[2,2,2,2,4,4,4,4] ; encoding: [0x00,0x05,0x3e,0xcc,0xe9,0x04,0x0e,0x44,0x01,0x92,0x44,0x92]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_fma_mixlo_bf16 v0, abs(v1), -v2, abs(v3) op_sel:[1,0,0] op_sel_hi:[1,0,0] dpp8:[2,2,2,2,4,4,4,4]
// GFX1250: v_fma_mixlo_bf16_e64_dpp v0, |v1|, -v2, |v3| op_sel:[1,0,0] op_sel_hi:[1,0,0] dpp8:[2,2,2,2,4,4,4,4] ; encoding: [0x00,0x0d,0x3e,0xcc,0xe9,0x04,0x0e,0x4c,0x01,0x92,0x44,0x92]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_fma_mixhi_bf16 v0, abs(v1), -v2, abs(v3) dpp8:[2,2,2,2,4,4,4,4]
// GFX1250: v_fma_mixhi_bf16_e64_dpp v0, |v1|, -v2, |v3| dpp8:[2,2,2,2,4,4,4,4] ; encoding: [0x00,0x05,0x3f,0xcc,0xe9,0x04,0x0e,0x44,0x01,0x92,0x44,0x92]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_fma_mixhi_bf16 v0, abs(v1), -v2, abs(v3) op_sel:[1,0,0] op_sel_hi:[1,0,0] dpp8:[2,2,2,2,4,4,4,4]
// GFX1250: v_fma_mixhi_bf16_e64_dpp v0, |v1|, -v2, |v3| op_sel:[1,0,0] op_sel_hi:[1,0,0] dpp8:[2,2,2,2,4,4,4,4] ; encoding: [0x00,0x0d,0x3f,0xcc,0xe9,0x04,0x0e,0x4c,0x01,0x92,0x44,0x92]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU
