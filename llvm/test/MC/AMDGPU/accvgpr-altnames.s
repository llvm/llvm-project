// RUN: llvm-mc -triple=amdgcn -mcpu=gfx908 -show-encoding %s | FileCheck -check-prefix=GFX908 %s

v_accvgpr_read_b32 v2, acc0
// GFX908: v_accvgpr_read_b32 v2, a0       ; encoding: [0x02,0x40,0xd8,0xd3,0x00,0x01,0x00,0x18]

v_accvgpr_write_b32 acc2, -2.0
// GFX908: v_accvgpr_write_b32 a2, -2.0    ; encoding: [0x02,0x40,0xd9,0xd3,0xf5,0x00,0x00,0x18]

v_mfma_f32_32x32x1f32 acc[0:31], acc0, acc1, acc[32:63]
// GFX908: v_mfma_f32_32x32x1f32 a[0:31], a0, a1, a[32:63] ; encoding: [0x00,0x00,0xc0,0xd3,0x00,0x03,0x82,0x1c]
