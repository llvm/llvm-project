# RUN: llvm-mc -triple=amdgcn -mcpu=gfx1010 -show-encoding < %s | FileCheck -check-prefix=GFX10 %s

v_add_co_ci_u32_e32 v3, vcc_lo, 12345, v3, vcc_lo
// GFX10: v_add_co_ci_u32_e32 v3, vcc_lo, 0x3039, v3, vcc_lo ; encoding: [0xff,0x06,0x06,0x50,0x39,0x30,0x00,0x00]

v_cndmask_b32 v0, 12345, v1, vcc_lo
// GFX10: v_cndmask_b32_e32 v0, 0x3039, v1, vcc_lo ; encoding: [0xff,0x02,0x00,0x02,0x39,0x30,0x00,0x00]
