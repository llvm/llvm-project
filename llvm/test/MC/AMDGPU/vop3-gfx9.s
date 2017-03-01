// RUN: llvm-mc -arch=amdgcn -mcpu=gfx901 -show-encoding %s | FileCheck -check-prefix=GFX9 %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=tahiti -show-encoding %s 2>&1 | FileCheck -check-prefix=NOVI %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=hawaii -show-encoding %s 2>&1 | FileCheck -check-prefix=NOVI %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=tonga -show-encoding %s 2>&1 | FileCheck -check-prefix=NOVI %s

v_lshl_add_u32 v1, v2, v3, v4
// GFX9: v_lshl_add_u32 v1, v2, v3, v4 ; encoding: [0x01,0x00,0xfd,0xd1,0x02,0x07,0x12,0x04]
// NOVI: :1: error: instruction not supported on this GPU

v_add_lshl_u32 v1, v2, v3, v4
// GFX9: v_add_lshl_u32 v1, v2, v3, v4 ; encoding: [0x01,0x00,0xfe,0xd1,0x02,0x07,0x12,0x04]
// NOVI: :1: error: instruction not supported on this GPU

v_add3_u32 v1, v2, v3, v4
// GFX9: v_add3_u32 v1, v2, v3, v4 ; encoding: [0x01,0x00,0xff,0xd1,0x02,0x07,0x12,0x04]
// NOVI: :1: error: instruction not supported on this GPU

v_lshl_or_b32 v1, v2, v3, v4
// GFX9: v_lshl_or_b32 v1, v2, v3, v4 ; encoding: [0x01,0x00,0x00,0xd2,0x02,0x07,0x12,0x04]
// NOVI: :1: error: instruction not supported on this GPU

v_and_or_b32 v1, v2, v3, v4
// GFX9: v_and_or_b32 v1, v2, v3, v4 ; encoding: [0x01,0x00,0x01,0xd2,0x02,0x07,0x12,0x04]
// NOVI: :1: error: instruction not supported on this GPU

v_or3_b32 v1, v2, v3, v4
// GFX9: v_or3_b32 v1, v2, v3, v4 ; encoding: [0x01,0x00,0x02,0xd2,0x02,0x07,0x12,0x04]
// NOVI: :1: error: instruction not supported on this GPU

v_pack_b32_f16 v1, v2, v3
// GFX9: v_pack_b32_f16 v1, v2, v3 ; encoding: [0x01,0x00,0xa0,0xd2,0x02,0x07,0x02,0x00]
// NOVI: :1: error: instruction not supported on this GPU

v_xad_u32 v1, v2, v3, v4
// GFX9: v_xad_u32 v1, v2, v3, v4 ; encoding: [0x01,0x00,0xf3,0xd1,0x02,0x07,0x12,0x04]
// NOVI: :1: error: instruction not supported on this GPU

v_med3_f16 v1, v2, v3, v4
// GFX9: v_med3_f16 v1, v2, v3, v4 ; encoding: [0x01,0x00,0xfa,0xd1,0x02,0x07,0x12,0x04]
// NOVI: :1: error: instruction not supported on this GPU

v_med3_i16 v1, v2, v3, v4
// GFX9: v_med3_i16 v1, v2, v3, v4 ; encoding: [0x01,0x00,0xfb,0xd1,0x02,0x07,0x12,0x04]
// NOVI: :1: error: instruction not supported on this GPU

v_med3_u16 v1, v2, v3, v4
// GFX9: v_med3_u16 v1, v2, v3, v4 ; encoding: [0x01,0x00,0xfc,0xd1,0x02,0x07,0x12,0x04]
// NOVI: :1: error: instruction not supported on this GPU
