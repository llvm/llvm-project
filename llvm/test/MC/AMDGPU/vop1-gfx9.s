// RUN: llvm-mc -triple=amdgcn -mcpu=gfx900 -show-encoding %s | FileCheck -check-prefix=GFX9 %s
// RUN: not llvm-mc -triple=amdgcn -mcpu=tahiti %s 2>&1 | FileCheck -check-prefix=NOVI --implicit-check-not=error: %s
// RUN: not llvm-mc -triple=amdgcn -mcpu=hawaii %s 2>&1 | FileCheck -check-prefix=NOVI --implicit-check-not=error: %s
// RUN: not llvm-mc -triple=amdgcn -mcpu=tonga %s 2>&1 | FileCheck -check-prefix=NOVI --implicit-check-not=error: %s

v_swap_b32 v1, v2
// GFX9: v_swap_b32 v1, v2 ; encoding: [0x02,0xa3,0x02,0x7e]
// NOVI: :[[@LINE-2]]:1: error: instruction not supported on this GPU

// FIXME: Error for it requiring VOP1 encoding
v_swap_b32_e32 v1, v2
// GFX9: v_swap_b32 v1, v2 ; encoding: [0x02,0xa3,0x02,0x7e]
// NOVI: :[[@LINE-2]]:1: error: instruction not supported on this GPU

v_cvt_norm_i16_f16 v5, v1
// GFX9: v_cvt_norm_i16_f16_e32 v5, v1 ; encoding: [0x01,0x9b,0x0a,0x7e]
// NOVI: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_norm_i16_f16 v5, -4.0
// GFX9: v_cvt_norm_i16_f16_e32 v5, -4.0 ; encoding: [0xf7,0x9a,0x0a,0x7e]
// NOVI: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_norm_i16_f16 v5, 0xfe0b
// GFX9: v_cvt_norm_i16_f16_e32 v5, 0xfe0b ; encoding: [0xff,0x9a,0x0a,0x7e,0x0b,0xfe,0x00,0x00]
// NOVI: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_norm_u16_f16 v5, s101
// GFX9: v_cvt_norm_u16_f16_e32 v5, s101 ; encoding: [0x65,0x9c,0x0a,0x7e]
// NOVI: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_sat_pk_u8_i16 v255, v1
// GFX9: v_sat_pk_u8_i16_e32 v255, v1 ; encoding: [0x01,0x9f,0xfe,0x7f]
// NOVI: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_sat_pk_u8_i16 v5, -1
// GFX9: v_sat_pk_u8_i16_e32 v5, -1 ; encoding: [0xc1,0x9e,0x0a,0x7e]
// NOVI: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_sat_pk_u8_i16 v5, 0x3f717273
// GFX9: v_sat_pk_u8_i16_e32 v5, 0x3f717273 ; encoding: [0xff,0x9e,0x0a,0x7e,0x73,0x72,0x71,0x3f]
// NOVI: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_screen_partition_4se_b32 v5, v255
// GFX9: v_screen_partition_4se_b32_e32 v5, v255 ; encoding: [0xff,0x6f,0x0a,0x7e]
// NOVI: :[[@LINE-2]]:1: error: instruction not supported on this GPU

v_screen_partition_4se_b32 v5, s101
// GFX9: v_screen_partition_4se_b32_e32 v5, s101 ; encoding: [0x65,0x6e,0x0a,0x7e]
// NOVI: :[[@LINE-2]]:1: error: instruction not supported on this GPU

v_screen_partition_4se_b32 v5, -1
// GFX9: v_screen_partition_4se_b32_e32 v5, -1 ; encoding: [0xc1,0x6e,0x0a,0x7e]
// NOVI: :[[@LINE-2]]:1: error: instruction not supported on this GPU

v_screen_partition_4se_b32 v5, 0x3f717273
// GFX9: v_screen_partition_4se_b32_e32 v5, 0x3f717273 ; encoding: [0xff,0x6e,0x0a,0x7e,0x73,0x72,0x71,0x3f]
// NOVI: :[[@LINE-2]]:1: error: instruction not supported on this GPU
