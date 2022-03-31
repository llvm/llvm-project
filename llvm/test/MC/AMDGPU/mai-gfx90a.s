// RUN: llvm-mc -arch=amdgcn -mcpu=gfx90a -show-encoding %s | FileCheck -check-prefix=GFX90A %s

v_accvgpr_read_b32 v2, a0
// GFX90A: v_accvgpr_read_b32 v2, a0       ; encoding: [0x02,0x40,0xd8,0xd3,0x00,0x01,0x00,0x18]

v_accvgpr_read_b32 v2, a1
// GFX90A: v_accvgpr_read_b32 v2, a1       ; encoding: [0x02,0x40,0xd8,0xd3,0x01,0x01,0x00,0x18]

v_accvgpr_read_b32 v2, a255
// GFX90A: v_accvgpr_read_b32 v2, a255     ; encoding: [0x02,0x40,0xd8,0xd3,0xff,0x01,0x00,0x18]

v_accvgpr_read v2, a10
// GFX90A: v_accvgpr_read_b32 v2, a10      ; encoding: [0x02,0x40,0xd8,0xd3,0x0a,0x01,0x00,0x18]

v_accvgpr_write_b32 a2, -2.0
// GFX90A: v_accvgpr_write_b32 a2, -2.0    ; encoding: [0x02,0x40,0xd9,0xd3,0xf5,0x00,0x00,0x18]

v_accvgpr_write_b32 a2, -2
// GFX90A: v_accvgpr_write_b32 a2, -2      ; encoding: [0x02,0x40,0xd9,0xd3,0xc2,0x00,0x00,0x18]

v_accvgpr_write_b32 a2, v1
// GFX90A: v_accvgpr_write_b32 a2, v1      ; encoding: [0x02,0x40,0xd9,0xd3,0x01,0x01,0x00,0x18]

v_accvgpr_write a2, v255
// GFX90A: v_accvgpr_write_b32 a2, v255    ; encoding: [0x02,0x40,0xd9,0xd3,0xff,0x01,0x00,0x18]

v_accvgpr_mov_b32 a1, a2
// GFX90A: v_accvgpr_mov_b32 a1, a2        ; encoding: [0x02,0xa5,0x02,0x7e]

v_accvgpr_write_b32 a10, s20
// GFX940: v_accvgpr_write_b32 a10, s20    ; encoding: [0x0a,0x40,0xd9,0xd3,0x14,0x00,0x00,0x18]

v_mfma_f32_32x32x1f32 a[0:31], v0, v1, a[34:65]
// GFX90A: v_mfma_f32_32x32x1f32 a[0:31], v0, v1, a[34:65] ; encoding: [0x00,0x80,0xc0,0xd3,0x00,0x03,0x8a,0x04]

v_mfma_f32_32x32x1f32 a[0:31], v0, v1, a[34:65] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_32x32x1f32 a[0:31], v0, v1, a[34:65] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xc0,0xd3,0x00,0x03,0x8a,0xe4]

v_mfma_f32_32x32x1f32 a[0:31], v0, a1, a[34:65]
// GFX90A: v_mfma_f32_32x32x1f32 a[0:31], v0, a1, a[34:65] ; encoding: [0x00,0x80,0xc0,0xd3,0x00,0x03,0x8a,0x14]

v_mfma_f32_32x32x1f32 a[0:31], v0, a1, a[34:65] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_32x32x1f32 a[0:31], v0, a1, a[34:65] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xc0,0xd3,0x00,0x03,0x8a,0xf4]

v_mfma_f32_32x32x1f32 a[0:31], a0, v1, a[34:65]
// GFX90A: v_mfma_f32_32x32x1f32 a[0:31], a0, v1, a[34:65] ; encoding: [0x00,0x80,0xc0,0xd3,0x00,0x03,0x8a,0x0c]

v_mfma_f32_32x32x1f32 a[0:31], a0, v1, a[34:65] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_32x32x1f32 a[0:31], a0, v1, a[34:65] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xc0,0xd3,0x00,0x03,0x8a,0xec]

v_mfma_f32_32x32x1f32 a[0:31], a0, a1, a[34:65]
// GFX90A: v_mfma_f32_32x32x1f32 a[0:31], a0, a1, a[34:65] ; encoding: [0x00,0x80,0xc0,0xd3,0x00,0x03,0x8a,0x1c]

v_mfma_f32_32x32x1f32 a[0:31], a0, a1, a[34:65] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_32x32x1f32 a[0:31], a0, a1, a[34:65] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xc0,0xd3,0x00,0x03,0x8a,0xfc]

v_mfma_f32_32x32x1f32 v[0:31], v0, v1, v[34:65]
// GFX90A: v_mfma_f32_32x32x1f32 v[0:31], v0, v1, v[34:65] ; encoding: [0x00,0x00,0xc0,0xd3,0x00,0x03,0x8a,0x04]

v_mfma_f32_32x32x1f32 v[0:31], v0, v1, v[34:65] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_32x32x1f32 v[0:31], v0, v1, v[34:65] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xc0,0xd3,0x00,0x03,0x8a,0xe4]

v_mfma_f32_32x32x1f32 v[0:31], v0, a1, v[34:65]
// GFX90A: v_mfma_f32_32x32x1f32 v[0:31], v0, a1, v[34:65] ; encoding: [0x00,0x00,0xc0,0xd3,0x00,0x03,0x8a,0x14]

v_mfma_f32_32x32x1f32 v[0:31], v0, a1, v[34:65] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_32x32x1f32 v[0:31], v0, a1, v[34:65] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xc0,0xd3,0x00,0x03,0x8a,0xf4]

v_mfma_f32_32x32x1f32 v[0:31], a0, v1, v[34:65]
// GFX90A: v_mfma_f32_32x32x1f32 v[0:31], a0, v1, v[34:65] ; encoding: [0x00,0x00,0xc0,0xd3,0x00,0x03,0x8a,0x0c]

v_mfma_f32_32x32x1f32 v[0:31], a0, v1, v[34:65] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_32x32x1f32 v[0:31], a0, v1, v[34:65] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xc0,0xd3,0x00,0x03,0x8a,0xec]

v_mfma_f32_32x32x1f32 v[0:31], a0, a1, v[34:65]
// GFX90A: v_mfma_f32_32x32x1f32 v[0:31], a0, a1, v[34:65] ; encoding: [0x00,0x00,0xc0,0xd3,0x00,0x03,0x8a,0x1c]

v_mfma_f32_32x32x1f32 v[0:31], a0, a1, v[34:65] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_32x32x1f32 v[0:31], a0, a1, v[34:65] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xc0,0xd3,0x00,0x03,0x8a,0xfc]

v_mfma_f32_32x32x1f32 a[0:31], v0, v1, -2.0
// GFX90A: v_mfma_f32_32x32x1f32 a[0:31], v0, v1, -2.0 ; encoding: [0x00,0x80,0xc0,0xd3,0x00,0x03,0xd6,0x03]

v_mfma_f32_32x32x1f32 v[0:31], v0, v1, -2.0
// GFX90A: v_mfma_f32_32x32x1f32 v[0:31], v0, v1, -2.0 ; encoding: [0x00,0x00,0xc0,0xd3,0x00,0x03,0xd6,0x03]

v_mfma_f32_32x32x1f32 a[0:31], v0, v1, -2.0 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_32x32x1f32 a[0:31], v0, v1, -2.0 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xc0,0xd3,0x00,0x03,0xd6,0xe3]

v_mfma_f32_32x32x1f32 v[0:31], v0, v1, -2.0 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_32x32x1f32 v[0:31], v0, v1, -2.0 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xc0,0xd3,0x00,0x03,0xd6,0xe3]

v_mfma_f32_32x32x1f32 a[0:31], v0, a1, -2.0
// GFX90A: v_mfma_f32_32x32x1f32 a[0:31], v0, a1, -2.0 ; encoding: [0x00,0x80,0xc0,0xd3,0x00,0x03,0xd6,0x13]

v_mfma_f32_32x32x1f32 v[0:31], v0, a1, -2.0
// GFX90A: v_mfma_f32_32x32x1f32 v[0:31], v0, a1, -2.0 ; encoding: [0x00,0x00,0xc0,0xd3,0x00,0x03,0xd6,0x13]

v_mfma_f32_32x32x1f32 a[0:31], v0, a1, -2.0 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_32x32x1f32 a[0:31], v0, a1, -2.0 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xc0,0xd3,0x00,0x03,0xd6,0xf3]

v_mfma_f32_32x32x1f32 v[0:31], v0, a1, -2.0 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_32x32x1f32 v[0:31], v0, a1, -2.0 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xc0,0xd3,0x00,0x03,0xd6,0xf3]

v_mfma_f32_32x32x1f32 a[0:31], a0, v1, -2.0
// GFX90A: v_mfma_f32_32x32x1f32 a[0:31], a0, v1, -2.0 ; encoding: [0x00,0x80,0xc0,0xd3,0x00,0x03,0xd6,0x0b]

v_mfma_f32_32x32x1f32 v[0:31], a0, v1, -2.0
// GFX90A: v_mfma_f32_32x32x1f32 v[0:31], a0, v1, -2.0 ; encoding: [0x00,0x00,0xc0,0xd3,0x00,0x03,0xd6,0x0b]

v_mfma_f32_32x32x1f32 a[0:31], a0, v1, -2.0 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_32x32x1f32 a[0:31], a0, v1, -2.0 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xc0,0xd3,0x00,0x03,0xd6,0xeb]

v_mfma_f32_32x32x1f32 v[0:31], a0, v1, -2.0 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_32x32x1f32 v[0:31], a0, v1, -2.0 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xc0,0xd3,0x00,0x03,0xd6,0xeb]

v_mfma_f32_32x32x1f32 a[0:31], a0, a1, -2.0
// GFX90A: v_mfma_f32_32x32x1f32 a[0:31], a0, a1, -2.0 ; encoding: [0x00,0x80,0xc0,0xd3,0x00,0x03,0xd6,0x1b]

v_mfma_f32_32x32x1f32 v[0:31], a0, a1, -2.0
// GFX90A: v_mfma_f32_32x32x1f32 v[0:31], a0, a1, -2.0 ; encoding: [0x00,0x00,0xc0,0xd3,0x00,0x03,0xd6,0x1b]

v_mfma_f32_32x32x1f32 a[0:31], a0, a1, -2.0 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_32x32x1f32 a[0:31], a0, a1, -2.0 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xc0,0xd3,0x00,0x03,0xd6,0xfb]

v_mfma_f32_32x32x1f32 v[0:31], a0, a1, -2.0 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_32x32x1f32 v[0:31], a0, a1, -2.0 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xc0,0xd3,0x00,0x03,0xd6,0xfb]

v_mfma_f32_16x16x1f32 a[0:15], v0, v1, a[18:33]
// GFX90A: v_mfma_f32_16x16x1f32 a[0:15], v0, v1, a[18:33] ; encoding: [0x00,0x80,0xc1,0xd3,0x00,0x03,0x4a,0x04]

v_mfma_f32_16x16x1f32 a[0:15], v0, v1, a[18:33] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_16x16x1f32 a[0:15], v0, v1, a[18:33] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xc1,0xd3,0x00,0x03,0x4a,0xe4]

v_mfma_f32_16x16x1f32 a[0:15], v0, a1, a[18:33]
// GFX90A: v_mfma_f32_16x16x1f32 a[0:15], v0, a1, a[18:33] ; encoding: [0x00,0x80,0xc1,0xd3,0x00,0x03,0x4a,0x14]

v_mfma_f32_16x16x1f32 a[0:15], v0, a1, a[18:33] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_16x16x1f32 a[0:15], v0, a1, a[18:33] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xc1,0xd3,0x00,0x03,0x4a,0xf4]

v_mfma_f32_16x16x1f32 a[0:15], a0, v1, a[18:33]
// GFX90A: v_mfma_f32_16x16x1f32 a[0:15], a0, v1, a[18:33] ; encoding: [0x00,0x80,0xc1,0xd3,0x00,0x03,0x4a,0x0c]

v_mfma_f32_16x16x1f32 a[0:15], a0, v1, a[18:33] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_16x16x1f32 a[0:15], a0, v1, a[18:33] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xc1,0xd3,0x00,0x03,0x4a,0xec]

v_mfma_f32_16x16x1f32 a[0:15], a0, a1, a[18:33]
// GFX90A: v_mfma_f32_16x16x1f32 a[0:15], a0, a1, a[18:33] ; encoding: [0x00,0x80,0xc1,0xd3,0x00,0x03,0x4a,0x1c]

v_mfma_f32_16x16x1f32 a[0:15], a0, a1, a[18:33] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_16x16x1f32 a[0:15], a0, a1, a[18:33] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xc1,0xd3,0x00,0x03,0x4a,0xfc]

v_mfma_f32_16x16x1f32 v[0:15], v0, v1, v[18:33]
// GFX90A: v_mfma_f32_16x16x1f32 v[0:15], v0, v1, v[18:33] ; encoding: [0x00,0x00,0xc1,0xd3,0x00,0x03,0x4a,0x04]

v_mfma_f32_16x16x1f32 v[0:15], v0, v1, v[18:33] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_16x16x1f32 v[0:15], v0, v1, v[18:33] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xc1,0xd3,0x00,0x03,0x4a,0xe4]

v_mfma_f32_16x16x1f32 v[0:15], v0, a1, v[18:33]
// GFX90A: v_mfma_f32_16x16x1f32 v[0:15], v0, a1, v[18:33] ; encoding: [0x00,0x00,0xc1,0xd3,0x00,0x03,0x4a,0x14]

v_mfma_f32_16x16x1f32 v[0:15], v0, a1, v[18:33] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_16x16x1f32 v[0:15], v0, a1, v[18:33] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xc1,0xd3,0x00,0x03,0x4a,0xf4]

v_mfma_f32_16x16x1f32 v[0:15], a0, v1, v[18:33]
// GFX90A: v_mfma_f32_16x16x1f32 v[0:15], a0, v1, v[18:33] ; encoding: [0x00,0x00,0xc1,0xd3,0x00,0x03,0x4a,0x0c]

v_mfma_f32_16x16x1f32 v[0:15], a0, v1, v[18:33] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_16x16x1f32 v[0:15], a0, v1, v[18:33] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xc1,0xd3,0x00,0x03,0x4a,0xec]

v_mfma_f32_16x16x1f32 v[0:15], a0, a1, v[18:33]
// GFX90A: v_mfma_f32_16x16x1f32 v[0:15], a0, a1, v[18:33] ; encoding: [0x00,0x00,0xc1,0xd3,0x00,0x03,0x4a,0x1c]

v_mfma_f32_16x16x1f32 v[0:15], a0, a1, v[18:33] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_16x16x1f32 v[0:15], a0, a1, v[18:33] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xc1,0xd3,0x00,0x03,0x4a,0xfc]

v_mfma_f32_16x16x1f32 a[0:15], v0, v1, -2.0
// GFX90A: v_mfma_f32_16x16x1f32 a[0:15], v0, v1, -2.0 ; encoding: [0x00,0x80,0xc1,0xd3,0x00,0x03,0xd6,0x03]

v_mfma_f32_16x16x1f32 v[0:15], v0, v1, -2.0
// GFX90A: v_mfma_f32_16x16x1f32 v[0:15], v0, v1, -2.0 ; encoding: [0x00,0x00,0xc1,0xd3,0x00,0x03,0xd6,0x03]

v_mfma_f32_16x16x1f32 a[0:15], v0, v1, -2.0 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_16x16x1f32 a[0:15], v0, v1, -2.0 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xc1,0xd3,0x00,0x03,0xd6,0xe3]

v_mfma_f32_16x16x1f32 v[0:15], v0, v1, -2.0 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_16x16x1f32 v[0:15], v0, v1, -2.0 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xc1,0xd3,0x00,0x03,0xd6,0xe3]

v_mfma_f32_16x16x1f32 a[0:15], v0, a1, -2.0
// GFX90A: v_mfma_f32_16x16x1f32 a[0:15], v0, a1, -2.0 ; encoding: [0x00,0x80,0xc1,0xd3,0x00,0x03,0xd6,0x13]

v_mfma_f32_16x16x1f32 v[0:15], v0, a1, -2.0
// GFX90A: v_mfma_f32_16x16x1f32 v[0:15], v0, a1, -2.0 ; encoding: [0x00,0x00,0xc1,0xd3,0x00,0x03,0xd6,0x13]

v_mfma_f32_16x16x1f32 a[0:15], v0, a1, -2.0 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_16x16x1f32 a[0:15], v0, a1, -2.0 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xc1,0xd3,0x00,0x03,0xd6,0xf3]

v_mfma_f32_16x16x1f32 v[0:15], v0, a1, -2.0 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_16x16x1f32 v[0:15], v0, a1, -2.0 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xc1,0xd3,0x00,0x03,0xd6,0xf3]

v_mfma_f32_16x16x1f32 a[0:15], a0, v1, -2.0
// GFX90A: v_mfma_f32_16x16x1f32 a[0:15], a0, v1, -2.0 ; encoding: [0x00,0x80,0xc1,0xd3,0x00,0x03,0xd6,0x0b]

v_mfma_f32_16x16x1f32 v[0:15], a0, v1, -2.0
// GFX90A: v_mfma_f32_16x16x1f32 v[0:15], a0, v1, -2.0 ; encoding: [0x00,0x00,0xc1,0xd3,0x00,0x03,0xd6,0x0b]

v_mfma_f32_16x16x1f32 a[0:15], a0, v1, -2.0 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_16x16x1f32 a[0:15], a0, v1, -2.0 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xc1,0xd3,0x00,0x03,0xd6,0xeb]

v_mfma_f32_16x16x1f32 v[0:15], a0, v1, -2.0 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_16x16x1f32 v[0:15], a0, v1, -2.0 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xc1,0xd3,0x00,0x03,0xd6,0xeb]

v_mfma_f32_16x16x1f32 a[0:15], a0, a1, -2.0
// GFX90A: v_mfma_f32_16x16x1f32 a[0:15], a0, a1, -2.0 ; encoding: [0x00,0x80,0xc1,0xd3,0x00,0x03,0xd6,0x1b]

v_mfma_f32_16x16x1f32 v[0:15], a0, a1, -2.0
// GFX90A: v_mfma_f32_16x16x1f32 v[0:15], a0, a1, -2.0 ; encoding: [0x00,0x00,0xc1,0xd3,0x00,0x03,0xd6,0x1b]

v_mfma_f32_16x16x1f32 a[0:15], a0, a1, -2.0 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_16x16x1f32 a[0:15], a0, a1, -2.0 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xc1,0xd3,0x00,0x03,0xd6,0xfb]

v_mfma_f32_16x16x1f32 v[0:15], a0, a1, -2.0 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_16x16x1f32 v[0:15], a0, a1, -2.0 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xc1,0xd3,0x00,0x03,0xd6,0xfb]

v_mfma_f32_4x4x1f32 a[0:3], v0, v1, a[2:5]
// GFX90A: v_mfma_f32_4x4x1f32 a[0:3], v0, v1, a[2:5] ; encoding: [0x00,0x80,0xc2,0xd3,0x00,0x03,0x0a,0x04]

v_mfma_f32_4x4x1f32 a[0:3], v0, v1, a[2:5] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_4x4x1f32 a[0:3], v0, v1, a[2:5] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xc2,0xd3,0x00,0x03,0x0a,0xe4]

v_mfma_f32_4x4x1f32 a[0:3], v0, a1, a[2:5]
// GFX90A: v_mfma_f32_4x4x1f32 a[0:3], v0, a1, a[2:5] ; encoding: [0x00,0x80,0xc2,0xd3,0x00,0x03,0x0a,0x14]

v_mfma_f32_4x4x1f32 a[0:3], v0, a1, a[2:5] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_4x4x1f32 a[0:3], v0, a1, a[2:5] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xc2,0xd3,0x00,0x03,0x0a,0xf4]

v_mfma_f32_4x4x1f32 a[0:3], a0, v1, a[2:5]
// GFX90A: v_mfma_f32_4x4x1f32 a[0:3], a0, v1, a[2:5] ; encoding: [0x00,0x80,0xc2,0xd3,0x00,0x03,0x0a,0x0c]

v_mfma_f32_4x4x1f32 a[0:3], a0, v1, a[2:5] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_4x4x1f32 a[0:3], a0, v1, a[2:5] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xc2,0xd3,0x00,0x03,0x0a,0xec]

v_mfma_f32_4x4x1f32 a[0:3], a0, a1, a[2:5]
// GFX90A: v_mfma_f32_4x4x1f32 a[0:3], a0, a1, a[2:5] ; encoding: [0x00,0x80,0xc2,0xd3,0x00,0x03,0x0a,0x1c]

v_mfma_f32_4x4x1f32 a[0:3], a0, a1, a[2:5] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_4x4x1f32 a[0:3], a0, a1, a[2:5] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xc2,0xd3,0x00,0x03,0x0a,0xfc]

v_mfma_f32_4x4x1f32 v[0:3], v0, v1, v[2:5]
// GFX90A: v_mfma_f32_4x4x1f32 v[0:3], v0, v1, v[2:5] ; encoding: [0x00,0x00,0xc2,0xd3,0x00,0x03,0x0a,0x04]

v_mfma_f32_4x4x1f32 v[0:3], v0, v1, v[2:5] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_4x4x1f32 v[0:3], v0, v1, v[2:5] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xc2,0xd3,0x00,0x03,0x0a,0xe4]

v_mfma_f32_4x4x1f32 v[0:3], v0, a1, v[2:5]
// GFX90A: v_mfma_f32_4x4x1f32 v[0:3], v0, a1, v[2:5] ; encoding: [0x00,0x00,0xc2,0xd3,0x00,0x03,0x0a,0x14]

v_mfma_f32_4x4x1f32 v[0:3], v0, a1, v[2:5] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_4x4x1f32 v[0:3], v0, a1, v[2:5] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xc2,0xd3,0x00,0x03,0x0a,0xf4]

v_mfma_f32_4x4x1f32 v[0:3], a0, v1, v[2:5]
// GFX90A: v_mfma_f32_4x4x1f32 v[0:3], a0, v1, v[2:5] ; encoding: [0x00,0x00,0xc2,0xd3,0x00,0x03,0x0a,0x0c]

v_mfma_f32_4x4x1f32 v[0:3], a0, v1, v[2:5] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_4x4x1f32 v[0:3], a0, v1, v[2:5] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xc2,0xd3,0x00,0x03,0x0a,0xec]

v_mfma_f32_4x4x1f32 v[0:3], a0, a1, v[2:5]
// GFX90A: v_mfma_f32_4x4x1f32 v[0:3], a0, a1, v[2:5] ; encoding: [0x00,0x00,0xc2,0xd3,0x00,0x03,0x0a,0x1c]

v_mfma_f32_4x4x1f32 v[0:3], a0, a1, v[2:5] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_4x4x1f32 v[0:3], a0, a1, v[2:5] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xc2,0xd3,0x00,0x03,0x0a,0xfc]

v_mfma_f32_4x4x1f32 a[0:3], v0, v1, -2.0
// GFX90A: v_mfma_f32_4x4x1f32 a[0:3], v0, v1, -2.0 ; encoding: [0x00,0x80,0xc2,0xd3,0x00,0x03,0xd6,0x03]

v_mfma_f32_4x4x1f32 v[0:3], v0, v1, -2.0
// GFX90A: v_mfma_f32_4x4x1f32 v[0:3], v0, v1, -2.0 ; encoding: [0x00,0x00,0xc2,0xd3,0x00,0x03,0xd6,0x03]

v_mfma_f32_4x4x1f32 a[0:3], v0, v1, -2.0 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_4x4x1f32 a[0:3], v0, v1, -2.0 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xc2,0xd3,0x00,0x03,0xd6,0xe3]

v_mfma_f32_4x4x1f32 v[0:3], v0, v1, -2.0 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_4x4x1f32 v[0:3], v0, v1, -2.0 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xc2,0xd3,0x00,0x03,0xd6,0xe3]

v_mfma_f32_4x4x1f32 a[0:3], v0, a1, -2.0
// GFX90A: v_mfma_f32_4x4x1f32 a[0:3], v0, a1, -2.0 ; encoding: [0x00,0x80,0xc2,0xd3,0x00,0x03,0xd6,0x13]

v_mfma_f32_4x4x1f32 v[0:3], v0, a1, -2.0
// GFX90A: v_mfma_f32_4x4x1f32 v[0:3], v0, a1, -2.0 ; encoding: [0x00,0x00,0xc2,0xd3,0x00,0x03,0xd6,0x13]

v_mfma_f32_4x4x1f32 a[0:3], v0, a1, -2.0 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_4x4x1f32 a[0:3], v0, a1, -2.0 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xc2,0xd3,0x00,0x03,0xd6,0xf3]

v_mfma_f32_4x4x1f32 v[0:3], v0, a1, -2.0 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_4x4x1f32 v[0:3], v0, a1, -2.0 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xc2,0xd3,0x00,0x03,0xd6,0xf3]

v_mfma_f32_4x4x1f32 a[0:3], a0, v1, -2.0
// GFX90A: v_mfma_f32_4x4x1f32 a[0:3], a0, v1, -2.0 ; encoding: [0x00,0x80,0xc2,0xd3,0x00,0x03,0xd6,0x0b]

v_mfma_f32_4x4x1f32 v[0:3], a0, v1, -2.0
// GFX90A: v_mfma_f32_4x4x1f32 v[0:3], a0, v1, -2.0 ; encoding: [0x00,0x00,0xc2,0xd3,0x00,0x03,0xd6,0x0b]

v_mfma_f32_4x4x1f32 a[0:3], a0, v1, -2.0 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_4x4x1f32 a[0:3], a0, v1, -2.0 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xc2,0xd3,0x00,0x03,0xd6,0xeb]

v_mfma_f32_4x4x1f32 v[0:3], a0, v1, -2.0 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_4x4x1f32 v[0:3], a0, v1, -2.0 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xc2,0xd3,0x00,0x03,0xd6,0xeb]

v_mfma_f32_4x4x1f32 a[0:3], a0, a1, -2.0
// GFX90A: v_mfma_f32_4x4x1f32 a[0:3], a0, a1, -2.0 ; encoding: [0x00,0x80,0xc2,0xd3,0x00,0x03,0xd6,0x1b]

v_mfma_f32_4x4x1f32 v[0:3], a0, a1, -2.0
// GFX90A: v_mfma_f32_4x4x1f32 v[0:3], a0, a1, -2.0 ; encoding: [0x00,0x00,0xc2,0xd3,0x00,0x03,0xd6,0x1b]

v_mfma_f32_4x4x1f32 a[0:3], a0, a1, -2.0 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_4x4x1f32 a[0:3], a0, a1, -2.0 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xc2,0xd3,0x00,0x03,0xd6,0xfb]

v_mfma_f32_4x4x1f32 v[0:3], a0, a1, -2.0 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_4x4x1f32 v[0:3], a0, a1, -2.0 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xc2,0xd3,0x00,0x03,0xd6,0xfb]

v_mfma_f32_32x32x2f32 a[0:15], v0, v1, a[18:33]
// GFX90A: v_mfma_f32_32x32x2f32 a[0:15], v0, v1, a[18:33] ; encoding: [0x00,0x80,0xc4,0xd3,0x00,0x03,0x4a,0x04]

v_mfma_f32_32x32x2f32 a[0:15], v0, v1, a[18:33] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_32x32x2f32 a[0:15], v0, v1, a[18:33] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xc4,0xd3,0x00,0x03,0x4a,0xe4]

v_mfma_f32_32x32x2f32 a[0:15], v0, a1, a[18:33]
// GFX90A: v_mfma_f32_32x32x2f32 a[0:15], v0, a1, a[18:33] ; encoding: [0x00,0x80,0xc4,0xd3,0x00,0x03,0x4a,0x14]

v_mfma_f32_32x32x2f32 a[0:15], v0, a1, a[18:33] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_32x32x2f32 a[0:15], v0, a1, a[18:33] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xc4,0xd3,0x00,0x03,0x4a,0xf4]

v_mfma_f32_32x32x2f32 a[0:15], a0, v1, a[18:33]
// GFX90A: v_mfma_f32_32x32x2f32 a[0:15], a0, v1, a[18:33] ; encoding: [0x00,0x80,0xc4,0xd3,0x00,0x03,0x4a,0x0c]

v_mfma_f32_32x32x2f32 a[0:15], a0, v1, a[18:33] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_32x32x2f32 a[0:15], a0, v1, a[18:33] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xc4,0xd3,0x00,0x03,0x4a,0xec]

v_mfma_f32_32x32x2f32 a[0:15], a0, a1, a[18:33]
// GFX90A: v_mfma_f32_32x32x2f32 a[0:15], a0, a1, a[18:33] ; encoding: [0x00,0x80,0xc4,0xd3,0x00,0x03,0x4a,0x1c]

v_mfma_f32_32x32x2f32 a[0:15], a0, a1, a[18:33] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_32x32x2f32 a[0:15], a0, a1, a[18:33] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xc4,0xd3,0x00,0x03,0x4a,0xfc]

v_mfma_f32_32x32x2f32 v[0:15], v0, v1, v[18:33]
// GFX90A: v_mfma_f32_32x32x2f32 v[0:15], v0, v1, v[18:33] ; encoding: [0x00,0x00,0xc4,0xd3,0x00,0x03,0x4a,0x04]

v_mfma_f32_32x32x2f32 v[0:15], v0, v1, v[18:33] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_32x32x2f32 v[0:15], v0, v1, v[18:33] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xc4,0xd3,0x00,0x03,0x4a,0xe4]

v_mfma_f32_32x32x2f32 v[0:15], v0, a1, v[18:33]
// GFX90A: v_mfma_f32_32x32x2f32 v[0:15], v0, a1, v[18:33] ; encoding: [0x00,0x00,0xc4,0xd3,0x00,0x03,0x4a,0x14]

v_mfma_f32_32x32x2f32 v[0:15], v0, a1, v[18:33] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_32x32x2f32 v[0:15], v0, a1, v[18:33] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xc4,0xd3,0x00,0x03,0x4a,0xf4]

v_mfma_f32_32x32x2f32 v[0:15], a0, v1, v[18:33]
// GFX90A: v_mfma_f32_32x32x2f32 v[0:15], a0, v1, v[18:33] ; encoding: [0x00,0x00,0xc4,0xd3,0x00,0x03,0x4a,0x0c]

v_mfma_f32_32x32x2f32 v[0:15], a0, v1, v[18:33] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_32x32x2f32 v[0:15], a0, v1, v[18:33] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xc4,0xd3,0x00,0x03,0x4a,0xec]

v_mfma_f32_32x32x2f32 v[0:15], a0, a1, v[18:33]
// GFX90A: v_mfma_f32_32x32x2f32 v[0:15], a0, a1, v[18:33] ; encoding: [0x00,0x00,0xc4,0xd3,0x00,0x03,0x4a,0x1c]

v_mfma_f32_32x32x2f32 v[0:15], a0, a1, v[18:33] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_32x32x2f32 v[0:15], a0, a1, v[18:33] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xc4,0xd3,0x00,0x03,0x4a,0xfc]

v_mfma_f32_32x32x2f32 a[0:15], v0, v1, -2.0
// GFX90A: v_mfma_f32_32x32x2f32 a[0:15], v0, v1, -2.0 ; encoding: [0x00,0x80,0xc4,0xd3,0x00,0x03,0xd6,0x03]

v_mfma_f32_32x32x2f32 v[0:15], v0, v1, -2.0
// GFX90A: v_mfma_f32_32x32x2f32 v[0:15], v0, v1, -2.0 ; encoding: [0x00,0x00,0xc4,0xd3,0x00,0x03,0xd6,0x03]

v_mfma_f32_32x32x2f32 a[0:15], v0, v1, -2.0 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_32x32x2f32 a[0:15], v0, v1, -2.0 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xc4,0xd3,0x00,0x03,0xd6,0xe3]

v_mfma_f32_32x32x2f32 v[0:15], v0, v1, -2.0 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_32x32x2f32 v[0:15], v0, v1, -2.0 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xc4,0xd3,0x00,0x03,0xd6,0xe3]

v_mfma_f32_32x32x2f32 a[0:15], v0, a1, -2.0
// GFX90A: v_mfma_f32_32x32x2f32 a[0:15], v0, a1, -2.0 ; encoding: [0x00,0x80,0xc4,0xd3,0x00,0x03,0xd6,0x13]

v_mfma_f32_32x32x2f32 v[0:15], v0, a1, -2.0
// GFX90A: v_mfma_f32_32x32x2f32 v[0:15], v0, a1, -2.0 ; encoding: [0x00,0x00,0xc4,0xd3,0x00,0x03,0xd6,0x13]

v_mfma_f32_32x32x2f32 a[0:15], v0, a1, -2.0 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_32x32x2f32 a[0:15], v0, a1, -2.0 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xc4,0xd3,0x00,0x03,0xd6,0xf3]

v_mfma_f32_32x32x2f32 v[0:15], v0, a1, -2.0 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_32x32x2f32 v[0:15], v0, a1, -2.0 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xc4,0xd3,0x00,0x03,0xd6,0xf3]

v_mfma_f32_32x32x2f32 a[0:15], a0, v1, -2.0
// GFX90A: v_mfma_f32_32x32x2f32 a[0:15], a0, v1, -2.0 ; encoding: [0x00,0x80,0xc4,0xd3,0x00,0x03,0xd6,0x0b]

v_mfma_f32_32x32x2f32 v[0:15], a0, v1, -2.0
// GFX90A: v_mfma_f32_32x32x2f32 v[0:15], a0, v1, -2.0 ; encoding: [0x00,0x00,0xc4,0xd3,0x00,0x03,0xd6,0x0b]

v_mfma_f32_32x32x2f32 a[0:15], a0, v1, -2.0 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_32x32x2f32 a[0:15], a0, v1, -2.0 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xc4,0xd3,0x00,0x03,0xd6,0xeb]

v_mfma_f32_32x32x2f32 v[0:15], a0, v1, -2.0 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_32x32x2f32 v[0:15], a0, v1, -2.0 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xc4,0xd3,0x00,0x03,0xd6,0xeb]

v_mfma_f32_32x32x2f32 a[0:15], a0, a1, -2.0
// GFX90A: v_mfma_f32_32x32x2f32 a[0:15], a0, a1, -2.0 ; encoding: [0x00,0x80,0xc4,0xd3,0x00,0x03,0xd6,0x1b]

v_mfma_f32_32x32x2f32 v[0:15], a0, a1, -2.0
// GFX90A: v_mfma_f32_32x32x2f32 v[0:15], a0, a1, -2.0 ; encoding: [0x00,0x00,0xc4,0xd3,0x00,0x03,0xd6,0x1b]

v_mfma_f32_32x32x2f32 a[0:15], a0, a1, -2.0 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_32x32x2f32 a[0:15], a0, a1, -2.0 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xc4,0xd3,0x00,0x03,0xd6,0xfb]

v_mfma_f32_32x32x2f32 v[0:15], a0, a1, -2.0 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_32x32x2f32 v[0:15], a0, a1, -2.0 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xc4,0xd3,0x00,0x03,0xd6,0xfb]

v_mfma_f32_16x16x4f32 a[0:3], v0, v1, a[2:5]
// GFX90A: v_mfma_f32_16x16x4f32 a[0:3], v0, v1, a[2:5] ; encoding: [0x00,0x80,0xc5,0xd3,0x00,0x03,0x0a,0x04]

v_mfma_f32_16x16x4f32 a[0:3], v0, v1, a[2:5] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_16x16x4f32 a[0:3], v0, v1, a[2:5] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xc5,0xd3,0x00,0x03,0x0a,0xe4]

v_mfma_f32_16x16x4f32 a[0:3], v0, a1, a[2:5]
// GFX90A: v_mfma_f32_16x16x4f32 a[0:3], v0, a1, a[2:5] ; encoding: [0x00,0x80,0xc5,0xd3,0x00,0x03,0x0a,0x14]

v_mfma_f32_16x16x4f32 a[0:3], v0, a1, a[2:5] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_16x16x4f32 a[0:3], v0, a1, a[2:5] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xc5,0xd3,0x00,0x03,0x0a,0xf4]

v_mfma_f32_16x16x4f32 a[0:3], a0, v1, a[2:5]
// GFX90A: v_mfma_f32_16x16x4f32 a[0:3], a0, v1, a[2:5] ; encoding: [0x00,0x80,0xc5,0xd3,0x00,0x03,0x0a,0x0c]

v_mfma_f32_16x16x4f32 a[0:3], a0, v1, a[2:5] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_16x16x4f32 a[0:3], a0, v1, a[2:5] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xc5,0xd3,0x00,0x03,0x0a,0xec]

v_mfma_f32_16x16x4f32 a[0:3], a0, a1, a[2:5]
// GFX90A: v_mfma_f32_16x16x4f32 a[0:3], a0, a1, a[2:5] ; encoding: [0x00,0x80,0xc5,0xd3,0x00,0x03,0x0a,0x1c]

v_mfma_f32_16x16x4f32 a[0:3], a0, a1, a[2:5] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_16x16x4f32 a[0:3], a0, a1, a[2:5] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xc5,0xd3,0x00,0x03,0x0a,0xfc]

v_mfma_f32_16x16x4f32 v[0:3], v0, v1, v[2:5]
// GFX90A: v_mfma_f32_16x16x4f32 v[0:3], v0, v1, v[2:5] ; encoding: [0x00,0x00,0xc5,0xd3,0x00,0x03,0x0a,0x04]

v_mfma_f32_16x16x4f32 v[0:3], v0, v1, v[2:5] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_16x16x4f32 v[0:3], v0, v1, v[2:5] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xc5,0xd3,0x00,0x03,0x0a,0xe4]

v_mfma_f32_16x16x4f32 v[0:3], v0, a1, v[2:5]
// GFX90A: v_mfma_f32_16x16x4f32 v[0:3], v0, a1, v[2:5] ; encoding: [0x00,0x00,0xc5,0xd3,0x00,0x03,0x0a,0x14]

v_mfma_f32_16x16x4f32 v[0:3], v0, a1, v[2:5] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_16x16x4f32 v[0:3], v0, a1, v[2:5] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xc5,0xd3,0x00,0x03,0x0a,0xf4]

v_mfma_f32_16x16x4f32 v[0:3], a0, v1, v[2:5]
// GFX90A: v_mfma_f32_16x16x4f32 v[0:3], a0, v1, v[2:5] ; encoding: [0x00,0x00,0xc5,0xd3,0x00,0x03,0x0a,0x0c]

v_mfma_f32_16x16x4f32 v[0:3], a0, v1, v[2:5] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_16x16x4f32 v[0:3], a0, v1, v[2:5] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xc5,0xd3,0x00,0x03,0x0a,0xec]

v_mfma_f32_16x16x4f32 v[0:3], a0, a1, v[2:5]
// GFX90A: v_mfma_f32_16x16x4f32 v[0:3], a0, a1, v[2:5] ; encoding: [0x00,0x00,0xc5,0xd3,0x00,0x03,0x0a,0x1c]

v_mfma_f32_16x16x4f32 v[0:3], a0, a1, v[2:5] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_16x16x4f32 v[0:3], a0, a1, v[2:5] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xc5,0xd3,0x00,0x03,0x0a,0xfc]

v_mfma_f32_16x16x4f32 a[0:3], v0, v1, -2.0
// GFX90A: v_mfma_f32_16x16x4f32 a[0:3], v0, v1, -2.0 ; encoding: [0x00,0x80,0xc5,0xd3,0x00,0x03,0xd6,0x03]

v_mfma_f32_16x16x4f32 v[0:3], v0, v1, -2.0
// GFX90A: v_mfma_f32_16x16x4f32 v[0:3], v0, v1, -2.0 ; encoding: [0x00,0x00,0xc5,0xd3,0x00,0x03,0xd6,0x03]

v_mfma_f32_16x16x4f32 a[0:3], v0, v1, -2.0 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_16x16x4f32 a[0:3], v0, v1, -2.0 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xc5,0xd3,0x00,0x03,0xd6,0xe3]

v_mfma_f32_16x16x4f32 v[0:3], v0, v1, -2.0 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_16x16x4f32 v[0:3], v0, v1, -2.0 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xc5,0xd3,0x00,0x03,0xd6,0xe3]

v_mfma_f32_16x16x4f32 a[0:3], v0, a1, -2.0
// GFX90A: v_mfma_f32_16x16x4f32 a[0:3], v0, a1, -2.0 ; encoding: [0x00,0x80,0xc5,0xd3,0x00,0x03,0xd6,0x13]

v_mfma_f32_16x16x4f32 v[0:3], v0, a1, -2.0
// GFX90A: v_mfma_f32_16x16x4f32 v[0:3], v0, a1, -2.0 ; encoding: [0x00,0x00,0xc5,0xd3,0x00,0x03,0xd6,0x13]

v_mfma_f32_16x16x4f32 a[0:3], v0, a1, -2.0 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_16x16x4f32 a[0:3], v0, a1, -2.0 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xc5,0xd3,0x00,0x03,0xd6,0xf3]

v_mfma_f32_16x16x4f32 v[0:3], v0, a1, -2.0 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_16x16x4f32 v[0:3], v0, a1, -2.0 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xc5,0xd3,0x00,0x03,0xd6,0xf3]

v_mfma_f32_16x16x4f32 a[0:3], a0, v1, -2.0
// GFX90A: v_mfma_f32_16x16x4f32 a[0:3], a0, v1, -2.0 ; encoding: [0x00,0x80,0xc5,0xd3,0x00,0x03,0xd6,0x0b]

v_mfma_f32_16x16x4f32 v[0:3], a0, v1, -2.0
// GFX90A: v_mfma_f32_16x16x4f32 v[0:3], a0, v1, -2.0 ; encoding: [0x00,0x00,0xc5,0xd3,0x00,0x03,0xd6,0x0b]

v_mfma_f32_16x16x4f32 a[0:3], a0, v1, -2.0 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_16x16x4f32 a[0:3], a0, v1, -2.0 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xc5,0xd3,0x00,0x03,0xd6,0xeb]

v_mfma_f32_16x16x4f32 v[0:3], a0, v1, -2.0 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_16x16x4f32 v[0:3], a0, v1, -2.0 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xc5,0xd3,0x00,0x03,0xd6,0xeb]

v_mfma_f32_16x16x4f32 a[0:3], a0, a1, -2.0
// GFX90A: v_mfma_f32_16x16x4f32 a[0:3], a0, a1, -2.0 ; encoding: [0x00,0x80,0xc5,0xd3,0x00,0x03,0xd6,0x1b]

v_mfma_f32_16x16x4f32 v[0:3], a0, a1, -2.0
// GFX90A: v_mfma_f32_16x16x4f32 v[0:3], a0, a1, -2.0 ; encoding: [0x00,0x00,0xc5,0xd3,0x00,0x03,0xd6,0x1b]

v_mfma_f32_16x16x4f32 a[0:3], a0, a1, -2.0 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_16x16x4f32 a[0:3], a0, a1, -2.0 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xc5,0xd3,0x00,0x03,0xd6,0xfb]

v_mfma_f32_16x16x4f32 v[0:3], a0, a1, -2.0 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_16x16x4f32 v[0:3], a0, a1, -2.0 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xc5,0xd3,0x00,0x03,0xd6,0xfb]

v_mfma_f32_32x32x4f16 a[0:31], v[0:1], v[2:3], a[34:65]
// GFX90A: v_mfma_f32_32x32x4f16 a[0:31], v[0:1], v[2:3], a[34:65] ; encoding: [0x00,0x80,0xc8,0xd3,0x00,0x05,0x8a,0x04]

v_mfma_f32_32x32x4f16 a[0:31], v[0:1], v[2:3], a[34:65] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_32x32x4f16 a[0:31], v[0:1], v[2:3], a[34:65] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xc8,0xd3,0x00,0x05,0x8a,0xe4]

v_mfma_f32_32x32x4f16 a[0:31], v[0:1], a[2:3], a[34:65]
// GFX90A: v_mfma_f32_32x32x4f16 a[0:31], v[0:1], a[2:3], a[34:65] ; encoding: [0x00,0x80,0xc8,0xd3,0x00,0x05,0x8a,0x14]

v_mfma_f32_32x32x4f16 a[0:31], v[0:1], a[2:3], a[34:65] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_32x32x4f16 a[0:31], v[0:1], a[2:3], a[34:65] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xc8,0xd3,0x00,0x05,0x8a,0xf4]

v_mfma_f32_32x32x4f16 a[0:31], a[0:1], v[2:3], a[34:65]
// GFX90A: v_mfma_f32_32x32x4f16 a[0:31], a[0:1], v[2:3], a[34:65] ; encoding: [0x00,0x80,0xc8,0xd3,0x00,0x05,0x8a,0x0c]

v_mfma_f32_32x32x4f16 a[0:31], a[0:1], v[2:3], a[34:65] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_32x32x4f16 a[0:31], a[0:1], v[2:3], a[34:65] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xc8,0xd3,0x00,0x05,0x8a,0xec]

v_mfma_f32_32x32x4f16 a[0:31], a[0:1], a[2:3], a[34:65]
// GFX90A: v_mfma_f32_32x32x4f16 a[0:31], a[0:1], a[2:3], a[34:65] ; encoding: [0x00,0x80,0xc8,0xd3,0x00,0x05,0x8a,0x1c]

v_mfma_f32_32x32x4f16 a[0:31], a[0:1], a[2:3], a[34:65] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_32x32x4f16 a[0:31], a[0:1], a[2:3], a[34:65] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xc8,0xd3,0x00,0x05,0x8a,0xfc]

v_mfma_f32_32x32x4f16 v[0:31], v[0:1], v[2:3], v[34:65]
// GFX90A: v_mfma_f32_32x32x4f16 v[0:31], v[0:1], v[2:3], v[34:65] ; encoding: [0x00,0x00,0xc8,0xd3,0x00,0x05,0x8a,0x04]

v_mfma_f32_32x32x4f16 v[0:31], v[0:1], v[2:3], v[34:65] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_32x32x4f16 v[0:31], v[0:1], v[2:3], v[34:65] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xc8,0xd3,0x00,0x05,0x8a,0xe4]

v_mfma_f32_32x32x4f16 v[0:31], v[0:1], a[2:3], v[34:65]
// GFX90A: v_mfma_f32_32x32x4f16 v[0:31], v[0:1], a[2:3], v[34:65] ; encoding: [0x00,0x00,0xc8,0xd3,0x00,0x05,0x8a,0x14]

v_mfma_f32_32x32x4f16 v[0:31], v[0:1], a[2:3], v[34:65] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_32x32x4f16 v[0:31], v[0:1], a[2:3], v[34:65] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xc8,0xd3,0x00,0x05,0x8a,0xf4]

v_mfma_f32_32x32x4f16 v[0:31], a[0:1], v[2:3], v[34:65]
// GFX90A: v_mfma_f32_32x32x4f16 v[0:31], a[0:1], v[2:3], v[34:65] ; encoding: [0x00,0x00,0xc8,0xd3,0x00,0x05,0x8a,0x0c]

v_mfma_f32_32x32x4f16 v[0:31], a[0:1], v[2:3], v[34:65] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_32x32x4f16 v[0:31], a[0:1], v[2:3], v[34:65] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xc8,0xd3,0x00,0x05,0x8a,0xec]

v_mfma_f32_32x32x4f16 v[0:31], a[0:1], a[2:3], v[34:65]
// GFX90A: v_mfma_f32_32x32x4f16 v[0:31], a[0:1], a[2:3], v[34:65] ; encoding: [0x00,0x00,0xc8,0xd3,0x00,0x05,0x8a,0x1c]

v_mfma_f32_32x32x4f16 v[0:31], a[0:1], a[2:3], v[34:65] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_32x32x4f16 v[0:31], a[0:1], a[2:3], v[34:65] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xc8,0xd3,0x00,0x05,0x8a,0xfc]

v_mfma_f32_32x32x4f16 a[0:31], v[0:1], v[2:3], -2.0
// GFX90A: v_mfma_f32_32x32x4f16 a[0:31], v[0:1], v[2:3], -2.0 ; encoding: [0x00,0x80,0xc8,0xd3,0x00,0x05,0xd6,0x03]

v_mfma_f32_32x32x4f16 v[0:31], v[0:1], v[2:3], -2.0
// GFX90A: v_mfma_f32_32x32x4f16 v[0:31], v[0:1], v[2:3], -2.0 ; encoding: [0x00,0x00,0xc8,0xd3,0x00,0x05,0xd6,0x03]

v_mfma_f32_32x32x4f16 a[0:31], v[0:1], v[2:3], -2.0 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_32x32x4f16 a[0:31], v[0:1], v[2:3], -2.0 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xc8,0xd3,0x00,0x05,0xd6,0xe3]

v_mfma_f32_32x32x4f16 v[0:31], v[0:1], v[2:3], -2.0 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_32x32x4f16 v[0:31], v[0:1], v[2:3], -2.0 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xc8,0xd3,0x00,0x05,0xd6,0xe3]

v_mfma_f32_32x32x4f16 a[0:31], v[0:1], a[2:3], -2.0
// GFX90A: v_mfma_f32_32x32x4f16 a[0:31], v[0:1], a[2:3], -2.0 ; encoding: [0x00,0x80,0xc8,0xd3,0x00,0x05,0xd6,0x13]

v_mfma_f32_32x32x4f16 v[0:31], v[0:1], a[2:3], -2.0
// GFX90A: v_mfma_f32_32x32x4f16 v[0:31], v[0:1], a[2:3], -2.0 ; encoding: [0x00,0x00,0xc8,0xd3,0x00,0x05,0xd6,0x13]

v_mfma_f32_32x32x4f16 a[0:31], v[0:1], a[2:3], -2.0 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_32x32x4f16 a[0:31], v[0:1], a[2:3], -2.0 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xc8,0xd3,0x00,0x05,0xd6,0xf3]

v_mfma_f32_32x32x4f16 v[0:31], v[0:1], a[2:3], -2.0 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_32x32x4f16 v[0:31], v[0:1], a[2:3], -2.0 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xc8,0xd3,0x00,0x05,0xd6,0xf3]

v_mfma_f32_32x32x4f16 a[0:31], a[0:1], v[2:3], -2.0
// GFX90A: v_mfma_f32_32x32x4f16 a[0:31], a[0:1], v[2:3], -2.0 ; encoding: [0x00,0x80,0xc8,0xd3,0x00,0x05,0xd6,0x0b]

v_mfma_f32_32x32x4f16 v[0:31], a[0:1], v[2:3], -2.0
// GFX90A: v_mfma_f32_32x32x4f16 v[0:31], a[0:1], v[2:3], -2.0 ; encoding: [0x00,0x00,0xc8,0xd3,0x00,0x05,0xd6,0x0b]

v_mfma_f32_32x32x4f16 a[0:31], a[0:1], v[2:3], -2.0 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_32x32x4f16 a[0:31], a[0:1], v[2:3], -2.0 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xc8,0xd3,0x00,0x05,0xd6,0xeb]

v_mfma_f32_32x32x4f16 v[0:31], a[0:1], v[2:3], -2.0 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_32x32x4f16 v[0:31], a[0:1], v[2:3], -2.0 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xc8,0xd3,0x00,0x05,0xd6,0xeb]

v_mfma_f32_32x32x4f16 a[0:31], a[0:1], a[2:3], -2.0
// GFX90A: v_mfma_f32_32x32x4f16 a[0:31], a[0:1], a[2:3], -2.0 ; encoding: [0x00,0x80,0xc8,0xd3,0x00,0x05,0xd6,0x1b]

v_mfma_f32_32x32x4f16 v[0:31], a[0:1], a[2:3], -2.0
// GFX90A: v_mfma_f32_32x32x4f16 v[0:31], a[0:1], a[2:3], -2.0 ; encoding: [0x00,0x00,0xc8,0xd3,0x00,0x05,0xd6,0x1b]

v_mfma_f32_32x32x4f16 a[0:31], a[0:1], a[2:3], -2.0 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_32x32x4f16 a[0:31], a[0:1], a[2:3], -2.0 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xc8,0xd3,0x00,0x05,0xd6,0xfb]

v_mfma_f32_32x32x4f16 v[0:31], a[0:1], a[2:3], -2.0 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_32x32x4f16 v[0:31], a[0:1], a[2:3], -2.0 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xc8,0xd3,0x00,0x05,0xd6,0xfb]

v_mfma_f32_16x16x4f16 a[0:15], v[0:1], v[2:3], a[18:33]
// GFX90A: v_mfma_f32_16x16x4f16 a[0:15], v[0:1], v[2:3], a[18:33] ; encoding: [0x00,0x80,0xc9,0xd3,0x00,0x05,0x4a,0x04]

v_mfma_f32_16x16x4f16 a[0:15], v[0:1], v[2:3], a[18:33] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_16x16x4f16 a[0:15], v[0:1], v[2:3], a[18:33] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xc9,0xd3,0x00,0x05,0x4a,0xe4]

v_mfma_f32_16x16x4f16 a[0:15], v[0:1], a[2:3], a[18:33]
// GFX90A: v_mfma_f32_16x16x4f16 a[0:15], v[0:1], a[2:3], a[18:33] ; encoding: [0x00,0x80,0xc9,0xd3,0x00,0x05,0x4a,0x14]

v_mfma_f32_16x16x4f16 a[0:15], v[0:1], a[2:3], a[18:33] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_16x16x4f16 a[0:15], v[0:1], a[2:3], a[18:33] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xc9,0xd3,0x00,0x05,0x4a,0xf4]

v_mfma_f32_16x16x4f16 a[0:15], a[0:1], v[2:3], a[18:33]
// GFX90A: v_mfma_f32_16x16x4f16 a[0:15], a[0:1], v[2:3], a[18:33] ; encoding: [0x00,0x80,0xc9,0xd3,0x00,0x05,0x4a,0x0c]

v_mfma_f32_16x16x4f16 a[0:15], a[0:1], v[2:3], a[18:33] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_16x16x4f16 a[0:15], a[0:1], v[2:3], a[18:33] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xc9,0xd3,0x00,0x05,0x4a,0xec]

v_mfma_f32_16x16x4f16 a[0:15], a[0:1], a[2:3], a[18:33]
// GFX90A: v_mfma_f32_16x16x4f16 a[0:15], a[0:1], a[2:3], a[18:33] ; encoding: [0x00,0x80,0xc9,0xd3,0x00,0x05,0x4a,0x1c]

v_mfma_f32_16x16x4f16 a[0:15], a[0:1], a[2:3], a[18:33] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_16x16x4f16 a[0:15], a[0:1], a[2:3], a[18:33] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xc9,0xd3,0x00,0x05,0x4a,0xfc]

v_mfma_f32_16x16x4f16 v[0:15], v[0:1], v[2:3], v[18:33]
// GFX90A: v_mfma_f32_16x16x4f16 v[0:15], v[0:1], v[2:3], v[18:33] ; encoding: [0x00,0x00,0xc9,0xd3,0x00,0x05,0x4a,0x04]

v_mfma_f32_16x16x4f16 v[0:15], v[0:1], v[2:3], v[18:33] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_16x16x4f16 v[0:15], v[0:1], v[2:3], v[18:33] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xc9,0xd3,0x00,0x05,0x4a,0xe4]

v_mfma_f32_16x16x4f16 v[0:15], v[0:1], a[2:3], v[18:33]
// GFX90A: v_mfma_f32_16x16x4f16 v[0:15], v[0:1], a[2:3], v[18:33] ; encoding: [0x00,0x00,0xc9,0xd3,0x00,0x05,0x4a,0x14]

v_mfma_f32_16x16x4f16 v[0:15], v[0:1], a[2:3], v[18:33] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_16x16x4f16 v[0:15], v[0:1], a[2:3], v[18:33] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xc9,0xd3,0x00,0x05,0x4a,0xf4]

v_mfma_f32_16x16x4f16 v[0:15], a[0:1], v[2:3], v[18:33]
// GFX90A: v_mfma_f32_16x16x4f16 v[0:15], a[0:1], v[2:3], v[18:33] ; encoding: [0x00,0x00,0xc9,0xd3,0x00,0x05,0x4a,0x0c]

v_mfma_f32_16x16x4f16 v[0:15], a[0:1], v[2:3], v[18:33] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_16x16x4f16 v[0:15], a[0:1], v[2:3], v[18:33] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xc9,0xd3,0x00,0x05,0x4a,0xec]

v_mfma_f32_16x16x4f16 v[0:15], a[0:1], a[2:3], v[18:33]
// GFX90A: v_mfma_f32_16x16x4f16 v[0:15], a[0:1], a[2:3], v[18:33] ; encoding: [0x00,0x00,0xc9,0xd3,0x00,0x05,0x4a,0x1c]

v_mfma_f32_16x16x4f16 v[0:15], a[0:1], a[2:3], v[18:33] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_16x16x4f16 v[0:15], a[0:1], a[2:3], v[18:33] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xc9,0xd3,0x00,0x05,0x4a,0xfc]

v_mfma_f32_16x16x4f16 a[0:15], v[0:1], v[2:3], -2.0
// GFX90A: v_mfma_f32_16x16x4f16 a[0:15], v[0:1], v[2:3], -2.0 ; encoding: [0x00,0x80,0xc9,0xd3,0x00,0x05,0xd6,0x03]

v_mfma_f32_16x16x4f16 v[0:15], v[0:1], v[2:3], -2.0
// GFX90A: v_mfma_f32_16x16x4f16 v[0:15], v[0:1], v[2:3], -2.0 ; encoding: [0x00,0x00,0xc9,0xd3,0x00,0x05,0xd6,0x03]

v_mfma_f32_16x16x4f16 a[0:15], v[0:1], v[2:3], -2.0 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_16x16x4f16 a[0:15], v[0:1], v[2:3], -2.0 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xc9,0xd3,0x00,0x05,0xd6,0xe3]

v_mfma_f32_16x16x4f16 v[0:15], v[0:1], v[2:3], -2.0 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_16x16x4f16 v[0:15], v[0:1], v[2:3], -2.0 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xc9,0xd3,0x00,0x05,0xd6,0xe3]

v_mfma_f32_16x16x4f16 a[0:15], v[0:1], a[2:3], -2.0
// GFX90A: v_mfma_f32_16x16x4f16 a[0:15], v[0:1], a[2:3], -2.0 ; encoding: [0x00,0x80,0xc9,0xd3,0x00,0x05,0xd6,0x13]

v_mfma_f32_16x16x4f16 v[0:15], v[0:1], a[2:3], -2.0
// GFX90A: v_mfma_f32_16x16x4f16 v[0:15], v[0:1], a[2:3], -2.0 ; encoding: [0x00,0x00,0xc9,0xd3,0x00,0x05,0xd6,0x13]

v_mfma_f32_16x16x4f16 a[0:15], v[0:1], a[2:3], -2.0 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_16x16x4f16 a[0:15], v[0:1], a[2:3], -2.0 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xc9,0xd3,0x00,0x05,0xd6,0xf3]

v_mfma_f32_16x16x4f16 v[0:15], v[0:1], a[2:3], -2.0 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_16x16x4f16 v[0:15], v[0:1], a[2:3], -2.0 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xc9,0xd3,0x00,0x05,0xd6,0xf3]

v_mfma_f32_16x16x4f16 a[0:15], a[0:1], v[2:3], -2.0
// GFX90A: v_mfma_f32_16x16x4f16 a[0:15], a[0:1], v[2:3], -2.0 ; encoding: [0x00,0x80,0xc9,0xd3,0x00,0x05,0xd6,0x0b]

v_mfma_f32_16x16x4f16 v[0:15], a[0:1], v[2:3], -2.0
// GFX90A: v_mfma_f32_16x16x4f16 v[0:15], a[0:1], v[2:3], -2.0 ; encoding: [0x00,0x00,0xc9,0xd3,0x00,0x05,0xd6,0x0b]

v_mfma_f32_16x16x4f16 a[0:15], a[0:1], v[2:3], -2.0 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_16x16x4f16 a[0:15], a[0:1], v[2:3], -2.0 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xc9,0xd3,0x00,0x05,0xd6,0xeb]

v_mfma_f32_16x16x4f16 v[0:15], a[0:1], v[2:3], -2.0 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_16x16x4f16 v[0:15], a[0:1], v[2:3], -2.0 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xc9,0xd3,0x00,0x05,0xd6,0xeb]

v_mfma_f32_16x16x4f16 a[0:15], a[0:1], a[2:3], -2.0
// GFX90A: v_mfma_f32_16x16x4f16 a[0:15], a[0:1], a[2:3], -2.0 ; encoding: [0x00,0x80,0xc9,0xd3,0x00,0x05,0xd6,0x1b]

v_mfma_f32_16x16x4f16 v[0:15], a[0:1], a[2:3], -2.0
// GFX90A: v_mfma_f32_16x16x4f16 v[0:15], a[0:1], a[2:3], -2.0 ; encoding: [0x00,0x00,0xc9,0xd3,0x00,0x05,0xd6,0x1b]

v_mfma_f32_16x16x4f16 a[0:15], a[0:1], a[2:3], -2.0 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_16x16x4f16 a[0:15], a[0:1], a[2:3], -2.0 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xc9,0xd3,0x00,0x05,0xd6,0xfb]

v_mfma_f32_16x16x4f16 v[0:15], a[0:1], a[2:3], -2.0 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_16x16x4f16 v[0:15], a[0:1], a[2:3], -2.0 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xc9,0xd3,0x00,0x05,0xd6,0xfb]

v_mfma_f32_4x4x4f16 a[0:3], v[0:1], v[2:3], a[2:5]
// GFX90A: v_mfma_f32_4x4x4f16 a[0:3], v[0:1], v[2:3], a[2:5] ; encoding: [0x00,0x80,0xca,0xd3,0x00,0x05,0x0a,0x04]

v_mfma_f32_4x4x4f16 a[0:3], v[0:1], v[2:3], a[2:5] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_4x4x4f16 a[0:3], v[0:1], v[2:3], a[2:5] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xca,0xd3,0x00,0x05,0x0a,0xe4]

v_mfma_f32_4x4x4f16 a[0:3], v[0:1], a[2:3], a[2:5]
// GFX90A: v_mfma_f32_4x4x4f16 a[0:3], v[0:1], a[2:3], a[2:5] ; encoding: [0x00,0x80,0xca,0xd3,0x00,0x05,0x0a,0x14]

v_mfma_f32_4x4x4f16 a[0:3], v[0:1], a[2:3], a[2:5] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_4x4x4f16 a[0:3], v[0:1], a[2:3], a[2:5] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xca,0xd3,0x00,0x05,0x0a,0xf4]

v_mfma_f32_4x4x4f16 a[0:3], a[0:1], v[2:3], a[2:5]
// GFX90A: v_mfma_f32_4x4x4f16 a[0:3], a[0:1], v[2:3], a[2:5] ; encoding: [0x00,0x80,0xca,0xd3,0x00,0x05,0x0a,0x0c]

v_mfma_f32_4x4x4f16 a[0:3], a[0:1], v[2:3], a[2:5] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_4x4x4f16 a[0:3], a[0:1], v[2:3], a[2:5] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xca,0xd3,0x00,0x05,0x0a,0xec]

v_mfma_f32_4x4x4f16 a[0:3], a[0:1], a[2:3], a[2:5]
// GFX90A: v_mfma_f32_4x4x4f16 a[0:3], a[0:1], a[2:3], a[2:5] ; encoding: [0x00,0x80,0xca,0xd3,0x00,0x05,0x0a,0x1c]

v_mfma_f32_4x4x4f16 a[0:3], a[0:1], a[2:3], a[2:5] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_4x4x4f16 a[0:3], a[0:1], a[2:3], a[2:5] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xca,0xd3,0x00,0x05,0x0a,0xfc]

v_mfma_f32_4x4x4f16 v[0:3], v[0:1], v[2:3], v[2:5]
// GFX90A: v_mfma_f32_4x4x4f16 v[0:3], v[0:1], v[2:3], v[2:5] ; encoding: [0x00,0x00,0xca,0xd3,0x00,0x05,0x0a,0x04]

v_mfma_f32_4x4x4f16 v[0:3], v[0:1], v[2:3], v[2:5] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_4x4x4f16 v[0:3], v[0:1], v[2:3], v[2:5] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xca,0xd3,0x00,0x05,0x0a,0xe4]

v_mfma_f32_4x4x4f16 v[0:3], v[0:1], a[2:3], v[2:5]
// GFX90A: v_mfma_f32_4x4x4f16 v[0:3], v[0:1], a[2:3], v[2:5] ; encoding: [0x00,0x00,0xca,0xd3,0x00,0x05,0x0a,0x14]

v_mfma_f32_4x4x4f16 v[0:3], v[0:1], a[2:3], v[2:5] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_4x4x4f16 v[0:3], v[0:1], a[2:3], v[2:5] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xca,0xd3,0x00,0x05,0x0a,0xf4]

v_mfma_f32_4x4x4f16 v[0:3], a[0:1], v[2:3], v[2:5]
// GFX90A: v_mfma_f32_4x4x4f16 v[0:3], a[0:1], v[2:3], v[2:5] ; encoding: [0x00,0x00,0xca,0xd3,0x00,0x05,0x0a,0x0c]

v_mfma_f32_4x4x4f16 v[0:3], a[0:1], v[2:3], v[2:5] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_4x4x4f16 v[0:3], a[0:1], v[2:3], v[2:5] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xca,0xd3,0x00,0x05,0x0a,0xec]

v_mfma_f32_4x4x4f16 v[0:3], a[0:1], a[2:3], v[2:5]
// GFX90A: v_mfma_f32_4x4x4f16 v[0:3], a[0:1], a[2:3], v[2:5] ; encoding: [0x00,0x00,0xca,0xd3,0x00,0x05,0x0a,0x1c]

v_mfma_f32_4x4x4f16 v[0:3], a[0:1], a[2:3], v[2:5] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_4x4x4f16 v[0:3], a[0:1], a[2:3], v[2:5] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xca,0xd3,0x00,0x05,0x0a,0xfc]

v_mfma_f32_4x4x4f16 a[0:3], v[0:1], v[2:3], -2.0
// GFX90A: v_mfma_f32_4x4x4f16 a[0:3], v[0:1], v[2:3], -2.0 ; encoding: [0x00,0x80,0xca,0xd3,0x00,0x05,0xd6,0x03]

v_mfma_f32_4x4x4f16 v[0:3], v[0:1], v[2:3], -2.0
// GFX90A: v_mfma_f32_4x4x4f16 v[0:3], v[0:1], v[2:3], -2.0 ; encoding: [0x00,0x00,0xca,0xd3,0x00,0x05,0xd6,0x03]

v_mfma_f32_4x4x4f16 a[0:3], v[0:1], v[2:3], -2.0 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_4x4x4f16 a[0:3], v[0:1], v[2:3], -2.0 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xca,0xd3,0x00,0x05,0xd6,0xe3]

v_mfma_f32_4x4x4f16 v[0:3], v[0:1], v[2:3], -2.0 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_4x4x4f16 v[0:3], v[0:1], v[2:3], -2.0 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xca,0xd3,0x00,0x05,0xd6,0xe3]

v_mfma_f32_4x4x4f16 a[0:3], v[0:1], a[2:3], -2.0
// GFX90A: v_mfma_f32_4x4x4f16 a[0:3], v[0:1], a[2:3], -2.0 ; encoding: [0x00,0x80,0xca,0xd3,0x00,0x05,0xd6,0x13]

v_mfma_f32_4x4x4f16 v[0:3], v[0:1], a[2:3], -2.0
// GFX90A: v_mfma_f32_4x4x4f16 v[0:3], v[0:1], a[2:3], -2.0 ; encoding: [0x00,0x00,0xca,0xd3,0x00,0x05,0xd6,0x13]

v_mfma_f32_4x4x4f16 a[0:3], v[0:1], a[2:3], -2.0 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_4x4x4f16 a[0:3], v[0:1], a[2:3], -2.0 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xca,0xd3,0x00,0x05,0xd6,0xf3]

v_mfma_f32_4x4x4f16 v[0:3], v[0:1], a[2:3], -2.0 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_4x4x4f16 v[0:3], v[0:1], a[2:3], -2.0 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xca,0xd3,0x00,0x05,0xd6,0xf3]

v_mfma_f32_4x4x4f16 a[0:3], a[0:1], v[2:3], -2.0
// GFX90A: v_mfma_f32_4x4x4f16 a[0:3], a[0:1], v[2:3], -2.0 ; encoding: [0x00,0x80,0xca,0xd3,0x00,0x05,0xd6,0x0b]

v_mfma_f32_4x4x4f16 v[0:3], a[0:1], v[2:3], -2.0
// GFX90A: v_mfma_f32_4x4x4f16 v[0:3], a[0:1], v[2:3], -2.0 ; encoding: [0x00,0x00,0xca,0xd3,0x00,0x05,0xd6,0x0b]

v_mfma_f32_4x4x4f16 a[0:3], a[0:1], v[2:3], -2.0 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_4x4x4f16 a[0:3], a[0:1], v[2:3], -2.0 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xca,0xd3,0x00,0x05,0xd6,0xeb]

v_mfma_f32_4x4x4f16 v[0:3], a[0:1], v[2:3], -2.0 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_4x4x4f16 v[0:3], a[0:1], v[2:3], -2.0 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xca,0xd3,0x00,0x05,0xd6,0xeb]

v_mfma_f32_4x4x4f16 a[0:3], a[0:1], a[2:3], -2.0
// GFX90A: v_mfma_f32_4x4x4f16 a[0:3], a[0:1], a[2:3], -2.0 ; encoding: [0x00,0x80,0xca,0xd3,0x00,0x05,0xd6,0x1b]

v_mfma_f32_4x4x4f16 v[0:3], a[0:1], a[2:3], -2.0
// GFX90A: v_mfma_f32_4x4x4f16 v[0:3], a[0:1], a[2:3], -2.0 ; encoding: [0x00,0x00,0xca,0xd3,0x00,0x05,0xd6,0x1b]

v_mfma_f32_4x4x4f16 a[0:3], a[0:1], a[2:3], -2.0 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_4x4x4f16 a[0:3], a[0:1], a[2:3], -2.0 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xca,0xd3,0x00,0x05,0xd6,0xfb]

v_mfma_f32_4x4x4f16 v[0:3], a[0:1], a[2:3], -2.0 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_4x4x4f16 v[0:3], a[0:1], a[2:3], -2.0 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xca,0xd3,0x00,0x05,0xd6,0xfb]

v_mfma_f32_32x32x8f16 a[0:15], v[0:1], v[2:3], a[18:33]
// GFX90A: v_mfma_f32_32x32x8f16 a[0:15], v[0:1], v[2:3], a[18:33] ; encoding: [0x00,0x80,0xcc,0xd3,0x00,0x05,0x4a,0x04]

v_mfma_f32_32x32x8f16 a[0:15], v[0:1], v[2:3], a[18:33] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_32x32x8f16 a[0:15], v[0:1], v[2:3], a[18:33] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xcc,0xd3,0x00,0x05,0x4a,0xe4]

v_mfma_f32_32x32x8f16 a[0:15], v[0:1], a[2:3], a[18:33]
// GFX90A: v_mfma_f32_32x32x8f16 a[0:15], v[0:1], a[2:3], a[18:33] ; encoding: [0x00,0x80,0xcc,0xd3,0x00,0x05,0x4a,0x14]

v_mfma_f32_32x32x8f16 a[0:15], v[0:1], a[2:3], a[18:33] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_32x32x8f16 a[0:15], v[0:1], a[2:3], a[18:33] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xcc,0xd3,0x00,0x05,0x4a,0xf4]

v_mfma_f32_32x32x8f16 a[0:15], a[0:1], v[2:3], a[18:33]
// GFX90A: v_mfma_f32_32x32x8f16 a[0:15], a[0:1], v[2:3], a[18:33] ; encoding: [0x00,0x80,0xcc,0xd3,0x00,0x05,0x4a,0x0c]

v_mfma_f32_32x32x8f16 a[0:15], a[0:1], v[2:3], a[18:33] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_32x32x8f16 a[0:15], a[0:1], v[2:3], a[18:33] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xcc,0xd3,0x00,0x05,0x4a,0xec]

v_mfma_f32_32x32x8f16 a[0:15], a[0:1], a[2:3], a[18:33]
// GFX90A: v_mfma_f32_32x32x8f16 a[0:15], a[0:1], a[2:3], a[18:33] ; encoding: [0x00,0x80,0xcc,0xd3,0x00,0x05,0x4a,0x1c]

v_mfma_f32_32x32x8f16 a[0:15], a[0:1], a[2:3], a[18:33] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_32x32x8f16 a[0:15], a[0:1], a[2:3], a[18:33] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xcc,0xd3,0x00,0x05,0x4a,0xfc]

v_mfma_f32_32x32x8f16 v[0:15], v[0:1], v[2:3], v[18:33]
// GFX90A: v_mfma_f32_32x32x8f16 v[0:15], v[0:1], v[2:3], v[18:33] ; encoding: [0x00,0x00,0xcc,0xd3,0x00,0x05,0x4a,0x04]

v_mfma_f32_32x32x8f16 v[0:15], v[0:1], v[2:3], v[18:33] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_32x32x8f16 v[0:15], v[0:1], v[2:3], v[18:33] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xcc,0xd3,0x00,0x05,0x4a,0xe4]

v_mfma_f32_32x32x8f16 v[0:15], v[0:1], a[2:3], v[18:33]
// GFX90A: v_mfma_f32_32x32x8f16 v[0:15], v[0:1], a[2:3], v[18:33] ; encoding: [0x00,0x00,0xcc,0xd3,0x00,0x05,0x4a,0x14]

v_mfma_f32_32x32x8f16 v[0:15], v[0:1], a[2:3], v[18:33] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_32x32x8f16 v[0:15], v[0:1], a[2:3], v[18:33] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xcc,0xd3,0x00,0x05,0x4a,0xf4]

v_mfma_f32_32x32x8f16 v[0:15], a[0:1], v[2:3], v[18:33]
// GFX90A: v_mfma_f32_32x32x8f16 v[0:15], a[0:1], v[2:3], v[18:33] ; encoding: [0x00,0x00,0xcc,0xd3,0x00,0x05,0x4a,0x0c]

v_mfma_f32_32x32x8f16 v[0:15], a[0:1], v[2:3], v[18:33] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_32x32x8f16 v[0:15], a[0:1], v[2:3], v[18:33] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xcc,0xd3,0x00,0x05,0x4a,0xec]

v_mfma_f32_32x32x8f16 v[0:15], a[0:1], a[2:3], v[18:33]
// GFX90A: v_mfma_f32_32x32x8f16 v[0:15], a[0:1], a[2:3], v[18:33] ; encoding: [0x00,0x00,0xcc,0xd3,0x00,0x05,0x4a,0x1c]

v_mfma_f32_32x32x8f16 v[0:15], a[0:1], a[2:3], v[18:33] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_32x32x8f16 v[0:15], a[0:1], a[2:3], v[18:33] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xcc,0xd3,0x00,0x05,0x4a,0xfc]

v_mfma_f32_32x32x8f16 a[0:15], v[0:1], v[2:3], -2.0
// GFX90A: v_mfma_f32_32x32x8f16 a[0:15], v[0:1], v[2:3], -2.0 ; encoding: [0x00,0x80,0xcc,0xd3,0x00,0x05,0xd6,0x03]

v_mfma_f32_32x32x8f16 v[0:15], v[0:1], v[2:3], -2.0
// GFX90A: v_mfma_f32_32x32x8f16 v[0:15], v[0:1], v[2:3], -2.0 ; encoding: [0x00,0x00,0xcc,0xd3,0x00,0x05,0xd6,0x03]

v_mfma_f32_32x32x8f16 a[0:15], v[0:1], v[2:3], -2.0 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_32x32x8f16 a[0:15], v[0:1], v[2:3], -2.0 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xcc,0xd3,0x00,0x05,0xd6,0xe3]

v_mfma_f32_32x32x8f16 v[0:15], v[0:1], v[2:3], -2.0 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_32x32x8f16 v[0:15], v[0:1], v[2:3], -2.0 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xcc,0xd3,0x00,0x05,0xd6,0xe3]

v_mfma_f32_32x32x8f16 a[0:15], v[0:1], a[2:3], -2.0
// GFX90A: v_mfma_f32_32x32x8f16 a[0:15], v[0:1], a[2:3], -2.0 ; encoding: [0x00,0x80,0xcc,0xd3,0x00,0x05,0xd6,0x13]

v_mfma_f32_32x32x8f16 v[0:15], v[0:1], a[2:3], -2.0
// GFX90A: v_mfma_f32_32x32x8f16 v[0:15], v[0:1], a[2:3], -2.0 ; encoding: [0x00,0x00,0xcc,0xd3,0x00,0x05,0xd6,0x13]

v_mfma_f32_32x32x8f16 a[0:15], v[0:1], a[2:3], -2.0 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_32x32x8f16 a[0:15], v[0:1], a[2:3], -2.0 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xcc,0xd3,0x00,0x05,0xd6,0xf3]

v_mfma_f32_32x32x8f16 v[0:15], v[0:1], a[2:3], -2.0 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_32x32x8f16 v[0:15], v[0:1], a[2:3], -2.0 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xcc,0xd3,0x00,0x05,0xd6,0xf3]

v_mfma_f32_32x32x8f16 a[0:15], a[0:1], v[2:3], -2.0
// GFX90A: v_mfma_f32_32x32x8f16 a[0:15], a[0:1], v[2:3], -2.0 ; encoding: [0x00,0x80,0xcc,0xd3,0x00,0x05,0xd6,0x0b]

v_mfma_f32_32x32x8f16 v[0:15], a[0:1], v[2:3], -2.0
// GFX90A: v_mfma_f32_32x32x8f16 v[0:15], a[0:1], v[2:3], -2.0 ; encoding: [0x00,0x00,0xcc,0xd3,0x00,0x05,0xd6,0x0b]

v_mfma_f32_32x32x8f16 a[0:15], a[0:1], v[2:3], -2.0 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_32x32x8f16 a[0:15], a[0:1], v[2:3], -2.0 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xcc,0xd3,0x00,0x05,0xd6,0xeb]

v_mfma_f32_32x32x8f16 v[0:15], a[0:1], v[2:3], -2.0 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_32x32x8f16 v[0:15], a[0:1], v[2:3], -2.0 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xcc,0xd3,0x00,0x05,0xd6,0xeb]

v_mfma_f32_32x32x8f16 a[0:15], a[0:1], a[2:3], -2.0
// GFX90A: v_mfma_f32_32x32x8f16 a[0:15], a[0:1], a[2:3], -2.0 ; encoding: [0x00,0x80,0xcc,0xd3,0x00,0x05,0xd6,0x1b]

v_mfma_f32_32x32x8f16 v[0:15], a[0:1], a[2:3], -2.0
// GFX90A: v_mfma_f32_32x32x8f16 v[0:15], a[0:1], a[2:3], -2.0 ; encoding: [0x00,0x00,0xcc,0xd3,0x00,0x05,0xd6,0x1b]

v_mfma_f32_32x32x8f16 a[0:15], a[0:1], a[2:3], -2.0 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_32x32x8f16 a[0:15], a[0:1], a[2:3], -2.0 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xcc,0xd3,0x00,0x05,0xd6,0xfb]

v_mfma_f32_32x32x8f16 v[0:15], a[0:1], a[2:3], -2.0 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_32x32x8f16 v[0:15], a[0:1], a[2:3], -2.0 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xcc,0xd3,0x00,0x05,0xd6,0xfb]

v_mfma_f32_16x16x16f16 a[0:3], v[0:1], v[2:3], a[2:5]
// GFX90A: v_mfma_f32_16x16x16f16 a[0:3], v[0:1], v[2:3], a[2:5] ; encoding: [0x00,0x80,0xcd,0xd3,0x00,0x05,0x0a,0x04]

v_mfma_f32_16x16x16f16 a[0:3], v[0:1], v[2:3], a[2:5] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_16x16x16f16 a[0:3], v[0:1], v[2:3], a[2:5] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xcd,0xd3,0x00,0x05,0x0a,0xe4]

v_mfma_f32_16x16x16f16 a[0:3], v[0:1], a[2:3], a[2:5]
// GFX90A: v_mfma_f32_16x16x16f16 a[0:3], v[0:1], a[2:3], a[2:5] ; encoding: [0x00,0x80,0xcd,0xd3,0x00,0x05,0x0a,0x14]

v_mfma_f32_16x16x16f16 a[0:3], v[0:1], a[2:3], a[2:5] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_16x16x16f16 a[0:3], v[0:1], a[2:3], a[2:5] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xcd,0xd3,0x00,0x05,0x0a,0xf4]

v_mfma_f32_16x16x16f16 a[0:3], a[0:1], v[2:3], a[2:5]
// GFX90A: v_mfma_f32_16x16x16f16 a[0:3], a[0:1], v[2:3], a[2:5] ; encoding: [0x00,0x80,0xcd,0xd3,0x00,0x05,0x0a,0x0c]

v_mfma_f32_16x16x16f16 a[0:3], a[0:1], v[2:3], a[2:5] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_16x16x16f16 a[0:3], a[0:1], v[2:3], a[2:5] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xcd,0xd3,0x00,0x05,0x0a,0xec]

v_mfma_f32_16x16x16f16 a[0:3], a[0:1], a[2:3], a[2:5]
// GFX90A: v_mfma_f32_16x16x16f16 a[0:3], a[0:1], a[2:3], a[2:5] ; encoding: [0x00,0x80,0xcd,0xd3,0x00,0x05,0x0a,0x1c]

v_mfma_f32_16x16x16f16 a[0:3], a[0:1], a[2:3], a[2:5] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_16x16x16f16 a[0:3], a[0:1], a[2:3], a[2:5] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xcd,0xd3,0x00,0x05,0x0a,0xfc]

v_mfma_f32_16x16x16f16 v[0:3], v[0:1], v[2:3], v[2:5]
// GFX90A: v_mfma_f32_16x16x16f16 v[0:3], v[0:1], v[2:3], v[2:5] ; encoding: [0x00,0x00,0xcd,0xd3,0x00,0x05,0x0a,0x04]

v_mfma_f32_16x16x16f16 v[0:3], v[0:1], v[2:3], v[2:5] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_16x16x16f16 v[0:3], v[0:1], v[2:3], v[2:5] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xcd,0xd3,0x00,0x05,0x0a,0xe4]

v_mfma_f32_16x16x16f16 v[0:3], v[0:1], a[2:3], v[2:5]
// GFX90A: v_mfma_f32_16x16x16f16 v[0:3], v[0:1], a[2:3], v[2:5] ; encoding: [0x00,0x00,0xcd,0xd3,0x00,0x05,0x0a,0x14]

v_mfma_f32_16x16x16f16 v[0:3], v[0:1], a[2:3], v[2:5] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_16x16x16f16 v[0:3], v[0:1], a[2:3], v[2:5] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xcd,0xd3,0x00,0x05,0x0a,0xf4]

v_mfma_f32_16x16x16f16 v[0:3], a[0:1], v[2:3], v[2:5]
// GFX90A: v_mfma_f32_16x16x16f16 v[0:3], a[0:1], v[2:3], v[2:5] ; encoding: [0x00,0x00,0xcd,0xd3,0x00,0x05,0x0a,0x0c]

v_mfma_f32_16x16x16f16 v[0:3], a[0:1], v[2:3], v[2:5] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_16x16x16f16 v[0:3], a[0:1], v[2:3], v[2:5] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xcd,0xd3,0x00,0x05,0x0a,0xec]

v_mfma_f32_16x16x16f16 v[0:3], a[0:1], a[2:3], v[2:5]
// GFX90A: v_mfma_f32_16x16x16f16 v[0:3], a[0:1], a[2:3], v[2:5] ; encoding: [0x00,0x00,0xcd,0xd3,0x00,0x05,0x0a,0x1c]

v_mfma_f32_16x16x16f16 v[0:3], a[0:1], a[2:3], v[2:5] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_16x16x16f16 v[0:3], a[0:1], a[2:3], v[2:5] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xcd,0xd3,0x00,0x05,0x0a,0xfc]

v_mfma_f32_16x16x16f16 a[0:3], v[0:1], v[2:3], -2.0
// GFX90A: v_mfma_f32_16x16x16f16 a[0:3], v[0:1], v[2:3], -2.0 ; encoding: [0x00,0x80,0xcd,0xd3,0x00,0x05,0xd6,0x03]

v_mfma_f32_16x16x16f16 v[0:3], v[0:1], v[2:3], -2.0
// GFX90A: v_mfma_f32_16x16x16f16 v[0:3], v[0:1], v[2:3], -2.0 ; encoding: [0x00,0x00,0xcd,0xd3,0x00,0x05,0xd6,0x03]

v_mfma_f32_16x16x16f16 a[0:3], v[0:1], v[2:3], -2.0 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_16x16x16f16 a[0:3], v[0:1], v[2:3], -2.0 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xcd,0xd3,0x00,0x05,0xd6,0xe3]

v_mfma_f32_16x16x16f16 v[0:3], v[0:1], v[2:3], -2.0 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_16x16x16f16 v[0:3], v[0:1], v[2:3], -2.0 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xcd,0xd3,0x00,0x05,0xd6,0xe3]

v_mfma_f32_16x16x16f16 a[0:3], v[0:1], a[2:3], -2.0
// GFX90A: v_mfma_f32_16x16x16f16 a[0:3], v[0:1], a[2:3], -2.0 ; encoding: [0x00,0x80,0xcd,0xd3,0x00,0x05,0xd6,0x13]

v_mfma_f32_16x16x16f16 v[0:3], v[0:1], a[2:3], -2.0
// GFX90A: v_mfma_f32_16x16x16f16 v[0:3], v[0:1], a[2:3], -2.0 ; encoding: [0x00,0x00,0xcd,0xd3,0x00,0x05,0xd6,0x13]

v_mfma_f32_16x16x16f16 a[0:3], v[0:1], a[2:3], -2.0 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_16x16x16f16 a[0:3], v[0:1], a[2:3], -2.0 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xcd,0xd3,0x00,0x05,0xd6,0xf3]

v_mfma_f32_16x16x16f16 v[0:3], v[0:1], a[2:3], -2.0 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_16x16x16f16 v[0:3], v[0:1], a[2:3], -2.0 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xcd,0xd3,0x00,0x05,0xd6,0xf3]

v_mfma_f32_16x16x16f16 a[0:3], a[0:1], v[2:3], -2.0
// GFX90A: v_mfma_f32_16x16x16f16 a[0:3], a[0:1], v[2:3], -2.0 ; encoding: [0x00,0x80,0xcd,0xd3,0x00,0x05,0xd6,0x0b]

v_mfma_f32_16x16x16f16 v[0:3], a[0:1], v[2:3], -2.0
// GFX90A: v_mfma_f32_16x16x16f16 v[0:3], a[0:1], v[2:3], -2.0 ; encoding: [0x00,0x00,0xcd,0xd3,0x00,0x05,0xd6,0x0b]

v_mfma_f32_16x16x16f16 a[0:3], a[0:1], v[2:3], -2.0 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_16x16x16f16 a[0:3], a[0:1], v[2:3], -2.0 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xcd,0xd3,0x00,0x05,0xd6,0xeb]

v_mfma_f32_16x16x16f16 v[0:3], a[0:1], v[2:3], -2.0 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_16x16x16f16 v[0:3], a[0:1], v[2:3], -2.0 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xcd,0xd3,0x00,0x05,0xd6,0xeb]

v_mfma_f32_16x16x16f16 a[0:3], a[0:1], a[2:3], -2.0
// GFX90A: v_mfma_f32_16x16x16f16 a[0:3], a[0:1], a[2:3], -2.0 ; encoding: [0x00,0x80,0xcd,0xd3,0x00,0x05,0xd6,0x1b]

v_mfma_f32_16x16x16f16 v[0:3], a[0:1], a[2:3], -2.0
// GFX90A: v_mfma_f32_16x16x16f16 v[0:3], a[0:1], a[2:3], -2.0 ; encoding: [0x00,0x00,0xcd,0xd3,0x00,0x05,0xd6,0x1b]

v_mfma_f32_16x16x16f16 a[0:3], a[0:1], a[2:3], -2.0 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_16x16x16f16 a[0:3], a[0:1], a[2:3], -2.0 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xcd,0xd3,0x00,0x05,0xd6,0xfb]

v_mfma_f32_16x16x16f16 v[0:3], a[0:1], a[2:3], -2.0 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_16x16x16f16 v[0:3], a[0:1], a[2:3], -2.0 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xcd,0xd3,0x00,0x05,0xd6,0xfb]

v_mfma_i32_32x32x4i8 a[0:31], v0, v1, a[34:65]
// GFX90A: v_mfma_i32_32x32x4i8 a[0:31], v0, v1, a[34:65] ; encoding: [0x00,0x80,0xd0,0xd3,0x00,0x03,0x8a,0x04]

v_mfma_i32_32x32x4i8 a[0:31], v0, v1, a[34:65] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_i32_32x32x4i8 a[0:31], v0, v1, a[34:65] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xd0,0xd3,0x00,0x03,0x8a,0xe4]

v_mfma_i32_32x32x4i8 a[0:31], v0, a1, a[34:65]
// GFX90A: v_mfma_i32_32x32x4i8 a[0:31], v0, a1, a[34:65] ; encoding: [0x00,0x80,0xd0,0xd3,0x00,0x03,0x8a,0x14]

v_mfma_i32_32x32x4i8 a[0:31], v0, a1, a[34:65] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_i32_32x32x4i8 a[0:31], v0, a1, a[34:65] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xd0,0xd3,0x00,0x03,0x8a,0xf4]

v_mfma_i32_32x32x4i8 a[0:31], a0, v1, a[34:65]
// GFX90A: v_mfma_i32_32x32x4i8 a[0:31], a0, v1, a[34:65] ; encoding: [0x00,0x80,0xd0,0xd3,0x00,0x03,0x8a,0x0c]

v_mfma_i32_32x32x4i8 a[0:31], a0, v1, a[34:65] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_i32_32x32x4i8 a[0:31], a0, v1, a[34:65] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xd0,0xd3,0x00,0x03,0x8a,0xec]

v_mfma_i32_32x32x4i8 a[0:31], a0, a1, a[34:65]
// GFX90A: v_mfma_i32_32x32x4i8 a[0:31], a0, a1, a[34:65] ; encoding: [0x00,0x80,0xd0,0xd3,0x00,0x03,0x8a,0x1c]

v_mfma_i32_32x32x4i8 a[0:31], a0, a1, a[34:65] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_i32_32x32x4i8 a[0:31], a0, a1, a[34:65] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xd0,0xd3,0x00,0x03,0x8a,0xfc]

v_mfma_i32_32x32x4i8 v[0:31], v0, v1, v[34:65]
// GFX90A: v_mfma_i32_32x32x4i8 v[0:31], v0, v1, v[34:65] ; encoding: [0x00,0x00,0xd0,0xd3,0x00,0x03,0x8a,0x04]

v_mfma_i32_32x32x4i8 v[0:31], v0, v1, v[34:65] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_i32_32x32x4i8 v[0:31], v0, v1, v[34:65] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xd0,0xd3,0x00,0x03,0x8a,0xe4]

v_mfma_i32_32x32x4i8 v[0:31], v0, a1, v[34:65]
// GFX90A: v_mfma_i32_32x32x4i8 v[0:31], v0, a1, v[34:65] ; encoding: [0x00,0x00,0xd0,0xd3,0x00,0x03,0x8a,0x14]

v_mfma_i32_32x32x4i8 v[0:31], v0, a1, v[34:65] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_i32_32x32x4i8 v[0:31], v0, a1, v[34:65] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xd0,0xd3,0x00,0x03,0x8a,0xf4]

v_mfma_i32_32x32x4i8 v[0:31], a0, v1, v[34:65]
// GFX90A: v_mfma_i32_32x32x4i8 v[0:31], a0, v1, v[34:65] ; encoding: [0x00,0x00,0xd0,0xd3,0x00,0x03,0x8a,0x0c]

v_mfma_i32_32x32x4i8 v[0:31], a0, v1, v[34:65] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_i32_32x32x4i8 v[0:31], a0, v1, v[34:65] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xd0,0xd3,0x00,0x03,0x8a,0xec]

v_mfma_i32_32x32x4i8 v[0:31], a0, a1, v[34:65]
// GFX90A: v_mfma_i32_32x32x4i8 v[0:31], a0, a1, v[34:65] ; encoding: [0x00,0x00,0xd0,0xd3,0x00,0x03,0x8a,0x1c]

v_mfma_i32_32x32x4i8 v[0:31], a0, a1, v[34:65] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_i32_32x32x4i8 v[0:31], a0, a1, v[34:65] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xd0,0xd3,0x00,0x03,0x8a,0xfc]

v_mfma_i32_32x32x4i8 a[0:31], v0, v1, 2
// GFX90A: v_mfma_i32_32x32x4i8 a[0:31], v0, v1, 2 ; encoding: [0x00,0x80,0xd0,0xd3,0x00,0x03,0x0a,0x02]

v_mfma_i32_32x32x4i8 v[0:31], v0, v1, 2
// GFX90A: v_mfma_i32_32x32x4i8 v[0:31], v0, v1, 2 ; encoding: [0x00,0x00,0xd0,0xd3,0x00,0x03,0x0a,0x02]

v_mfma_i32_32x32x4i8 a[0:31], v0, v1, 2 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_i32_32x32x4i8 a[0:31], v0, v1, 2 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xd0,0xd3,0x00,0x03,0x0a,0xe2]

v_mfma_i32_32x32x4i8 v[0:31], v0, v1, 2 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_i32_32x32x4i8 v[0:31], v0, v1, 2 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xd0,0xd3,0x00,0x03,0x0a,0xe2]

v_mfma_i32_32x32x4i8 a[0:31], v0, a1, 2
// GFX90A: v_mfma_i32_32x32x4i8 a[0:31], v0, a1, 2 ; encoding: [0x00,0x80,0xd0,0xd3,0x00,0x03,0x0a,0x12]

v_mfma_i32_32x32x4i8 v[0:31], v0, a1, 2
// GFX90A: v_mfma_i32_32x32x4i8 v[0:31], v0, a1, 2 ; encoding: [0x00,0x00,0xd0,0xd3,0x00,0x03,0x0a,0x12]

v_mfma_i32_32x32x4i8 a[0:31], v0, a1, 2 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_i32_32x32x4i8 a[0:31], v0, a1, 2 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xd0,0xd3,0x00,0x03,0x0a,0xf2]

v_mfma_i32_32x32x4i8 v[0:31], v0, a1, 2 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_i32_32x32x4i8 v[0:31], v0, a1, 2 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xd0,0xd3,0x00,0x03,0x0a,0xf2]

v_mfma_i32_32x32x4i8 a[0:31], a0, v1, 2
// GFX90A: v_mfma_i32_32x32x4i8 a[0:31], a0, v1, 2 ; encoding: [0x00,0x80,0xd0,0xd3,0x00,0x03,0x0a,0x0a]

v_mfma_i32_32x32x4i8 v[0:31], a0, v1, 2
// GFX90A: v_mfma_i32_32x32x4i8 v[0:31], a0, v1, 2 ; encoding: [0x00,0x00,0xd0,0xd3,0x00,0x03,0x0a,0x0a]

v_mfma_i32_32x32x4i8 a[0:31], a0, v1, 2 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_i32_32x32x4i8 a[0:31], a0, v1, 2 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xd0,0xd3,0x00,0x03,0x0a,0xea]

v_mfma_i32_32x32x4i8 v[0:31], a0, v1, 2 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_i32_32x32x4i8 v[0:31], a0, v1, 2 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xd0,0xd3,0x00,0x03,0x0a,0xea]

v_mfma_i32_32x32x4i8 a[0:31], a0, a1, 2
// GFX90A: v_mfma_i32_32x32x4i8 a[0:31], a0, a1, 2 ; encoding: [0x00,0x80,0xd0,0xd3,0x00,0x03,0x0a,0x1a]

v_mfma_i32_32x32x4i8 v[0:31], a0, a1, 2
// GFX90A: v_mfma_i32_32x32x4i8 v[0:31], a0, a1, 2 ; encoding: [0x00,0x00,0xd0,0xd3,0x00,0x03,0x0a,0x1a]

v_mfma_i32_32x32x4i8 a[0:31], a0, a1, 2 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_i32_32x32x4i8 a[0:31], a0, a1, 2 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xd0,0xd3,0x00,0x03,0x0a,0xfa]

v_mfma_i32_32x32x4i8 v[0:31], a0, a1, 2 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_i32_32x32x4i8 v[0:31], a0, a1, 2 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xd0,0xd3,0x00,0x03,0x0a,0xfa]

v_mfma_i32_16x16x4i8 a[0:15], v0, v1, a[18:33]
// GFX90A: v_mfma_i32_16x16x4i8 a[0:15], v0, v1, a[18:33] ; encoding: [0x00,0x80,0xd1,0xd3,0x00,0x03,0x4a,0x04]

v_mfma_i32_16x16x4i8 a[0:15], v0, v1, a[18:33] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_i32_16x16x4i8 a[0:15], v0, v1, a[18:33] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xd1,0xd3,0x00,0x03,0x4a,0xe4]

v_mfma_i32_16x16x4i8 a[0:15], v0, a1, a[18:33]
// GFX90A: v_mfma_i32_16x16x4i8 a[0:15], v0, a1, a[18:33] ; encoding: [0x00,0x80,0xd1,0xd3,0x00,0x03,0x4a,0x14]

v_mfma_i32_16x16x4i8 a[0:15], v0, a1, a[18:33] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_i32_16x16x4i8 a[0:15], v0, a1, a[18:33] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xd1,0xd3,0x00,0x03,0x4a,0xf4]

v_mfma_i32_16x16x4i8 a[0:15], a0, v1, a[18:33]
// GFX90A: v_mfma_i32_16x16x4i8 a[0:15], a0, v1, a[18:33] ; encoding: [0x00,0x80,0xd1,0xd3,0x00,0x03,0x4a,0x0c]

v_mfma_i32_16x16x4i8 a[0:15], a0, v1, a[18:33] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_i32_16x16x4i8 a[0:15], a0, v1, a[18:33] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xd1,0xd3,0x00,0x03,0x4a,0xec]

v_mfma_i32_16x16x4i8 a[0:15], a0, a1, a[18:33]
// GFX90A: v_mfma_i32_16x16x4i8 a[0:15], a0, a1, a[18:33] ; encoding: [0x00,0x80,0xd1,0xd3,0x00,0x03,0x4a,0x1c]

v_mfma_i32_16x16x4i8 a[0:15], a0, a1, a[18:33] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_i32_16x16x4i8 a[0:15], a0, a1, a[18:33] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xd1,0xd3,0x00,0x03,0x4a,0xfc]

v_mfma_i32_16x16x4i8 v[0:15], v0, v1, v[18:33]
// GFX90A: v_mfma_i32_16x16x4i8 v[0:15], v0, v1, v[18:33] ; encoding: [0x00,0x00,0xd1,0xd3,0x00,0x03,0x4a,0x04]

v_mfma_i32_16x16x4i8 v[0:15], v0, v1, v[18:33] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_i32_16x16x4i8 v[0:15], v0, v1, v[18:33] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xd1,0xd3,0x00,0x03,0x4a,0xe4]

v_mfma_i32_16x16x4i8 v[0:15], v0, a1, v[18:33]
// GFX90A: v_mfma_i32_16x16x4i8 v[0:15], v0, a1, v[18:33] ; encoding: [0x00,0x00,0xd1,0xd3,0x00,0x03,0x4a,0x14]

v_mfma_i32_16x16x4i8 v[0:15], v0, a1, v[18:33] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_i32_16x16x4i8 v[0:15], v0, a1, v[18:33] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xd1,0xd3,0x00,0x03,0x4a,0xf4]

v_mfma_i32_16x16x4i8 v[0:15], a0, v1, v[18:33]
// GFX90A: v_mfma_i32_16x16x4i8 v[0:15], a0, v1, v[18:33] ; encoding: [0x00,0x00,0xd1,0xd3,0x00,0x03,0x4a,0x0c]

v_mfma_i32_16x16x4i8 v[0:15], a0, v1, v[18:33] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_i32_16x16x4i8 v[0:15], a0, v1, v[18:33] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xd1,0xd3,0x00,0x03,0x4a,0xec]

v_mfma_i32_16x16x4i8 v[0:15], a0, a1, v[18:33]
// GFX90A: v_mfma_i32_16x16x4i8 v[0:15], a0, a1, v[18:33] ; encoding: [0x00,0x00,0xd1,0xd3,0x00,0x03,0x4a,0x1c]

v_mfma_i32_16x16x4i8 v[0:15], a0, a1, v[18:33] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_i32_16x16x4i8 v[0:15], a0, a1, v[18:33] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xd1,0xd3,0x00,0x03,0x4a,0xfc]

v_mfma_i32_16x16x4i8 a[0:15], v0, v1, 2
// GFX90A: v_mfma_i32_16x16x4i8 a[0:15], v0, v1, 2 ; encoding: [0x00,0x80,0xd1,0xd3,0x00,0x03,0x0a,0x02]

v_mfma_i32_16x16x4i8 v[0:15], v0, v1, 2
// GFX90A: v_mfma_i32_16x16x4i8 v[0:15], v0, v1, 2 ; encoding: [0x00,0x00,0xd1,0xd3,0x00,0x03,0x0a,0x02]

v_mfma_i32_16x16x4i8 a[0:15], v0, v1, 2 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_i32_16x16x4i8 a[0:15], v0, v1, 2 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xd1,0xd3,0x00,0x03,0x0a,0xe2]

v_mfma_i32_16x16x4i8 v[0:15], v0, v1, 2 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_i32_16x16x4i8 v[0:15], v0, v1, 2 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xd1,0xd3,0x00,0x03,0x0a,0xe2]

v_mfma_i32_16x16x4i8 a[0:15], v0, a1, 2
// GFX90A: v_mfma_i32_16x16x4i8 a[0:15], v0, a1, 2 ; encoding: [0x00,0x80,0xd1,0xd3,0x00,0x03,0x0a,0x12]

v_mfma_i32_16x16x4i8 v[0:15], v0, a1, 2
// GFX90A: v_mfma_i32_16x16x4i8 v[0:15], v0, a1, 2 ; encoding: [0x00,0x00,0xd1,0xd3,0x00,0x03,0x0a,0x12]

v_mfma_i32_16x16x4i8 a[0:15], v0, a1, 2 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_i32_16x16x4i8 a[0:15], v0, a1, 2 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xd1,0xd3,0x00,0x03,0x0a,0xf2]

v_mfma_i32_16x16x4i8 v[0:15], v0, a1, 2 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_i32_16x16x4i8 v[0:15], v0, a1, 2 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xd1,0xd3,0x00,0x03,0x0a,0xf2]

v_mfma_i32_16x16x4i8 a[0:15], a0, v1, 2
// GFX90A: v_mfma_i32_16x16x4i8 a[0:15], a0, v1, 2 ; encoding: [0x00,0x80,0xd1,0xd3,0x00,0x03,0x0a,0x0a]

v_mfma_i32_16x16x4i8 v[0:15], a0, v1, 2
// GFX90A: v_mfma_i32_16x16x4i8 v[0:15], a0, v1, 2 ; encoding: [0x00,0x00,0xd1,0xd3,0x00,0x03,0x0a,0x0a]

v_mfma_i32_16x16x4i8 a[0:15], a0, v1, 2 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_i32_16x16x4i8 a[0:15], a0, v1, 2 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xd1,0xd3,0x00,0x03,0x0a,0xea]

v_mfma_i32_16x16x4i8 v[0:15], a0, v1, 2 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_i32_16x16x4i8 v[0:15], a0, v1, 2 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xd1,0xd3,0x00,0x03,0x0a,0xea]

v_mfma_i32_16x16x4i8 a[0:15], a0, a1, 2
// GFX90A: v_mfma_i32_16x16x4i8 a[0:15], a0, a1, 2 ; encoding: [0x00,0x80,0xd1,0xd3,0x00,0x03,0x0a,0x1a]

v_mfma_i32_16x16x4i8 v[0:15], a0, a1, 2
// GFX90A: v_mfma_i32_16x16x4i8 v[0:15], a0, a1, 2 ; encoding: [0x00,0x00,0xd1,0xd3,0x00,0x03,0x0a,0x1a]

v_mfma_i32_16x16x4i8 a[0:15], a0, a1, 2 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_i32_16x16x4i8 a[0:15], a0, a1, 2 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xd1,0xd3,0x00,0x03,0x0a,0xfa]

v_mfma_i32_16x16x4i8 v[0:15], a0, a1, 2 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_i32_16x16x4i8 v[0:15], a0, a1, 2 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xd1,0xd3,0x00,0x03,0x0a,0xfa]

v_mfma_i32_4x4x4i8 a[0:3], v0, v1, a[2:5]
// GFX90A: v_mfma_i32_4x4x4i8 a[0:3], v0, v1, a[2:5] ; encoding: [0x00,0x80,0xd2,0xd3,0x00,0x03,0x0a,0x04]

v_mfma_i32_4x4x4i8 a[0:3], v0, v1, a[2:5] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_i32_4x4x4i8 a[0:3], v0, v1, a[2:5] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xd2,0xd3,0x00,0x03,0x0a,0xe4]

v_mfma_i32_4x4x4i8 a[0:3], v0, a1, a[2:5]
// GFX90A: v_mfma_i32_4x4x4i8 a[0:3], v0, a1, a[2:5] ; encoding: [0x00,0x80,0xd2,0xd3,0x00,0x03,0x0a,0x14]

v_mfma_i32_4x4x4i8 a[0:3], v0, a1, a[2:5] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_i32_4x4x4i8 a[0:3], v0, a1, a[2:5] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xd2,0xd3,0x00,0x03,0x0a,0xf4]

v_mfma_i32_4x4x4i8 a[0:3], a0, v1, a[2:5]
// GFX90A: v_mfma_i32_4x4x4i8 a[0:3], a0, v1, a[2:5] ; encoding: [0x00,0x80,0xd2,0xd3,0x00,0x03,0x0a,0x0c]

v_mfma_i32_4x4x4i8 a[0:3], a0, v1, a[2:5] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_i32_4x4x4i8 a[0:3], a0, v1, a[2:5] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xd2,0xd3,0x00,0x03,0x0a,0xec]

v_mfma_i32_4x4x4i8 a[0:3], a0, a1, a[2:5]
// GFX90A: v_mfma_i32_4x4x4i8 a[0:3], a0, a1, a[2:5] ; encoding: [0x00,0x80,0xd2,0xd3,0x00,0x03,0x0a,0x1c]

v_mfma_i32_4x4x4i8 a[0:3], a0, a1, a[2:5] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_i32_4x4x4i8 a[0:3], a0, a1, a[2:5] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xd2,0xd3,0x00,0x03,0x0a,0xfc]

v_mfma_i32_4x4x4i8 v[0:3], v0, v1, v[2:5]
// GFX90A: v_mfma_i32_4x4x4i8 v[0:3], v0, v1, v[2:5] ; encoding: [0x00,0x00,0xd2,0xd3,0x00,0x03,0x0a,0x04]

v_mfma_i32_4x4x4i8 v[0:3], v0, v1, v[2:5] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_i32_4x4x4i8 v[0:3], v0, v1, v[2:5] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xd2,0xd3,0x00,0x03,0x0a,0xe4]

v_mfma_i32_4x4x4i8 v[0:3], v0, a1, v[2:5]
// GFX90A: v_mfma_i32_4x4x4i8 v[0:3], v0, a1, v[2:5] ; encoding: [0x00,0x00,0xd2,0xd3,0x00,0x03,0x0a,0x14]

v_mfma_i32_4x4x4i8 v[0:3], v0, a1, v[2:5] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_i32_4x4x4i8 v[0:3], v0, a1, v[2:5] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xd2,0xd3,0x00,0x03,0x0a,0xf4]

v_mfma_i32_4x4x4i8 v[0:3], a0, v1, v[2:5]
// GFX90A: v_mfma_i32_4x4x4i8 v[0:3], a0, v1, v[2:5] ; encoding: [0x00,0x00,0xd2,0xd3,0x00,0x03,0x0a,0x0c]

v_mfma_i32_4x4x4i8 v[0:3], a0, v1, v[2:5] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_i32_4x4x4i8 v[0:3], a0, v1, v[2:5] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xd2,0xd3,0x00,0x03,0x0a,0xec]

v_mfma_i32_4x4x4i8 v[0:3], a0, a1, v[2:5]
// GFX90A: v_mfma_i32_4x4x4i8 v[0:3], a0, a1, v[2:5] ; encoding: [0x00,0x00,0xd2,0xd3,0x00,0x03,0x0a,0x1c]

v_mfma_i32_4x4x4i8 v[0:3], a0, a1, v[2:5] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_i32_4x4x4i8 v[0:3], a0, a1, v[2:5] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xd2,0xd3,0x00,0x03,0x0a,0xfc]

v_mfma_i32_4x4x4i8 a[0:3], v0, v1, 2
// GFX90A: v_mfma_i32_4x4x4i8 a[0:3], v0, v1, 2 ; encoding: [0x00,0x80,0xd2,0xd3,0x00,0x03,0x0a,0x02]

v_mfma_i32_4x4x4i8 v[0:3], v0, v1, 2
// GFX90A: v_mfma_i32_4x4x4i8 v[0:3], v0, v1, 2 ; encoding: [0x00,0x00,0xd2,0xd3,0x00,0x03,0x0a,0x02]

v_mfma_i32_4x4x4i8 a[0:3], v0, v1, 2 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_i32_4x4x4i8 a[0:3], v0, v1, 2 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xd2,0xd3,0x00,0x03,0x0a,0xe2]

v_mfma_i32_4x4x4i8 v[0:3], v0, v1, 2 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_i32_4x4x4i8 v[0:3], v0, v1, 2 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xd2,0xd3,0x00,0x03,0x0a,0xe2]

v_mfma_i32_4x4x4i8 a[0:3], v0, a1, 2
// GFX90A: v_mfma_i32_4x4x4i8 a[0:3], v0, a1, 2 ; encoding: [0x00,0x80,0xd2,0xd3,0x00,0x03,0x0a,0x12]

v_mfma_i32_4x4x4i8 v[0:3], v0, a1, 2
// GFX90A: v_mfma_i32_4x4x4i8 v[0:3], v0, a1, 2 ; encoding: [0x00,0x00,0xd2,0xd3,0x00,0x03,0x0a,0x12]

v_mfma_i32_4x4x4i8 a[0:3], v0, a1, 2 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_i32_4x4x4i8 a[0:3], v0, a1, 2 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xd2,0xd3,0x00,0x03,0x0a,0xf2]

v_mfma_i32_4x4x4i8 v[0:3], v0, a1, 2 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_i32_4x4x4i8 v[0:3], v0, a1, 2 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xd2,0xd3,0x00,0x03,0x0a,0xf2]

v_mfma_i32_4x4x4i8 a[0:3], a0, v1, 2
// GFX90A: v_mfma_i32_4x4x4i8 a[0:3], a0, v1, 2 ; encoding: [0x00,0x80,0xd2,0xd3,0x00,0x03,0x0a,0x0a]

v_mfma_i32_4x4x4i8 v[0:3], a0, v1, 2
// GFX90A: v_mfma_i32_4x4x4i8 v[0:3], a0, v1, 2 ; encoding: [0x00,0x00,0xd2,0xd3,0x00,0x03,0x0a,0x0a]

v_mfma_i32_4x4x4i8 a[0:3], a0, v1, 2 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_i32_4x4x4i8 a[0:3], a0, v1, 2 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xd2,0xd3,0x00,0x03,0x0a,0xea]

v_mfma_i32_4x4x4i8 v[0:3], a0, v1, 2 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_i32_4x4x4i8 v[0:3], a0, v1, 2 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xd2,0xd3,0x00,0x03,0x0a,0xea]

v_mfma_i32_4x4x4i8 a[0:3], a0, a1, 2
// GFX90A: v_mfma_i32_4x4x4i8 a[0:3], a0, a1, 2 ; encoding: [0x00,0x80,0xd2,0xd3,0x00,0x03,0x0a,0x1a]

v_mfma_i32_4x4x4i8 v[0:3], a0, a1, 2
// GFX90A: v_mfma_i32_4x4x4i8 v[0:3], a0, a1, 2 ; encoding: [0x00,0x00,0xd2,0xd3,0x00,0x03,0x0a,0x1a]

v_mfma_i32_4x4x4i8 a[0:3], a0, a1, 2 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_i32_4x4x4i8 a[0:3], a0, a1, 2 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xd2,0xd3,0x00,0x03,0x0a,0xfa]

v_mfma_i32_4x4x4i8 v[0:3], a0, a1, 2 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_i32_4x4x4i8 v[0:3], a0, a1, 2 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xd2,0xd3,0x00,0x03,0x0a,0xfa]

v_mfma_i32_32x32x8i8 a[0:15], v0, v1, a[18:33]
// GFX90A: v_mfma_i32_32x32x8i8 a[0:15], v0, v1, a[18:33] ; encoding: [0x00,0x80,0xd4,0xd3,0x00,0x03,0x4a,0x04]

v_mfma_i32_32x32x8i8 a[0:15], v0, v1, a[18:33] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_i32_32x32x8i8 a[0:15], v0, v1, a[18:33] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xd4,0xd3,0x00,0x03,0x4a,0xe4]

v_mfma_i32_32x32x8i8 a[0:15], v0, a1, a[18:33]
// GFX90A: v_mfma_i32_32x32x8i8 a[0:15], v0, a1, a[18:33] ; encoding: [0x00,0x80,0xd4,0xd3,0x00,0x03,0x4a,0x14]

v_mfma_i32_32x32x8i8 a[0:15], v0, a1, a[18:33] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_i32_32x32x8i8 a[0:15], v0, a1, a[18:33] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xd4,0xd3,0x00,0x03,0x4a,0xf4]

v_mfma_i32_32x32x8i8 a[0:15], a0, v1, a[18:33]
// GFX90A: v_mfma_i32_32x32x8i8 a[0:15], a0, v1, a[18:33] ; encoding: [0x00,0x80,0xd4,0xd3,0x00,0x03,0x4a,0x0c]

v_mfma_i32_32x32x8i8 a[0:15], a0, v1, a[18:33] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_i32_32x32x8i8 a[0:15], a0, v1, a[18:33] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xd4,0xd3,0x00,0x03,0x4a,0xec]

v_mfma_i32_32x32x8i8 a[0:15], a0, a1, a[18:33]
// GFX90A: v_mfma_i32_32x32x8i8 a[0:15], a0, a1, a[18:33] ; encoding: [0x00,0x80,0xd4,0xd3,0x00,0x03,0x4a,0x1c]

v_mfma_i32_32x32x8i8 a[0:15], a0, a1, a[18:33] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_i32_32x32x8i8 a[0:15], a0, a1, a[18:33] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xd4,0xd3,0x00,0x03,0x4a,0xfc]

v_mfma_i32_32x32x8i8 v[0:15], v0, v1, v[18:33]
// GFX90A: v_mfma_i32_32x32x8i8 v[0:15], v0, v1, v[18:33] ; encoding: [0x00,0x00,0xd4,0xd3,0x00,0x03,0x4a,0x04]

v_mfma_i32_32x32x8i8 v[0:15], v0, v1, v[18:33] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_i32_32x32x8i8 v[0:15], v0, v1, v[18:33] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xd4,0xd3,0x00,0x03,0x4a,0xe4]

v_mfma_i32_32x32x8i8 v[0:15], v0, a1, v[18:33]
// GFX90A: v_mfma_i32_32x32x8i8 v[0:15], v0, a1, v[18:33] ; encoding: [0x00,0x00,0xd4,0xd3,0x00,0x03,0x4a,0x14]

v_mfma_i32_32x32x8i8 v[0:15], v0, a1, v[18:33] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_i32_32x32x8i8 v[0:15], v0, a1, v[18:33] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xd4,0xd3,0x00,0x03,0x4a,0xf4]

v_mfma_i32_32x32x8i8 v[0:15], a0, v1, v[18:33]
// GFX90A: v_mfma_i32_32x32x8i8 v[0:15], a0, v1, v[18:33] ; encoding: [0x00,0x00,0xd4,0xd3,0x00,0x03,0x4a,0x0c]

v_mfma_i32_32x32x8i8 v[0:15], a0, v1, v[18:33] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_i32_32x32x8i8 v[0:15], a0, v1, v[18:33] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xd4,0xd3,0x00,0x03,0x4a,0xec]

v_mfma_i32_32x32x8i8 v[0:15], a0, a1, v[18:33]
// GFX90A: v_mfma_i32_32x32x8i8 v[0:15], a0, a1, v[18:33] ; encoding: [0x00,0x00,0xd4,0xd3,0x00,0x03,0x4a,0x1c]

v_mfma_i32_32x32x8i8 v[0:15], a0, a1, v[18:33] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_i32_32x32x8i8 v[0:15], a0, a1, v[18:33] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xd4,0xd3,0x00,0x03,0x4a,0xfc]

v_mfma_i32_32x32x8i8 a[0:15], v0, v1, 2
// GFX90A: v_mfma_i32_32x32x8i8 a[0:15], v0, v1, 2 ; encoding: [0x00,0x80,0xd4,0xd3,0x00,0x03,0x0a,0x02]

v_mfma_i32_32x32x8i8 v[0:15], v0, v1, 2
// GFX90A: v_mfma_i32_32x32x8i8 v[0:15], v0, v1, 2 ; encoding: [0x00,0x00,0xd4,0xd3,0x00,0x03,0x0a,0x02]

v_mfma_i32_32x32x8i8 a[0:15], v0, v1, 2 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_i32_32x32x8i8 a[0:15], v0, v1, 2 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xd4,0xd3,0x00,0x03,0x0a,0xe2]

v_mfma_i32_32x32x8i8 v[0:15], v0, v1, 2 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_i32_32x32x8i8 v[0:15], v0, v1, 2 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xd4,0xd3,0x00,0x03,0x0a,0xe2]

v_mfma_i32_32x32x8i8 a[0:15], v0, a1, 2
// GFX90A: v_mfma_i32_32x32x8i8 a[0:15], v0, a1, 2 ; encoding: [0x00,0x80,0xd4,0xd3,0x00,0x03,0x0a,0x12]

v_mfma_i32_32x32x8i8 v[0:15], v0, a1, 2
// GFX90A: v_mfma_i32_32x32x8i8 v[0:15], v0, a1, 2 ; encoding: [0x00,0x00,0xd4,0xd3,0x00,0x03,0x0a,0x12]

v_mfma_i32_32x32x8i8 a[0:15], v0, a1, 2 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_i32_32x32x8i8 a[0:15], v0, a1, 2 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xd4,0xd3,0x00,0x03,0x0a,0xf2]

v_mfma_i32_32x32x8i8 v[0:15], v0, a1, 2 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_i32_32x32x8i8 v[0:15], v0, a1, 2 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xd4,0xd3,0x00,0x03,0x0a,0xf2]

v_mfma_i32_32x32x8i8 a[0:15], a0, v1, 2
// GFX90A: v_mfma_i32_32x32x8i8 a[0:15], a0, v1, 2 ; encoding: [0x00,0x80,0xd4,0xd3,0x00,0x03,0x0a,0x0a]

v_mfma_i32_32x32x8i8 v[0:15], a0, v1, 2
// GFX90A: v_mfma_i32_32x32x8i8 v[0:15], a0, v1, 2 ; encoding: [0x00,0x00,0xd4,0xd3,0x00,0x03,0x0a,0x0a]

v_mfma_i32_32x32x8i8 a[0:15], a0, v1, 2 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_i32_32x32x8i8 a[0:15], a0, v1, 2 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xd4,0xd3,0x00,0x03,0x0a,0xea]

v_mfma_i32_32x32x8i8 v[0:15], a0, v1, 2 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_i32_32x32x8i8 v[0:15], a0, v1, 2 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xd4,0xd3,0x00,0x03,0x0a,0xea]

v_mfma_i32_32x32x8i8 a[0:15], a0, a1, 2
// GFX90A: v_mfma_i32_32x32x8i8 a[0:15], a0, a1, 2 ; encoding: [0x00,0x80,0xd4,0xd3,0x00,0x03,0x0a,0x1a]

v_mfma_i32_32x32x8i8 v[0:15], a0, a1, 2
// GFX90A: v_mfma_i32_32x32x8i8 v[0:15], a0, a1, 2 ; encoding: [0x00,0x00,0xd4,0xd3,0x00,0x03,0x0a,0x1a]

v_mfma_i32_32x32x8i8 a[0:15], a0, a1, 2 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_i32_32x32x8i8 a[0:15], a0, a1, 2 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xd4,0xd3,0x00,0x03,0x0a,0xfa]

v_mfma_i32_32x32x8i8 v[0:15], a0, a1, 2 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_i32_32x32x8i8 v[0:15], a0, a1, 2 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xd4,0xd3,0x00,0x03,0x0a,0xfa]

v_mfma_i32_16x16x16i8 a[0:3], v0, v1, a[2:5]
// GFX90A: v_mfma_i32_16x16x16i8 a[0:3], v0, v1, a[2:5] ; encoding: [0x00,0x80,0xd5,0xd3,0x00,0x03,0x0a,0x04]

v_mfma_i32_16x16x16i8 a[0:3], v0, v1, a[2:5] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_i32_16x16x16i8 a[0:3], v0, v1, a[2:5] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xd5,0xd3,0x00,0x03,0x0a,0xe4]

v_mfma_i32_16x16x16i8 a[0:3], v0, a1, a[2:5]
// GFX90A: v_mfma_i32_16x16x16i8 a[0:3], v0, a1, a[2:5] ; encoding: [0x00,0x80,0xd5,0xd3,0x00,0x03,0x0a,0x14]

v_mfma_i32_16x16x16i8 a[0:3], v0, a1, a[2:5] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_i32_16x16x16i8 a[0:3], v0, a1, a[2:5] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xd5,0xd3,0x00,0x03,0x0a,0xf4]

v_mfma_i32_16x16x16i8 a[0:3], a0, v1, a[2:5]
// GFX90A: v_mfma_i32_16x16x16i8 a[0:3], a0, v1, a[2:5] ; encoding: [0x00,0x80,0xd5,0xd3,0x00,0x03,0x0a,0x0c]

v_mfma_i32_16x16x16i8 a[0:3], a0, v1, a[2:5] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_i32_16x16x16i8 a[0:3], a0, v1, a[2:5] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xd5,0xd3,0x00,0x03,0x0a,0xec]

v_mfma_i32_16x16x16i8 a[0:3], a0, a1, a[2:5]
// GFX90A: v_mfma_i32_16x16x16i8 a[0:3], a0, a1, a[2:5] ; encoding: [0x00,0x80,0xd5,0xd3,0x00,0x03,0x0a,0x1c]

v_mfma_i32_16x16x16i8 a[0:3], a0, a1, a[2:5] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_i32_16x16x16i8 a[0:3], a0, a1, a[2:5] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xd5,0xd3,0x00,0x03,0x0a,0xfc]

v_mfma_i32_16x16x16i8 v[0:3], v0, v1, v[2:5]
// GFX90A: v_mfma_i32_16x16x16i8 v[0:3], v0, v1, v[2:5] ; encoding: [0x00,0x00,0xd5,0xd3,0x00,0x03,0x0a,0x04]

v_mfma_i32_16x16x16i8 v[0:3], v0, v1, v[2:5] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_i32_16x16x16i8 v[0:3], v0, v1, v[2:5] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xd5,0xd3,0x00,0x03,0x0a,0xe4]

v_mfma_i32_16x16x16i8 v[0:3], v0, a1, v[2:5]
// GFX90A: v_mfma_i32_16x16x16i8 v[0:3], v0, a1, v[2:5] ; encoding: [0x00,0x00,0xd5,0xd3,0x00,0x03,0x0a,0x14]

v_mfma_i32_16x16x16i8 v[0:3], v0, a1, v[2:5] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_i32_16x16x16i8 v[0:3], v0, a1, v[2:5] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xd5,0xd3,0x00,0x03,0x0a,0xf4]

v_mfma_i32_16x16x16i8 v[0:3], a0, v1, v[2:5]
// GFX90A: v_mfma_i32_16x16x16i8 v[0:3], a0, v1, v[2:5] ; encoding: [0x00,0x00,0xd5,0xd3,0x00,0x03,0x0a,0x0c]

v_mfma_i32_16x16x16i8 v[0:3], a0, v1, v[2:5] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_i32_16x16x16i8 v[0:3], a0, v1, v[2:5] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xd5,0xd3,0x00,0x03,0x0a,0xec]

v_mfma_i32_16x16x16i8 v[0:3], a0, a1, v[2:5]
// GFX90A: v_mfma_i32_16x16x16i8 v[0:3], a0, a1, v[2:5] ; encoding: [0x00,0x00,0xd5,0xd3,0x00,0x03,0x0a,0x1c]

v_mfma_i32_16x16x16i8 v[0:3], a0, a1, v[2:5] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_i32_16x16x16i8 v[0:3], a0, a1, v[2:5] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xd5,0xd3,0x00,0x03,0x0a,0xfc]

v_mfma_i32_16x16x16i8 a[0:3], v0, v1, 2
// GFX90A: v_mfma_i32_16x16x16i8 a[0:3], v0, v1, 2 ; encoding: [0x00,0x80,0xd5,0xd3,0x00,0x03,0x0a,0x02]

v_mfma_i32_16x16x16i8 v[0:3], v0, v1, 2
// GFX90A: v_mfma_i32_16x16x16i8 v[0:3], v0, v1, 2 ; encoding: [0x00,0x00,0xd5,0xd3,0x00,0x03,0x0a,0x02]

v_mfma_i32_16x16x16i8 a[0:3], v0, v1, 2 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_i32_16x16x16i8 a[0:3], v0, v1, 2 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xd5,0xd3,0x00,0x03,0x0a,0xe2]

v_mfma_i32_16x16x16i8 v[0:3], v0, v1, 2 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_i32_16x16x16i8 v[0:3], v0, v1, 2 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xd5,0xd3,0x00,0x03,0x0a,0xe2]

v_mfma_i32_16x16x16i8 a[0:3], v0, a1, 2
// GFX90A: v_mfma_i32_16x16x16i8 a[0:3], v0, a1, 2 ; encoding: [0x00,0x80,0xd5,0xd3,0x00,0x03,0x0a,0x12]

v_mfma_i32_16x16x16i8 v[0:3], v0, a1, 2
// GFX90A: v_mfma_i32_16x16x16i8 v[0:3], v0, a1, 2 ; encoding: [0x00,0x00,0xd5,0xd3,0x00,0x03,0x0a,0x12]

v_mfma_i32_16x16x16i8 a[0:3], v0, a1, 2 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_i32_16x16x16i8 a[0:3], v0, a1, 2 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xd5,0xd3,0x00,0x03,0x0a,0xf2]

v_mfma_i32_16x16x16i8 v[0:3], v0, a1, 2 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_i32_16x16x16i8 v[0:3], v0, a1, 2 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xd5,0xd3,0x00,0x03,0x0a,0xf2]

v_mfma_i32_16x16x16i8 a[0:3], a0, v1, 2
// GFX90A: v_mfma_i32_16x16x16i8 a[0:3], a0, v1, 2 ; encoding: [0x00,0x80,0xd5,0xd3,0x00,0x03,0x0a,0x0a]

v_mfma_i32_16x16x16i8 v[0:3], a0, v1, 2
// GFX90A: v_mfma_i32_16x16x16i8 v[0:3], a0, v1, 2 ; encoding: [0x00,0x00,0xd5,0xd3,0x00,0x03,0x0a,0x0a]

v_mfma_i32_16x16x16i8 a[0:3], a0, v1, 2 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_i32_16x16x16i8 a[0:3], a0, v1, 2 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xd5,0xd3,0x00,0x03,0x0a,0xea]

v_mfma_i32_16x16x16i8 v[0:3], a0, v1, 2 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_i32_16x16x16i8 v[0:3], a0, v1, 2 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xd5,0xd3,0x00,0x03,0x0a,0xea]

v_mfma_i32_16x16x16i8 a[0:3], a0, a1, 2
// GFX90A: v_mfma_i32_16x16x16i8 a[0:3], a0, a1, 2 ; encoding: [0x00,0x80,0xd5,0xd3,0x00,0x03,0x0a,0x1a]

v_mfma_i32_16x16x16i8 v[0:3], a0, a1, 2
// GFX90A: v_mfma_i32_16x16x16i8 v[0:3], a0, a1, 2 ; encoding: [0x00,0x00,0xd5,0xd3,0x00,0x03,0x0a,0x1a]

v_mfma_i32_16x16x16i8 a[0:3], a0, a1, 2 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_i32_16x16x16i8 a[0:3], a0, a1, 2 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xd5,0xd3,0x00,0x03,0x0a,0xfa]

v_mfma_i32_16x16x16i8 v[0:3], a0, a1, 2 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_i32_16x16x16i8 v[0:3], a0, a1, 2 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xd5,0xd3,0x00,0x03,0x0a,0xfa]

v_mfma_f32_32x32x2bf16 a[0:31], v0, v1, a[34:65]
// GFX90A: v_mfma_f32_32x32x2bf16 a[0:31], v0, v1, a[34:65] ; encoding: [0x00,0x80,0xe8,0xd3,0x00,0x03,0x8a,0x04]

v_mfma_f32_32x32x2bf16 a[0:31], v0, v1, a[34:65] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_32x32x2bf16 a[0:31], v0, v1, a[34:65] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xe8,0xd3,0x00,0x03,0x8a,0xe4]

v_mfma_f32_32x32x2bf16 a[0:31], v0, a1, a[34:65]
// GFX90A: v_mfma_f32_32x32x2bf16 a[0:31], v0, a1, a[34:65] ; encoding: [0x00,0x80,0xe8,0xd3,0x00,0x03,0x8a,0x14]

v_mfma_f32_32x32x2bf16 a[0:31], v0, a1, a[34:65] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_32x32x2bf16 a[0:31], v0, a1, a[34:65] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xe8,0xd3,0x00,0x03,0x8a,0xf4]

v_mfma_f32_32x32x2bf16 a[0:31], a0, v1, a[34:65]
// GFX90A: v_mfma_f32_32x32x2bf16 a[0:31], a0, v1, a[34:65] ; encoding: [0x00,0x80,0xe8,0xd3,0x00,0x03,0x8a,0x0c]

v_mfma_f32_32x32x2bf16 a[0:31], a0, v1, a[34:65] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_32x32x2bf16 a[0:31], a0, v1, a[34:65] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xe8,0xd3,0x00,0x03,0x8a,0xec]

v_mfma_f32_32x32x2bf16 a[0:31], a0, a1, a[34:65]
// GFX90A: v_mfma_f32_32x32x2bf16 a[0:31], a0, a1, a[34:65] ; encoding: [0x00,0x80,0xe8,0xd3,0x00,0x03,0x8a,0x1c]

v_mfma_f32_32x32x2bf16 a[0:31], a0, a1, a[34:65] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_32x32x2bf16 a[0:31], a0, a1, a[34:65] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xe8,0xd3,0x00,0x03,0x8a,0xfc]

v_mfma_f32_32x32x2bf16 v[0:31], v0, v1, v[34:65]
// GFX90A: v_mfma_f32_32x32x2bf16 v[0:31], v0, v1, v[34:65] ; encoding: [0x00,0x00,0xe8,0xd3,0x00,0x03,0x8a,0x04]

v_mfma_f32_32x32x2bf16 v[0:31], v0, v1, v[34:65] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_32x32x2bf16 v[0:31], v0, v1, v[34:65] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xe8,0xd3,0x00,0x03,0x8a,0xe4]

v_mfma_f32_32x32x2bf16 v[0:31], v0, a1, v[34:65]
// GFX90A: v_mfma_f32_32x32x2bf16 v[0:31], v0, a1, v[34:65] ; encoding: [0x00,0x00,0xe8,0xd3,0x00,0x03,0x8a,0x14]

v_mfma_f32_32x32x2bf16 v[0:31], v0, a1, v[34:65] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_32x32x2bf16 v[0:31], v0, a1, v[34:65] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xe8,0xd3,0x00,0x03,0x8a,0xf4]

v_mfma_f32_32x32x2bf16 v[0:31], a0, v1, v[34:65]
// GFX90A: v_mfma_f32_32x32x2bf16 v[0:31], a0, v1, v[34:65] ; encoding: [0x00,0x00,0xe8,0xd3,0x00,0x03,0x8a,0x0c]

v_mfma_f32_32x32x2bf16 v[0:31], a0, v1, v[34:65] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_32x32x2bf16 v[0:31], a0, v1, v[34:65] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xe8,0xd3,0x00,0x03,0x8a,0xec]

v_mfma_f32_32x32x2bf16 v[0:31], a0, a1, v[34:65]
// GFX90A: v_mfma_f32_32x32x2bf16 v[0:31], a0, a1, v[34:65] ; encoding: [0x00,0x00,0xe8,0xd3,0x00,0x03,0x8a,0x1c]

v_mfma_f32_32x32x2bf16 v[0:31], a0, a1, v[34:65] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_32x32x2bf16 v[0:31], a0, a1, v[34:65] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xe8,0xd3,0x00,0x03,0x8a,0xfc]

v_mfma_f32_32x32x2bf16 a[0:31], v0, v1, -2.0
// GFX90A: v_mfma_f32_32x32x2bf16 a[0:31], v0, v1, -2.0 ; encoding: [0x00,0x80,0xe8,0xd3,0x00,0x03,0xd6,0x03]

v_mfma_f32_32x32x2bf16 v[0:31], v0, v1, -2.0
// GFX90A: v_mfma_f32_32x32x2bf16 v[0:31], v0, v1, -2.0 ; encoding: [0x00,0x00,0xe8,0xd3,0x00,0x03,0xd6,0x03]

v_mfma_f32_32x32x2bf16 a[0:31], v0, v1, -2.0 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_32x32x2bf16 a[0:31], v0, v1, -2.0 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xe8,0xd3,0x00,0x03,0xd6,0xe3]

v_mfma_f32_32x32x2bf16 v[0:31], v0, v1, -2.0 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_32x32x2bf16 v[0:31], v0, v1, -2.0 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xe8,0xd3,0x00,0x03,0xd6,0xe3]

v_mfma_f32_32x32x2bf16 a[0:31], v0, a1, -2.0
// GFX90A: v_mfma_f32_32x32x2bf16 a[0:31], v0, a1, -2.0 ; encoding: [0x00,0x80,0xe8,0xd3,0x00,0x03,0xd6,0x13]

v_mfma_f32_32x32x2bf16 v[0:31], v0, a1, -2.0
// GFX90A: v_mfma_f32_32x32x2bf16 v[0:31], v0, a1, -2.0 ; encoding: [0x00,0x00,0xe8,0xd3,0x00,0x03,0xd6,0x13]

v_mfma_f32_32x32x2bf16 a[0:31], v0, a1, -2.0 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_32x32x2bf16 a[0:31], v0, a1, -2.0 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xe8,0xd3,0x00,0x03,0xd6,0xf3]

v_mfma_f32_32x32x2bf16 v[0:31], v0, a1, -2.0 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_32x32x2bf16 v[0:31], v0, a1, -2.0 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xe8,0xd3,0x00,0x03,0xd6,0xf3]

v_mfma_f32_32x32x2bf16 a[0:31], a0, v1, -2.0
// GFX90A: v_mfma_f32_32x32x2bf16 a[0:31], a0, v1, -2.0 ; encoding: [0x00,0x80,0xe8,0xd3,0x00,0x03,0xd6,0x0b]

v_mfma_f32_32x32x2bf16 v[0:31], a0, v1, -2.0
// GFX90A: v_mfma_f32_32x32x2bf16 v[0:31], a0, v1, -2.0 ; encoding: [0x00,0x00,0xe8,0xd3,0x00,0x03,0xd6,0x0b]

v_mfma_f32_32x32x2bf16 a[0:31], a0, v1, -2.0 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_32x32x2bf16 a[0:31], a0, v1, -2.0 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xe8,0xd3,0x00,0x03,0xd6,0xeb]

v_mfma_f32_32x32x2bf16 v[0:31], a0, v1, -2.0 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_32x32x2bf16 v[0:31], a0, v1, -2.0 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xe8,0xd3,0x00,0x03,0xd6,0xeb]

v_mfma_f32_32x32x2bf16 a[0:31], a0, a1, -2.0
// GFX90A: v_mfma_f32_32x32x2bf16 a[0:31], a0, a1, -2.0 ; encoding: [0x00,0x80,0xe8,0xd3,0x00,0x03,0xd6,0x1b]

v_mfma_f32_32x32x2bf16 v[0:31], a0, a1, -2.0
// GFX90A: v_mfma_f32_32x32x2bf16 v[0:31], a0, a1, -2.0 ; encoding: [0x00,0x00,0xe8,0xd3,0x00,0x03,0xd6,0x1b]

v_mfma_f32_32x32x2bf16 a[0:31], a0, a1, -2.0 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_32x32x2bf16 a[0:31], a0, a1, -2.0 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xe8,0xd3,0x00,0x03,0xd6,0xfb]

v_mfma_f32_32x32x2bf16 v[0:31], a0, a1, -2.0 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_32x32x2bf16 v[0:31], a0, a1, -2.0 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xe8,0xd3,0x00,0x03,0xd6,0xfb]

v_mfma_f32_16x16x2bf16 a[0:15], v0, v1, a[18:33]
// GFX90A: v_mfma_f32_16x16x2bf16 a[0:15], v0, v1, a[18:33] ; encoding: [0x00,0x80,0xe9,0xd3,0x00,0x03,0x4a,0x04]

v_mfma_f32_16x16x2bf16 a[0:15], v0, v1, a[18:33] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_16x16x2bf16 a[0:15], v0, v1, a[18:33] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xe9,0xd3,0x00,0x03,0x4a,0xe4]

v_mfma_f32_16x16x2bf16 a[0:15], v0, a1, a[18:33]
// GFX90A: v_mfma_f32_16x16x2bf16 a[0:15], v0, a1, a[18:33] ; encoding: [0x00,0x80,0xe9,0xd3,0x00,0x03,0x4a,0x14]

v_mfma_f32_16x16x2bf16 a[0:15], v0, a1, a[18:33] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_16x16x2bf16 a[0:15], v0, a1, a[18:33] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xe9,0xd3,0x00,0x03,0x4a,0xf4]

v_mfma_f32_16x16x2bf16 a[0:15], a0, v1, a[18:33]
// GFX90A: v_mfma_f32_16x16x2bf16 a[0:15], a0, v1, a[18:33] ; encoding: [0x00,0x80,0xe9,0xd3,0x00,0x03,0x4a,0x0c]

v_mfma_f32_16x16x2bf16 a[0:15], a0, v1, a[18:33] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_16x16x2bf16 a[0:15], a0, v1, a[18:33] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xe9,0xd3,0x00,0x03,0x4a,0xec]

v_mfma_f32_16x16x2bf16 a[0:15], a0, a1, a[18:33]
// GFX90A: v_mfma_f32_16x16x2bf16 a[0:15], a0, a1, a[18:33] ; encoding: [0x00,0x80,0xe9,0xd3,0x00,0x03,0x4a,0x1c]

v_mfma_f32_16x16x2bf16 a[0:15], a0, a1, a[18:33] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_16x16x2bf16 a[0:15], a0, a1, a[18:33] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xe9,0xd3,0x00,0x03,0x4a,0xfc]

v_mfma_f32_16x16x2bf16 v[0:15], v0, v1, v[18:33]
// GFX90A: v_mfma_f32_16x16x2bf16 v[0:15], v0, v1, v[18:33] ; encoding: [0x00,0x00,0xe9,0xd3,0x00,0x03,0x4a,0x04]

v_mfma_f32_16x16x2bf16 v[0:15], v0, v1, v[18:33] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_16x16x2bf16 v[0:15], v0, v1, v[18:33] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xe9,0xd3,0x00,0x03,0x4a,0xe4]

v_mfma_f32_16x16x2bf16 v[0:15], v0, a1, v[18:33]
// GFX90A: v_mfma_f32_16x16x2bf16 v[0:15], v0, a1, v[18:33] ; encoding: [0x00,0x00,0xe9,0xd3,0x00,0x03,0x4a,0x14]

v_mfma_f32_16x16x2bf16 v[0:15], v0, a1, v[18:33] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_16x16x2bf16 v[0:15], v0, a1, v[18:33] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xe9,0xd3,0x00,0x03,0x4a,0xf4]

v_mfma_f32_16x16x2bf16 v[0:15], a0, v1, v[18:33]
// GFX90A: v_mfma_f32_16x16x2bf16 v[0:15], a0, v1, v[18:33] ; encoding: [0x00,0x00,0xe9,0xd3,0x00,0x03,0x4a,0x0c]

v_mfma_f32_16x16x2bf16 v[0:15], a0, v1, v[18:33] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_16x16x2bf16 v[0:15], a0, v1, v[18:33] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xe9,0xd3,0x00,0x03,0x4a,0xec]

v_mfma_f32_16x16x2bf16 v[0:15], a0, a1, v[18:33]
// GFX90A: v_mfma_f32_16x16x2bf16 v[0:15], a0, a1, v[18:33] ; encoding: [0x00,0x00,0xe9,0xd3,0x00,0x03,0x4a,0x1c]

v_mfma_f32_16x16x2bf16 v[0:15], a0, a1, v[18:33] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_16x16x2bf16 v[0:15], a0, a1, v[18:33] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xe9,0xd3,0x00,0x03,0x4a,0xfc]

v_mfma_f32_16x16x2bf16 a[0:15], v0, v1, -2.0
// GFX90A: v_mfma_f32_16x16x2bf16 a[0:15], v0, v1, -2.0 ; encoding: [0x00,0x80,0xe9,0xd3,0x00,0x03,0xd6,0x03]

v_mfma_f32_16x16x2bf16 v[0:15], v0, v1, -2.0
// GFX90A: v_mfma_f32_16x16x2bf16 v[0:15], v0, v1, -2.0 ; encoding: [0x00,0x00,0xe9,0xd3,0x00,0x03,0xd6,0x03]

v_mfma_f32_16x16x2bf16 a[0:15], v0, v1, -2.0 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_16x16x2bf16 a[0:15], v0, v1, -2.0 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xe9,0xd3,0x00,0x03,0xd6,0xe3]

v_mfma_f32_16x16x2bf16 v[0:15], v0, v1, -2.0 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_16x16x2bf16 v[0:15], v0, v1, -2.0 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xe9,0xd3,0x00,0x03,0xd6,0xe3]

v_mfma_f32_16x16x2bf16 a[0:15], v0, a1, -2.0
// GFX90A: v_mfma_f32_16x16x2bf16 a[0:15], v0, a1, -2.0 ; encoding: [0x00,0x80,0xe9,0xd3,0x00,0x03,0xd6,0x13]

v_mfma_f32_16x16x2bf16 v[0:15], v0, a1, -2.0
// GFX90A: v_mfma_f32_16x16x2bf16 v[0:15], v0, a1, -2.0 ; encoding: [0x00,0x00,0xe9,0xd3,0x00,0x03,0xd6,0x13]

v_mfma_f32_16x16x2bf16 a[0:15], v0, a1, -2.0 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_16x16x2bf16 a[0:15], v0, a1, -2.0 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xe9,0xd3,0x00,0x03,0xd6,0xf3]

v_mfma_f32_16x16x2bf16 v[0:15], v0, a1, -2.0 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_16x16x2bf16 v[0:15], v0, a1, -2.0 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xe9,0xd3,0x00,0x03,0xd6,0xf3]

v_mfma_f32_16x16x2bf16 a[0:15], a0, v1, -2.0
// GFX90A: v_mfma_f32_16x16x2bf16 a[0:15], a0, v1, -2.0 ; encoding: [0x00,0x80,0xe9,0xd3,0x00,0x03,0xd6,0x0b]

v_mfma_f32_16x16x2bf16 v[0:15], a0, v1, -2.0
// GFX90A: v_mfma_f32_16x16x2bf16 v[0:15], a0, v1, -2.0 ; encoding: [0x00,0x00,0xe9,0xd3,0x00,0x03,0xd6,0x0b]

v_mfma_f32_16x16x2bf16 a[0:15], a0, v1, -2.0 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_16x16x2bf16 a[0:15], a0, v1, -2.0 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xe9,0xd3,0x00,0x03,0xd6,0xeb]

v_mfma_f32_16x16x2bf16 v[0:15], a0, v1, -2.0 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_16x16x2bf16 v[0:15], a0, v1, -2.0 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xe9,0xd3,0x00,0x03,0xd6,0xeb]

v_mfma_f32_16x16x2bf16 a[0:15], a0, a1, -2.0
// GFX90A: v_mfma_f32_16x16x2bf16 a[0:15], a0, a1, -2.0 ; encoding: [0x00,0x80,0xe9,0xd3,0x00,0x03,0xd6,0x1b]

v_mfma_f32_16x16x2bf16 v[0:15], a0, a1, -2.0
// GFX90A: v_mfma_f32_16x16x2bf16 v[0:15], a0, a1, -2.0 ; encoding: [0x00,0x00,0xe9,0xd3,0x00,0x03,0xd6,0x1b]

v_mfma_f32_16x16x2bf16 a[0:15], a0, a1, -2.0 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_16x16x2bf16 a[0:15], a0, a1, -2.0 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xe9,0xd3,0x00,0x03,0xd6,0xfb]

v_mfma_f32_16x16x2bf16 v[0:15], a0, a1, -2.0 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_16x16x2bf16 v[0:15], a0, a1, -2.0 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xe9,0xd3,0x00,0x03,0xd6,0xfb]

v_mfma_f32_4x4x2bf16 a[0:3], v0, v1, a[2:5]
// GFX90A: v_mfma_f32_4x4x2bf16 a[0:3], v0, v1, a[2:5] ; encoding: [0x00,0x80,0xeb,0xd3,0x00,0x03,0x0a,0x04]

v_mfma_f32_4x4x2bf16 a[0:3], v0, v1, a[2:5] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_4x4x2bf16 a[0:3], v0, v1, a[2:5] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xeb,0xd3,0x00,0x03,0x0a,0xe4]

v_mfma_f32_4x4x2bf16 a[0:3], v0, a1, a[2:5]
// GFX90A: v_mfma_f32_4x4x2bf16 a[0:3], v0, a1, a[2:5] ; encoding: [0x00,0x80,0xeb,0xd3,0x00,0x03,0x0a,0x14]

v_mfma_f32_4x4x2bf16 a[0:3], v0, a1, a[2:5] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_4x4x2bf16 a[0:3], v0, a1, a[2:5] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xeb,0xd3,0x00,0x03,0x0a,0xf4]

v_mfma_f32_4x4x2bf16 a[0:3], a0, v1, a[2:5]
// GFX90A: v_mfma_f32_4x4x2bf16 a[0:3], a0, v1, a[2:5] ; encoding: [0x00,0x80,0xeb,0xd3,0x00,0x03,0x0a,0x0c]

v_mfma_f32_4x4x2bf16 a[0:3], a0, v1, a[2:5] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_4x4x2bf16 a[0:3], a0, v1, a[2:5] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xeb,0xd3,0x00,0x03,0x0a,0xec]

v_mfma_f32_4x4x2bf16 a[0:3], a0, a1, a[2:5]
// GFX90A: v_mfma_f32_4x4x2bf16 a[0:3], a0, a1, a[2:5] ; encoding: [0x00,0x80,0xeb,0xd3,0x00,0x03,0x0a,0x1c]

v_mfma_f32_4x4x2bf16 a[0:3], a0, a1, a[2:5] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_4x4x2bf16 a[0:3], a0, a1, a[2:5] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xeb,0xd3,0x00,0x03,0x0a,0xfc]

v_mfma_f32_4x4x2bf16 v[0:3], v0, v1, v[2:5]
// GFX90A: v_mfma_f32_4x4x2bf16 v[0:3], v0, v1, v[2:5] ; encoding: [0x00,0x00,0xeb,0xd3,0x00,0x03,0x0a,0x04]

v_mfma_f32_4x4x2bf16 v[0:3], v0, v1, v[2:5] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_4x4x2bf16 v[0:3], v0, v1, v[2:5] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xeb,0xd3,0x00,0x03,0x0a,0xe4]

v_mfma_f32_4x4x2bf16 v[0:3], v0, a1, v[2:5]
// GFX90A: v_mfma_f32_4x4x2bf16 v[0:3], v0, a1, v[2:5] ; encoding: [0x00,0x00,0xeb,0xd3,0x00,0x03,0x0a,0x14]

v_mfma_f32_4x4x2bf16 v[0:3], v0, a1, v[2:5] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_4x4x2bf16 v[0:3], v0, a1, v[2:5] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xeb,0xd3,0x00,0x03,0x0a,0xf4]

v_mfma_f32_4x4x2bf16 v[0:3], a0, v1, v[2:5]
// GFX90A: v_mfma_f32_4x4x2bf16 v[0:3], a0, v1, v[2:5] ; encoding: [0x00,0x00,0xeb,0xd3,0x00,0x03,0x0a,0x0c]

v_mfma_f32_4x4x2bf16 v[0:3], a0, v1, v[2:5] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_4x4x2bf16 v[0:3], a0, v1, v[2:5] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xeb,0xd3,0x00,0x03,0x0a,0xec]

v_mfma_f32_4x4x2bf16 v[0:3], a0, a1, v[2:5]
// GFX90A: v_mfma_f32_4x4x2bf16 v[0:3], a0, a1, v[2:5] ; encoding: [0x00,0x00,0xeb,0xd3,0x00,0x03,0x0a,0x1c]

v_mfma_f32_4x4x2bf16 v[0:3], a0, a1, v[2:5] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_4x4x2bf16 v[0:3], a0, a1, v[2:5] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xeb,0xd3,0x00,0x03,0x0a,0xfc]

v_mfma_f32_4x4x2bf16 a[0:3], v0, v1, -2.0
// GFX90A: v_mfma_f32_4x4x2bf16 a[0:3], v0, v1, -2.0 ; encoding: [0x00,0x80,0xeb,0xd3,0x00,0x03,0xd6,0x03]

v_mfma_f32_4x4x2bf16 v[0:3], v0, v1, -2.0
// GFX90A: v_mfma_f32_4x4x2bf16 v[0:3], v0, v1, -2.0 ; encoding: [0x00,0x00,0xeb,0xd3,0x00,0x03,0xd6,0x03]

v_mfma_f32_4x4x2bf16 a[0:3], v0, v1, -2.0 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_4x4x2bf16 a[0:3], v0, v1, -2.0 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xeb,0xd3,0x00,0x03,0xd6,0xe3]

v_mfma_f32_4x4x2bf16 v[0:3], v0, v1, -2.0 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_4x4x2bf16 v[0:3], v0, v1, -2.0 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xeb,0xd3,0x00,0x03,0xd6,0xe3]

v_mfma_f32_4x4x2bf16 a[0:3], v0, a1, -2.0
// GFX90A: v_mfma_f32_4x4x2bf16 a[0:3], v0, a1, -2.0 ; encoding: [0x00,0x80,0xeb,0xd3,0x00,0x03,0xd6,0x13]

v_mfma_f32_4x4x2bf16 v[0:3], v0, a1, -2.0
// GFX90A: v_mfma_f32_4x4x2bf16 v[0:3], v0, a1, -2.0 ; encoding: [0x00,0x00,0xeb,0xd3,0x00,0x03,0xd6,0x13]

v_mfma_f32_4x4x2bf16 a[0:3], v0, a1, -2.0 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_4x4x2bf16 a[0:3], v0, a1, -2.0 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xeb,0xd3,0x00,0x03,0xd6,0xf3]

v_mfma_f32_4x4x2bf16 v[0:3], v0, a1, -2.0 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_4x4x2bf16 v[0:3], v0, a1, -2.0 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xeb,0xd3,0x00,0x03,0xd6,0xf3]

v_mfma_f32_4x4x2bf16 a[0:3], a0, v1, -2.0
// GFX90A: v_mfma_f32_4x4x2bf16 a[0:3], a0, v1, -2.0 ; encoding: [0x00,0x80,0xeb,0xd3,0x00,0x03,0xd6,0x0b]

v_mfma_f32_4x4x2bf16 v[0:3], a0, v1, -2.0
// GFX90A: v_mfma_f32_4x4x2bf16 v[0:3], a0, v1, -2.0 ; encoding: [0x00,0x00,0xeb,0xd3,0x00,0x03,0xd6,0x0b]

v_mfma_f32_4x4x2bf16 a[0:3], a0, v1, -2.0 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_4x4x2bf16 a[0:3], a0, v1, -2.0 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xeb,0xd3,0x00,0x03,0xd6,0xeb]

v_mfma_f32_4x4x2bf16 v[0:3], a0, v1, -2.0 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_4x4x2bf16 v[0:3], a0, v1, -2.0 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xeb,0xd3,0x00,0x03,0xd6,0xeb]

v_mfma_f32_4x4x2bf16 a[0:3], a0, a1, -2.0
// GFX90A: v_mfma_f32_4x4x2bf16 a[0:3], a0, a1, -2.0 ; encoding: [0x00,0x80,0xeb,0xd3,0x00,0x03,0xd6,0x1b]

v_mfma_f32_4x4x2bf16 v[0:3], a0, a1, -2.0
// GFX90A: v_mfma_f32_4x4x2bf16 v[0:3], a0, a1, -2.0 ; encoding: [0x00,0x00,0xeb,0xd3,0x00,0x03,0xd6,0x1b]

v_mfma_f32_4x4x2bf16 a[0:3], a0, a1, -2.0 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_4x4x2bf16 a[0:3], a0, a1, -2.0 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xeb,0xd3,0x00,0x03,0xd6,0xfb]

v_mfma_f32_4x4x2bf16 v[0:3], a0, a1, -2.0 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_4x4x2bf16 v[0:3], a0, a1, -2.0 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xeb,0xd3,0x00,0x03,0xd6,0xfb]

v_mfma_f32_32x32x4bf16 a[0:15], v0, v1, a[18:33]
// GFX90A: v_mfma_f32_32x32x4bf16 a[0:15], v0, v1, a[18:33] ; encoding: [0x00,0x80,0xec,0xd3,0x00,0x03,0x4a,0x04]

v_mfma_f32_32x32x4bf16 a[0:15], v0, v1, a[18:33] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_32x32x4bf16 a[0:15], v0, v1, a[18:33] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xec,0xd3,0x00,0x03,0x4a,0xe4]

v_mfma_f32_32x32x4bf16 a[0:15], v0, a1, a[18:33]
// GFX90A: v_mfma_f32_32x32x4bf16 a[0:15], v0, a1, a[18:33] ; encoding: [0x00,0x80,0xec,0xd3,0x00,0x03,0x4a,0x14]

v_mfma_f32_32x32x4bf16 a[0:15], v0, a1, a[18:33] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_32x32x4bf16 a[0:15], v0, a1, a[18:33] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xec,0xd3,0x00,0x03,0x4a,0xf4]

v_mfma_f32_32x32x4bf16 a[0:15], a0, v1, a[18:33]
// GFX90A: v_mfma_f32_32x32x4bf16 a[0:15], a0, v1, a[18:33] ; encoding: [0x00,0x80,0xec,0xd3,0x00,0x03,0x4a,0x0c]

v_mfma_f32_32x32x4bf16 a[0:15], a0, v1, a[18:33] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_32x32x4bf16 a[0:15], a0, v1, a[18:33] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xec,0xd3,0x00,0x03,0x4a,0xec]

v_mfma_f32_32x32x4bf16 a[0:15], a0, a1, a[18:33]
// GFX90A: v_mfma_f32_32x32x4bf16 a[0:15], a0, a1, a[18:33] ; encoding: [0x00,0x80,0xec,0xd3,0x00,0x03,0x4a,0x1c]

v_mfma_f32_32x32x4bf16 a[0:15], a0, a1, a[18:33] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_32x32x4bf16 a[0:15], a0, a1, a[18:33] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xec,0xd3,0x00,0x03,0x4a,0xfc]

v_mfma_f32_32x32x4bf16 v[0:15], v0, v1, v[18:33]
// GFX90A: v_mfma_f32_32x32x4bf16 v[0:15], v0, v1, v[18:33] ; encoding: [0x00,0x00,0xec,0xd3,0x00,0x03,0x4a,0x04]

v_mfma_f32_32x32x4bf16 v[0:15], v0, v1, v[18:33] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_32x32x4bf16 v[0:15], v0, v1, v[18:33] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xec,0xd3,0x00,0x03,0x4a,0xe4]

v_mfma_f32_32x32x4bf16 v[0:15], v0, a1, v[18:33]
// GFX90A: v_mfma_f32_32x32x4bf16 v[0:15], v0, a1, v[18:33] ; encoding: [0x00,0x00,0xec,0xd3,0x00,0x03,0x4a,0x14]

v_mfma_f32_32x32x4bf16 v[0:15], v0, a1, v[18:33] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_32x32x4bf16 v[0:15], v0, a1, v[18:33] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xec,0xd3,0x00,0x03,0x4a,0xf4]

v_mfma_f32_32x32x4bf16 v[0:15], a0, v1, v[18:33]
// GFX90A: v_mfma_f32_32x32x4bf16 v[0:15], a0, v1, v[18:33] ; encoding: [0x00,0x00,0xec,0xd3,0x00,0x03,0x4a,0x0c]

v_mfma_f32_32x32x4bf16 v[0:15], a0, v1, v[18:33] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_32x32x4bf16 v[0:15], a0, v1, v[18:33] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xec,0xd3,0x00,0x03,0x4a,0xec]

v_mfma_f32_32x32x4bf16 v[0:15], a0, a1, v[18:33]
// GFX90A: v_mfma_f32_32x32x4bf16 v[0:15], a0, a1, v[18:33] ; encoding: [0x00,0x00,0xec,0xd3,0x00,0x03,0x4a,0x1c]

v_mfma_f32_32x32x4bf16 v[0:15], a0, a1, v[18:33] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_32x32x4bf16 v[0:15], a0, a1, v[18:33] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xec,0xd3,0x00,0x03,0x4a,0xfc]

v_mfma_f32_32x32x4bf16 a[0:15], v0, v1, -2.0
// GFX90A: v_mfma_f32_32x32x4bf16 a[0:15], v0, v1, -2.0 ; encoding: [0x00,0x80,0xec,0xd3,0x00,0x03,0xd6,0x03]

v_mfma_f32_32x32x4bf16 v[0:15], v0, v1, -2.0
// GFX90A: v_mfma_f32_32x32x4bf16 v[0:15], v0, v1, -2.0 ; encoding: [0x00,0x00,0xec,0xd3,0x00,0x03,0xd6,0x03]

v_mfma_f32_32x32x4bf16 a[0:15], v0, v1, -2.0 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_32x32x4bf16 a[0:15], v0, v1, -2.0 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xec,0xd3,0x00,0x03,0xd6,0xe3]

v_mfma_f32_32x32x4bf16 v[0:15], v0, v1, -2.0 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_32x32x4bf16 v[0:15], v0, v1, -2.0 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xec,0xd3,0x00,0x03,0xd6,0xe3]

v_mfma_f32_32x32x4bf16 a[0:15], v0, a1, -2.0
// GFX90A: v_mfma_f32_32x32x4bf16 a[0:15], v0, a1, -2.0 ; encoding: [0x00,0x80,0xec,0xd3,0x00,0x03,0xd6,0x13]

v_mfma_f32_32x32x4bf16 v[0:15], v0, a1, -2.0
// GFX90A: v_mfma_f32_32x32x4bf16 v[0:15], v0, a1, -2.0 ; encoding: [0x00,0x00,0xec,0xd3,0x00,0x03,0xd6,0x13]

v_mfma_f32_32x32x4bf16 a[0:15], v0, a1, -2.0 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_32x32x4bf16 a[0:15], v0, a1, -2.0 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xec,0xd3,0x00,0x03,0xd6,0xf3]

v_mfma_f32_32x32x4bf16 v[0:15], v0, a1, -2.0 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_32x32x4bf16 v[0:15], v0, a1, -2.0 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xec,0xd3,0x00,0x03,0xd6,0xf3]

v_mfma_f32_32x32x4bf16 a[0:15], a0, v1, -2.0
// GFX90A: v_mfma_f32_32x32x4bf16 a[0:15], a0, v1, -2.0 ; encoding: [0x00,0x80,0xec,0xd3,0x00,0x03,0xd6,0x0b]

v_mfma_f32_32x32x4bf16 v[0:15], a0, v1, -2.0
// GFX90A: v_mfma_f32_32x32x4bf16 v[0:15], a0, v1, -2.0 ; encoding: [0x00,0x00,0xec,0xd3,0x00,0x03,0xd6,0x0b]

v_mfma_f32_32x32x4bf16 a[0:15], a0, v1, -2.0 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_32x32x4bf16 a[0:15], a0, v1, -2.0 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xec,0xd3,0x00,0x03,0xd6,0xeb]

v_mfma_f32_32x32x4bf16 v[0:15], a0, v1, -2.0 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_32x32x4bf16 v[0:15], a0, v1, -2.0 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xec,0xd3,0x00,0x03,0xd6,0xeb]

v_mfma_f32_32x32x4bf16 a[0:15], a0, a1, -2.0
// GFX90A: v_mfma_f32_32x32x4bf16 a[0:15], a0, a1, -2.0 ; encoding: [0x00,0x80,0xec,0xd3,0x00,0x03,0xd6,0x1b]

v_mfma_f32_32x32x4bf16 v[0:15], a0, a1, -2.0
// GFX90A: v_mfma_f32_32x32x4bf16 v[0:15], a0, a1, -2.0 ; encoding: [0x00,0x00,0xec,0xd3,0x00,0x03,0xd6,0x1b]

v_mfma_f32_32x32x4bf16 a[0:15], a0, a1, -2.0 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_32x32x4bf16 a[0:15], a0, a1, -2.0 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xec,0xd3,0x00,0x03,0xd6,0xfb]

v_mfma_f32_32x32x4bf16 v[0:15], a0, a1, -2.0 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_32x32x4bf16 v[0:15], a0, a1, -2.0 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xec,0xd3,0x00,0x03,0xd6,0xfb]

v_mfma_f32_16x16x8bf16 a[0:3], v0, v1, a[2:5]
// GFX90A: v_mfma_f32_16x16x8bf16 a[0:3], v0, v1, a[2:5] ; encoding: [0x00,0x80,0xed,0xd3,0x00,0x03,0x0a,0x04]

v_mfma_f32_16x16x8bf16 a[0:3], v0, v1, a[2:5] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_16x16x8bf16 a[0:3], v0, v1, a[2:5] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xed,0xd3,0x00,0x03,0x0a,0xe4]

v_mfma_f32_16x16x8bf16 a[0:3], v0, a1, a[2:5]
// GFX90A: v_mfma_f32_16x16x8bf16 a[0:3], v0, a1, a[2:5] ; encoding: [0x00,0x80,0xed,0xd3,0x00,0x03,0x0a,0x14]

v_mfma_f32_16x16x8bf16 a[0:3], v0, a1, a[2:5] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_16x16x8bf16 a[0:3], v0, a1, a[2:5] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xed,0xd3,0x00,0x03,0x0a,0xf4]

v_mfma_f32_16x16x8bf16 a[0:3], a0, v1, a[2:5]
// GFX90A: v_mfma_f32_16x16x8bf16 a[0:3], a0, v1, a[2:5] ; encoding: [0x00,0x80,0xed,0xd3,0x00,0x03,0x0a,0x0c]

v_mfma_f32_16x16x8bf16 a[0:3], a0, v1, a[2:5] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_16x16x8bf16 a[0:3], a0, v1, a[2:5] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xed,0xd3,0x00,0x03,0x0a,0xec]

v_mfma_f32_16x16x8bf16 a[0:3], a0, a1, a[2:5]
// GFX90A: v_mfma_f32_16x16x8bf16 a[0:3], a0, a1, a[2:5] ; encoding: [0x00,0x80,0xed,0xd3,0x00,0x03,0x0a,0x1c]

v_mfma_f32_16x16x8bf16 a[0:3], a0, a1, a[2:5] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_16x16x8bf16 a[0:3], a0, a1, a[2:5] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xed,0xd3,0x00,0x03,0x0a,0xfc]

v_mfma_f32_16x16x8bf16 v[0:3], v0, v1, v[2:5]
// GFX90A: v_mfma_f32_16x16x8bf16 v[0:3], v0, v1, v[2:5] ; encoding: [0x00,0x00,0xed,0xd3,0x00,0x03,0x0a,0x04]

v_mfma_f32_16x16x8bf16 v[0:3], v0, v1, v[2:5] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_16x16x8bf16 v[0:3], v0, v1, v[2:5] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xed,0xd3,0x00,0x03,0x0a,0xe4]

v_mfma_f32_16x16x8bf16 v[0:3], v0, a1, v[2:5]
// GFX90A: v_mfma_f32_16x16x8bf16 v[0:3], v0, a1, v[2:5] ; encoding: [0x00,0x00,0xed,0xd3,0x00,0x03,0x0a,0x14]

v_mfma_f32_16x16x8bf16 v[0:3], v0, a1, v[2:5] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_16x16x8bf16 v[0:3], v0, a1, v[2:5] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xed,0xd3,0x00,0x03,0x0a,0xf4]

v_mfma_f32_16x16x8bf16 v[0:3], a0, v1, v[2:5]
// GFX90A: v_mfma_f32_16x16x8bf16 v[0:3], a0, v1, v[2:5] ; encoding: [0x00,0x00,0xed,0xd3,0x00,0x03,0x0a,0x0c]

v_mfma_f32_16x16x8bf16 v[0:3], a0, v1, v[2:5] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_16x16x8bf16 v[0:3], a0, v1, v[2:5] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xed,0xd3,0x00,0x03,0x0a,0xec]

v_mfma_f32_16x16x8bf16 v[0:3], a0, a1, v[2:5]
// GFX90A: v_mfma_f32_16x16x8bf16 v[0:3], a0, a1, v[2:5] ; encoding: [0x00,0x00,0xed,0xd3,0x00,0x03,0x0a,0x1c]

v_mfma_f32_16x16x8bf16 v[0:3], a0, a1, v[2:5] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_16x16x8bf16 v[0:3], a0, a1, v[2:5] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xed,0xd3,0x00,0x03,0x0a,0xfc]

v_mfma_f32_16x16x8bf16 a[0:3], v0, v1, -2.0
// GFX90A: v_mfma_f32_16x16x8bf16 a[0:3], v0, v1, -2.0 ; encoding: [0x00,0x80,0xed,0xd3,0x00,0x03,0xd6,0x03]

v_mfma_f32_16x16x8bf16 v[0:3], v0, v1, -2.0
// GFX90A: v_mfma_f32_16x16x8bf16 v[0:3], v0, v1, -2.0 ; encoding: [0x00,0x00,0xed,0xd3,0x00,0x03,0xd6,0x03]

v_mfma_f32_16x16x8bf16 a[0:3], v0, v1, -2.0 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_16x16x8bf16 a[0:3], v0, v1, -2.0 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xed,0xd3,0x00,0x03,0xd6,0xe3]

v_mfma_f32_16x16x8bf16 v[0:3], v0, v1, -2.0 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_16x16x8bf16 v[0:3], v0, v1, -2.0 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xed,0xd3,0x00,0x03,0xd6,0xe3]

v_mfma_f32_16x16x8bf16 a[0:3], v0, a1, -2.0
// GFX90A: v_mfma_f32_16x16x8bf16 a[0:3], v0, a1, -2.0 ; encoding: [0x00,0x80,0xed,0xd3,0x00,0x03,0xd6,0x13]

v_mfma_f32_16x16x8bf16 v[0:3], v0, a1, -2.0
// GFX90A: v_mfma_f32_16x16x8bf16 v[0:3], v0, a1, -2.0 ; encoding: [0x00,0x00,0xed,0xd3,0x00,0x03,0xd6,0x13]

v_mfma_f32_16x16x8bf16 a[0:3], v0, a1, -2.0 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_16x16x8bf16 a[0:3], v0, a1, -2.0 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xed,0xd3,0x00,0x03,0xd6,0xf3]

v_mfma_f32_16x16x8bf16 v[0:3], v0, a1, -2.0 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_16x16x8bf16 v[0:3], v0, a1, -2.0 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xed,0xd3,0x00,0x03,0xd6,0xf3]

v_mfma_f32_16x16x8bf16 a[0:3], a0, v1, -2.0
// GFX90A: v_mfma_f32_16x16x8bf16 a[0:3], a0, v1, -2.0 ; encoding: [0x00,0x80,0xed,0xd3,0x00,0x03,0xd6,0x0b]

v_mfma_f32_16x16x8bf16 v[0:3], a0, v1, -2.0
// GFX90A: v_mfma_f32_16x16x8bf16 v[0:3], a0, v1, -2.0 ; encoding: [0x00,0x00,0xed,0xd3,0x00,0x03,0xd6,0x0b]

v_mfma_f32_16x16x8bf16 a[0:3], a0, v1, -2.0 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_16x16x8bf16 a[0:3], a0, v1, -2.0 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xed,0xd3,0x00,0x03,0xd6,0xeb]

v_mfma_f32_16x16x8bf16 v[0:3], a0, v1, -2.0 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_16x16x8bf16 v[0:3], a0, v1, -2.0 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xed,0xd3,0x00,0x03,0xd6,0xeb]

v_mfma_f32_16x16x8bf16 a[0:3], a0, a1, -2.0
// GFX90A: v_mfma_f32_16x16x8bf16 a[0:3], a0, a1, -2.0 ; encoding: [0x00,0x80,0xed,0xd3,0x00,0x03,0xd6,0x1b]

v_mfma_f32_16x16x8bf16 v[0:3], a0, a1, -2.0
// GFX90A: v_mfma_f32_16x16x8bf16 v[0:3], a0, a1, -2.0 ; encoding: [0x00,0x00,0xed,0xd3,0x00,0x03,0xd6,0x1b]

v_mfma_f32_16x16x8bf16 a[0:3], a0, a1, -2.0 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_16x16x8bf16 a[0:3], a0, a1, -2.0 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xed,0xd3,0x00,0x03,0xd6,0xfb]

v_mfma_f32_16x16x8bf16 v[0:3], a0, a1, -2.0 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_16x16x8bf16 v[0:3], a0, a1, -2.0 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xed,0xd3,0x00,0x03,0xd6,0xfb]

v_mfma_f32_32x32x4bf16_1k a[0:31], v[0:1], v[2:3], a[34:65]
// GFX90A: v_mfma_f32_32x32x4bf16_1k a[0:31], v[0:1], v[2:3], a[34:65] ; encoding: [0x00,0x80,0xe3,0xd3,0x00,0x05,0x8a,0x04]

v_mfma_f32_32x32x4bf16_1k a[0:31], v[0:1], v[2:3], a[34:65] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_32x32x4bf16_1k a[0:31], v[0:1], v[2:3], a[34:65] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xe3,0xd3,0x00,0x05,0x8a,0xe4]

v_mfma_f32_32x32x4bf16_1k a[0:31], v[0:1], a[2:3], a[34:65]
// GFX90A: v_mfma_f32_32x32x4bf16_1k a[0:31], v[0:1], a[2:3], a[34:65] ; encoding: [0x00,0x80,0xe3,0xd3,0x00,0x05,0x8a,0x14]

v_mfma_f32_32x32x4bf16_1k a[0:31], v[0:1], a[2:3], a[34:65] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_32x32x4bf16_1k a[0:31], v[0:1], a[2:3], a[34:65] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xe3,0xd3,0x00,0x05,0x8a,0xf4]

v_mfma_f32_32x32x4bf16_1k a[0:31], a[0:1], v[2:3], a[34:65]
// GFX90A: v_mfma_f32_32x32x4bf16_1k a[0:31], a[0:1], v[2:3], a[34:65] ; encoding: [0x00,0x80,0xe3,0xd3,0x00,0x05,0x8a,0x0c]

v_mfma_f32_32x32x4bf16_1k a[0:31], a[0:1], v[2:3], a[34:65] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_32x32x4bf16_1k a[0:31], a[0:1], v[2:3], a[34:65] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xe3,0xd3,0x00,0x05,0x8a,0xec]

v_mfma_f32_32x32x4bf16_1k a[0:31], a[0:1], a[2:3], a[34:65]
// GFX90A: v_mfma_f32_32x32x4bf16_1k a[0:31], a[0:1], a[2:3], a[34:65] ; encoding: [0x00,0x80,0xe3,0xd3,0x00,0x05,0x8a,0x1c]

v_mfma_f32_32x32x4bf16_1k a[0:31], a[0:1], a[2:3], a[34:65] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_32x32x4bf16_1k a[0:31], a[0:1], a[2:3], a[34:65] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xe3,0xd3,0x00,0x05,0x8a,0xfc]

v_mfma_f32_32x32x4bf16_1k v[0:31], v[0:1], v[2:3], v[34:65]
// GFX90A: v_mfma_f32_32x32x4bf16_1k v[0:31], v[0:1], v[2:3], v[34:65] ; encoding: [0x00,0x00,0xe3,0xd3,0x00,0x05,0x8a,0x04]

v_mfma_f32_32x32x4bf16_1k v[0:31], v[0:1], v[2:3], v[34:65] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_32x32x4bf16_1k v[0:31], v[0:1], v[2:3], v[34:65] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xe3,0xd3,0x00,0x05,0x8a,0xe4]

v_mfma_f32_32x32x4bf16_1k v[0:31], v[0:1], a[2:3], v[34:65]
// GFX90A: v_mfma_f32_32x32x4bf16_1k v[0:31], v[0:1], a[2:3], v[34:65] ; encoding: [0x00,0x00,0xe3,0xd3,0x00,0x05,0x8a,0x14]

v_mfma_f32_32x32x4bf16_1k v[0:31], v[0:1], a[2:3], v[34:65] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_32x32x4bf16_1k v[0:31], v[0:1], a[2:3], v[34:65] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xe3,0xd3,0x00,0x05,0x8a,0xf4]

v_mfma_f32_32x32x4bf16_1k v[0:31], a[0:1], v[2:3], v[34:65]
// GFX90A: v_mfma_f32_32x32x4bf16_1k v[0:31], a[0:1], v[2:3], v[34:65] ; encoding: [0x00,0x00,0xe3,0xd3,0x00,0x05,0x8a,0x0c]

v_mfma_f32_32x32x4bf16_1k v[0:31], a[0:1], v[2:3], v[34:65] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_32x32x4bf16_1k v[0:31], a[0:1], v[2:3], v[34:65] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xe3,0xd3,0x00,0x05,0x8a,0xec]

v_mfma_f32_32x32x4bf16_1k v[0:31], a[0:1], a[2:3], v[34:65]
// GFX90A: v_mfma_f32_32x32x4bf16_1k v[0:31], a[0:1], a[2:3], v[34:65] ; encoding: [0x00,0x00,0xe3,0xd3,0x00,0x05,0x8a,0x1c]

v_mfma_f32_32x32x4bf16_1k v[0:31], a[0:1], a[2:3], v[34:65] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_32x32x4bf16_1k v[0:31], a[0:1], a[2:3], v[34:65] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xe3,0xd3,0x00,0x05,0x8a,0xfc]

v_mfma_f32_32x32x4bf16_1k a[0:31], v[0:1], v[2:3], -2.0
// GFX90A: v_mfma_f32_32x32x4bf16_1k a[0:31], v[0:1], v[2:3], -2.0 ; encoding: [0x00,0x80,0xe3,0xd3,0x00,0x05,0xd6,0x03]

v_mfma_f32_32x32x4bf16_1k v[0:31], v[0:1], v[2:3], -2.0
// GFX90A: v_mfma_f32_32x32x4bf16_1k v[0:31], v[0:1], v[2:3], -2.0 ; encoding: [0x00,0x00,0xe3,0xd3,0x00,0x05,0xd6,0x03]

v_mfma_f32_32x32x4bf16_1k a[0:31], v[0:1], v[2:3], -2.0 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_32x32x4bf16_1k a[0:31], v[0:1], v[2:3], -2.0 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xe3,0xd3,0x00,0x05,0xd6,0xe3]

v_mfma_f32_32x32x4bf16_1k v[0:31], v[0:1], v[2:3], -2.0 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_32x32x4bf16_1k v[0:31], v[0:1], v[2:3], -2.0 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xe3,0xd3,0x00,0x05,0xd6,0xe3]

v_mfma_f32_32x32x4bf16_1k a[0:31], v[0:1], a[2:3], -2.0
// GFX90A: v_mfma_f32_32x32x4bf16_1k a[0:31], v[0:1], a[2:3], -2.0 ; encoding: [0x00,0x80,0xe3,0xd3,0x00,0x05,0xd6,0x13]

v_mfma_f32_32x32x4bf16_1k v[0:31], v[0:1], a[2:3], -2.0
// GFX90A: v_mfma_f32_32x32x4bf16_1k v[0:31], v[0:1], a[2:3], -2.0 ; encoding: [0x00,0x00,0xe3,0xd3,0x00,0x05,0xd6,0x13]

v_mfma_f32_32x32x4bf16_1k a[0:31], v[0:1], a[2:3], -2.0 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_32x32x4bf16_1k a[0:31], v[0:1], a[2:3], -2.0 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xe3,0xd3,0x00,0x05,0xd6,0xf3]

v_mfma_f32_32x32x4bf16_1k v[0:31], v[0:1], a[2:3], -2.0 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_32x32x4bf16_1k v[0:31], v[0:1], a[2:3], -2.0 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xe3,0xd3,0x00,0x05,0xd6,0xf3]

v_mfma_f32_32x32x4bf16_1k a[0:31], a[0:1], v[2:3], -2.0
// GFX90A: v_mfma_f32_32x32x4bf16_1k a[0:31], a[0:1], v[2:3], -2.0 ; encoding: [0x00,0x80,0xe3,0xd3,0x00,0x05,0xd6,0x0b]

v_mfma_f32_32x32x4bf16_1k v[0:31], a[0:1], v[2:3], -2.0
// GFX90A: v_mfma_f32_32x32x4bf16_1k v[0:31], a[0:1], v[2:3], -2.0 ; encoding: [0x00,0x00,0xe3,0xd3,0x00,0x05,0xd6,0x0b]

v_mfma_f32_32x32x4bf16_1k a[0:31], a[0:1], v[2:3], -2.0 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_32x32x4bf16_1k a[0:31], a[0:1], v[2:3], -2.0 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xe3,0xd3,0x00,0x05,0xd6,0xeb]

v_mfma_f32_32x32x4bf16_1k v[0:31], a[0:1], v[2:3], -2.0 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_32x32x4bf16_1k v[0:31], a[0:1], v[2:3], -2.0 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xe3,0xd3,0x00,0x05,0xd6,0xeb]

v_mfma_f32_32x32x4bf16_1k a[0:31], a[0:1], a[2:3], -2.0
// GFX90A: v_mfma_f32_32x32x4bf16_1k a[0:31], a[0:1], a[2:3], -2.0 ; encoding: [0x00,0x80,0xe3,0xd3,0x00,0x05,0xd6,0x1b]

v_mfma_f32_32x32x4bf16_1k v[0:31], a[0:1], a[2:3], -2.0
// GFX90A: v_mfma_f32_32x32x4bf16_1k v[0:31], a[0:1], a[2:3], -2.0 ; encoding: [0x00,0x00,0xe3,0xd3,0x00,0x05,0xd6,0x1b]

v_mfma_f32_32x32x4bf16_1k a[0:31], a[0:1], a[2:3], -2.0 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_32x32x4bf16_1k a[0:31], a[0:1], a[2:3], -2.0 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xe3,0xd3,0x00,0x05,0xd6,0xfb]

v_mfma_f32_32x32x4bf16_1k v[0:31], a[0:1], a[2:3], -2.0 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_32x32x4bf16_1k v[0:31], a[0:1], a[2:3], -2.0 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xe3,0xd3,0x00,0x05,0xd6,0xfb]

v_mfma_f32_16x16x4bf16_1k a[0:15], v[0:1], v[2:3], a[18:33]
// GFX90A: v_mfma_f32_16x16x4bf16_1k a[0:15], v[0:1], v[2:3], a[18:33] ; encoding: [0x00,0x80,0xe4,0xd3,0x00,0x05,0x4a,0x04]

v_mfma_f32_16x16x4bf16_1k a[0:15], v[0:1], v[2:3], a[18:33] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_16x16x4bf16_1k a[0:15], v[0:1], v[2:3], a[18:33] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xe4,0xd3,0x00,0x05,0x4a,0xe4]

v_mfma_f32_16x16x4bf16_1k a[0:15], v[0:1], a[2:3], a[18:33]
// GFX90A: v_mfma_f32_16x16x4bf16_1k a[0:15], v[0:1], a[2:3], a[18:33] ; encoding: [0x00,0x80,0xe4,0xd3,0x00,0x05,0x4a,0x14]

v_mfma_f32_16x16x4bf16_1k a[0:15], v[0:1], a[2:3], a[18:33] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_16x16x4bf16_1k a[0:15], v[0:1], a[2:3], a[18:33] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xe4,0xd3,0x00,0x05,0x4a,0xf4]

v_mfma_f32_16x16x4bf16_1k a[0:15], a[0:1], v[2:3], a[18:33]
// GFX90A: v_mfma_f32_16x16x4bf16_1k a[0:15], a[0:1], v[2:3], a[18:33] ; encoding: [0x00,0x80,0xe4,0xd3,0x00,0x05,0x4a,0x0c]

v_mfma_f32_16x16x4bf16_1k a[0:15], a[0:1], v[2:3], a[18:33] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_16x16x4bf16_1k a[0:15], a[0:1], v[2:3], a[18:33] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xe4,0xd3,0x00,0x05,0x4a,0xec]

v_mfma_f32_16x16x4bf16_1k a[0:15], a[0:1], a[2:3], a[18:33]
// GFX90A: v_mfma_f32_16x16x4bf16_1k a[0:15], a[0:1], a[2:3], a[18:33] ; encoding: [0x00,0x80,0xe4,0xd3,0x00,0x05,0x4a,0x1c]

v_mfma_f32_16x16x4bf16_1k a[0:15], a[0:1], a[2:3], a[18:33] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_16x16x4bf16_1k a[0:15], a[0:1], a[2:3], a[18:33] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xe4,0xd3,0x00,0x05,0x4a,0xfc]

v_mfma_f32_16x16x4bf16_1k v[0:15], v[0:1], v[2:3], v[18:33]
// GFX90A: v_mfma_f32_16x16x4bf16_1k v[0:15], v[0:1], v[2:3], v[18:33] ; encoding: [0x00,0x00,0xe4,0xd3,0x00,0x05,0x4a,0x04]

v_mfma_f32_16x16x4bf16_1k v[0:15], v[0:1], v[2:3], v[18:33] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_16x16x4bf16_1k v[0:15], v[0:1], v[2:3], v[18:33] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xe4,0xd3,0x00,0x05,0x4a,0xe4]

v_mfma_f32_16x16x4bf16_1k v[0:15], v[0:1], a[2:3], v[18:33]
// GFX90A: v_mfma_f32_16x16x4bf16_1k v[0:15], v[0:1], a[2:3], v[18:33] ; encoding: [0x00,0x00,0xe4,0xd3,0x00,0x05,0x4a,0x14]

v_mfma_f32_16x16x4bf16_1k v[0:15], v[0:1], a[2:3], v[18:33] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_16x16x4bf16_1k v[0:15], v[0:1], a[2:3], v[18:33] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xe4,0xd3,0x00,0x05,0x4a,0xf4]

v_mfma_f32_16x16x4bf16_1k v[0:15], a[0:1], v[2:3], v[18:33]
// GFX90A: v_mfma_f32_16x16x4bf16_1k v[0:15], a[0:1], v[2:3], v[18:33] ; encoding: [0x00,0x00,0xe4,0xd3,0x00,0x05,0x4a,0x0c]

v_mfma_f32_16x16x4bf16_1k v[0:15], a[0:1], v[2:3], v[18:33] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_16x16x4bf16_1k v[0:15], a[0:1], v[2:3], v[18:33] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xe4,0xd3,0x00,0x05,0x4a,0xec]

v_mfma_f32_16x16x4bf16_1k v[0:15], a[0:1], a[2:3], v[18:33]
// GFX90A: v_mfma_f32_16x16x4bf16_1k v[0:15], a[0:1], a[2:3], v[18:33] ; encoding: [0x00,0x00,0xe4,0xd3,0x00,0x05,0x4a,0x1c]

v_mfma_f32_16x16x4bf16_1k v[0:15], a[0:1], a[2:3], v[18:33] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_16x16x4bf16_1k v[0:15], a[0:1], a[2:3], v[18:33] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xe4,0xd3,0x00,0x05,0x4a,0xfc]

v_mfma_f32_16x16x4bf16_1k a[0:15], v[0:1], v[2:3], -2.0
// GFX90A: v_mfma_f32_16x16x4bf16_1k a[0:15], v[0:1], v[2:3], -2.0 ; encoding: [0x00,0x80,0xe4,0xd3,0x00,0x05,0xd6,0x03]

v_mfma_f32_16x16x4bf16_1k v[0:15], v[0:1], v[2:3], -2.0
// GFX90A: v_mfma_f32_16x16x4bf16_1k v[0:15], v[0:1], v[2:3], -2.0 ; encoding: [0x00,0x00,0xe4,0xd3,0x00,0x05,0xd6,0x03]

v_mfma_f32_16x16x4bf16_1k a[0:15], v[0:1], v[2:3], -2.0 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_16x16x4bf16_1k a[0:15], v[0:1], v[2:3], -2.0 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xe4,0xd3,0x00,0x05,0xd6,0xe3]

v_mfma_f32_16x16x4bf16_1k v[0:15], v[0:1], v[2:3], -2.0 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_16x16x4bf16_1k v[0:15], v[0:1], v[2:3], -2.0 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xe4,0xd3,0x00,0x05,0xd6,0xe3]

v_mfma_f32_16x16x4bf16_1k a[0:15], v[0:1], a[2:3], -2.0
// GFX90A: v_mfma_f32_16x16x4bf16_1k a[0:15], v[0:1], a[2:3], -2.0 ; encoding: [0x00,0x80,0xe4,0xd3,0x00,0x05,0xd6,0x13]

v_mfma_f32_16x16x4bf16_1k v[0:15], v[0:1], a[2:3], -2.0
// GFX90A: v_mfma_f32_16x16x4bf16_1k v[0:15], v[0:1], a[2:3], -2.0 ; encoding: [0x00,0x00,0xe4,0xd3,0x00,0x05,0xd6,0x13]

v_mfma_f32_16x16x4bf16_1k a[0:15], v[0:1], a[2:3], -2.0 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_16x16x4bf16_1k a[0:15], v[0:1], a[2:3], -2.0 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xe4,0xd3,0x00,0x05,0xd6,0xf3]

v_mfma_f32_16x16x4bf16_1k v[0:15], v[0:1], a[2:3], -2.0 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_16x16x4bf16_1k v[0:15], v[0:1], a[2:3], -2.0 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xe4,0xd3,0x00,0x05,0xd6,0xf3]

v_mfma_f32_16x16x4bf16_1k a[0:15], a[0:1], v[2:3], -2.0
// GFX90A: v_mfma_f32_16x16x4bf16_1k a[0:15], a[0:1], v[2:3], -2.0 ; encoding: [0x00,0x80,0xe4,0xd3,0x00,0x05,0xd6,0x0b]

v_mfma_f32_16x16x4bf16_1k v[0:15], a[0:1], v[2:3], -2.0
// GFX90A: v_mfma_f32_16x16x4bf16_1k v[0:15], a[0:1], v[2:3], -2.0 ; encoding: [0x00,0x00,0xe4,0xd3,0x00,0x05,0xd6,0x0b]

v_mfma_f32_16x16x4bf16_1k a[0:15], a[0:1], v[2:3], -2.0 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_16x16x4bf16_1k a[0:15], a[0:1], v[2:3], -2.0 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xe4,0xd3,0x00,0x05,0xd6,0xeb]

v_mfma_f32_16x16x4bf16_1k v[0:15], a[0:1], v[2:3], -2.0 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_16x16x4bf16_1k v[0:15], a[0:1], v[2:3], -2.0 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xe4,0xd3,0x00,0x05,0xd6,0xeb]

v_mfma_f32_16x16x4bf16_1k a[0:15], a[0:1], a[2:3], -2.0
// GFX90A: v_mfma_f32_16x16x4bf16_1k a[0:15], a[0:1], a[2:3], -2.0 ; encoding: [0x00,0x80,0xe4,0xd3,0x00,0x05,0xd6,0x1b]

v_mfma_f32_16x16x4bf16_1k v[0:15], a[0:1], a[2:3], -2.0
// GFX90A: v_mfma_f32_16x16x4bf16_1k v[0:15], a[0:1], a[2:3], -2.0 ; encoding: [0x00,0x00,0xe4,0xd3,0x00,0x05,0xd6,0x1b]

v_mfma_f32_16x16x4bf16_1k a[0:15], a[0:1], a[2:3], -2.0 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_16x16x4bf16_1k a[0:15], a[0:1], a[2:3], -2.0 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xe4,0xd3,0x00,0x05,0xd6,0xfb]

v_mfma_f32_16x16x4bf16_1k v[0:15], a[0:1], a[2:3], -2.0 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_16x16x4bf16_1k v[0:15], a[0:1], a[2:3], -2.0 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xe4,0xd3,0x00,0x05,0xd6,0xfb]

v_mfma_f32_4x4x4bf16_1k a[0:3], v[0:1], v[2:3], a[2:5]
// GFX90A: v_mfma_f32_4x4x4bf16_1k a[0:3], v[0:1], v[2:3], a[2:5] ; encoding: [0x00,0x80,0xe5,0xd3,0x00,0x05,0x0a,0x04]

v_mfma_f32_4x4x4bf16_1k a[0:3], v[0:1], v[2:3], a[2:5] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_4x4x4bf16_1k a[0:3], v[0:1], v[2:3], a[2:5] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xe5,0xd3,0x00,0x05,0x0a,0xe4]

v_mfma_f32_4x4x4bf16_1k a[0:3], v[0:1], a[2:3], a[2:5]
// GFX90A: v_mfma_f32_4x4x4bf16_1k a[0:3], v[0:1], a[2:3], a[2:5] ; encoding: [0x00,0x80,0xe5,0xd3,0x00,0x05,0x0a,0x14]

v_mfma_f32_4x4x4bf16_1k a[0:3], v[0:1], a[2:3], a[2:5] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_4x4x4bf16_1k a[0:3], v[0:1], a[2:3], a[2:5] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xe5,0xd3,0x00,0x05,0x0a,0xf4]

v_mfma_f32_4x4x4bf16_1k a[0:3], a[0:1], v[2:3], a[2:5]
// GFX90A: v_mfma_f32_4x4x4bf16_1k a[0:3], a[0:1], v[2:3], a[2:5] ; encoding: [0x00,0x80,0xe5,0xd3,0x00,0x05,0x0a,0x0c]

v_mfma_f32_4x4x4bf16_1k a[0:3], a[0:1], v[2:3], a[2:5] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_4x4x4bf16_1k a[0:3], a[0:1], v[2:3], a[2:5] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xe5,0xd3,0x00,0x05,0x0a,0xec]

v_mfma_f32_4x4x4bf16_1k a[0:3], a[0:1], a[2:3], a[2:5]
// GFX90A: v_mfma_f32_4x4x4bf16_1k a[0:3], a[0:1], a[2:3], a[2:5] ; encoding: [0x00,0x80,0xe5,0xd3,0x00,0x05,0x0a,0x1c]

v_mfma_f32_4x4x4bf16_1k a[0:3], a[0:1], a[2:3], a[2:5] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_4x4x4bf16_1k a[0:3], a[0:1], a[2:3], a[2:5] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xe5,0xd3,0x00,0x05,0x0a,0xfc]

v_mfma_f32_4x4x4bf16_1k v[0:3], v[0:1], v[2:3], v[2:5]
// GFX90A: v_mfma_f32_4x4x4bf16_1k v[0:3], v[0:1], v[2:3], v[2:5] ; encoding: [0x00,0x00,0xe5,0xd3,0x00,0x05,0x0a,0x04]

v_mfma_f32_4x4x4bf16_1k v[0:3], v[0:1], v[2:3], v[2:5] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_4x4x4bf16_1k v[0:3], v[0:1], v[2:3], v[2:5] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xe5,0xd3,0x00,0x05,0x0a,0xe4]

v_mfma_f32_4x4x4bf16_1k v[0:3], v[0:1], a[2:3], v[2:5]
// GFX90A: v_mfma_f32_4x4x4bf16_1k v[0:3], v[0:1], a[2:3], v[2:5] ; encoding: [0x00,0x00,0xe5,0xd3,0x00,0x05,0x0a,0x14]

v_mfma_f32_4x4x4bf16_1k v[0:3], v[0:1], a[2:3], v[2:5] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_4x4x4bf16_1k v[0:3], v[0:1], a[2:3], v[2:5] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xe5,0xd3,0x00,0x05,0x0a,0xf4]

v_mfma_f32_4x4x4bf16_1k v[0:3], a[0:1], v[2:3], v[2:5]
// GFX90A: v_mfma_f32_4x4x4bf16_1k v[0:3], a[0:1], v[2:3], v[2:5] ; encoding: [0x00,0x00,0xe5,0xd3,0x00,0x05,0x0a,0x0c]

v_mfma_f32_4x4x4bf16_1k v[0:3], a[0:1], v[2:3], v[2:5] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_4x4x4bf16_1k v[0:3], a[0:1], v[2:3], v[2:5] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xe5,0xd3,0x00,0x05,0x0a,0xec]

v_mfma_f32_4x4x4bf16_1k v[0:3], a[0:1], a[2:3], v[2:5]
// GFX90A: v_mfma_f32_4x4x4bf16_1k v[0:3], a[0:1], a[2:3], v[2:5] ; encoding: [0x00,0x00,0xe5,0xd3,0x00,0x05,0x0a,0x1c]

v_mfma_f32_4x4x4bf16_1k v[0:3], a[0:1], a[2:3], v[2:5] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_4x4x4bf16_1k v[0:3], a[0:1], a[2:3], v[2:5] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xe5,0xd3,0x00,0x05,0x0a,0xfc]

v_mfma_f32_4x4x4bf16_1k a[0:3], v[0:1], v[2:3], -2.0
// GFX90A: v_mfma_f32_4x4x4bf16_1k a[0:3], v[0:1], v[2:3], -2.0 ; encoding: [0x00,0x80,0xe5,0xd3,0x00,0x05,0xd6,0x03]

v_mfma_f32_4x4x4bf16_1k v[0:3], v[0:1], v[2:3], -2.0
// GFX90A: v_mfma_f32_4x4x4bf16_1k v[0:3], v[0:1], v[2:3], -2.0 ; encoding: [0x00,0x00,0xe5,0xd3,0x00,0x05,0xd6,0x03]

v_mfma_f32_4x4x4bf16_1k a[0:3], v[0:1], v[2:3], -2.0 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_4x4x4bf16_1k a[0:3], v[0:1], v[2:3], -2.0 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xe5,0xd3,0x00,0x05,0xd6,0xe3]

v_mfma_f32_4x4x4bf16_1k v[0:3], v[0:1], v[2:3], -2.0 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_4x4x4bf16_1k v[0:3], v[0:1], v[2:3], -2.0 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xe5,0xd3,0x00,0x05,0xd6,0xe3]

v_mfma_f32_4x4x4bf16_1k a[0:3], v[0:1], a[2:3], -2.0
// GFX90A: v_mfma_f32_4x4x4bf16_1k a[0:3], v[0:1], a[2:3], -2.0 ; encoding: [0x00,0x80,0xe5,0xd3,0x00,0x05,0xd6,0x13]

v_mfma_f32_4x4x4bf16_1k v[0:3], v[0:1], a[2:3], -2.0
// GFX90A: v_mfma_f32_4x4x4bf16_1k v[0:3], v[0:1], a[2:3], -2.0 ; encoding: [0x00,0x00,0xe5,0xd3,0x00,0x05,0xd6,0x13]

v_mfma_f32_4x4x4bf16_1k a[0:3], v[0:1], a[2:3], -2.0 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_4x4x4bf16_1k a[0:3], v[0:1], a[2:3], -2.0 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xe5,0xd3,0x00,0x05,0xd6,0xf3]

v_mfma_f32_4x4x4bf16_1k v[0:3], v[0:1], a[2:3], -2.0 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_4x4x4bf16_1k v[0:3], v[0:1], a[2:3], -2.0 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xe5,0xd3,0x00,0x05,0xd6,0xf3]

v_mfma_f32_4x4x4bf16_1k a[0:3], a[0:1], v[2:3], -2.0
// GFX90A: v_mfma_f32_4x4x4bf16_1k a[0:3], a[0:1], v[2:3], -2.0 ; encoding: [0x00,0x80,0xe5,0xd3,0x00,0x05,0xd6,0x0b]

v_mfma_f32_4x4x4bf16_1k v[0:3], a[0:1], v[2:3], -2.0
// GFX90A: v_mfma_f32_4x4x4bf16_1k v[0:3], a[0:1], v[2:3], -2.0 ; encoding: [0x00,0x00,0xe5,0xd3,0x00,0x05,0xd6,0x0b]

v_mfma_f32_4x4x4bf16_1k a[0:3], a[0:1], v[2:3], -2.0 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_4x4x4bf16_1k a[0:3], a[0:1], v[2:3], -2.0 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xe5,0xd3,0x00,0x05,0xd6,0xeb]

v_mfma_f32_4x4x4bf16_1k v[0:3], a[0:1], v[2:3], -2.0 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_4x4x4bf16_1k v[0:3], a[0:1], v[2:3], -2.0 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xe5,0xd3,0x00,0x05,0xd6,0xeb]

v_mfma_f32_4x4x4bf16_1k a[0:3], a[0:1], a[2:3], -2.0
// GFX90A: v_mfma_f32_4x4x4bf16_1k a[0:3], a[0:1], a[2:3], -2.0 ; encoding: [0x00,0x80,0xe5,0xd3,0x00,0x05,0xd6,0x1b]

v_mfma_f32_4x4x4bf16_1k v[0:3], a[0:1], a[2:3], -2.0
// GFX90A: v_mfma_f32_4x4x4bf16_1k v[0:3], a[0:1], a[2:3], -2.0 ; encoding: [0x00,0x00,0xe5,0xd3,0x00,0x05,0xd6,0x1b]

v_mfma_f32_4x4x4bf16_1k a[0:3], a[0:1], a[2:3], -2.0 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_4x4x4bf16_1k a[0:3], a[0:1], a[2:3], -2.0 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xe5,0xd3,0x00,0x05,0xd6,0xfb]

v_mfma_f32_4x4x4bf16_1k v[0:3], a[0:1], a[2:3], -2.0 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_4x4x4bf16_1k v[0:3], a[0:1], a[2:3], -2.0 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xe5,0xd3,0x00,0x05,0xd6,0xfb]

v_mfma_f32_32x32x8bf16_1k a[0:15], v[0:1], v[2:3], a[18:33]
// GFX90A: v_mfma_f32_32x32x8bf16_1k a[0:15], v[0:1], v[2:3], a[18:33] ; encoding: [0x00,0x80,0xe6,0xd3,0x00,0x05,0x4a,0x04]

v_mfma_f32_32x32x8bf16_1k a[0:15], v[0:1], v[2:3], a[18:33] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_32x32x8bf16_1k a[0:15], v[0:1], v[2:3], a[18:33] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xe6,0xd3,0x00,0x05,0x4a,0xe4]

v_mfma_f32_32x32x8bf16_1k a[0:15], v[0:1], a[2:3], a[18:33]
// GFX90A: v_mfma_f32_32x32x8bf16_1k a[0:15], v[0:1], a[2:3], a[18:33] ; encoding: [0x00,0x80,0xe6,0xd3,0x00,0x05,0x4a,0x14]

v_mfma_f32_32x32x8bf16_1k a[0:15], v[0:1], a[2:3], a[18:33] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_32x32x8bf16_1k a[0:15], v[0:1], a[2:3], a[18:33] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xe6,0xd3,0x00,0x05,0x4a,0xf4]

v_mfma_f32_32x32x8bf16_1k a[0:15], a[0:1], v[2:3], a[18:33]
// GFX90A: v_mfma_f32_32x32x8bf16_1k a[0:15], a[0:1], v[2:3], a[18:33] ; encoding: [0x00,0x80,0xe6,0xd3,0x00,0x05,0x4a,0x0c]

v_mfma_f32_32x32x8bf16_1k a[0:15], a[0:1], v[2:3], a[18:33] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_32x32x8bf16_1k a[0:15], a[0:1], v[2:3], a[18:33] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xe6,0xd3,0x00,0x05,0x4a,0xec]

v_mfma_f32_32x32x8bf16_1k a[0:15], a[0:1], a[2:3], a[18:33]
// GFX90A: v_mfma_f32_32x32x8bf16_1k a[0:15], a[0:1], a[2:3], a[18:33] ; encoding: [0x00,0x80,0xe6,0xd3,0x00,0x05,0x4a,0x1c]

v_mfma_f32_32x32x8bf16_1k a[0:15], a[0:1], a[2:3], a[18:33] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_32x32x8bf16_1k a[0:15], a[0:1], a[2:3], a[18:33] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xe6,0xd3,0x00,0x05,0x4a,0xfc]

v_mfma_f32_32x32x8bf16_1k v[0:15], v[0:1], v[2:3], v[18:33]
// GFX90A: v_mfma_f32_32x32x8bf16_1k v[0:15], v[0:1], v[2:3], v[18:33] ; encoding: [0x00,0x00,0xe6,0xd3,0x00,0x05,0x4a,0x04]

v_mfma_f32_32x32x8bf16_1k v[0:15], v[0:1], v[2:3], v[18:33] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_32x32x8bf16_1k v[0:15], v[0:1], v[2:3], v[18:33] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xe6,0xd3,0x00,0x05,0x4a,0xe4]

v_mfma_f32_32x32x8bf16_1k v[0:15], v[0:1], a[2:3], v[18:33]
// GFX90A: v_mfma_f32_32x32x8bf16_1k v[0:15], v[0:1], a[2:3], v[18:33] ; encoding: [0x00,0x00,0xe6,0xd3,0x00,0x05,0x4a,0x14]

v_mfma_f32_32x32x8bf16_1k v[0:15], v[0:1], a[2:3], v[18:33] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_32x32x8bf16_1k v[0:15], v[0:1], a[2:3], v[18:33] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xe6,0xd3,0x00,0x05,0x4a,0xf4]

v_mfma_f32_32x32x8bf16_1k v[0:15], a[0:1], v[2:3], v[18:33]
// GFX90A: v_mfma_f32_32x32x8bf16_1k v[0:15], a[0:1], v[2:3], v[18:33] ; encoding: [0x00,0x00,0xe6,0xd3,0x00,0x05,0x4a,0x0c]

v_mfma_f32_32x32x8bf16_1k v[0:15], a[0:1], v[2:3], v[18:33] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_32x32x8bf16_1k v[0:15], a[0:1], v[2:3], v[18:33] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xe6,0xd3,0x00,0x05,0x4a,0xec]

v_mfma_f32_32x32x8bf16_1k v[0:15], a[0:1], a[2:3], v[18:33]
// GFX90A: v_mfma_f32_32x32x8bf16_1k v[0:15], a[0:1], a[2:3], v[18:33] ; encoding: [0x00,0x00,0xe6,0xd3,0x00,0x05,0x4a,0x1c]

v_mfma_f32_32x32x8bf16_1k v[0:15], a[0:1], a[2:3], v[18:33] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_32x32x8bf16_1k v[0:15], a[0:1], a[2:3], v[18:33] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xe6,0xd3,0x00,0x05,0x4a,0xfc]

v_mfma_f32_32x32x8bf16_1k a[0:15], v[0:1], v[2:3], -2.0
// GFX90A: v_mfma_f32_32x32x8bf16_1k a[0:15], v[0:1], v[2:3], -2.0 ; encoding: [0x00,0x80,0xe6,0xd3,0x00,0x05,0xd6,0x03]

v_mfma_f32_32x32x8bf16_1k v[0:15], v[0:1], v[2:3], -2.0
// GFX90A: v_mfma_f32_32x32x8bf16_1k v[0:15], v[0:1], v[2:3], -2.0 ; encoding: [0x00,0x00,0xe6,0xd3,0x00,0x05,0xd6,0x03]

v_mfma_f32_32x32x8bf16_1k a[0:15], v[0:1], v[2:3], -2.0 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_32x32x8bf16_1k a[0:15], v[0:1], v[2:3], -2.0 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xe6,0xd3,0x00,0x05,0xd6,0xe3]

v_mfma_f32_32x32x8bf16_1k v[0:15], v[0:1], v[2:3], -2.0 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_32x32x8bf16_1k v[0:15], v[0:1], v[2:3], -2.0 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xe6,0xd3,0x00,0x05,0xd6,0xe3]

v_mfma_f32_32x32x8bf16_1k a[0:15], v[0:1], a[2:3], -2.0
// GFX90A: v_mfma_f32_32x32x8bf16_1k a[0:15], v[0:1], a[2:3], -2.0 ; encoding: [0x00,0x80,0xe6,0xd3,0x00,0x05,0xd6,0x13]

v_mfma_f32_32x32x8bf16_1k v[0:15], v[0:1], a[2:3], -2.0
// GFX90A: v_mfma_f32_32x32x8bf16_1k v[0:15], v[0:1], a[2:3], -2.0 ; encoding: [0x00,0x00,0xe6,0xd3,0x00,0x05,0xd6,0x13]

v_mfma_f32_32x32x8bf16_1k a[0:15], v[0:1], a[2:3], -2.0 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_32x32x8bf16_1k a[0:15], v[0:1], a[2:3], -2.0 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xe6,0xd3,0x00,0x05,0xd6,0xf3]

v_mfma_f32_32x32x8bf16_1k v[0:15], v[0:1], a[2:3], -2.0 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_32x32x8bf16_1k v[0:15], v[0:1], a[2:3], -2.0 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xe6,0xd3,0x00,0x05,0xd6,0xf3]

v_mfma_f32_32x32x8bf16_1k a[0:15], a[0:1], v[2:3], -2.0
// GFX90A: v_mfma_f32_32x32x8bf16_1k a[0:15], a[0:1], v[2:3], -2.0 ; encoding: [0x00,0x80,0xe6,0xd3,0x00,0x05,0xd6,0x0b]

v_mfma_f32_32x32x8bf16_1k v[0:15], a[0:1], v[2:3], -2.0
// GFX90A: v_mfma_f32_32x32x8bf16_1k v[0:15], a[0:1], v[2:3], -2.0 ; encoding: [0x00,0x00,0xe6,0xd3,0x00,0x05,0xd6,0x0b]

v_mfma_f32_32x32x8bf16_1k a[0:15], a[0:1], v[2:3], -2.0 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_32x32x8bf16_1k a[0:15], a[0:1], v[2:3], -2.0 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xe6,0xd3,0x00,0x05,0xd6,0xeb]

v_mfma_f32_32x32x8bf16_1k v[0:15], a[0:1], v[2:3], -2.0 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_32x32x8bf16_1k v[0:15], a[0:1], v[2:3], -2.0 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xe6,0xd3,0x00,0x05,0xd6,0xeb]

v_mfma_f32_32x32x8bf16_1k a[0:15], a[0:1], a[2:3], -2.0
// GFX90A: v_mfma_f32_32x32x8bf16_1k a[0:15], a[0:1], a[2:3], -2.0 ; encoding: [0x00,0x80,0xe6,0xd3,0x00,0x05,0xd6,0x1b]

v_mfma_f32_32x32x8bf16_1k v[0:15], a[0:1], a[2:3], -2.0
// GFX90A: v_mfma_f32_32x32x8bf16_1k v[0:15], a[0:1], a[2:3], -2.0 ; encoding: [0x00,0x00,0xe6,0xd3,0x00,0x05,0xd6,0x1b]

v_mfma_f32_32x32x8bf16_1k a[0:15], a[0:1], a[2:3], -2.0 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_32x32x8bf16_1k a[0:15], a[0:1], a[2:3], -2.0 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xe6,0xd3,0x00,0x05,0xd6,0xfb]

v_mfma_f32_32x32x8bf16_1k v[0:15], a[0:1], a[2:3], -2.0 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_32x32x8bf16_1k v[0:15], a[0:1], a[2:3], -2.0 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xe6,0xd3,0x00,0x05,0xd6,0xfb]

v_mfma_f32_16x16x16bf16_1k a[0:3], v[0:1], v[2:3], a[2:5]
// GFX90A: v_mfma_f32_16x16x16bf16_1k a[0:3], v[0:1], v[2:3], a[2:5] ; encoding: [0x00,0x80,0xe7,0xd3,0x00,0x05,0x0a,0x04]

v_mfma_f32_16x16x16bf16_1k a[0:3], v[0:1], v[2:3], a[2:5] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_16x16x16bf16_1k a[0:3], v[0:1], v[2:3], a[2:5] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xe7,0xd3,0x00,0x05,0x0a,0xe4]

v_mfma_f32_16x16x16bf16_1k a[0:3], v[0:1], a[2:3], a[2:5]
// GFX90A: v_mfma_f32_16x16x16bf16_1k a[0:3], v[0:1], a[2:3], a[2:5] ; encoding: [0x00,0x80,0xe7,0xd3,0x00,0x05,0x0a,0x14]

v_mfma_f32_16x16x16bf16_1k a[0:3], v[0:1], a[2:3], a[2:5] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_16x16x16bf16_1k a[0:3], v[0:1], a[2:3], a[2:5] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xe7,0xd3,0x00,0x05,0x0a,0xf4]

v_mfma_f32_16x16x16bf16_1k a[0:3], a[0:1], v[2:3], a[2:5]
// GFX90A: v_mfma_f32_16x16x16bf16_1k a[0:3], a[0:1], v[2:3], a[2:5] ; encoding: [0x00,0x80,0xe7,0xd3,0x00,0x05,0x0a,0x0c]

v_mfma_f32_16x16x16bf16_1k a[0:3], a[0:1], v[2:3], a[2:5] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_16x16x16bf16_1k a[0:3], a[0:1], v[2:3], a[2:5] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xe7,0xd3,0x00,0x05,0x0a,0xec]

v_mfma_f32_16x16x16bf16_1k a[0:3], a[0:1], a[2:3], a[2:5]
// GFX90A: v_mfma_f32_16x16x16bf16_1k a[0:3], a[0:1], a[2:3], a[2:5] ; encoding: [0x00,0x80,0xe7,0xd3,0x00,0x05,0x0a,0x1c]

v_mfma_f32_16x16x16bf16_1k a[0:3], a[0:1], a[2:3], a[2:5] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_16x16x16bf16_1k a[0:3], a[0:1], a[2:3], a[2:5] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xe7,0xd3,0x00,0x05,0x0a,0xfc]

v_mfma_f32_16x16x16bf16_1k v[0:3], v[0:1], v[2:3], v[2:5]
// GFX90A: v_mfma_f32_16x16x16bf16_1k v[0:3], v[0:1], v[2:3], v[2:5] ; encoding: [0x00,0x00,0xe7,0xd3,0x00,0x05,0x0a,0x04]

v_mfma_f32_16x16x16bf16_1k v[0:3], v[0:1], v[2:3], v[2:5] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_16x16x16bf16_1k v[0:3], v[0:1], v[2:3], v[2:5] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xe7,0xd3,0x00,0x05,0x0a,0xe4]

v_mfma_f32_16x16x16bf16_1k v[0:3], v[0:1], a[2:3], v[2:5]
// GFX90A: v_mfma_f32_16x16x16bf16_1k v[0:3], v[0:1], a[2:3], v[2:5] ; encoding: [0x00,0x00,0xe7,0xd3,0x00,0x05,0x0a,0x14]

v_mfma_f32_16x16x16bf16_1k v[0:3], v[0:1], a[2:3], v[2:5] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_16x16x16bf16_1k v[0:3], v[0:1], a[2:3], v[2:5] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xe7,0xd3,0x00,0x05,0x0a,0xf4]

v_mfma_f32_16x16x16bf16_1k v[0:3], a[0:1], v[2:3], v[2:5]
// GFX90A: v_mfma_f32_16x16x16bf16_1k v[0:3], a[0:1], v[2:3], v[2:5] ; encoding: [0x00,0x00,0xe7,0xd3,0x00,0x05,0x0a,0x0c]

v_mfma_f32_16x16x16bf16_1k v[0:3], a[0:1], v[2:3], v[2:5] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_16x16x16bf16_1k v[0:3], a[0:1], v[2:3], v[2:5] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xe7,0xd3,0x00,0x05,0x0a,0xec]

v_mfma_f32_16x16x16bf16_1k v[0:3], a[0:1], a[2:3], v[2:5]
// GFX90A: v_mfma_f32_16x16x16bf16_1k v[0:3], a[0:1], a[2:3], v[2:5] ; encoding: [0x00,0x00,0xe7,0xd3,0x00,0x05,0x0a,0x1c]

v_mfma_f32_16x16x16bf16_1k v[0:3], a[0:1], a[2:3], v[2:5] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_16x16x16bf16_1k v[0:3], a[0:1], a[2:3], v[2:5] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xe7,0xd3,0x00,0x05,0x0a,0xfc]

v_mfma_f32_16x16x16bf16_1k a[0:3], v[0:1], v[2:3], -2.0
// GFX90A: v_mfma_f32_16x16x16bf16_1k a[0:3], v[0:1], v[2:3], -2.0 ; encoding: [0x00,0x80,0xe7,0xd3,0x00,0x05,0xd6,0x03]

v_mfma_f32_16x16x16bf16_1k v[0:3], v[0:1], v[2:3], -2.0
// GFX90A: v_mfma_f32_16x16x16bf16_1k v[0:3], v[0:1], v[2:3], -2.0 ; encoding: [0x00,0x00,0xe7,0xd3,0x00,0x05,0xd6,0x03]

v_mfma_f32_16x16x16bf16_1k a[0:3], v[0:1], v[2:3], -2.0 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_16x16x16bf16_1k a[0:3], v[0:1], v[2:3], -2.0 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xe7,0xd3,0x00,0x05,0xd6,0xe3]

v_mfma_f32_16x16x16bf16_1k v[0:3], v[0:1], v[2:3], -2.0 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_16x16x16bf16_1k v[0:3], v[0:1], v[2:3], -2.0 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xe7,0xd3,0x00,0x05,0xd6,0xe3]

v_mfma_f32_16x16x16bf16_1k a[0:3], v[0:1], a[2:3], -2.0
// GFX90A: v_mfma_f32_16x16x16bf16_1k a[0:3], v[0:1], a[2:3], -2.0 ; encoding: [0x00,0x80,0xe7,0xd3,0x00,0x05,0xd6,0x13]

v_mfma_f32_16x16x16bf16_1k v[0:3], v[0:1], a[2:3], -2.0
// GFX90A: v_mfma_f32_16x16x16bf16_1k v[0:3], v[0:1], a[2:3], -2.0 ; encoding: [0x00,0x00,0xe7,0xd3,0x00,0x05,0xd6,0x13]

v_mfma_f32_16x16x16bf16_1k a[0:3], v[0:1], a[2:3], -2.0 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_16x16x16bf16_1k a[0:3], v[0:1], a[2:3], -2.0 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xe7,0xd3,0x00,0x05,0xd6,0xf3]

v_mfma_f32_16x16x16bf16_1k v[0:3], v[0:1], a[2:3], -2.0 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_16x16x16bf16_1k v[0:3], v[0:1], a[2:3], -2.0 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xe7,0xd3,0x00,0x05,0xd6,0xf3]

v_mfma_f32_16x16x16bf16_1k a[0:3], a[0:1], v[2:3], -2.0
// GFX90A: v_mfma_f32_16x16x16bf16_1k a[0:3], a[0:1], v[2:3], -2.0 ; encoding: [0x00,0x80,0xe7,0xd3,0x00,0x05,0xd6,0x0b]

v_mfma_f32_16x16x16bf16_1k v[0:3], a[0:1], v[2:3], -2.0
// GFX90A: v_mfma_f32_16x16x16bf16_1k v[0:3], a[0:1], v[2:3], -2.0 ; encoding: [0x00,0x00,0xe7,0xd3,0x00,0x05,0xd6,0x0b]

v_mfma_f32_16x16x16bf16_1k a[0:3], a[0:1], v[2:3], -2.0 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_16x16x16bf16_1k a[0:3], a[0:1], v[2:3], -2.0 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xe7,0xd3,0x00,0x05,0xd6,0xeb]

v_mfma_f32_16x16x16bf16_1k v[0:3], a[0:1], v[2:3], -2.0 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_16x16x16bf16_1k v[0:3], a[0:1], v[2:3], -2.0 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xe7,0xd3,0x00,0x05,0xd6,0xeb]

v_mfma_f32_16x16x16bf16_1k a[0:3], a[0:1], a[2:3], -2.0
// GFX90A: v_mfma_f32_16x16x16bf16_1k a[0:3], a[0:1], a[2:3], -2.0 ; encoding: [0x00,0x80,0xe7,0xd3,0x00,0x05,0xd6,0x1b]

v_mfma_f32_16x16x16bf16_1k v[0:3], a[0:1], a[2:3], -2.0
// GFX90A: v_mfma_f32_16x16x16bf16_1k v[0:3], a[0:1], a[2:3], -2.0 ; encoding: [0x00,0x00,0xe7,0xd3,0x00,0x05,0xd6,0x1b]

v_mfma_f32_16x16x16bf16_1k a[0:3], a[0:1], a[2:3], -2.0 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_16x16x16bf16_1k a[0:3], a[0:1], a[2:3], -2.0 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xe7,0xd3,0x00,0x05,0xd6,0xfb]

v_mfma_f32_16x16x16bf16_1k v[0:3], a[0:1], a[2:3], -2.0 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f32_16x16x16bf16_1k v[0:3], a[0:1], a[2:3], -2.0 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xe7,0xd3,0x00,0x05,0xd6,0xfb]

v_mfma_f64_16x16x4f64 v[0:7], v[0:1], v[2:3], v[10:17]
// GFX90A: v_mfma_f64_16x16x4f64 v[0:7], v[0:1], v[2:3], v[10:17] ; encoding: [0x00,0x00,0xee,0xd3,0x00,0x05,0x2a,0x04]

v_mfma_f64_16x16x4f64 v[0:7], v[0:1], v[2:3], v[10:17] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f64_16x16x4f64 v[0:7], v[0:1], v[2:3], v[10:17] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xee,0xd3,0x00,0x05,0x2a,0xe4]

v_mfma_f64_16x16x4f64 v[0:7], v[0:1], v[2:3], -2.0
// GFX90A: v_mfma_f64_16x16x4f64 v[0:7], v[0:1], v[2:3], -2.0 ; encoding: [0x00,0x00,0xee,0xd3,0x00,0x05,0xd6,0x03]

v_mfma_f64_16x16x4f64 v[0:7], v[0:1], v[2:3], -2.0 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f64_16x16x4f64 v[0:7], v[0:1], v[2:3], -2.0 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xee,0xd3,0x00,0x05,0xd6,0xe3]

v_mfma_f64_4x4x4f64 v[0:1], v[0:1], v[2:3], v[2:3]
// GFX90A: v_mfma_f64_4x4x4f64 v[0:1], v[0:1], v[2:3], v[2:3] ; encoding: [0x00,0x00,0xef,0xd3,0x00,0x05,0x0a,0x04]

v_mfma_f64_4x4x4f64 v[0:1], v[0:1], v[2:3], v[2:3] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f64_4x4x4f64 v[0:1], v[0:1], v[2:3], v[2:3] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xef,0xd3,0x00,0x05,0x0a,0xe4]

v_mfma_f64_4x4x4f64 v[0:1], v[0:1], v[2:3], -2.0
// GFX90A: v_mfma_f64_4x4x4f64 v[0:1], v[0:1], v[2:3], -2.0 ; encoding: [0x00,0x00,0xef,0xd3,0x00,0x05,0xd6,0x03]

v_mfma_f64_4x4x4f64 v[0:1], v[0:1], v[2:3], -2.0 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f64_4x4x4f64 v[0:1], v[0:1], v[2:3], -2.0 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xef,0xd3,0x00,0x05,0xd6,0xe3]

v_mfma_f64_16x16x4f64 v[0:7], a[0:1], v[2:3], v[10:17]
// GFX90A: v_mfma_f64_16x16x4f64 v[0:7], a[0:1], v[2:3], v[10:17] ; encoding: [0x00,0x00,0xee,0xd3,0x00,0x05,0x2a,0x0c]

v_mfma_f64_16x16x4f64 v[0:7], v[0:1], a[2:3], v[10:17] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f64_16x16x4f64 v[0:7], v[0:1], a[2:3], v[10:17] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xee,0xd3,0x00,0x05,0x2a,0xf4]

v_mfma_f64_16x16x4f64 v[0:7], a[0:1], a[2:3], -2.0
// GFX90A: v_mfma_f64_16x16x4f64 v[0:7], a[0:1], a[2:3], -2.0 ; encoding: [0x00,0x00,0xee,0xd3,0x00,0x05,0xd6,0x1b]

v_mfma_f64_4x4x4f64 v[0:1], a[0:1], v[2:3], v[2:3]
// GFX90A: v_mfma_f64_4x4x4f64 v[0:1], a[0:1], v[2:3], v[2:3] ; encoding: [0x00,0x00,0xef,0xd3,0x00,0x05,0x0a,0x0c]

v_mfma_f64_4x4x4f64 v[0:1], v[0:1], a[2:3], v[2:3] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f64_4x4x4f64 v[0:1], v[0:1], a[2:3], v[2:3] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x13,0xef,0xd3,0x00,0x05,0x0a,0xf4]

v_mfma_f64_4x4x4f64 v[0:1], a[0:1], a[2:3], -2.0
// GFX90A: v_mfma_f64_4x4x4f64 v[0:1], a[0:1], a[2:3], -2.0 ; encoding: [0x00,0x00,0xef,0xd3,0x00,0x05,0xd6,0x1b]

v_mfma_f64_16x16x4f64 a[0:7], v[0:1], v[2:3], a[10:17]
// GFX90A: v_mfma_f64_16x16x4f64 a[0:7], v[0:1], v[2:3], a[10:17] ; encoding: [0x00,0x80,0xee,0xd3,0x00,0x05,0x2a,0x04]

v_mfma_f64_16x16x4f64 a[0:7], v[0:1], v[2:3], a[10:17] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f64_16x16x4f64 a[0:7], v[0:1], v[2:3], a[10:17] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xee,0xd3,0x00,0x05,0x2a,0xe4]

v_mfma_f64_16x16x4f64 a[0:7], v[0:1], v[2:3], -2.0
// GFX90A: v_mfma_f64_16x16x4f64 a[0:7], v[0:1], v[2:3], -2.0 ; encoding: [0x00,0x80,0xee,0xd3,0x00,0x05,0xd6,0x03]

v_mfma_f64_16x16x4f64 a[0:7], v[0:1], v[2:3], -2.0 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f64_16x16x4f64 a[0:7], v[0:1], v[2:3], -2.0 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xee,0xd3,0x00,0x05,0xd6,0xe3]

v_mfma_f64_4x4x4f64 a[0:1], v[0:1], v[2:3], a[2:3]
// GFX90A: v_mfma_f64_4x4x4f64 a[0:1], v[0:1], v[2:3], a[2:3] ; encoding: [0x00,0x80,0xef,0xd3,0x00,0x05,0x0a,0x04]

v_mfma_f64_4x4x4f64 a[0:1], v[0:1], v[2:3], a[2:3] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f64_4x4x4f64 a[0:1], v[0:1], v[2:3], a[2:3] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xef,0xd3,0x00,0x05,0x0a,0xe4]

v_mfma_f64_4x4x4f64 a[0:1], v[0:1], v[2:3], -2.0
// GFX90A: v_mfma_f64_4x4x4f64 a[0:1], v[0:1], v[2:3], -2.0 ; encoding: [0x00,0x80,0xef,0xd3,0x00,0x05,0xd6,0x03]

v_mfma_f64_4x4x4f64 a[0:1], v[0:1], v[2:3], 0
// GFX90A: v_mfma_f64_4x4x4f64 a[0:1], v[0:1], v[2:3], 0 ; encoding: [0x00,0x80,0xef,0xd3,0x00,0x05,0x02,0x02]

v_mfma_f64_4x4x4f64 a[0:1], v[0:1], v[2:3], -2.0 cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f64_4x4x4f64 a[0:1], v[0:1], v[2:3], -2.0 cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xef,0xd3,0x00,0x05,0xd6,0xe3]

v_mfma_f64_16x16x4f64 a[0:7], a[0:1], v[2:3], a[10:17]
// GFX90A: v_mfma_f64_16x16x4f64 a[0:7], a[0:1], v[2:3], a[10:17] ; encoding: [0x00,0x80,0xee,0xd3,0x00,0x05,0x2a,0x0c]

v_mfma_f64_16x16x4f64 a[0:7], v[0:1], a[2:3], a[10:17] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f64_16x16x4f64 a[0:7], v[0:1], a[2:3], a[10:17] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xee,0xd3,0x00,0x05,0x2a,0xf4]

v_mfma_f64_16x16x4f64 a[0:7], a[0:1], a[2:3], -2.0
// GFX90A: v_mfma_f64_16x16x4f64 a[0:7], a[0:1], a[2:3], -2.0 ; encoding: [0x00,0x80,0xee,0xd3,0x00,0x05,0xd6,0x1b]

v_mfma_f64_4x4x4f64 a[0:1], a[0:1], v[2:3], a[2:3]
// GFX90A: v_mfma_f64_4x4x4f64 a[0:1], a[0:1], v[2:3], a[2:3] ; encoding: [0x00,0x80,0xef,0xd3,0x00,0x05,0x0a,0x0c]

v_mfma_f64_4x4x4f64 a[0:1], v[0:1], a[2:3], a[2:3] cbsz:3 abid:2 blgp:7
// GFX90A: v_mfma_f64_4x4x4f64 a[0:1], v[0:1], a[2:3], a[2:3] cbsz:3 abid:2 blgp:7 ; encoding: [0x00,0x93,0xef,0xd3,0x00,0x05,0x0a,0xf4]

v_mfma_f64_4x4x4f64 a[0:1], a[0:1], a[2:3], -2.0
// GFX90A: v_mfma_f64_4x4x4f64 a[0:1], a[0:1], a[2:3], -2.0 ; encoding: [0x00,0x80,0xef,0xd3,0x00,0x05,0xd6,0x1b]

v_mfma_f64_16x16x4f64 a[0:7], a[0:1], a[2:3], 0
// GFX90A: v_mfma_f64_16x16x4f64 a[0:7], a[0:1], a[2:3], 0 ; encoding: [0x00,0x80,0xee,0xd3,0x00,0x05,0x02,0x1a]
