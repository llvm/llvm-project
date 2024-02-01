// RUN: llvm-mc -triple=amdgcn -mcpu=gfx950 -show-encoding %s | FileCheck -check-prefix=GFX950 %s
// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx940 %s 2>&1 | FileCheck -check-prefix=ERR %s
// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx90a %s 2>&1 | FileCheck -check-prefix=ERR %s

//===----------------------------------------------------------------------===//
// MFMA opcodes.
//===----------------------------------------------------------------------===//


//===----------------------------------------------------------------------===//
// v_mfma_f32_16x16x32_f16
//===----------------------------------------------------------------------===//

// GFX950: v_mfma_f32_16x16x32_f16 v[0:3], v[0:3], v[0:3], v[0:3] ; encoding: [0x00,0x00,0xd4,0xd3,0x00,0x01,0x02,0x04]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_mfma_f32_16x16x32_f16 v[0:3], v[0:3], v[0:3], v[0:3]

// GFX950: v_mfma_f32_16x16x32_f16 v[0:3], v[0:3], v[0:3], v[0:3] ; encoding: [0x00,0x00,0xd4,0xd3,0x00,0x01,0x02,0x04]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_mfma_f32_16x16x32f16 v[0:3], v[0:3], v[0:3], v[0:3]

// GFX950: v_mfma_f32_16x16x32_f16 a[0:3], a[0:3], a[0:3], a[0:3] ; encoding: [0x00,0x80,0xd4,0xd3,0x00,0x01,0x02,0x1c]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_mfma_f32_16x16x32_f16 a[0:3], a[0:3], a[0:3], a[0:3]

// GFX950: v_mfma_f32_16x16x32_f16 a[0:3], a[0:3], a[0:3], a[0:3] ; encoding: [0x00,0x80,0xd4,0xd3,0x00,0x01,0x02,0x1c]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_mfma_f32_16x16x32f16 a[0:3], a[0:3], a[0:3], a[0:3]

// GFX950: v_mfma_f32_16x16x32_f16 v[0:3], a[0:3], v[0:3], 1.0 ; encoding: [0x00,0x00,0xd4,0xd3,0x00,0x01,0xca,0x0b]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_mfma_f32_16x16x32_f16 v[0:3], a[0:3], v[0:3], 1.0

// GFX950: v_mfma_f32_16x16x32_f16 a[0:3], v[0:3], a[0:3], 1.0 ; encoding: [0x00,0x80,0xd4,0xd3,0x00,0x01,0xca,0x13]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_mfma_f32_16x16x32_f16 a[0:3], v[0:3], a[0:3], 1.0

// GFX950: v_mfma_f32_16x16x32_f16 v[0:3], v[0:3], v[0:3], v[0:3] blgp:5 ; encoding: [0x00,0x00,0xd4,0xd3,0x00,0x01,0x02,0xa4]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_mfma_f32_16x16x32_f16 v[0:3], v[0:3], v[0:3], v[0:3] blgp:5

// GFX950: v_mfma_f32_16x16x32_f16 a[0:3], a[0:3], a[0:3], a[0:3] blgp:1 ; encoding: [0x00,0x80,0xd4,0xd3,0x00,0x01,0x02,0x3c]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_mfma_f32_16x16x32_f16 a[0:3], a[0:3], a[0:3], a[0:3] blgp:1

// GFX950: v_mfma_f32_16x16x32_f16 a[0:3], a[0:3], a[0:3], a[0:3] cbsz:3 ; encoding: [0x00,0x83,0xd4,0xd3,0x00,0x01,0x02,0x1c]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_mfma_f32_16x16x32_f16 a[0:3], a[0:3], a[0:3], a[0:3] cbsz:3

// GFX950:  v_mfma_f32_16x16x32_f16 a[0:3], a[0:3], a[0:3], a[0:3] abid:1 ; encoding: [0x00,0x88,0xd4,0xd3,0x00,0x01,0x02,0x1c]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_mfma_f32_16x16x32_f16 a[0:3], a[0:3], a[0:3], a[0:3] abid:1

// GFX950:  v_mfma_f32_16x16x32_f16 a[0:3], a[0:3], a[0:3], a[0:3] cbsz:3 abid:1 ; encoding: [0x00,0x8b,0xd4,0xd3,0x00,0x01,0x02,0x1c]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_mfma_f32_16x16x32_f16 a[0:3], a[0:3], a[0:3], a[0:3] cbsz:3 abid:1

// GFX950: v_mfma_f32_16x16x32_f16 a[0:3], v[0:3], v[0:3], a[4:7] ; encoding: [0x00,0x80,0xd4,0xd3,0x00,0x01,0x12,0x04]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_mfma_f32_16x16x32_f16 a[0:3], v[0:3], v[0:3], a[4:7]

// GFX950: v_mfma_f32_16x16x32_f16 v[0:3], a[0:3], a[0:3], v[4:7] ; encoding: [0x00,0x00,0xd4,0xd3,0x00,0x01,0x12,0x1c]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_mfma_f32_16x16x32_f16 v[0:3], a[0:3], a[0:3], v[4:7]

//===----------------------------------------------------------------------===//
// v_mfma_f32_32x32x16_f16
//===----------------------------------------------------------------------===//

// GFX950: v_mfma_f32_32x32x16_f16 v[0:15], v[0:3], v[0:3], v[0:15] ; encoding: [0x00,0x00,0xd5,0xd3,0x00,0x01,0x02,0x04]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_mfma_f32_32x32x16_f16 v[0:15], v[0:3], v[0:3], v[0:15]

// GFX950: v_mfma_f32_32x32x16_f16 a[0:15], a[0:3], a[0:3], a[0:15] ; encoding: [0x00,0x80,0xd5,0xd3,0x00,0x01,0x02,0x1c]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_mfma_f32_32x32x16_f16 a[0:15], a[0:3], a[0:3], a[0:15]

// GFX950:  v_mfma_f32_32x32x16_f16 v[0:15], v[0:3], v[0:3], v[0:15] ; encoding: [0x00,0x00,0xd5,0xd3,0x00,0x01,0x02,0x04]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_mfma_f32_32x32x16f16 v[0:15], v[0:3], v[0:3], v[0:15]

// GFX950: v_mfma_f32_32x32x16_f16 a[0:15], a[0:3], a[0:3], a[0:15] ; encoding: [0x00,0x80,0xd5,0xd3,0x00,0x01,0x02,0x1c]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_mfma_f32_32x32x16f16 a[0:15], a[0:3], a[0:3], a[0:15]

// GFX950: v_mfma_f32_32x32x16_f16 v[0:15], v[0:3], v[0:3], 1.0 ; encoding: [0x00,0x00,0xd5,0xd3,0x00,0x01,0xca,0x03]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_mfma_f32_32x32x16_f16 v[0:15], v[0:3], v[0:3], 1.0

// GFX950: v_mfma_f32_32x32x16_f16 a[0:15], a[0:3], a[0:3], 1.0 ; encoding: [0x00,0x80,0xd5,0xd3,0x00,0x01,0xca,0x1b]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_mfma_f32_32x32x16_f16 a[0:15], a[0:3], a[0:3], 1.0

// GFX950:  v_mfma_f32_32x32x16_f16 v[0:15], v[0:3], v[0:3], v[0:15] blgp:5 ; encoding: [0x00,0x00,0xd5,0xd3,0x00,0x01,0x02,0xa4]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_mfma_f32_32x32x16_f16 v[0:15], v[0:3], v[0:3], v[0:15] blgp:5

// GFX950: v_mfma_f32_32x32x16_f16 a[0:15], a[0:3], a[0:3], a[0:15] blgp:2 ; encoding: [0x00,0x80,0xd5,0xd3,0x00,0x01,0x02,0x5c]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_mfma_f32_32x32x16_f16 a[0:15], a[0:3], a[0:3], a[0:15] blgp:2

// GFX950: v_mfma_f32_32x32x16_f16 v[0:15], v[0:3], v[0:3], v[0:15] cbsz:3 ; encoding: [0x00,0x03,0xd5,0xd3,0x00,0x01,0x02,0x04]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_mfma_f32_32x32x16_f16 v[0:15], v[0:3], v[0:3], v[0:15] cbsz:3

// GFX950: v_mfma_f32_32x32x16_f16 v[0:15], v[0:3], v[0:3], v[0:15] abid:1 ; encoding: [0x00,0x08,0xd5,0xd3,0x00,0x01,0x02,0x04]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_mfma_f32_32x32x16_f16 v[0:15], v[0:3], v[0:3], v[0:15] abid:1

// GFX950: v_mfma_f32_32x32x16_f16 a[0:15], a[0:3], a[0:3], a[0:15] cbsz:3 abid:1 ; encoding: [0x00,0x8b,0xd5,0xd3,0x00,0x01,0x02,0x1c]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_mfma_f32_32x32x16_f16 a[0:15], a[0:3], a[0:3], a[0:15] cbsz:3 abid:1

//===----------------------------------------------------------------------===//
// v_mfma_f32_32x32x16_bf16
//===----------------------------------------------------------------------===//

// GFX950: v_mfma_f32_32x32x16_bf16 v[0:15], v[0:3], v[0:3], v[0:15] ; encoding: [0x00,0x00,0xb7,0xd3,0x00,0x01,0x02,0x04]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_mfma_f32_32x32x16_bf16 v[0:15], v[0:3], v[0:3], v[0:15]

// GFX950: v_mfma_f32_32x32x16_bf16 a[0:15], a[0:3], a[0:3], a[0:15] ; encoding: [0x00,0x80,0xb7,0xd3,0x00,0x01,0x02,0x1c]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_mfma_f32_32x32x16_bf16 a[0:15], a[0:3], a[0:3], a[0:15]

// GFX950:  v_mfma_f32_32x32x16_bf16 v[0:15], v[0:3], v[0:3], v[0:15] ; encoding: [0x00,0x00,0xb7,0xd3,0x00,0x01,0x02,0x04]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_mfma_f32_32x32x16bf16 v[0:15], v[0:3], v[0:3], v[0:15]

// GFX950: v_mfma_f32_32x32x16_bf16 a[0:15], a[0:3], a[0:3], a[0:15] ; encoding: [0x00,0x80,0xb7,0xd3,0x00,0x01,0x02,0x1c]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_mfma_f32_32x32x16bf16 a[0:15], a[0:3], a[0:3], a[0:15]

// GFX950: v_mfma_f32_32x32x16_bf16 v[0:15], v[0:3], v[0:3], 1.0 ; encoding: [0x00,0x00,0xb7,0xd3,0x00,0x01,0xca,0x03]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_mfma_f32_32x32x16_bf16 v[0:15], v[0:3], v[0:3], 1.0

// GFX950: v_mfma_f32_32x32x16_bf16 a[0:15], a[0:3], a[0:3], 1.0 ; encoding: [0x00,0x80,0xb7,0xd3,0x00,0x01,0xca,0x1b]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_mfma_f32_32x32x16_bf16 a[0:15], a[0:3], a[0:3], 1.0

// GFX950:  v_mfma_f32_32x32x16_bf16 v[0:15], v[0:3], v[0:3], v[0:15] blgp:5 ; encoding: [0x00,0x00,0xb7,0xd3,0x00,0x01,0x02,0xa4]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_mfma_f32_32x32x16_bf16 v[0:15], v[0:3], v[0:3], v[0:15] blgp:5

// GFX950: v_mfma_f32_32x32x16_bf16 a[0:15], a[0:3], a[0:3], a[0:15] blgp:2 ; encoding: [0x00,0x80,0xb7,0xd3,0x00,0x01,0x02,0x5c]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_mfma_f32_32x32x16_bf16 a[0:15], a[0:3], a[0:3], a[0:15] blgp:2

// GFX950: v_mfma_f32_32x32x16_bf16 v[0:15], v[0:3], v[0:3], v[0:15] cbsz:3 ; encoding: [0x00,0x03,0xb7,0xd3,0x00,0x01,0x02,0x04]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_mfma_f32_32x32x16_bf16 v[0:15], v[0:3], v[0:3], v[0:15] cbsz:3

// GFX950: v_mfma_f32_32x32x16_bf16 v[0:15], v[0:3], v[0:3], v[0:15] abid:1 ; encoding: [0x00,0x08,0xb7,0xd3,0x00,0x01,0x02,0x04]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_mfma_f32_32x32x16_bf16 v[0:15], v[0:3], v[0:3], v[0:15] abid:1

// GFX950: v_mfma_f32_32x32x16_bf16 a[0:15], a[0:3], a[0:3], a[0:15] cbsz:3 abid:1 ; encoding: [0x00,0x8b,0xb7,0xd3,0x00,0x01,0x02,0x1c]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_mfma_f32_32x32x16_bf16 a[0:15], a[0:3], a[0:3], a[0:15] cbsz:3 abid:1

//===----------------------------------------------------------------------===//
// v_mfma_ld_scale_b32
//===----------------------------------------------------------------------===//

// GFX950: v_mfma_ld_scale_b32 v0, 64             ; encoding: [0x00,0x40,0xac,0xd3,0x00,0x81,0x01,0x18]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_mfma_ld_scale_b32 v0, 64

// GFX950: v_mfma_ld_scale_b32 64, v0             ; encoding: [0x00,0x40,0xac,0xd3,0xc0,0x00,0x02,0x18]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_mfma_ld_scale_b32 64, v0

// GFX950: v_mfma_ld_scale_b32 64, 64             ; encoding: [0x00,0x40,0xac,0xd3,0xc0,0x80,0x01,0x18]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_mfma_ld_scale_b32 64, 64

// GFX950: v_mfma_ld_scale_b32 s0, s0             ; encoding: [0x00,0x40,0xac,0xd3,0x00,0x00,0x00,0x18]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_mfma_ld_scale_b32 s0, s0

// GFX950: v_mfma_ld_scale_b32 s0, v0             ; encoding: [0x00,0x40,0xac,0xd3,0x00,0x00,0x02,0x18]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_mfma_ld_scale_b32 s0, v0

// GFX950: v_mfma_ld_scale_b32 v0, s0             ; encoding: [0x00,0x40,0xac,0xd3,0x00,0x01,0x00,0x18]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_mfma_ld_scale_b32 v0, s0

// GFX950: v_mfma_ld_scale_b32 vcc_lo, vcc_lo     ; encoding: [0x00,0x40,0xac,0xd3,0x6a,0xd4,0x00,0x18]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_mfma_ld_scale_b32 vcc_lo, vcc_lo

// GFX950: v_mfma_ld_scale_b32 m0, m0             ; encoding: [0x00,0x40,0xac,0xd3,0x7c,0xf8,0x00,0x18]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_mfma_ld_scale_b32 m0, m0

// GFX950: v_mfma_ld_scale_b32 src_vccz, src_vccz ; encoding: [0x00,0x40,0xac,0xd3,0xfb,0xf6,0x01,0x18]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_mfma_ld_scale_b32 vccz, vccz

// GFX950: v_mfma_ld_scale_b32 src_execz, src_execz ; encoding: [0x00,0x40,0xac,0xd3,0xfc,0xf8,0x01,0x18]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_mfma_ld_scale_b32 execz, execz

// GFX950:  v_mfma_ld_scale_b32 v0, v0 ; encoding: [0x00,0x40,0xac,0xd3,0x00,0x01,0x02,0x18]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_mfma_ld_scale_b32 v0, v0

// GFX950: v_mfma_ld_scale_b32 v1, v1 ; encoding: [0x00,0x40,0xac,0xd3,0x01,0x03,0x02,0x18]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_mfma_ld_scale_b32 v1, v1

// GFX950: v_mfma_ld_scale_b32 0, 0 ; encoding: [0x00,0x40,0xac,0xd3,0x80,0x00,0x01,0x18]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_mfma_ld_scale_b32 0, 0

// GFX950: v_mfma_ld_scale_b32 1, 0               ; encoding: [0x00,0x40,0xac,0xd3,0x81,0x00,0x01,0x18]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_mfma_ld_scale_b32 1, 0

// GFX950: v_mfma_ld_scale_b32 v1, v1 op_sel:[1,0] ; encoding: [0x00,0x48,0xac,0xd3,0x01,0x03,0x02,0x18]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_mfma_ld_scale_b32 v1, v1 op_sel:[1, 0]

// GFX950: v_mfma_ld_scale_b32 v1, v1 op_sel:[0,1] ; encoding: [0x00,0x50,0xac,0xd3,0x01,0x03,0x02,0x18]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_mfma_ld_scale_b32 v1, v1 op_sel:[0, 1]

// GFX950: v_mfma_ld_scale_b32 v1, v1 op_sel:[1,1] ; encoding: [0x00,0x58,0xac,0xd3,0x01,0x03,0x02,0x18]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_mfma_ld_scale_b32 v1, v1 op_sel:[1, 1]

// GFX950: v_mfma_ld_scale_b32 v1, v1 op_sel_hi:[1,0] ; encoding: [0x00,0x40,0xac,0xd3,0x01,0x03,0x02,0x08]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_mfma_ld_scale_b32 v1, v1 op_sel_hi:[1, 0]

// GFX950: v_mfma_ld_scale_b32 v1, v1 op_sel_hi:[0,1] ; encoding: [0x00,0x40,0xac,0xd3,0x01,0x03,0x02,0x10]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_mfma_ld_scale_b32 v1, v1 op_sel_hi:[0, 1]

// GFX950: v_mfma_ld_scale_b32 v1, v1             ; encoding: [0x00,0x40,0xac,0xd3,0x01,0x03,0x02,0x18]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_mfma_ld_scale_b32 v1, v1 op_sel_hi:[1, 1]

// GFX950: v_mfma_ld_scale_b32 v1, v1 op_sel_hi:[0,0] ; encoding: [0x00,0x40,0xac,0xd3,0x01,0x03,0x02,0x00]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_mfma_ld_scale_b32 v1, v1 op_sel:[0,0] op_sel_hi:[0,0]

// GFX950: v_mfma_ld_scale_b32 v1, v1 op_sel:[1,0] op_sel_hi:[1,0] ; encoding: [0x00,0x48,0xac,0xd3,0x01,0x03,0x02,0x08]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_mfma_ld_scale_b32 v1, v1 op_sel:[1,0] op_sel_hi:[1,0]

// GFX950: v_mfma_ld_scale_b32 v1, v1 op_sel:[0,1] op_sel_hi:[0,1] ; encoding: [0x00,0x50,0xac,0xd3,0x01,0x03,0x02,0x10]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_mfma_ld_scale_b32 v1, v1 op_sel:[0,1] op_sel_hi:[0,1]

// GFX950: v_mfma_ld_scale_b32 v1, v1 op_sel:[0,1] ; encoding: [0x00,0x50,0xac,0xd3,0x01,0x03,0x02,0x18]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_mfma_ld_scale_b32 v1, v1 op_sel:[0,1] op_sel_hi:[1,1]

// GFX950: v_mfma_ld_scale_b32 v1, v1 op_sel:[0,1] ; encoding: [0x00,0x50,0xac,0xd3,0x01,0x03,0x02,0x18]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_mfma_ld_scale_b32 v1, v1 op_sel:[0,1] op_sel_hi:[1,1]

// GFX950: v_mfma_ld_scale_b32 v1, v1 op_sel:[1,1] ; encoding: [0x00,0x58,0xac,0xd3,0x01,0x03,0x02,0x18]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_mfma_ld_scale_b32 v1, v1 op_sel:[1,1] op_sel_hi:[1,1]

// GFX950: v_mfma_ld_scale_b32 v1, v1 op_sel:[1,0] op_sel_hi:[0,1] ; encoding: [0x00,0x48,0xac,0xd3,0x01,0x03,0x02,0x10]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_mfma_ld_scale_b32 v1, v1 op_sel:[1,0] op_sel_hi:[0,1]

// GFX950: v_mfma_ld_scale_b32 v1, v1 op_sel:[0,1] op_sel_hi:[1,0] ; encoding: [0x00,0x50,0xac,0xd3,0x01,0x03,0x02,0x08]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_mfma_ld_scale_b32 v1, v1 op_sel:[0,1] op_sel_hi:[1,0]


//===----------------------------------------------------------------------===//
// v_mfma_f32_16x16x128_f8f6f4
//===----------------------------------------------------------------------===//

// GFX950: v_mfma_f32_16x16x128_f8f6f4 v[0:3], v[4:11], v[4:11], v[0:3] ; encoding: [0x00,0x00,0xad,0xd3,0x04,0x09,0x02,0x04]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_mfma_f32_16x16x128_f8f6f4 v[0:3], v[4:11], v[4:11], v[0:3]

// GFX950: v_mfma_f32_16x16x128_f8f6f4 v[0:3], v[4:11], v[4:11], v[0:3] ; encoding: [0x00,0x00,0xad,0xd3,0x04,0x09,0x02,0x04]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_mfma_f32_16x16x128f8f6f4 v[0:3], v[4:11], v[4:11], v[0:3]

// GFX950: v_mfma_f32_16x16x128_f8f6f4 v[0:3], v[4:11], v[4:11], v[0:3] blgp:1 ; encoding: [0x00,0x00,0xad,0xd3,0x04,0x09,0x02,0x24]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_mfma_f32_16x16x128_f8f6f4 v[0:3], v[4:11], v[4:11], v[0:3] blgp:1

// GFX950: v_mfma_f32_16x16x128_f8f6f4 v[0:3], v[4:11], v[4:11], v[0:3] cbsz:3 ; encoding: [0x00,0x03,0xad,0xd3,0x04,0x09,0x02,0x04]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_mfma_f32_16x16x128_f8f6f4 v[0:3], v[4:11], v[4:11], v[0:3] cbsz:3

// GFX950: v_mfma_f32_16x16x128_f8f6f4 v[0:3], v[4:11], v[4:11], v[0:3] abid:1 ; encoding: [0x00,0x08,0xad,0xd3,0x04,0x09,0x02,0x04]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_mfma_f32_16x16x128_f8f6f4 v[0:3], v[4:11], v[4:11], v[0:3] abid:1

// GFX950: v_mfma_f32_16x16x128_f8f6f4 v[0:3], v[4:11], v[4:11], v[0:3] cbsz:3 abid:2 blgp:1 ; encoding: [0x00,0x13,0xad,0xd3,0x04,0x09,0x02,0x24]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_mfma_f32_16x16x128_f8f6f4 v[0:3], v[4:11], v[4:11], v[0:3] cbsz:3 abid:2 blgp:1

// GFX950: v_mfma_f32_16x16x128_f8f6f4 a[0:3], a[4:11], a[4:11], a[0:3] ; encoding: [0x00,0x80,0xad,0xd3,0x04,0x09,0x02,0x1c]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_mfma_f32_16x16x128_f8f6f4 a[0:3], a[4:11], a[4:11], a[0:3]

// GFX950: v_mfma_f32_16x16x128_f8f6f4 a[0:3], a[4:11], a[4:11], a[0:3] ; encoding: [0x00,0x80,0xad,0xd3,0x04,0x09,0x02,0x1c]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_mfma_f32_16x16x128f8f6f4 a[0:3], a[4:11], a[4:11], a[0:3]

// GFX950: v_mfma_f32_16x16x128_f8f6f4 a[0:3], a[4:11], a[4:11], a[0:3] blgp:1 ; encoding: [0x00,0x80,0xad,0xd3,0x04,0x09,0x02,0x3c]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_mfma_f32_16x16x128_f8f6f4 a[0:3], a[4:11], a[4:11], a[0:3] blgp:1

// GFX950: v_mfma_f32_16x16x128_f8f6f4 a[0:3], a[4:11], a[4:11], a[0:3] cbsz:3 ; encoding: [0x00,0x83,0xad,0xd3,0x04,0x09,0x02,0x1c]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_mfma_f32_16x16x128_f8f6f4 a[0:3], a[4:11], a[4:11], a[0:3] cbsz:3

// GFX950: v_mfma_f32_16x16x128_f8f6f4 a[0:3], a[4:11], a[4:11], a[0:3] abid:1 ; encoding: [0x00,0x88,0xad,0xd3,0x04,0x09,0x02,0x1c]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_mfma_f32_16x16x128_f8f6f4 a[0:3], a[4:11], a[4:11], a[0:3] abid:1

// GFX950: v_mfma_f32_16x16x128_f8f6f4 a[0:3], a[4:11], a[4:11], a[0:3] cbsz:1 abid:1 blgp:3 ; encoding: [0x00,0x89,0xad,0xd3,0x04,0x09,0x02,0x7c]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_mfma_f32_16x16x128_f8f6f4 a[0:3], a[4:11], a[4:11], a[0:3] cbsz:1 abid:1 blgp:3

//===----------------------------------------------------------------------===//
// v_mfma_f32_32x32x64_f8f6f4
//===----------------------------------------------------------------------===//

// GFX950: v_mfma_f32_32x32x64_f8f6f4 v[0:15], v[4:11], v[4:11], v[0:15] ; encoding: [0x00,0x00,0xae,0xd3,0x04,0x09,0x02,0x04]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_mfma_f32_32x32x64_f8f6f4 v[0:15], v[4:11], v[4:11], v[0:15]

// GFX950: v_mfma_f32_32x32x64_f8f6f4 v[0:15], v[4:11], v[4:11], v[0:15] ; encoding: [0x00,0x00,0xae,0xd3,0x04,0x09,0x02,0x04]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_mfma_f32_32x32x64f8f6f4 v[0:15], v[4:11], v[4:11], v[0:15]

// GFX950: v_mfma_f32_32x32x64_f8f6f4 v[0:15], v[4:11], v[4:11], v[0:15] blgp:1 ; encoding: [0x00,0x00,0xae,0xd3,0x04,0x09,0x02,0x24]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_mfma_f32_32x32x64_f8f6f4 v[0:15], v[4:11], v[4:11], v[0:15] blgp:1

// GFX950: v_mfma_f32_32x32x64_f8f6f4 v[0:15], v[4:11], v[4:11], v[0:15] cbsz:3 ; encoding: [0x00,0x03,0xae,0xd3,0x04,0x09,0x02,0x04]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_mfma_f32_32x32x64_f8f6f4 v[0:15], v[4:11], v[4:11], v[0:15] cbsz:3

// GFX950: v_mfma_f32_32x32x64_f8f6f4 v[0:15], v[4:11], v[4:11], v[0:15] abid:1 ; encoding: [0x00,0x08,0xae,0xd3,0x04,0x09,0x02,0x04]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_mfma_f32_32x32x64_f8f6f4 v[0:15], v[4:11], v[4:11], v[0:15] abid:1

// GFX950: v_mfma_f32_32x32x64_f8f6f4 v[0:15], v[4:11], v[4:11], v[0:15] cbsz:3 abid:2 blgp:1 ; encoding: [0x00,0x13,0xae,0xd3,0x04,0x09,0x02,0x24]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_mfma_f32_32x32x64_f8f6f4 v[0:15], v[4:11], v[4:11], v[0:15] cbsz:3 abid:2 blgp:1

// GFX950: v_mfma_f32_32x32x64_f8f6f4 a[0:15], a[4:11], a[4:11], a[0:15] ; encoding: [0x00,0x80,0xae,0xd3,0x04,0x09,0x02,0x1c]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_mfma_f32_32x32x64_f8f6f4 a[0:15], a[4:11], a[4:11], a[0:15]

// GFX950: v_mfma_f32_32x32x64_f8f6f4 a[0:15], a[4:11], a[4:11], a[0:15] ; encoding: [0x00,0x80,0xae,0xd3,0x04,0x09,0x02,0x1c]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_mfma_f32_32x32x64f8f6f4 a[0:15], a[4:11], a[4:11], a[0:15]

// GFX950: v_mfma_f32_32x32x64_f8f6f4 a[0:15], a[4:11], a[4:11], a[0:15] blgp:1 ; encoding: [0x00,0x80,0xae,0xd3,0x04,0x09,0x02,0x3c]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_mfma_f32_32x32x64_f8f6f4 a[0:15], a[4:11], a[4:11], a[0:15] blgp:1

// GFX950: v_mfma_f32_32x32x64_f8f6f4 a[0:15], a[4:11], a[4:11], a[0:15] cbsz:3 ; encoding: [0x00,0x83,0xae,0xd3,0x04,0x09,0x02,0x1c]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_mfma_f32_32x32x64_f8f6f4 a[0:15], a[4:11], a[4:11], a[0:15] cbsz:3

// GFX950: v_mfma_f32_32x32x64_f8f6f4 a[0:15], a[4:11], a[4:11], a[0:15] abid:1 ; encoding: [0x00,0x88,0xae,0xd3,0x04,0x09,0x02,0x1c]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_mfma_f32_32x32x64_f8f6f4 a[0:15], a[4:11], a[4:11], a[0:15] abid:1

// GFX950: v_mfma_f32_32x32x64_f8f6f4 a[0:15], a[4:11], a[4:11], a[0:15] cbsz:1 abid:1 blgp:3 ; encoding: [0x00,0x89,0xae,0xd3,0x04,0x09,0x02,0x7c]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_mfma_f32_32x32x64_f8f6f4 a[0:15], a[4:11], a[4:11], a[0:15] cbsz:1 abid:1 blgp:3

//===----------------------------------------------------------------------===//
// v_mfma_i32_16x16x64_i8
//===----------------------------------------------------------------------===//

// GFX950: v_mfma_i32_16x16x64_i8 v[0:3], v[0:3], v[0:3], v[0:3] ; encoding: [0x00,0x00,0xb6,0xd3,0x00,0x01,0x02,0x04]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_mfma_i32_16x16x64_i8 v[0:3], v[0:3], v[0:3], v[0:3]

// GFX950: v_mfma_i32_16x16x64_i8 v[0:3], v[0:3], v[0:3], v[0:3] ; encoding: [0x00,0x00,0xb6,0xd3,0x00,0x01,0x02,0x04]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_mfma_i32_16x16x64i8 v[0:3], v[0:3], v[0:3], v[0:3]

// GFX950: v_mfma_i32_16x16x64_i8 a[0:3], a[0:3], a[0:3], a[0:3] ; encoding: [0x00,0x80,0xb6,0xd3,0x00,0x01,0x02,0x1c]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_mfma_i32_16x16x64_i8 a[0:3], a[0:3], a[0:3], a[0:3]

// GFX950: v_mfma_i32_16x16x64_i8 a[0:3], a[0:3], a[0:3], a[0:3] ; encoding: [0x00,0x80,0xb6,0xd3,0x00,0x01,0x02,0x1c]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_mfma_i32_16x16x64i8 a[0:3], a[0:3], a[0:3], a[0:3]

// GFX950: v_mfma_i32_16x16x64_i8 v[0:3], a[0:3], v[0:3], 1.0 ; encoding: [0x00,0x00,0xb6,0xd3,0x00,0x01,0xca,0x0b]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_mfma_i32_16x16x64_i8 v[0:3], a[0:3], v[0:3], 1.0

// GFX950: v_mfma_i32_16x16x64_i8 a[0:3], v[0:3], a[0:3], 1.0 ; encoding: [0x00,0x80,0xb6,0xd3,0x00,0x01,0xca,0x13]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_mfma_i32_16x16x64_i8 a[0:3], v[0:3], a[0:3], 1.0

// GFX950: v_mfma_i32_16x16x64_i8 v[0:3], v[0:3], v[0:3], v[0:3] blgp:5 ; encoding: [0x00,0x00,0xb6,0xd3,0x00,0x01,0x02,0xa4]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_mfma_i32_16x16x64_i8 v[0:3], v[0:3], v[0:3], v[0:3] blgp:5

// GFX950: v_mfma_i32_16x16x64_i8 a[0:3], a[0:3], a[0:3], a[0:3] blgp:1 ; encoding: [0x00,0x80,0xb6,0xd3,0x00,0x01,0x02,0x3c]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_mfma_i32_16x16x64_i8 a[0:3], a[0:3], a[0:3], a[0:3] blgp:1

// GFX950: v_mfma_i32_16x16x64_i8 a[0:3], a[0:3], a[0:3], a[0:3] cbsz:3 ; encoding: [0x00,0x83,0xb6,0xd3,0x00,0x01,0x02,0x1c]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_mfma_i32_16x16x64_i8 a[0:3], a[0:3], a[0:3], a[0:3] cbsz:3

// GFX950:  v_mfma_i32_16x16x64_i8 a[0:3], a[0:3], a[0:3], a[0:3] abid:1 ; encoding: [0x00,0x88,0xb6,0xd3,0x00,0x01,0x02,0x1c]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_mfma_i32_16x16x64_i8 a[0:3], a[0:3], a[0:3], a[0:3] abid:1

// GFX950:  v_mfma_i32_16x16x64_i8 a[0:3], a[0:3], a[0:3], a[0:3] cbsz:3 abid:1 ; encoding: [0x00,0x8b,0xb6,0xd3,0x00,0x01,0x02,0x1c]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_mfma_i32_16x16x64_i8 a[0:3], a[0:3], a[0:3], a[0:3] cbsz:3 abid:1

// GFX950: v_mfma_i32_16x16x64_i8 a[0:3], v[0:3], v[0:3], a[4:7] ; encoding: [0x00,0x80,0xb6,0xd3,0x00,0x01,0x12,0x04]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_mfma_i32_16x16x64_i8 a[0:3], v[0:3], v[0:3], a[4:7]

// GFX950: v_mfma_i32_16x16x64_i8 v[0:3], a[0:3], a[0:3], v[4:7] ; encoding: [0x00,0x00,0xb6,0xd3,0x00,0x01,0x12,0x1c]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_mfma_i32_16x16x64_i8 v[0:3], a[0:3], a[0:3], v[4:7]

//===----------------------------------------------------------------------===//
// v_mfma_i32_32x32x32_i8
//===----------------------------------------------------------------------===//

// GFX950: v_mfma_i32_32x32x32_i8 v[0:15], v[0:3], v[0:3], v[0:15] ; encoding: [0x00,0x00,0xb8,0xd3,0x00,0x01,0x02,0x04]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_mfma_i32_32x32x32_i8 v[0:15], v[0:3], v[0:3], v[0:15]

// GFX950: v_mfma_i32_32x32x32_i8 a[0:15], a[0:3], a[0:3], a[0:15] ; encoding: [0x00,0x80,0xb8,0xd3,0x00,0x01,0x02,0x1c]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_mfma_i32_32x32x32_i8 a[0:15], a[0:3], a[0:3], a[0:15]

// GFX950:  v_mfma_i32_32x32x32_i8 v[0:15], v[0:3], v[0:3], v[0:15] ; encoding: [0x00,0x00,0xb8,0xd3,0x00,0x01,0x02,0x04]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_mfma_i32_32x32x32i8 v[0:15], v[0:3], v[0:3], v[0:15]

// GFX950: v_mfma_i32_32x32x32_i8 a[0:15], a[0:3], a[0:3], a[0:15] ; encoding: [0x00,0x80,0xb8,0xd3,0x00,0x01,0x02,0x1c]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_mfma_i32_32x32x32i8 a[0:15], a[0:3], a[0:3], a[0:15]

// GFX950: v_mfma_i32_32x32x32_i8 v[0:15], v[0:3], v[0:3], 1.0 ; encoding: [0x00,0x00,0xb8,0xd3,0x00,0x01,0xca,0x03]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_mfma_i32_32x32x32_i8 v[0:15], v[0:3], v[0:3], 1.0

// GFX950: v_mfma_i32_32x32x32_i8 a[0:15], a[0:3], a[0:3], 1.0 ; encoding: [0x00,0x80,0xb8,0xd3,0x00,0x01,0xca,0x1b]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_mfma_i32_32x32x32_i8 a[0:15], a[0:3], a[0:3], 1.0

// GFX950:  v_mfma_i32_32x32x32_i8 v[0:15], v[0:3], v[0:3], v[0:15] blgp:5 ; encoding: [0x00,0x00,0xb8,0xd3,0x00,0x01,0x02,0xa4]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_mfma_i32_32x32x32_i8 v[0:15], v[0:3], v[0:3], v[0:15] blgp:5

// GFX950: v_mfma_i32_32x32x32_i8 a[0:15], a[0:3], a[0:3], a[0:15] blgp:2 ; encoding: [0x00,0x80,0xb8,0xd3,0x00,0x01,0x02,0x5c]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_mfma_i32_32x32x32_i8 a[0:15], a[0:3], a[0:3], a[0:15] blgp:2

// GFX950: v_mfma_i32_32x32x32_i8 v[0:15], v[0:3], v[0:3], v[0:15] cbsz:3 ; encoding: [0x00,0x03,0xb8,0xd3,0x00,0x01,0x02,0x04]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_mfma_i32_32x32x32_i8 v[0:15], v[0:3], v[0:3], v[0:15] cbsz:3

// GFX950: v_mfma_i32_32x32x32_i8 v[0:15], v[0:3], v[0:3], v[0:15] abid:1 ; encoding: [0x00,0x08,0xb8,0xd3,0x00,0x01,0x02,0x04]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_mfma_i32_32x32x32_i8 v[0:15], v[0:3], v[0:3], v[0:15] abid:1

// GFX950: v_mfma_i32_32x32x32_i8 a[0:15], a[0:3], a[0:3], a[0:15] cbsz:3 abid:1 ; encoding: [0x00,0x8b,0xb8,0xd3,0x00,0x01,0x02,0x1c]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_mfma_i32_32x32x32_i8 a[0:15], a[0:3], a[0:3], a[0:15] cbsz:3 abid:1

//===----------------------------------------------------------------------===//
// v_mfma_f32_16x16x32_bf16
//===----------------------------------------------------------------------===//

// GFX950: v_mfma_f32_16x16x32_bf16 v[0:3], v[0:3], v[0:3], v[0:3] ; encoding: [0x00,0x00,0xb5,0xd3,0x00,0x01,0x02,0x04]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_mfma_f32_16x16x32_bf16 v[0:3], v[0:3], v[0:3], v[0:3]

// GFX950: v_mfma_f32_16x16x32_bf16 v[0:3], v[0:3], v[0:3], v[0:3] ; encoding: [0x00,0x00,0xb5,0xd3,0x00,0x01,0x02,0x04]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_mfma_f32_16x16x32bf16 v[0:3], v[0:3], v[0:3], v[0:3]

// GFX950: v_mfma_f32_16x16x32_bf16 a[0:3], a[0:3], a[0:3], a[0:3] ; encoding: [0x00,0x80,0xb5,0xd3,0x00,0x01,0x02,0x1c]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_mfma_f32_16x16x32_bf16 a[0:3], a[0:3], a[0:3], a[0:3]

// GFX950: v_mfma_f32_16x16x32_bf16 a[0:3], a[0:3], a[0:3], a[0:3] ; encoding: [0x00,0x80,0xb5,0xd3,0x00,0x01,0x02,0x1c]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_mfma_f32_16x16x32bf16 a[0:3], a[0:3], a[0:3], a[0:3]

// GFX950: v_mfma_f32_16x16x32_bf16 v[0:3], a[0:3], v[0:3], 1.0 ; encoding: [0x00,0x00,0xb5,0xd3,0x00,0x01,0xca,0x0b]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_mfma_f32_16x16x32_bf16 v[0:3], a[0:3], v[0:3], 1.0

// GFX950: v_mfma_f32_16x16x32_bf16 a[0:3], v[0:3], a[0:3], 1.0 ; encoding: [0x00,0x80,0xb5,0xd3,0x00,0x01,0xca,0x13]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_mfma_f32_16x16x32_bf16 a[0:3], v[0:3], a[0:3], 1.0

// GFX950: v_mfma_f32_16x16x32_bf16 v[0:3], v[0:3], v[0:3], v[0:3] blgp:5 ; encoding: [0x00,0x00,0xb5,0xd3,0x00,0x01,0x02,0xa4]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_mfma_f32_16x16x32_bf16 v[0:3], v[0:3], v[0:3], v[0:3] blgp:5

// GFX950: v_mfma_f32_16x16x32_bf16 a[0:3], a[0:3], a[0:3], a[0:3] blgp:1 ; encoding: [0x00,0x80,0xb5,0xd3,0x00,0x01,0x02,0x3c]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_mfma_f32_16x16x32_bf16 a[0:3], a[0:3], a[0:3], a[0:3] blgp:1

// GFX950: v_mfma_f32_16x16x32_bf16 a[0:3], a[0:3], a[0:3], a[0:3] cbsz:3 ; encoding: [0x00,0x83,0xb5,0xd3,0x00,0x01,0x02,0x1c]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_mfma_f32_16x16x32_bf16 a[0:3], a[0:3], a[0:3], a[0:3] cbsz:3

// GFX950:  v_mfma_f32_16x16x32_bf16 a[0:3], a[0:3], a[0:3], a[0:3] abid:1 ; encoding: [0x00,0x88,0xb5,0xd3,0x00,0x01,0x02,0x1c]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_mfma_f32_16x16x32_bf16 a[0:3], a[0:3], a[0:3], a[0:3] abid:1

// GFX950:  v_mfma_f32_16x16x32_bf16 a[0:3], a[0:3], a[0:3], a[0:3] cbsz:3 abid:1 ; encoding: [0x00,0x8b,0xb5,0xd3,0x00,0x01,0x02,0x1c]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_mfma_f32_16x16x32_bf16 a[0:3], a[0:3], a[0:3], a[0:3] cbsz:3 abid:1

// GFX950: v_mfma_f32_16x16x32_bf16 a[0:3], v[0:3], v[0:3], a[4:7] ; encoding: [0x00,0x80,0xb5,0xd3,0x00,0x01,0x12,0x04]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_mfma_f32_16x16x32_bf16 a[0:3], v[0:3], v[0:3], a[4:7]

// GFX950: v_mfma_f32_16x16x32_bf16 v[0:3], a[0:3], a[0:3], v[4:7] ; encoding: [0x00,0x00,0xb5,0xd3,0x00,0x01,0x12,0x1c]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_mfma_f32_16x16x32_bf16 v[0:3], a[0:3], a[0:3], v[4:7]
