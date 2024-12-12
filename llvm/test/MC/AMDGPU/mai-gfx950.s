// RUN: llvm-mc -triple=amdgcn -mcpu=gfx950 -show-encoding %s | FileCheck -check-prefix=GFX950 %s
// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx940 %s 2>&1 | FileCheck -check-prefix=ERR %s
// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx90a %s 2>&1 | FileCheck -check-prefix=ERR %s
// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx1010 %s 2>&1 | FileCheck -check-prefix=ERR %s

// cbsz = SrcA
// blgp = SrcB

// 0 = fp8. 1 = bf8
// 2 = fp6, 3 = bf6
// 4 = fp4

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

// GFX950: v_mfma_f32_16x16x128_f8f6f4 v[0:3], v[4:11], v[4:11], v[0:3] blgp:1 ; encoding: [0x00,0x00,0xad,0xd3,0x04,0x09,0x02,0x24]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_mfma_f32_16x16x128_f8f6f4 v[0:3], v[4:11], v[4:11], v[0:3] blgp:1

// GFX950: v_mfma_f32_16x16x128_f8f6f4 v[0:3], v[4:9], v[4:11], v[0:3] cbsz:3 ; encoding: [0x00,0x03,0xad,0xd3,0x04,0x09,0x02,0x04]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_mfma_f32_16x16x128_f8f6f4 v[0:3], v[4:9], v[4:11], v[0:3] cbsz:3

// GFX950: v_mfma_f32_16x16x128_f8f6f4 v[0:3], v[4:11], v[4:11], v[0:3] ; encoding: [0x00,0x00,0xad,0xd3,0x04,0x09,0x02,0x04]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_mfma_f32_16x16x128_f8f6f4 v[0:3], v[4:11], v[4:11], v[0:3]

// GFX950: v_mfma_f32_16x16x128_f8f6f4 v[0:3], v[4:9], v[4:11], v[0:3] cbsz:3 blgp:1 ; encoding: [0x00,0x03,0xad,0xd3,0x04,0x09,0x02,0x24]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_mfma_f32_16x16x128_f8f6f4 v[0:3], v[4:9], v[4:11], v[0:3] cbsz:3 blgp:1

// GFX950: v_mfma_f32_16x16x128_f8f6f4 a[0:3], a[4:11], a[4:11], a[0:3] ; encoding: [0x00,0x80,0xad,0xd3,0x04,0x09,0x02,0x1c]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_mfma_f32_16x16x128_f8f6f4 a[0:3], a[4:11], a[4:11], a[0:3]

// GFX950: v_mfma_f32_16x16x128_f8f6f4 a[0:3], a[4:11], a[4:11], a[0:3] blgp:1 ; encoding: [0x00,0x80,0xad,0xd3,0x04,0x09,0x02,0x3c]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_mfma_f32_16x16x128_f8f6f4 a[0:3], a[4:11], a[4:11], a[0:3] blgp:1

// GFX950: v_mfma_f32_16x16x128_f8f6f4 a[0:3], a[4:9], a[4:11], a[0:3] cbsz:3 ; encoding: [0x00,0x83,0xad,0xd3,0x04,0x09,0x02,0x1c]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_mfma_f32_16x16x128_f8f6f4 a[0:3], a[4:9], a[4:11], a[0:3] cbsz:3

// GFX950: v_mfma_f32_16x16x128_f8f6f4 a[0:3], a[4:11], a[4:11], a[0:3] ; encoding: [0x00,0x80,0xad,0xd3,0x04,0x09,0x02,0x1c]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_mfma_f32_16x16x128_f8f6f4 a[0:3], a[4:11], a[4:11], a[0:3]

// GFX950: v_mfma_f32_16x16x128_f8f6f4 a[0:3], a[4:11], a[4:9], a[0:3] cbsz:1 blgp:3 ; encoding: [0x00,0x81,0xad,0xd3,0x04,0x09,0x02,0x7c]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_mfma_f32_16x16x128_f8f6f4 a[0:3], a[4:11], a[4:9], a[0:3] cbsz:1 blgp:3

//===----------------------------------------------------------------------===//
// v_mfma_f32_32x32x64_f8f6f4
//===----------------------------------------------------------------------===//

// GFX950: v_mfma_f32_32x32x64_f8f6f4 v[0:15], v[4:11], v[4:11], v[0:15] ; encoding: [0x00,0x00,0xae,0xd3,0x04,0x09,0x02,0x04]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_mfma_f32_32x32x64_f8f6f4 v[0:15], v[4:11], v[4:11], v[0:15]

// GFX950: v_mfma_f32_32x32x64_f8f6f4 v[0:15], v[4:11], v[4:11], v[0:15] blgp:1 ; encoding: [0x00,0x00,0xae,0xd3,0x04,0x09,0x02,0x24]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_mfma_f32_32x32x64_f8f6f4 v[0:15], v[4:11], v[4:11], v[0:15] blgp:1

// GFX950: v_mfma_f32_32x32x64_f8f6f4 v[0:15], v[4:9], v[4:11], v[0:15] cbsz:3 ; encoding: [0x00,0x03,0xae,0xd3,0x04,0x09,0x02,0x04]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_mfma_f32_32x32x64_f8f6f4 v[0:15], v[4:9], v[4:11], v[0:15] cbsz:3

// GFX950: v_mfma_f32_32x32x64_f8f6f4 v[0:15], v[4:11], v[4:11], v[0:15] ; encoding: [0x00,0x00,0xae,0xd3,0x04,0x09,0x02,0x04]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_mfma_f32_32x32x64_f8f6f4 v[0:15], v[4:11], v[4:11], v[0:15]

// GFX950: v_mfma_f32_32x32x64_f8f6f4 v[0:15], v[4:9], v[4:11], v[0:15] cbsz:3 blgp:1 ; encoding: [0x00,0x03,0xae,0xd3,0x04,0x09,0x02,0x24]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_mfma_f32_32x32x64_f8f6f4 v[0:15], v[4:9], v[4:11], v[0:15] cbsz:3 blgp:1

// GFX950: v_mfma_f32_32x32x64_f8f6f4 a[0:15], a[4:11], a[4:11], a[0:15] ; encoding: [0x00,0x80,0xae,0xd3,0x04,0x09,0x02,0x1c]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_mfma_f32_32x32x64_f8f6f4 a[0:15], a[4:11], a[4:11], a[0:15]

// GFX950: v_mfma_f32_32x32x64_f8f6f4 a[0:15], a[4:11], a[4:11], a[0:15] blgp:1 ; encoding: [0x00,0x80,0xae,0xd3,0x04,0x09,0x02,0x3c]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_mfma_f32_32x32x64_f8f6f4 a[0:15], a[4:11], a[4:11], a[0:15] blgp:1

// GFX950: v_mfma_f32_32x32x64_f8f6f4 a[0:15], a[4:9], a[4:11], a[0:15] cbsz:3 ; encoding: [0x00,0x83,0xae,0xd3,0x04,0x09,0x02,0x1c]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_mfma_f32_32x32x64_f8f6f4 a[0:15], a[4:9], a[4:11], a[0:15] cbsz:3

// GFX950: v_mfma_f32_32x32x64_f8f6f4 a[0:15], a[4:11], a[4:11], a[0:15] ; encoding: [0x00,0x80,0xae,0xd3,0x04,0x09,0x02,0x1c]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_mfma_f32_32x32x64_f8f6f4 a[0:15], a[4:11], a[4:11], a[0:15]

// GFX950: v_mfma_f32_32x32x64_f8f6f4 a[0:15], a[4:11], a[4:9], a[0:15] cbsz:1 blgp:3 ; encoding: [0x00,0x81,0xae,0xd3,0x04,0x09,0x02,0x7c]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_mfma_f32_32x32x64_f8f6f4 a[0:15], a[4:11], a[4:9], a[0:15] cbsz:1 blgp:3

//===----------------------------------------------------------------------===//
// v_mfma_scale_f32_16x16x128_f8f6f4
//===----------------------------------------------------------------------===//
// FIXME: Test op_sel, neg, clamp

// GFX950: v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[4:11], v[12:19], v[20:23], v24, v25 op_sel_hi:[0,0,0] ; encoding: [0x00,0x00,0xac,0xd3,0x18,0x33,0x02,0x00,0x00,0x08,0xad,0xd3,0x04,0x19,0x52,0x04]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[4:11], v[12:19], v[20:23], v24, v25

// GFX950: v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], a[4:11], a[12:19], a[20:23], v24, v25 op_sel_hi:[0,0,0] ; encoding: [0x00,0x00,0xac,0xd3,0x18,0x33,0x02,0x00,0x00,0x88,0xad,0xd3,0x04,0x19,0x52,0x1c]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], a[4:11], a[12:19], a[20:23], v24, v25

// GFX950: v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[4:11], v[12:19], a[20:23], v24, v25 op_sel_hi:[0,0,0] ; encoding: [0x00,0x00,0xac,0xd3,0x18,0x33,0x02,0x00,0x00,0x88,0xad,0xd3,0x04,0x19,0x52,0x04]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[4:11], v[12:19], a[20:23], v24, v25

// GFX950: v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], a[4:11], a[12:19], v[20:23], v24, v25 op_sel_hi:[0,0,0] ; encoding: [0x00,0x00,0xac,0xd3,0x18,0x33,0x02,0x00,0x00,0x08,0xad,0xd3,0x04,0x19,0x52,0x1c]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], a[4:11], a[12:19], v[20:23], v24, v25

// GFX950: v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[4:11], a[12:19], v[20:23], v24, v25 op_sel_hi:[0,0,0] ; encoding: [0x00,0x00,0xac,0xd3,0x18,0x33,0x02,0x00,0x00,0x08,0xad,0xd3,0x04,0x19,0x52,0x14]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[4:11], a[12:19], v[20:23], v24, v25

// GFX950: v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], a[4:11], v[12:19], v[20:23], v24, v25 op_sel_hi:[0,0,0] ; encoding: [0x00,0x00,0xac,0xd3,0x18,0x33,0x02,0x00,0x00,0x08,0xad,0xd3,0x04,0x19,0x52,0x0c]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], a[4:11], v[12:19], v[20:23], v24, v25

// GFX950: v_mfma_scale_f32_16x16x128_f8f6f4 v[50:53], v[4:11], v[12:19], v[20:23], v24, v25 op_sel_hi:[0,0,0] ; encoding: [0x00,0x00,0xac,0xd3,0x18,0x33,0x02,0x00,0x32,0x08,0xad,0xd3,0x04,0x19,0x52,0x04]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_mfma_scale_f32_16x16x128_f8f6f4 v[50:53], v[4:11], v[12:19], v[20:23], v24, v25

// GFX950: v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[4:11], v[12:19], v[20:23], v44, s24 op_sel_hi:[0,0,0] ; encoding: [0x00,0x00,0xac,0xd3,0x2c,0x31,0x00,0x00,0x00,0x08,0xad,0xd3,0x04,0x19,0x52,0x04]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[4:11], v[12:19], v[20:23], v44, s24

// GFX950: v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[4:11], v[12:19], v[20:23], s24, s24 op_sel_hi:[0,0,0] ; encoding: [0x00,0x00,0xac,0xd3,0x18,0x30,0x00,0x00,0x00,0x08,0xad,0xd3,0x04,0x19,0x52,0x04]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[4:11], v[12:19], v[20:23], s24, s24

// GFX950: v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[4:11], v[12:19], v[20:23], s24, v44 op_sel_hi:[0,0,0] ; encoding: [0x00,0x00,0xac,0xd3,0x18,0x58,0x02,0x00,0x00,0x08,0xad,0xd3,0x04,0x19,0x52,0x04]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[4:11], v[12:19], v[20:23], s24, v44

// GFX950: v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[4:11], v[12:19], v[20:23], m0, m0 op_sel_hi:[0,0,0]  ; encoding: [0x00,0x00,0xac,0xd3,0x7c,0xf8,0x00,0x00,0x00,0x08,0xad,0xd3,0x04,0x19,0x52,0x04]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[4:11], v[12:19], v[20:23], m0, m0

// GFX950: v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[4:11], v[12:19], v[20:23], vcc_lo, v2 op_sel_hi:[0,0,0] ; encoding: [0x00,0x00,0xac,0xd3,0x6a,0x04,0x02,0x00,0x00,0x08,0xad,0xd3,0x04,0x19,0x52,0x04]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[4:11], v[12:19], v[20:23], vcc_lo, v2

// GFX950: v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[4:11], v[12:19], v[20:23], 9, v2 op_sel_hi:[0,0,0] ; encoding: [0x00,0x00,0xac,0xd3,0x89,0x04,0x02,0x00,0x00,0x08,0xad,0xd3,0x04,0x19,0x52,0x04]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[4:11], v[12:19], v[20:23], 9, v2

// GFX950: v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[4:11], v[12:19], v[20:23], v2, 9 op_sel_hi:[0,0,0] ; encoding: [0x00,0x00,0xac,0xd3,0x02,0x13,0x01,0x00,0x00,0x08,0xad,0xd3,0x04,0x19,0x52,0x04]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[4:11], v[12:19], v[20:23], v2, 9

// GFX950: v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[4:11], v[12:19], v[20:23], s20, 9 op_sel_hi:[0,0,0] ; encoding: [0x00,0x00,0xac,0xd3,0x14,0x12,0x01,0x00,0x00,0x08,0xad,0xd3,0x04,0x19,0x52,0x04]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[4:11], v[12:19], v[20:23], s20, 9

// GFX950: v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[4:11], v[12:19], v[20:23], 33, 9 op_sel_hi:[0,0,0] ; encoding: [0x00,0x00,0xac,0xd3,0xa1,0x12,0x01,0x00,0x00,0x08,0xad,0xd3,0x04,0x19,0x52,0x04]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[4:11], v[12:19], v[20:23], 33, 9

// GFX950: v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[4:9], v[12:19], v[20:23], v24, v25 op_sel_hi:[0,0,0] cbsz:3 blgp:1 ; encoding: [0x00,0x00,0xac,0xd3,0x18,0x33,0x02,0x00,0x00,0x0b,0xad,0xd3,0x04,0x19,0x52,0x24]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[4:9], v[12:19], v[20:23], v24, v25 cbsz:3 blgp:1

// GFX950: v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[4:9], v[12:19], v[20:23], v24, v25 op_sel_hi:[0,0,0] cbsz:3 blgp:1 ; encoding: [0x00,0x00,0xac,0xd3,0x18,0x33,0x02,0x08,0x00,0x0b,0xad,0xd3,0x04,0x19,0x52,0x24]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[4:9], v[12:19], v[20:23], v24, v25 op_sel:[0,1,0] cbsz:3 blgp:1

// GFX950: v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[4:9], v[12:19], v[20:23], v24, v25 op_sel_hi:[0,0,0] cbsz:3 blgp:1 ; encoding: [0x00,0x00,0xac,0xd3,0x18,0x33,0x02,0x18,0x00,0x0b,0xad,0xd3,0x04,0x19,0x52,0x24]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[4:9], v[12:19], v[20:23], v24, v25 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3 blgp:1

// GFX950: v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[4:11], v[12:17], v[20:23], v24, v25 op_sel_hi:[0,0,0] blgp:2 ; encoding: [0x00,0x00,0xac,0xd3,0x18,0x33,0x02,0x18,0x00,0x08,0xad,0xd3,0x04,0x19,0x52,0x44]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[4:11], v[12:17], v[20:23], v24, v25 op_sel:[0,1,0] op_sel_hi:[0,1,0] blgp:2

// GFX950: v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[4:11], v[12:19], v[20:23], v24, v25 op_sel_hi:[0,0,0] ; encoding: [0x00,0x00,0xac,0xd3,0x18,0x33,0x02,0x18,0x00,0x08,0xad,0xd3,0x04,0x19,0x52,0x04]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[4:11], v[12:19], v[20:23], v24, v25 op_sel:[0,1,0] op_sel_hi:[0,1,0]

// GFX950: v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[4:9], v[12:19], v[20:23], v24, v25 op_sel_hi:[0,0,0] cbsz:3 ; encoding: [0x00,0x00,0xac,0xd3,0x18,0x33,0x02,0x18,0x00,0x0b,0xad,0xd3,0x04,0x19,0x52,0x04]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[4:9], v[12:19], v[20:23], v24, v25 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:3

//===----------------------------------------------------------------------===//
// v_mfma_scale_f32_32x32x64_f8f6f4
//===----------------------------------------------------------------------===//
// FIXME: Test op_sel, neg, clamp

// GFX950: v_mfma_scale_f32_32x32x64_f8f6f4 v[0:15], v[16:23], v[24:31], v[32:47], v48, v49 op_sel_hi:[0,0,0] ; encoding: [0x00,0x00,0xac,0xd3,0x30,0x63,0x02,0x00,0x00,0x08,0xae,0xd3,0x10,0x31,0x82,0x04]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_mfma_scale_f32_32x32x64_f8f6f4 v[0:15], v[16:23], v[24:31], v[32:47], v48, v49

// GFX950: v_mfma_scale_f32_32x32x64_f8f6f4 a[0:15], a[16:23], a[24:31], a[32:47], v48, v49 op_sel_hi:[0,0,0] ; encoding: [0x00,0x00,0xac,0xd3,0x30,0x63,0x02,0x00,0x00,0x88,0xae,0xd3,0x10,0x31,0x82,0x1c]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_mfma_scale_f32_32x32x64_f8f6f4 a[0:15], a[16:23], a[24:31], a[32:47], v48, v49

// GFX950: v_mfma_scale_f32_32x32x64_f8f6f4 a[0:15], v[16:23], v[24:31], a[32:47], v48, v49 op_sel_hi:[0,0,0] ; encoding: [0x00,0x00,0xac,0xd3,0x30,0x63,0x02,0x00,0x00,0x88,0xae,0xd3,0x10,0x31,0x82,0x04]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_mfma_scale_f32_32x32x64_f8f6f4 a[0:15], v[16:23], v[24:31], a[32:47], v48, v49

// GFX950: v_mfma_scale_f32_32x32x64_f8f6f4 v[0:15], a[16:23], a[24:31], v[32:47], v48, v49 op_sel_hi:[0,0,0] ; encoding: [0x00,0x00,0xac,0xd3,0x30,0x63,0x02,0x00,0x00,0x08,0xae,0xd3,0x10,0x31,0x82,0x1c]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_mfma_scale_f32_32x32x64_f8f6f4 v[0:15], a[16:23], a[24:31], v[32:47], v48, v49

// GFX950: v_mfma_scale_f32_32x32x64_f8f6f4 v[0:15], v[16:23], a[24:31], v[32:47], v48, v49 op_sel_hi:[0,0,0] ; encoding: [0x00,0x00,0xac,0xd3,0x30,0x63,0x02,0x00,0x00,0x08,0xae,0xd3,0x10,0x31,0x82,0x14]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_mfma_scale_f32_32x32x64_f8f6f4 v[0:15], v[16:23], a[24:31], v[32:47], v48, v49

// GFX950: v_mfma_scale_f32_32x32x64_f8f6f4 a[0:15], a[16:23], v[24:31], a[32:47], v48, v49 op_sel_hi:[0,0,0] ; encoding: [0x00,0x00,0xac,0xd3,0x30,0x63,0x02,0x00,0x00,0x88,0xae,0xd3,0x10,0x31,0x82,0x0c]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_mfma_scale_f32_32x32x64_f8f6f4 a[0:15], a[16:23], v[24:31], a[32:47], v48, v49

// GFX950: v_mfma_scale_f32_32x32x64_f8f6f4 v[50:65], v[16:23], v[24:31], v[32:47], v48, v49  op_sel_hi:[0,0,0] ; encoding: [0x00,0x00,0xac,0xd3,0x30,0x63,0x02,0x00,0x32,0x08,0xae,0xd3,0x10,0x31,0x82,0x04]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_mfma_scale_f32_32x32x64_f8f6f4 v[50:65], v[16:23], v[24:31], v[32:47], v48, v49

// GFX950: v_mfma_scale_f32_32x32x64_f8f6f4 v[0:15], v[16:21], v[24:31], v[32:47], v48, v49 op_sel_hi:[0,0,0] cbsz:3 blgp:1 ; encoding: [0x00,0x00,0xac,0xd3,0x30,0x63,0x02,0x00,0x00,0x0b,0xae,0xd3,0x10,0x31,0x82,0x24]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_mfma_scale_f32_32x32x64_f8f6f4 v[0:15], v[16:21], v[24:31], v[32:47], v48, v49 cbsz:3 blgp:1

// GFX950: 	v_mfma_scale_f32_32x32x64_f8f6f4 v[0:15], v[16:21], v[24:31], v[32:47], v48, v49 op_sel_hi:[0,0,0] cbsz:2 ; encoding: [0x00,0x00,0xac,0xd3,0x30,0x63,0x02,0x00,0x00,0x0a,0xae,0xd3,0x10,0x31,0x82,0x04]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_mfma_scale_f32_32x32x64_f8f6f4 v[0:15], v[16:21], v[24:31], v[32:47], v48, v49 cbsz:2

// GFX950: v_mfma_scale_f32_32x32x64_f8f6f4 v[0:15], v[16:23], v[24:31], v[32:47], v48, v49 op_sel_hi:[0,0,0] ; encoding: [0x00,0x00,0xac,0xd3,0x30,0x63,0x02,0x00,0x00,0x08,0xae,0xd3,0x10,0x31,0x82,0x04]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_mfma_scale_f32_32x32x64_f8f6f4 v[0:15], v[16:23], v[24:31], v[32:47], v48, v49

// GFX950: v_mfma_scale_f32_32x32x64_f8f6f4 v[0:15], v[16:23], v[24:29], v[32:47], v48, v49 op_sel_hi:[0,0,0] blgp:2 ; encoding: [0x00,0x00,0xac,0xd3,0x30,0x63,0x02,0x00,0x00,0x08,0xae,0xd3,0x10,0x31,0x82,0x44]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_mfma_scale_f32_32x32x64_f8f6f4 v[0:15], v[16:23], v[24:29], v[32:47], v48, v49 blgp:2

// GFX950: v_mfma_scale_f32_32x32x64_f8f6f4 v[0:15], v[16:21], v[24:29], v[32:47], v48, v49 op_sel_hi:[0,0,0] cbsz:2 blgp:3 ; encoding: [0x00,0x00,0xac,0xd3,0x30,0x63,0x02,0x18,0x00,0x0a,0xae,0xd3,0x10,0x31,0x82,0x64]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_mfma_scale_f32_32x32x64_f8f6f4 v[0:15], v[16:21], v[24:29], v[32:47], v48, v49 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:3

//===----------------------------------------------------------------------===//
// v_mfma_f32_16x16x128_f8f6f4 with appropriate register widths
//===----------------------------------------------------------------------===//

// bf8 x fp4
// GFX950: v_mfma_f32_16x16x128_f8f6f4 v[0:3], v[4:11], v[12:15], v[20:23] cbsz:1 blgp:4 ; encoding: [0x00,0x01,0xad,0xd3,0x04,0x19,0x52,0x84]
v_mfma_f32_16x16x128_f8f6f4 v[0:3], v[4:11], v[12:15], v[20:23] cbsz:1 blgp:4

// GFX950: v_mfma_f32_16x16x128_f8f6f4 v[0:3], v[4:11], v[12:15], v[20:23] cbsz:1 blgp:4 ; encoding: [0x00,0x01,0xad,0xd3,0x04,0x19,0x52,0x84]
v_mfma_f32_16x16x128_f8f6f4 v[0:3], v[4:11], v[12:15], v[20:23] cbsz:1 blgp:4

// GFX950: v_mfma_f32_16x16x128_f8f6f4 a[0:3], a[4:11], a[12:15], a[20:23] cbsz:1 blgp:4 ; encoding: [0x00,0x81,0xad,0xd3,0x04,0x19,0x52,0x9c]
v_mfma_f32_16x16x128_f8f6f4 a[0:3], a[4:11], a[12:15], a[20:23] cbsz:1 blgp:4


// fp4 x bf8
// GFX950: v_mfma_f32_16x16x128_f8f6f4 v[0:3], v[4:7], v[12:19], v[20:23] cbsz:4 blgp:1 ; encoding: [0x00,0x04,0xad,0xd3,0x04,0x19,0x52,0x24]
v_mfma_f32_16x16x128_f8f6f4 v[0:3], v[4:7], v[12:19], v[20:23] cbsz:4 blgp:1

// GFX950: v_mfma_f32_16x16x128_f8f6f4 a[0:3], a[4:7], a[12:19], a[20:23] cbsz:4 blgp:1 ; encoding: [0x00,0x84,0xad,0xd3,0x04,0x19,0x52,0x3c]
v_mfma_f32_16x16x128_f8f6f4 a[0:3], a[4:7], a[12:19], a[20:23] cbsz:4 blgp:1


// bf6 x bf8
// GFX950: v_mfma_f32_16x16x128_f8f6f4 v[0:3], v[4:9], v[12:19], v[20:23] cbsz:3 blgp:1 ; encoding: [0x00,0x03,0xad,0xd3,0x04,0x19,0x52,0x24]
v_mfma_f32_16x16x128_f8f6f4 v[0:3], v[4:9], v[12:19], v[20:23] cbsz:3 blgp:1

// GFX950: v_mfma_f32_16x16x128_f8f6f4 a[0:3], a[4:9], a[12:19], a[20:23] cbsz:3 blgp:1 ; encoding: [0x00,0x83,0xad,0xd3,0x04,0x19,0x52,0x3c]
v_mfma_f32_16x16x128_f8f6f4 a[0:3], a[4:9], a[12:19], a[20:23] cbsz:3 blgp:1


// bf8 x bf6
// GFX950: v_mfma_f32_16x16x128_f8f6f4 v[0:3], v[4:11], v[12:17], v[20:23] cbsz:1 blgp:3 ; encoding: [0x00,0x01,0xad,0xd3,0x04,0x19,0x52,0x64]
v_mfma_f32_16x16x128_f8f6f4 v[0:3], v[4:11], v[12:17], v[20:23] cbsz:1 blgp:3

// GFX950: v_mfma_f32_16x16x128_f8f6f4 a[0:3], a[4:11], a[12:17], a[20:23] cbsz:1 blgp:3 ; encoding: [0x00,0x81,0xad,0xd3,0x04,0x19,0x52,0x7c]
v_mfma_f32_16x16x128_f8f6f4 a[0:3], a[4:11], a[12:17], a[20:23] cbsz:1 blgp:3


// bf6 x bf6
// GFX950: v_mfma_f32_16x16x128_f8f6f4 v[0:3], v[4:9], v[12:17], v[20:23] cbsz:3 blgp:3 ; encoding: [0x00,0x03,0xad,0xd3,0x04,0x19,0x52,0x64]
v_mfma_f32_16x16x128_f8f6f4 v[0:3], v[4:9], v[12:17], v[20:23] cbsz:3 blgp:3

// GFX950: v_mfma_f32_16x16x128_f8f6f4 a[0:3], a[4:9], a[12:17], a[20:23] cbsz:3 blgp:3 ; encoding: [0x00,0x83,0xad,0xd3,0x04,0x19,0x52,0x7c]
v_mfma_f32_16x16x128_f8f6f4 a[0:3], a[4:9], a[12:17], a[20:23] cbsz:3 blgp:3

// bf6 x fp6
// GFX950: v_mfma_f32_16x16x128_f8f6f4 v[0:3], v[4:9], v[12:17], v[20:23] cbsz:3 blgp:2 ; encoding: [0x00,0x03,0xad,0xd3,0x04,0x19,0x52,0x44]
v_mfma_f32_16x16x128_f8f6f4 v[0:3], v[4:9], v[12:17], v[20:23] cbsz:3 blgp:2

// GFX950: v_mfma_f32_16x16x128_f8f6f4 a[0:3], a[4:9], a[12:17], a[20:23] cbsz:3 blgp:2 ; encoding: [0x00,0x83,0xad,0xd3,0x04,0x19,0x52,0x5c]
v_mfma_f32_16x16x128_f8f6f4 a[0:3], a[4:9], a[12:17], a[20:23] cbsz:3 blgp:2


// fp6 x bf6
// GFX950: v_mfma_f32_16x16x128_f8f6f4 v[0:3], v[4:9], v[12:17], v[20:23] cbsz:2 blgp:3 ; encoding: [0x00,0x02,0xad,0xd3,0x04,0x19,0x52,0x64]
v_mfma_f32_16x16x128_f8f6f4 v[0:3], v[4:9], v[12:17], v[20:23] cbsz:2 blgp:3

// GFX950: v_mfma_f32_16x16x128_f8f6f4 a[0:3], a[4:9], a[12:17], a[20:23] cbsz:2 blgp:3 ; encoding: [0x00,0x82,0xad,0xd3,0x04,0x19,0x52,0x7c]
v_mfma_f32_16x16x128_f8f6f4 a[0:3], a[4:9], a[12:17], a[20:23] cbsz:2 blgp:3


// fp6 x fp4
// GFX950: v_mfma_f32_16x16x128_f8f6f4 v[0:3], v[4:9], v[12:15], v[20:23] cbsz:2 blgp:4 ; encoding: [0x00,0x02,0xad,0xd3,0x04,0x19,0x52,0x84]
v_mfma_f32_16x16x128_f8f6f4 v[0:3], v[4:9], v[12:15], v[20:23] cbsz:2 blgp:4

// GFX950: v_mfma_f32_16x16x128_f8f6f4 a[0:3], a[4:9], a[12:15], a[20:23] cbsz:2 blgp:4 ; encoding: [0x00,0x82,0xad,0xd3,0x04,0x19,0x52,0x9c]
v_mfma_f32_16x16x128_f8f6f4 a[0:3], a[4:9], a[12:15], a[20:23] cbsz:2 blgp:4


// fp4 x fp6
// GFX950: v_mfma_f32_16x16x128_f8f6f4 v[0:3], v[4:7], v[12:17], v[20:23] cbsz:4 blgp:2 ; encoding: [0x00,0x04,0xad,0xd3,0x04,0x19,0x52,0x44]
v_mfma_f32_16x16x128_f8f6f4 v[0:3], v[4:7], v[12:17], v[20:23] cbsz:4 blgp:2

// GFX950: v_mfma_f32_16x16x128_f8f6f4 a[0:3], a[4:7], a[12:17], a[20:23] cbsz:4 blgp:2 ; encoding: [0x00,0x84,0xad,0xd3,0x04,0x19,0x52,0x5c]
v_mfma_f32_16x16x128_f8f6f4 a[0:3], a[4:7], a[12:17], a[20:23] cbsz:4 blgp:2


// fp4 x fp4
// GFX950: v_mfma_f32_16x16x128_f8f6f4 v[0:3], v[4:7], v[12:15], v[20:23] cbsz:4 blgp:4 ; encoding: [0x00,0x04,0xad,0xd3,0x04,0x19,0x52,0x84]
v_mfma_f32_16x16x128_f8f6f4 v[0:3], v[4:7], v[12:15], v[20:23] cbsz:4 blgp:4

// GFX950: v_mfma_f32_16x16x128_f8f6f4 a[0:3], a[4:7], a[12:15], a[20:23] cbsz:4 blgp:4 ; encoding: [0x00,0x84,0xad,0xd3,0x04,0x19,0x52,0x9c]
v_mfma_f32_16x16x128_f8f6f4 a[0:3], a[4:7], a[12:15], a[20:23] cbsz:4 blgp:4

//===----------------------------------------------------------------------===//
// v_mfma_f32_32x32x64_f8f6f4 with appropriate register widths
//===----------------------------------------------------------------------===//

// bf8 x fp4
// GFX950: v_mfma_f32_32x32x64_f8f6f4 v[0:15], v[4:11], v[12:15], v[16:31] cbsz:1 blgp:4 ; encoding: [0x00,0x01,0xae,0xd3,0x04,0x19,0x42,0x84]
v_mfma_f32_32x32x64_f8f6f4 v[0:15], v[4:11], v[12:15], v[16:31] cbsz:1 blgp:4

// GFX950: v_mfma_f32_32x32x64_f8f6f4 a[0:15], a[4:11], a[12:15], a[16:31] cbsz:1 blgp:4 ; encoding: [0x00,0x81,0xae,0xd3,0x04,0x19,0x42,0x9c]
v_mfma_f32_32x32x64_f8f6f4 a[0:15], a[4:11], a[12:15], a[16:31] cbsz:1 blgp:4


// fp4 x bf8
// GFX950: v_mfma_f32_32x32x64_f8f6f4 v[0:15], v[4:7], v[8:15], v[16:31] cbsz:4 blgp:1 ; encoding: [0x00,0x04,0xae,0xd3,0x04,0x11,0x42,0x24]
v_mfma_f32_32x32x64_f8f6f4 v[0:15], v[4:7], v[8:15], v[16:31] cbsz:4 blgp:1

// GFX950: v_mfma_f32_32x32x64_f8f6f4 a[0:15], a[4:7], a[8:15], a[16:31] cbsz:4 blgp:1 ; encoding: [0x00,0x84,0xae,0xd3,0x04,0x11,0x42,0x3c]
v_mfma_f32_32x32x64_f8f6f4 a[0:15], a[4:7], a[8:15], a[16:31] cbsz:4 blgp:1


// bf6 x bf8
// GFX950: v_mfma_f32_32x32x64_f8f6f4 v[0:15], v[4:9], v[12:19], v[16:31] cbsz:3 blgp:1 ; encoding: [0x00,0x03,0xae,0xd3,0x04,0x19,0x42,0x24]
v_mfma_f32_32x32x64_f8f6f4 v[0:15], v[4:9], v[12:19], v[16:31] cbsz:3 blgp:1

// GFX950: v_mfma_f32_32x32x64_f8f6f4 a[0:15], a[4:9], a[12:19], a[16:31] cbsz:3 blgp:1 ; encoding: [0x00,0x83,0xae,0xd3,0x04,0x19,0x42,0x3c]
v_mfma_f32_32x32x64_f8f6f4 a[0:15], a[4:9], a[12:19], a[16:31] cbsz:3 blgp:1

// bf8 x bf6
// GFX950: v_mfma_f32_32x32x64_f8f6f4 v[0:15], v[4:11], v[12:17], v[16:31] cbsz:1 blgp:3 ; encoding: [0x00,0x01,0xae,0xd3,0x04,0x19,0x42,0x64]
v_mfma_f32_32x32x64_f8f6f4 v[0:15], v[4:11], v[12:17], v[16:31] cbsz:1 blgp:3

// GFX950: v_mfma_f32_32x32x64_f8f6f4 a[0:15], a[4:11], a[12:17], a[16:31] cbsz:1 blgp:3 ; encoding: [0x00,0x81,0xae,0xd3,0x04,0x19,0x42,0x7c]
v_mfma_f32_32x32x64_f8f6f4 a[0:15], a[4:11], a[12:17], a[16:31] cbsz:1 blgp:3


// bf6 x bf6
// GFX950: v_mfma_f32_32x32x64_f8f6f4 v[0:15], v[4:9], v[12:17], v[16:31] cbsz:3 blgp:3 ; encoding: [0x00,0x03,0xae,0xd3,0x04,0x19,0x42,0x64]
v_mfma_f32_32x32x64_f8f6f4 v[0:15], v[4:9], v[12:17], v[16:31] cbsz:3 blgp:3

// GFX950: v_mfma_f32_32x32x64_f8f6f4 a[0:15], a[4:9], a[12:17], a[16:31] cbsz:3 blgp:3 ; encoding: [0x00,0x83,0xae,0xd3,0x04,0x19,0x42,0x7c]
v_mfma_f32_32x32x64_f8f6f4 a[0:15], a[4:9], a[12:17], a[16:31] cbsz:3 blgp:3


// bf6 x fp6
// GFX950: v_mfma_f32_32x32x64_f8f6f4 v[0:15], v[4:9], v[12:17], v[16:31] cbsz:3 blgp:2 ; encoding: [0x00,0x03,0xae,0xd3,0x04,0x19,0x42,0x44]
v_mfma_f32_32x32x64_f8f6f4 v[0:15], v[4:9], v[12:17], v[16:31] cbsz:3 blgp:2

// GFX950: v_mfma_f32_32x32x64_f8f6f4 a[0:15], a[4:9], a[12:17], a[16:31] cbsz:3 blgp:2 ; encoding: [0x00,0x83,0xae,0xd3,0x04,0x19,0x42,0x5c]
v_mfma_f32_32x32x64_f8f6f4 a[0:15], a[4:9], a[12:17], a[16:31] cbsz:3 blgp:2


// fp6 x bf6
// GFX950: v_mfma_f32_32x32x64_f8f6f4 v[0:15], v[4:9], v[12:17], v[16:31] cbsz:2 blgp:3 ; encoding: [0x00,0x02,0xae,0xd3,0x04,0x19,0x42,0x64]
v_mfma_f32_32x32x64_f8f6f4 v[0:15], v[4:9], v[12:17], v[16:31] cbsz:2 blgp:3

// GFX950: v_mfma_f32_32x32x64_f8f6f4 a[0:15], a[4:9], a[12:17], a[16:31] cbsz:2 blgp:3 ; encoding: [0x00,0x82,0xae,0xd3,0x04,0x19,0x42,0x7c]
v_mfma_f32_32x32x64_f8f6f4 a[0:15], a[4:9], a[12:17], a[16:31] cbsz:2 blgp:3


// fp6 x fp4
// GFX950: v_mfma_f32_32x32x64_f8f6f4 v[0:15], v[4:9], v[12:15], v[16:31] cbsz:2 blgp:4 ; encoding: [0x00,0x02,0xae,0xd3,0x04,0x19,0x42,0x84]
v_mfma_f32_32x32x64_f8f6f4 v[0:15], v[4:9], v[12:15], v[16:31] cbsz:2 blgp:4

// GFX950: v_mfma_f32_32x32x64_f8f6f4 a[0:15], a[4:9], a[12:15], a[16:31] cbsz:2 blgp:4 ; encoding: [0x00,0x82,0xae,0xd3,0x04,0x19,0x42,0x9c]
v_mfma_f32_32x32x64_f8f6f4 a[0:15], a[4:9], a[12:15], a[16:31] cbsz:2 blgp:4


// fp4 x fp6
// GFX950: v_mfma_f32_32x32x64_f8f6f4 v[0:15], v[4:7], v[12:17], v[16:31] cbsz:4 blgp:2 ; encoding: [0x00,0x04,0xae,0xd3,0x04,0x19,0x42,0x44]
v_mfma_f32_32x32x64_f8f6f4 v[0:15], v[4:7], v[12:17], v[16:31] cbsz:4 blgp:2

// GFX950: v_mfma_f32_32x32x64_f8f6f4 a[0:15], a[4:7], a[12:17], a[16:31] cbsz:4 blgp:2 ; encoding: [0x00,0x84,0xae,0xd3,0x04,0x19,0x42,0x5c]
v_mfma_f32_32x32x64_f8f6f4 a[0:15], a[4:7], a[12:17], a[16:31] cbsz:4 blgp:2


// fp4 x fp4
// GFX950: v_mfma_f32_32x32x64_f8f6f4 v[0:15], v[4:7], v[12:15], v[16:31] cbsz:4 blgp:4 ; encoding: [0x00,0x04,0xae,0xd3,0x04,0x19,0x42,0x84]
v_mfma_f32_32x32x64_f8f6f4 v[0:15], v[4:7], v[12:15], v[16:31] cbsz:4 blgp:4

// GFX950: v_mfma_f32_32x32x64_f8f6f4 a[0:15], a[4:7], a[12:15], a[16:31] cbsz:4 blgp:4 ; encoding: [0x00,0x84,0xae,0xd3,0x04,0x19,0x42,0x9c]
v_mfma_f32_32x32x64_f8f6f4 a[0:15], a[4:7], a[12:15], a[16:31] cbsz:4 blgp:4

//===----------------------------------------------------------------------===//
// v_mfma_scale_f32_16x16x128_f8f6f4 corrected widths
//===----------------------------------------------------------------------===//

// fp8 x fp8
// GFX950: v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[4:11], v[12:19], v[20:23], v24, v25 op_sel_hi:[0,0,0] ; encoding: [0x00,0x00,0xac,0xd3,0x18,0x33,0x02,0x00,0x00,0x08,0xad,0xd3,0x04,0x19,0x52,0x04]
v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[4:11], v[12:19], v[20:23], v24, v25 cbsz:0 blgp:0

// GFX950: v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], a[4:11], a[12:19], a[20:23], v24, v25 op_sel_hi:[0,0,0] ; encoding: [0x00,0x00,0xac,0xd3,0x18,0x33,0x02,0x00,0x00,0x88,0xad,0xd3,0x04,0x19,0x52,0x1c]
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], a[4:11], a[12:19], a[20:23], v24, v25 cbsz:0 blgp:0


// bf8 x fp8
// GFX950: v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[4:11], v[12:19], v[20:23], v24, v25 op_sel_hi:[0,0,0] cbsz:1 ; encoding: [0x00,0x00,0xac,0xd3,0x18,0x33,0x02,0x00,0x00,0x09,0xad,0xd3,0x04,0x19,0x52,0x04]
v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[4:11], v[12:19], v[20:23], v24, v25 cbsz:1 blgp:0

// GFX950: v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], a[4:11], a[12:19], a[20:23], v24, v25 op_sel_hi:[0,0,0] cbsz:1 ; encoding: [0x00,0x00,0xac,0xd3,0x18,0x33,0x02,0x00,0x00,0x89,0xad,0xd3,0x04,0x19,0x52,0x1c]
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], a[4:11], a[12:19], a[20:23], v24, v25 cbsz:1 blgp:0

// fp8 x bf8
// GFX950: v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[4:11], v[12:19], v[20:23], v24, v25 op_sel_hi:[0,0,0] blgp:1 ; encoding: [0x00,0x00,0xac,0xd3,0x18,0x33,0x02,0x00,0x00,0x08,0xad,0xd3,0x04,0x19,0x52,0x24]
v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[4:11], v[12:19], v[20:23], v24, v25 cbsz:0 blgp:1

// GFX950: v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], a[4:11], a[12:19], a[20:23], v24, v25 op_sel_hi:[0,0,0] blgp:1 ; encoding: [0x00,0x00,0xac,0xd3,0x18,0x33,0x02,0x00,0x00,0x88,0xad,0xd3,0x04,0x19,0x52,0x3c]
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], a[4:11], a[12:19], a[20:23], v24, v25 cbsz:0 blgp:1


// bf8 x fp4
// GFX950: v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[4:11], v[12:15], v[20:23], v24, v25 op_sel_hi:[0,0,0] cbsz:1 blgp:4 ; encoding: [0x00,0x00,0xac,0xd3,0x18,0x33,0x02,0x00,0x00,0x09,0xad,0xd3,0x04,0x19,0x52,0x84]
v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[4:11], v[12:15], v[20:23], v24, v25 cbsz:1 blgp:4

// GFX950: v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], a[4:11], a[12:15], a[20:23], v24, v25 op_sel_hi:[0,0,0] cbsz:1 blgp:4 ; encoding: [0x00,0x00,0xac,0xd3,0x18,0x33,0x02,0x00,0x00,0x89,0xad,0xd3,0x04,0x19,0x52,0x9c]
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], a[4:11], a[12:15], a[20:23], v24, v25 cbsz:1 blgp:4

// fp4 x bf8
// GFX950: v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[4:7], v[12:19], v[20:23], v24, v25 op_sel_hi:[0,0,0] cbsz:4 blgp:1 ; encoding: [0x00,0x00,0xac,0xd3,0x18,0x33,0x02,0x00,0x00,0x0c,0xad,0xd3,0x04,0x19,0x52,0x24]
v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[4:7], v[12:19], v[20:23], v24, v25 cbsz:4 blgp:1

// GFX950: v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], a[4:7], a[12:19], a[20:23], v24, v25 op_sel_hi:[0,0,0] cbsz:4 blgp:1 ; encoding: [0x00,0x00,0xac,0xd3,0x18,0x33,0x02,0x00,0x00,0x8c,0xad,0xd3,0x04,0x19,0x52,0x3c]
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], a[4:7], a[12:19], a[20:23], v24, v25 cbsz:4 blgp:1


// bf6 x bf8
// GFX950: v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[4:9], v[12:19], v[20:23], v24, v25 op_sel_hi:[0,0,0] cbsz:3 blgp:1 ; encoding: [0x00,0x00,0xac,0xd3,0x18,0x33,0x02,0x00,0x00,0x0b,0xad,0xd3,0x04,0x19,0x52,0x24]
v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[4:9], v[12:19], v[20:23], v24, v25 cbsz:3 blgp:1

// GFX950: v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], a[4:9], a[12:19], a[20:23], v24, v25 op_sel_hi:[0,0,0] cbsz:3 blgp:1 ; encoding: [0x00,0x00,0xac,0xd3,0x18,0x33,0x02,0x00,0x00,0x8b,0xad,0xd3,0x04,0x19,0x52,0x3c]
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], a[4:9], a[12:19], a[20:23], v24, v25 cbsz:3 blgp:1


// bf8 x bf6
// GFX950: v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[4:11], v[12:17], v[20:23], v24, v25 op_sel_hi:[0,0,0] cbsz:1 blgp:3 ; encoding: [0x00,0x00,0xac,0xd3,0x18,0x33,0x02,0x00,0x00,0x09,0xad,0xd3,0x04,0x19,0x52,0x64]
v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[4:11], v[12:17], v[20:23], v24, v25 cbsz:1 blgp:3

// GFX950: v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], a[4:11], a[12:17], a[20:23], v24, v25 op_sel_hi:[0,0,0] cbsz:1 blgp:3 ; encoding: [0x00,0x00,0xac,0xd3,0x18,0x33,0x02,0x00,0x00,0x89,0xad,0xd3,0x04,0x19,0x52,0x7c]
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], a[4:11], a[12:17], a[20:23], v24, v25 cbsz:1 blgp:3


// bf6 x bf6
// GFX950: v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[4:9], v[12:17], v[20:23], v24, v25 op_sel_hi:[0,0,0] cbsz:3 blgp:3 ; encoding: [0x00,0x00,0xac,0xd3,0x18,0x33,0x02,0x00,0x00,0x0b,0xad,0xd3,0x04,0x19,0x52,0x64]
v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[4:9], v[12:17], v[20:23], v24, v25 cbsz:3 blgp:3

// GFX950: v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], a[4:9], a[12:17], a[20:23], v24, v25 op_sel_hi:[0,0,0] cbsz:3 blgp:3 ; encoding: [0x00,0x00,0xac,0xd3,0x18,0x33,0x02,0x00,0x00,0x8b,0xad,0xd3,0x04,0x19,0x52,0x7c]
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], a[4:9], a[12:17], a[20:23], v24, v25 cbsz:3 blgp:3


// bf6 x fp6
// GFX950: v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[4:9], v[12:17], v[20:23], v24, v25 op_sel_hi:[0,0,0] cbsz:3 blgp:2 ; encoding: [0x00,0x00,0xac,0xd3,0x18,0x33,0x02,0x00,0x00,0x0b,0xad,0xd3,0x04,0x19,0x52,0x44]
v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[4:9], v[12:17], v[20:23], v24, v25 cbsz:3 blgp:2

// GFX950: v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], a[4:9], a[12:17], a[20:23], v24, v25 op_sel_hi:[0,0,0] cbsz:3 blgp:2 ; encoding: [0x00,0x00,0xac,0xd3,0x18,0x33,0x02,0x00,0x00,0x8b,0xad,0xd3,0x04,0x19,0x52,0x5c]
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], a[4:9], a[12:17], a[20:23], v24, v25 cbsz:3 blgp:2


// fp6 x bf6
// GFX950: v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[4:9], v[12:17], v[20:23], v24, v25 op_sel_hi:[0,0,0] cbsz:2 blgp:3 ; encoding: [0x00,0x00,0xac,0xd3,0x18,0x33,0x02,0x00,0x00,0x0a,0xad,0xd3,0x04,0x19,0x52,0x64
v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[4:9], v[12:17], v[20:23], v24, v25 cbsz:2 blgp:3

// GFX950: v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], a[4:9], a[12:17], a[20:23], v24, v25 op_sel_hi:[0,0,0] cbsz:2 blgp:3 ; encoding: [0x00,0x00,0xac,0xd3,0x18,0x33,0x02,0x00,0x00,0x8a,0xad,0xd3,0x04,0x19,0x52,0x7c]
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], a[4:9], a[12:17], a[20:23], v24, v25 cbsz:2 blgp:3


// fp6 x fp4
// GFX950: v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[4:9], v[12:15], v[20:23], v24, v25 op_sel_hi:[0,0,0] cbsz:2 blgp:4 ; encoding: [0x00,0x00,0xac,0xd3,0x18,0x33,0x02,0x00,0x00,0x0a,0xad,0xd3,0x04,0x19,0x52,0x84]
v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[4:9], v[12:15], v[20:23], v24, v25 cbsz:2 blgp:4

// GFX950: v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], a[4:9], a[12:15], a[20:23], v24, v25 op_sel_hi:[0,0,0] cbsz:2 blgp:4 ; encoding: [0x00,0x00,0xac,0xd3,0x18,0x33,0x02,0x00,0x00,0x8a,0xad,0xd3,0x04,0x19,0x52,0x9c]
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], a[4:9], a[12:15], a[20:23], v24, v25 cbsz:2 blgp:4

// fp4 x fp6
// GFX950: v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[4:7], v[12:17], v[20:23], v24, v25 op_sel_hi:[0,0,0] cbsz:4 blgp:2 ; encoding: [0x00,0x00,0xac,0xd3,0x18,0x33,0x02,0x00,0x00,0x0c,0xad,0xd3,0x04,0x19,0x52,0x44]
v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[4:7], v[12:17], v[20:23], v24, v25 cbsz:4 blgp:2

// GFX950: v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], a[4:7], a[12:17], a[20:23], v24, v25 op_sel_hi:[0,0,0] cbsz:4 blgp:2 ; encoding: [0x00,0x00,0xac,0xd3,0x18,0x33,0x02,0x00,0x00,0x8c,0xad,0xd3,0x04,0x19,0x52,0x5c]
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], a[4:7], a[12:17], a[20:23], v24, v25 cbsz:4 blgp:2

// fp4 x fp4
// GFX950: v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[4:7], v[12:15], v[20:23], v24, v25 op_sel_hi:[0,0,0] cbsz:4 blgp:4 ; encoding: [0x00,0x00,0xac,0xd3,0x18,0x33,0x02,0x00,0x00,0x0c,0xad,0xd3,0x04,0x19,0x52,0x84]
v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[4:7], v[12:15], v[20:23], v24, v25 cbsz:4 blgp:4

// GFX950: v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], a[4:7], a[12:15], a[20:23], v24, v25 op_sel_hi:[0,0,0] cbsz:4 blgp:4 ; encoding: [0x00,0x00,0xac,0xd3,0x18,0x33,0x02,0x00,0x00,0x8c,0xad,0xd3,0x04,0x19,0x52,0x9c]
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], a[4:7], a[12:15], a[20:23], v24, v25 cbsz:4 blgp:4


//===----------------------------------------------------------------------===//
// v_mfma_scale_f32_32x32x64_f8f6f4 corrected widths
//===----------------------------------------------------------------------===//

// fp8 x fp8
// GFX950: v_mfma_scale_f32_32x32x64_f8f6f4 v[0:15], v[16:23], v[24:31], v[32:47], v48, v49 op_sel_hi:[0,0,0] ; encoding: [0x00,0x00,0xac,0xd3,0x30,0x63,0x02,0x00,0x00,0x08,0xae,0xd3,0x10,0x31,0x82,0x04]
v_mfma_scale_f32_32x32x64_f8f6f4 v[0:15], v[16:23], v[24:31], v[32:47], v48, v49 cbsz:0 blgp:0

// GFX950: v_mfma_scale_f32_32x32x64_f8f6f4 a[0:15], a[16:23], a[24:31], a[32:47], v48, v49 op_sel_hi:[0,0,0] ; encoding: [0x00,0x00,0xac,0xd3,0x30,0x63,0x02,0x00,0x00,0x88,0xae,0xd3,0x10,0x31,0x82,0x1c]
v_mfma_scale_f32_32x32x64_f8f6f4 a[0:15], a[16:23], a[24:31], a[32:47], v48, v49 cbsz:0 blgp:0

// bf8 x fp8
// GFX950: v_mfma_scale_f32_32x32x64_f8f6f4 v[0:15], v[16:23], v[24:31], v[32:47], v48, v49 op_sel_hi:[0,0,0] cbsz:1 ; encoding: [0x00,0x00,0xac,0xd3,0x30,0x63,0x02,0x00,0x00,0x09,0xae,0xd3,0x10,0x31,0x82,0x04]
v_mfma_scale_f32_32x32x64_f8f6f4 v[0:15], v[16:23], v[24:31], v[32:47], v48, v49 cbsz:1 blgp:0

// GFX950: v_mfma_scale_f32_32x32x64_f8f6f4 a[0:15], a[16:23], a[24:31], a[32:47], v48, v49 op_sel_hi:[0,0,0] cbsz:1 ; encoding: [0x00,0x00,0xac,0xd3,0x30,0x63,0x02,0x00,0x00,0x89,0xae,0xd3,0x10,0x31,0x82,0x1c]
v_mfma_scale_f32_32x32x64_f8f6f4 a[0:15], a[16:23], a[24:31], a[32:47], v48, v49 cbsz:1 blgp:0


// fp8 x bf8
// GFX950: v_mfma_scale_f32_32x32x64_f8f6f4 v[0:15], v[16:23], v[24:31], v[32:47], v48, v49 op_sel_hi:[0,0,0] blgp:1 ; encoding: [0x00,0x00,0xac,0xd3,0x30,0x63,0x02,0x00,0x00,0x08,0xae,0xd3,0x10,0x31,0x82,0x24]
v_mfma_scale_f32_32x32x64_f8f6f4 v[0:15], v[16:23], v[24:31], v[32:47], v48, v49 cbsz:0 blgp:1

// GFX950: v_mfma_scale_f32_32x32x64_f8f6f4 a[0:15], a[16:23], a[24:31], a[32:47], v48, v49 op_sel_hi:[0,0,0] blgp:1 ; encoding: [0x00,0x00,0xac,0xd3,0x30,0x63,0x02,0x00,0x00,0x88,0xae,0xd3,0x10,0x31,0x82,0x3c]
v_mfma_scale_f32_32x32x64_f8f6f4 a[0:15], a[16:23], a[24:31], a[32:47], v48, v49 cbsz:0 blgp:1

// bf8 x fp4
// GFX950: v_mfma_scale_f32_32x32x64_f8f6f4 v[0:15], v[16:23], v[24:27], v[32:47], v48, v49 op_sel_hi:[0,0,0] cbsz:1 blgp:4 ; encoding: [0x00,0x00,0xac,0xd3,0x30,0x63,0x02,0x00,0x00,0x09,0xae,0xd3,0x10,0x31,0x82,0x84]
v_mfma_scale_f32_32x32x64_f8f6f4 v[0:15], v[16:23], v[24:27], v[32:47], v48, v49 cbsz:1 blgp:4

// GFX950: v_mfma_scale_f32_32x32x64_f8f6f4 a[0:15], a[16:23], a[24:27], a[32:47], v48, v49 op_sel_hi:[0,0,0] cbsz:1 blgp:4 ; encoding: [0x00,0x00,0xac,0xd3,0x30,0x63,0x02,0x00,0x00,0x89,0xae,0xd3,0x10,0x31,0x82,0x9c]
v_mfma_scale_f32_32x32x64_f8f6f4 a[0:15], a[16:23], a[24:27], a[32:47], v48, v49 cbsz:1 blgp:4


// fp4 x bf8
// GFX950: v_mfma_scale_f32_32x32x64_f8f6f4 v[0:15], v[16:19], v[24:31], v[32:47], v48, v49 op_sel_hi:[0,0,0] cbsz:4 blgp:1 ; encoding: [0x00,0x00,0xac,0xd3,0x30,0x63,0x02,0x00,0x00,0x0c,0xae,0xd3,0x10,0x31,0x82,0x24]
v_mfma_scale_f32_32x32x64_f8f6f4 v[0:15], v[16:19], v[24:31], v[32:47], v48, v49 cbsz:4 blgp:1

// GFX950: v_mfma_scale_f32_32x32x64_f8f6f4 a[0:15], a[16:19], a[24:31], a[32:47], v48, v49 op_sel_hi:[0,0,0] cbsz:4 blgp:1 ; encoding: [0x00,0x00,0xac,0xd3,0x30,0x63,0x02,0x00,0x00,0x8c,0xae,0xd3,0x10,0x31,0x82,0x3c]
v_mfma_scale_f32_32x32x64_f8f6f4 a[0:15], a[16:19], a[24:31], a[32:47], v48, v49 cbsz:4 blgp:1


// bf6 x bf8
// GFX950: v_mfma_scale_f32_32x32x64_f8f6f4 v[0:15], v[16:21], v[24:31], v[32:47], v48, v49 op_sel_hi:[0,0,0] cbsz:3 blgp:1 ; encoding: [0x00,0x00,0xac,0xd3,0x30,0x63,0x02,0x00,0x00,0x0b,0xae,0xd3,0x10,0x31,0x82,0x24]
v_mfma_scale_f32_32x32x64_f8f6f4 v[0:15], v[16:21], v[24:31], v[32:47], v48, v49 cbsz:3 blgp:1

// GFX950: v_mfma_scale_f32_32x32x64_f8f6f4 a[0:15], a[16:21], a[24:31], a[32:47], v48, v49 op_sel_hi:[0,0,0] cbsz:3 blgp:1 ; encoding: [0x00,0x00,0xac,0xd3,0x30,0x63,0x02,0x00,0x00,0x8b,0xae,0xd3,0x10,0x31,0x82,0x3c]
v_mfma_scale_f32_32x32x64_f8f6f4 a[0:15], a[16:21], a[24:31], a[32:47], v48, v49 cbsz:3 blgp:1


// bf8 x bf6
// GFX950: v_mfma_scale_f32_32x32x64_f8f6f4 v[0:15], v[16:23], v[24:29], v[32:47], v48, v49 op_sel_hi:[0,0,0] cbsz:1 blgp:3 ; encoding: [0x00,0x00,0xac,0xd3,0x30,0x63,0x02,0x00,0x00,0x09,0xae,0xd3,0x10,0x31,0x82,0x64]
v_mfma_scale_f32_32x32x64_f8f6f4 v[0:15], v[16:23], v[24:29], v[32:47], v48, v49 cbsz:1 blgp:3

// GFX950: v_mfma_scale_f32_32x32x64_f8f6f4 a[0:15], a[16:23], a[24:29], a[32:47], v48, v49 op_sel_hi:[0,0,0] cbsz:1 blgp:3 ; encoding: [0x00,0x00,0xac,0xd3,0x30,0x63,0x02,0x00,0x00,0x89,0xae,0xd3,0x10,0x31,0x82,0x7c]
v_mfma_scale_f32_32x32x64_f8f6f4 a[0:15], a[16:23], a[24:29], a[32:47], v48, v49 cbsz:1 blgp:3

// bf6 x bf6
// GFX950: v_mfma_scale_f32_32x32x64_f8f6f4 v[0:15], v[16:21], v[24:29], v[32:47], v48, v49 op_sel_hi:[0,0,0] cbsz:3 blgp:3 ; encoding: [0x00,0x00,0xac,0xd3,0x30,0x63,0x02,0x00,0x00,0x0b,0xae,0xd3,0x10,0x31,0x82,0x64]
v_mfma_scale_f32_32x32x64_f8f6f4 v[0:15], v[16:21], v[24:29], v[32:47], v48, v49 cbsz:3 blgp:3

// GFX950: v_mfma_scale_f32_32x32x64_f8f6f4 a[0:15], a[16:21], a[24:29], a[32:47], v48, v49 op_sel_hi:[0,0,0] cbsz:3 blgp:3 ; encoding: [0x00,0x00,0xac,0xd3,0x30,0x63,0x02,0x00,0x00,0x8b,0xae,0xd3,0x10,0x31,0x82,0x7c]
v_mfma_scale_f32_32x32x64_f8f6f4 a[0:15], a[16:21], a[24:29], a[32:47], v48, v49 cbsz:3 blgp:3

// bf6 x fp6
// GFX950: v_mfma_scale_f32_32x32x64_f8f6f4 v[0:15], v[16:21], v[24:29], v[32:47], v48, v49 op_sel_hi:[0,0,0] cbsz:3 blgp:2 ; encoding: [0x00,0x00,0xac,0xd3,0x30,0x63,0x02,0x00,0x00,0x0b,0xae,0xd3,0x10,0x31,0x82,0x44]
v_mfma_scale_f32_32x32x64_f8f6f4 v[0:15], v[16:21], v[24:29], v[32:47], v48, v49 cbsz:3 blgp:2

// GFX950: v_mfma_scale_f32_32x32x64_f8f6f4 a[0:15], a[16:21], a[24:29], a[32:47], v48, v49 op_sel_hi:[0,0,0] cbsz:3 blgp:2 ; encoding: [0x00,0x00,0xac,0xd3,0x30,0x63,0x02,0x00,0x00,0x8b,0xae,0xd3,0x10,0x31,0x82,0x5c]
v_mfma_scale_f32_32x32x64_f8f6f4 a[0:15], a[16:21], a[24:29], a[32:47], v48, v49 cbsz:3 blgp:2


// fp6 x bf6
// GFX950: v_mfma_scale_f32_32x32x64_f8f6f4 v[0:15], v[16:21], v[24:29], v[32:47], v48, v49 op_sel_hi:[0,0,0] cbsz:2 blgp:3 ; encoding: [0x00,0x00,0xac,0xd3,0x30,0x63,0x02,0x00,0x00,0x0a,0xae,0xd3,0x10,0x31,0x82,0x64]
v_mfma_scale_f32_32x32x64_f8f6f4 v[0:15], v[16:21], v[24:29], v[32:47], v48, v49 cbsz:2 blgp:3

// GFX950: v_mfma_scale_f32_32x32x64_f8f6f4 a[0:15], a[16:21], a[24:29], a[32:47], v48, v49 op_sel_hi:[0,0,0] cbsz:2 blgp:3 ; encoding: [0x00,0x00,0xac,0xd3,0x30,0x63,0x02,0x00,0x00,0x8a,0xae,0xd3,0x10,0x31,0x82,0x7c]
v_mfma_scale_f32_32x32x64_f8f6f4 a[0:15], a[16:21], a[24:29], a[32:47], v48, v49 cbsz:2 blgp:3


// fp6 x fp4
// GFX950: v_mfma_scale_f32_32x32x64_f8f6f4 v[0:15], v[16:21], v[24:27], v[32:47], v48, v49 op_sel_hi:[0,0,0] cbsz:2 blgp:4 ; encoding: [0x00,0x00,0xac,0xd3,0x30,0x63,0x02,0x00,0x00,0x0a,0xae,0xd3,0x10,0x31,0x82,0x84]
v_mfma_scale_f32_32x32x64_f8f6f4 v[0:15], v[16:21], v[24:27], v[32:47], v48, v49 cbsz:2 blgp:4

// GFX950: v_mfma_scale_f32_32x32x64_f8f6f4 a[0:15], a[16:21], a[24:27], a[32:47], v48, v49 op_sel_hi:[0,0,0] cbsz:2 blgp:4 ; encoding: [0x00,0x00,0xac,0xd3,0x30,0x63,0x02,0x00,0x00,0x8a,0xae,0xd3,0x10,0x31,0x82,0x9c]
v_mfma_scale_f32_32x32x64_f8f6f4 a[0:15], a[16:21], a[24:27], a[32:47], v48, v49 cbsz:2 blgp:4


// fp4 x fp6
// GFX950: v_mfma_scale_f32_32x32x64_f8f6f4 v[0:15], v[16:19], v[24:29], v[32:47], v48, v49 op_sel_hi:[0,0,0] cbsz:4 blgp:2 ; encoding: [0x00,0x00,0xac,0xd3,0x30,0x63,0x02,0x00,0x00,0x0c,0xae,0xd3,0x10,0x31,0x82,0x44]
v_mfma_scale_f32_32x32x64_f8f6f4 v[0:15], v[16:19], v[24:29], v[32:47], v48, v49 cbsz:4 blgp:2

// GFX950: v_mfma_scale_f32_32x32x64_f8f6f4 a[0:15], a[16:19], a[24:29], a[32:47], v48, v49 op_sel_hi:[0,0,0] cbsz:4 blgp:2 ; encoding: [0x00,0x00,0xac,0xd3,0x30,0x63,0x02,0x00,0x00,0x8c,0xae,0xd3,0x10,0x31,0x82,0x5c]
v_mfma_scale_f32_32x32x64_f8f6f4 a[0:15], a[16:19], a[24:29], a[32:47], v48, v49 cbsz:4 blgp:2


// fp4 x fp4
// GFX950: v_mfma_scale_f32_32x32x64_f8f6f4 v[0:15], v[16:19], v[24:27], v[32:47], v48, v49 op_sel_hi:[0,0,0] cbsz:4 blgp:4 ; encoding: [0x00,0x00,0xac,0xd3,0x30,0x63,0x02,0x00,0x00,0x0c,0xae,0xd3,0x10,0x31,0x82,0x84]
v_mfma_scale_f32_32x32x64_f8f6f4 v[0:15], v[16:19], v[24:27], v[32:47], v48, v49 cbsz:4 blgp:4

// GFX950: v_mfma_scale_f32_32x32x64_f8f6f4 a[0:15], a[16:19], a[24:27], a[32:47], v48, v49 op_sel_hi:[0,0,0] cbsz:4 blgp:4 ; encoding: [0x00,0x00,0xac,0xd3,0x30,0x63,0x02,0x00,0x00,0x8c,0xae,0xd3,0x10,0x31,0x82,0x9c]
v_mfma_scale_f32_32x32x64_f8f6f4 a[0:15], a[16:19], a[24:27], a[32:47], v48, v49 cbsz:4 blgp:4

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


//===----------------------------------------------------------------------===//
// SMFMAC opcodes.
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// v_smfmac_f32_16x16x64_f16
//===----------------------------------------------------------------------===//

// GFX950: v_smfmac_f32_16x16x64_f16 v[10:13], a[2:5], v[4:11], v3 cbsz:3 abid:1 ; encoding: [0x0a,0x0b,0xda,0xd3,0x02,0x09,0x0e,0x0c]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_smfmac_f32_16x16x64_f16 v[10:13], a[2:5], v[4:11], v3 cbsz:3 abid:1

// GFX950: v_smfmac_f32_16x16x64_f16 v[10:13], a[2:5], v[4:11], v3 cbsz:3 abid:1 ; encoding: [0x0a,0x0b,0xda,0xd3,0x02,0x09,0x0e,0x0c]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_smfmac_f32_16x16x64f16 v[10:13], a[2:5], v[4:11], v3 cbsz:3 abid:1

// GFX950: v_smfmac_f32_16x16x64_f16 a[10:13], v[2:5], a[4:11], v1 ; encoding: [0x0a,0x80,0xda,0xd3,0x02,0x09,0x06,0x14]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_smfmac_f32_16x16x64_f16 a[10:13], v[2:5], a[4:11], v1

// GFX950: v_smfmac_f32_16x16x64_f16 v[10:13], a[2:5], v[4:11], v2 cbsz:3 abid:1 ; encoding: [0x0a,0x0b,0xda,0xd3,0x02,0x09,0x0a,0x0c]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_smfmac_f32_16x16x64_f16 v[10:13], a[2:5], v[4:11], v2 cbsz:3 abid:1

// GFX950: v_smfmac_f32_16x16x64_f16 a[10:13], v[2:5], a[4:11], v3 ; encoding: [0x0a,0x80,0xda,0xd3,0x02,0x09,0x0e,0x14]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_smfmac_f32_16x16x64_f16 a[10:13], v[2:5], a[4:11], v3

// GFX950: v_smfmac_f32_16x16x64_f16 v[10:13], v[2:5], v[6:13], v2 cbsz:3 abid:1 ; encoding: [0x0a,0x0b,0xda,0xd3,0x02,0x0d,0x0a,0x04]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_smfmac_f32_16x16x64_f16 v[10:13], v[2:5], v[6:13], v2 cbsz:3 abid:1

// GFX950: v_smfmac_f32_16x16x64_f16 a[10:13], a[2:5], a[6:13], v2 cbsz:3 abid:1 ; encoding: [0x0a,0x8b,0xda,0xd3,0x02,0x0d,0x0a,0x1c]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_smfmac_f32_16x16x64_f16 a[10:13], a[2:5], a[6:13], v2 cbsz:3 abid:1

// GFX950: v_smfmac_f32_16x16x64_f16 a[10:13], a[2:5], a[6:13], v3 cbsz:1 abid:3 ; encoding: [0x0a,0x99,0xda,0xd3,0x02,0x0d,0x0e,0x1c]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_smfmac_f32_16x16x64_f16 a[10:13], a[2:5], a[6:13], v3 cbsz:1 abid:3

//===----------------------------------------------------------------------===//
// v_smfmac_f32_32x32x32_f16
//===----------------------------------------------------------------------===//

// GFX950: v_smfmac_f32_32x32x32_f16 v[10:25], a[2:5], v[4:11], v3 cbsz:3 abid:1 ; encoding: [0x0a,0x0b,0xdb,0xd3,0x02,0x09,0x0e,0x0c]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_smfmac_f32_32x32x32_f16 v[10:25], a[2:5], v[4:11], v3 cbsz:3 abid:1

// GFX950: v_smfmac_f32_32x32x32_f16 v[10:25], a[2:5], v[4:11], v3 cbsz:3 abid:1 ; encoding: [0x0a,0x0b,0xdb,0xd3,0x02,0x09,0x0e,0x0c]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_smfmac_f32_32x32x32f16 v[10:25], a[2:5], v[4:11], v3 cbsz:3 abid:1

// GFX950: v_smfmac_f32_32x32x32_f16 a[10:25], v[2:5], a[4:11], v1 ; encoding: [0x0a,0x80,0xdb,0xd3,0x02,0x09,0x06,0x14]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_smfmac_f32_32x32x32_f16 a[10:25], v[2:5], a[4:11], v1

// GFX950: v_smfmac_f32_32x32x32_f16 v[10:25], a[2:5], v[4:11], v2 cbsz:3 abid:1 ; encoding: [0x0a,0x0b,0xdb,0xd3,0x02,0x09,0x0a,0x0c]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_smfmac_f32_32x32x32_f16 v[10:25], a[2:5], v[4:11], v2 cbsz:3 abid:1

// GFX950: v_smfmac_f32_32x32x32_f16 a[10:25], v[2:5], a[4:11], v3 ; encoding: [0x0a,0x80,0xdb,0xd3,0x02,0x09,0x0e,0x14]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_smfmac_f32_32x32x32_f16 a[10:25], v[2:5], a[4:11], v3

// GFX950: v_smfmac_f32_32x32x32_f16 v[10:25], v[2:5], v[6:13], v2 cbsz:3 abid:1 ; encoding: [0x0a,0x0b,0xdb,0xd3,0x02,0x0d,0x0a,0x04]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_smfmac_f32_32x32x32_f16 v[10:25], v[2:5], v[6:13], v2 cbsz:3 abid:1

// GFX950: v_smfmac_f32_32x32x32_f16 a[10:25], a[2:5], a[6:13], v2 cbsz:3 abid:1 ; encoding: [0x0a,0x8b,0xdb,0xd3,0x02,0x0d,0x0a,0x1c]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_smfmac_f32_32x32x32_f16 a[10:25], a[2:5], a[6:13], v2 cbsz:3 abid:1

// GFX950: v_smfmac_f32_32x32x32_f16 a[10:25], a[2:5], a[6:13], v3 cbsz:1 abid:3 ; encoding: [0x0a,0x99,0xdb,0xd3,0x02,0x0d,0x0e,0x1c]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_smfmac_f32_32x32x32_f16 a[10:25], a[2:5], a[6:13], v3 cbsz:1 abid:3

//===----------------------------------------------------------------------===//
// v_smfmac_f32_16x16x64_bf16
//===----------------------------------------------------------------------===//

// GFX950: v_smfmac_f32_16x16x64_bf16 v[10:13], a[2:5], v[4:11], v3 cbsz:3 abid:1 ; encoding: [0x0a,0x0b,0xb9,0xd3,0x02,0x09,0x0e,0x0c]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_smfmac_f32_16x16x64_bf16 v[10:13], a[2:5], v[4:11], v3 cbsz:3 abid:1

// GFX950: v_smfmac_f32_16x16x64_bf16 v[10:13], a[2:5], v[4:11], v3 cbsz:3 abid:1 ; encoding: [0x0a,0x0b,0xb9,0xd3,0x02,0x09,0x0e,0x0c]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_smfmac_f32_16x16x64bf16 v[10:13], a[2:5], v[4:11], v3 cbsz:3 abid:1

// GFX950: v_smfmac_f32_16x16x64_bf16 a[10:13], v[2:5], a[4:11], v1 ; encoding: [0x0a,0x80,0xb9,0xd3,0x02,0x09,0x06,0x14]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_smfmac_f32_16x16x64_bf16 a[10:13], v[2:5], a[4:11], v1

// GFX950: v_smfmac_f32_16x16x64_bf16 v[10:13], a[2:5], v[4:11], v2 cbsz:3 abid:1 ; encoding: [0x0a,0x0b,0xb9,0xd3,0x02,0x09,0x0a,0x0c]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_smfmac_f32_16x16x64_bf16 v[10:13], a[2:5], v[4:11], v2 cbsz:3 abid:1

// GFX950: v_smfmac_f32_16x16x64_bf16 a[10:13], v[2:5], a[4:11], v3 ; encoding: [0x0a,0x80,0xb9,0xd3,0x02,0x09,0x0e,0x14]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_smfmac_f32_16x16x64_bf16 a[10:13], v[2:5], a[4:11], v3

// GFX950: v_smfmac_f32_16x16x64_bf16 v[10:13], v[2:5], v[6:13], v2 cbsz:3 abid:1 ; encoding: [0x0a,0x0b,0xb9,0xd3,0x02,0x0d,0x0a,0x04]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_smfmac_f32_16x16x64_bf16 v[10:13], v[2:5], v[6:13], v2 cbsz:3 abid:1

// GFX950: v_smfmac_f32_16x16x64_bf16 a[10:13], a[2:5], a[6:13], v2 cbsz:3 abid:1 ; encoding: [0x0a,0x8b,0xb9,0xd3,0x02,0x0d,0x0a,0x1c]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_smfmac_f32_16x16x64_bf16 a[10:13], a[2:5], a[6:13], v2 cbsz:3 abid:1

// GFX950: v_smfmac_f32_16x16x64_bf16 a[10:13], a[2:5], a[6:13], v3 cbsz:1 abid:3 ; encoding: [0x0a,0x99,0xb9,0xd3,0x02,0x0d,0x0e,0x1c]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_smfmac_f32_16x16x64_bf16 a[10:13], a[2:5], a[6:13], v3 cbsz:1 abid:3

//===----------------------------------------------------------------------===//
// v_smfmac_f32_32x32x32_bf16
//===----------------------------------------------------------------------===//

// GFX950: v_smfmac_f32_32x32x32_bf16 v[10:25], a[2:5], v[4:11], v3 cbsz:3 abid:1 ; encoding: [0x0a,0x0b,0xc6,0xd3,0x02,0x09,0x0e,0x0c]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_smfmac_f32_32x32x32_bf16 v[10:25], a[2:5], v[4:11], v3 cbsz:3 abid:1

// GFX950: v_smfmac_f32_32x32x32_bf16 v[10:25], a[2:5], v[4:11], v3 cbsz:3 abid:1 ; encoding: [0x0a,0x0b,0xc6,0xd3,0x02,0x09,0x0e,0x0c]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_smfmac_f32_32x32x32bf16 v[10:25], a[2:5], v[4:11], v3 cbsz:3 abid:1

// GFX950: v_smfmac_f32_32x32x32_bf16 a[10:25], v[2:5], a[4:11], v1 ; encoding: [0x0a,0x80,0xc6,0xd3,0x02,0x09,0x06,0x14]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_smfmac_f32_32x32x32_bf16 a[10:25], v[2:5], a[4:11], v1

// GFX950: v_smfmac_f32_32x32x32_bf16 v[10:25], a[2:5], v[4:11], v2 cbsz:3 abid:1 ; encoding: [0x0a,0x0b,0xc6,0xd3,0x02,0x09,0x0a,0x0c]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_smfmac_f32_32x32x32_bf16 v[10:25], a[2:5], v[4:11], v2 cbsz:3 abid:1

// GFX950: v_smfmac_f32_32x32x32_bf16 a[10:25], v[2:5], a[4:11], v3 ; encoding: [0x0a,0x80,0xc6,0xd3,0x02,0x09,0x0e,0x14]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_smfmac_f32_32x32x32_bf16 a[10:25], v[2:5], a[4:11], v3

// GFX950: v_smfmac_f32_32x32x32_bf16 v[10:25], v[2:5], v[6:13], v2 cbsz:3 abid:1 ; encoding: [0x0a,0x0b,0xc6,0xd3,0x02,0x0d,0x0a,0x04]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_smfmac_f32_32x32x32_bf16 v[10:25], v[2:5], v[6:13], v2 cbsz:3 abid:1

// GFX950: v_smfmac_f32_32x32x32_bf16 a[10:25], a[2:5], a[6:13], v2 cbsz:3 abid:1 ; encoding: [0x0a,0x8b,0xc6,0xd3,0x02,0x0d,0x0a,0x1c]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_smfmac_f32_32x32x32_bf16 a[10:25], a[2:5], a[6:13], v2 cbsz:3 abid:1

// GFX950: v_smfmac_f32_32x32x32_bf16 a[10:25], a[2:5], a[6:13], v3 cbsz:1 abid:3 ; encoding: [0x0a,0x99,0xc6,0xd3,0x02,0x0d,0x0e,0x1c]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_smfmac_f32_32x32x32_bf16 a[10:25], a[2:5], a[6:13], v3 cbsz:1 abid:3

//===----------------------------------------------------------------------===//
// v_smfmac_i32_16x16x128_i8
//===----------------------------------------------------------------------===//

// GFX950: v_smfmac_i32_16x16x128_i8 v[10:13], a[2:5], v[4:11], v3 cbsz:3 abid:1 ; encoding: [0x0a,0x0b,0xba,0xd3,0x02,0x09,0x0e,0x0c]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_smfmac_i32_16x16x128_i8 v[10:13], a[2:5], v[4:11], v3 cbsz:3 abid:1

// GFX950: v_smfmac_i32_16x16x128_i8 v[10:13], a[2:5], v[4:11], v3 cbsz:3 abid:1 ; encoding: [0x0a,0x0b,0xba,0xd3,0x02,0x09,0x0e,0x0c]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_smfmac_i32_16x16x128i8 v[10:13], a[2:5], v[4:11], v3 cbsz:3 abid:1

// GFX950: v_smfmac_i32_16x16x128_i8 a[10:13], v[2:5], a[4:11], v1 ; encoding: [0x0a,0x80,0xba,0xd3,0x02,0x09,0x06,0x14]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_smfmac_i32_16x16x128_i8 a[10:13], v[2:5], a[4:11], v1

// GFX950: v_smfmac_i32_16x16x128_i8 v[10:13], a[2:5], v[4:11], v2 cbsz:3 abid:1 ; encoding: [0x0a,0x0b,0xba,0xd3,0x02,0x09,0x0a,0x0c]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_smfmac_i32_16x16x128_i8 v[10:13], a[2:5], v[4:11], v2 cbsz:3 abid:1

// GFX950: v_smfmac_i32_16x16x128_i8 a[10:13], v[2:5], a[4:11], v3 ; encoding: [0x0a,0x80,0xba,0xd3,0x02,0x09,0x0e,0x14]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_smfmac_i32_16x16x128_i8 a[10:13], v[2:5], a[4:11], v3

// GFX950: v_smfmac_i32_16x16x128_i8 v[10:13], v[2:5], v[6:13], v2 cbsz:3 abid:1 ; encoding: [0x0a,0x0b,0xba,0xd3,0x02,0x0d,0x0a,0x04]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_smfmac_i32_16x16x128_i8 v[10:13], v[2:5], v[6:13], v2 cbsz:3 abid:1

// GFX950: v_smfmac_i32_16x16x128_i8 a[10:13], a[2:5], a[6:13], v2 cbsz:3 abid:1 ; encoding: [0x0a,0x8b,0xba,0xd3,0x02,0x0d,0x0a,0x1c]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_smfmac_i32_16x16x128_i8 a[10:13], a[2:5], a[6:13], v2 cbsz:3 abid:1

// GFX950: v_smfmac_i32_16x16x128_i8 a[10:13], a[2:5], a[6:13], v3 cbsz:1 abid:3 ; encoding: [0x0a,0x99,0xba,0xd3,0x02,0x0d,0x0e,0x1c]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_smfmac_i32_16x16x128_i8 a[10:13], a[2:5], a[6:13], v3 cbsz:1 abid:3

//===----------------------------------------------------------------------===//
// v_smfmac_i32_32x32x64_i8
//===----------------------------------------------------------------------===//

// GFX950: v_smfmac_i32_32x32x64_i8 v[10:25], a[2:5], v[4:11], v3 cbsz:3 abid:1 ; encoding: [0x0a,0x0b,0xc7,0xd3,0x02,0x09,0x0e,0x0c]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_smfmac_i32_32x32x64_i8 v[10:25], a[2:5], v[4:11], v3 cbsz:3 abid:1

// GFX950: v_smfmac_i32_32x32x64_i8 v[10:25], a[2:5], v[4:11], v3 cbsz:3 abid:1 ; encoding: [0x0a,0x0b,0xc7,0xd3,0x02,0x09,0x0e,0x0c]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_smfmac_i32_32x32x64i8 v[10:25], a[2:5], v[4:11], v3 cbsz:3 abid:1

// GFX950: v_smfmac_i32_32x32x64_i8 a[10:25], v[2:5], a[4:11], v1 ; encoding: [0x0a,0x80,0xc7,0xd3,0x02,0x09,0x06,0x14]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_smfmac_i32_32x32x64_i8 a[10:25], v[2:5], a[4:11], v1

// GFX950: v_smfmac_i32_32x32x64_i8 v[10:25], a[2:5], v[4:11], v2 cbsz:3 abid:1 ; encoding: [0x0a,0x0b,0xc7,0xd3,0x02,0x09,0x0a,0x0c]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_smfmac_i32_32x32x64_i8 v[10:25], a[2:5], v[4:11], v2 cbsz:3 abid:1

// GFX950: v_smfmac_i32_32x32x64_i8 a[10:25], v[2:5], a[4:11], v3 ; encoding: [0x0a,0x80,0xc7,0xd3,0x02,0x09,0x0e,0x14]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_smfmac_i32_32x32x64_i8 a[10:25], v[2:5], a[4:11], v3

// GFX950: v_smfmac_i32_32x32x64_i8 v[10:25], v[2:5], v[6:13], v2 cbsz:3 abid:1 ; encoding: [0x0a,0x0b,0xc7,0xd3,0x02,0x0d,0x0a,0x04]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_smfmac_i32_32x32x64_i8 v[10:25], v[2:5], v[6:13], v2 cbsz:3 abid:1

// GFX950: v_smfmac_i32_32x32x64_i8 a[10:25], a[2:5], a[6:13], v2 cbsz:3 abid:1 ; encoding: [0x0a,0x8b,0xc7,0xd3,0x02,0x0d,0x0a,0x1c]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_smfmac_i32_32x32x64_i8 a[10:25], a[2:5], a[6:13], v2 cbsz:3 abid:1

// GFX950: v_smfmac_i32_32x32x64_i8 a[10:25], a[2:5], a[6:13], v3 cbsz:1 abid:3 ; encoding: [0x0a,0x99,0xc7,0xd3,0x02,0x0d,0x0e,0x1c]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_smfmac_i32_32x32x64_i8 a[10:25], a[2:5], a[6:13], v3 cbsz:1 abid:3

//===----------------------------------------------------------------------===//
// v_smfmac_f32_16x16x128_bf8_bf8
//===----------------------------------------------------------------------===//

// GFX950: v_smfmac_f32_16x16x128_bf8_bf8 v[10:13], a[2:5], v[4:11], v3 cbsz:3 abid:1 ; encoding: [0x0a,0x0b,0xbb,0xd3,0x02,0x09,0x0e,0x0c]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_smfmac_f32_16x16x128_bf8_bf8 v[10:13], a[2:5], v[4:11], v3 cbsz:3 abid:1

// GFX950: v_smfmac_f32_16x16x128_bf8_bf8 v[10:13], a[2:5], v[4:11], v3 cbsz:3 abid:1 ; encoding: [0x0a,0x0b,0xbb,0xd3,0x02,0x09,0x0e,0x0c]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_smfmac_f32_16x16x128bf8bf8 v[10:13], a[2:5], v[4:11], v3 cbsz:3 abid:1

// GFX950: v_smfmac_f32_16x16x128_bf8_bf8 a[10:13], v[2:5], a[4:11], v1 ; encoding: [0x0a,0x80,0xbb,0xd3,0x02,0x09,0x06,0x14]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_smfmac_f32_16x16x128_bf8_bf8 a[10:13], v[2:5], a[4:11], v1

// GFX950: v_smfmac_f32_16x16x128_bf8_bf8 v[10:13], a[2:5], v[4:11], v2 cbsz:3 abid:1 ; encoding: [0x0a,0x0b,0xbb,0xd3,0x02,0x09,0x0a,0x0c]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_smfmac_f32_16x16x128_bf8_bf8 v[10:13], a[2:5], v[4:11], v2 cbsz:3 abid:1

// GFX950: v_smfmac_f32_16x16x128_bf8_bf8 a[10:13], v[2:5], a[4:11], v3 ; encoding: [0x0a,0x80,0xbb,0xd3,0x02,0x09,0x0e,0x14]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_smfmac_f32_16x16x128_bf8_bf8 a[10:13], v[2:5], a[4:11], v3

// GFX950: v_smfmac_f32_16x16x128_bf8_bf8 v[10:13], v[2:5], v[6:13], v2 cbsz:3 abid:1 ; encoding: [0x0a,0x0b,0xbb,0xd3,0x02,0x0d,0x0a,0x04]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_smfmac_f32_16x16x128_bf8_bf8 v[10:13], v[2:5], v[6:13], v2 cbsz:3 abid:1

// GFX950: v_smfmac_f32_16x16x128_bf8_bf8 a[10:13], a[2:5], a[6:13], v2 cbsz:3 abid:1 ; encoding: [0x0a,0x8b,0xbb,0xd3,0x02,0x0d,0x0a,0x1c]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_smfmac_f32_16x16x128_bf8_bf8 a[10:13], a[2:5], a[6:13], v2 cbsz:3 abid:1

// GFX950: v_smfmac_f32_16x16x128_bf8_bf8 a[10:13], a[2:5], a[6:13], v3 cbsz:1 abid:3 ; encoding: [0x0a,0x99,0xbb,0xd3,0x02,0x0d,0x0e,0x1c]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_smfmac_f32_16x16x128_bf8_bf8 a[10:13], a[2:5], a[6:13], v3 cbsz:1 abid:3

//===----------------------------------------------------------------------===//
// v_smfmac_f32_16x16x128_bf8_fp8
//===----------------------------------------------------------------------===//

// GFX950: v_smfmac_f32_16x16x128_bf8_fp8 v[10:13], a[2:5], v[4:11], v3 cbsz:3 abid:1 ; encoding: [0x0a,0x0b,0xbc,0xd3,0x02,0x09,0x0e,0x0c]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_smfmac_f32_16x16x128_bf8_fp8 v[10:13], a[2:5], v[4:11], v3 cbsz:3 abid:1

// GFX950: v_smfmac_f32_16x16x128_bf8_fp8 v[10:13], a[2:5], v[4:11], v3 cbsz:3 abid:1 ; encoding: [0x0a,0x0b,0xbc,0xd3,0x02,0x09,0x0e,0x0c]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_smfmac_f32_16x16x128bf8fp8 v[10:13], a[2:5], v[4:11], v3 cbsz:3 abid:1

// GFX950: v_smfmac_f32_16x16x128_bf8_fp8 a[10:13], v[2:5], a[4:11], v1 ; encoding: [0x0a,0x80,0xbc,0xd3,0x02,0x09,0x06,0x14]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_smfmac_f32_16x16x128_bf8_fp8 a[10:13], v[2:5], a[4:11], v1

// GFX950: v_smfmac_f32_16x16x128_bf8_fp8 v[10:13], a[2:5], v[4:11], v2 cbsz:3 abid:1 ; encoding: [0x0a,0x0b,0xbc,0xd3,0x02,0x09,0x0a,0x0c]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_smfmac_f32_16x16x128_bf8_fp8 v[10:13], a[2:5], v[4:11], v2 cbsz:3 abid:1

// GFX950: v_smfmac_f32_16x16x128_bf8_fp8 a[10:13], v[2:5], a[4:11], v3 ; encoding: [0x0a,0x80,0xbc,0xd3,0x02,0x09,0x0e,0x14]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_smfmac_f32_16x16x128_bf8_fp8 a[10:13], v[2:5], a[4:11], v3

// GFX950: v_smfmac_f32_16x16x128_bf8_fp8 v[10:13], v[2:5], v[6:13], v2 cbsz:3 abid:1 ; encoding: [0x0a,0x0b,0xbc,0xd3,0x02,0x0d,0x0a,0x04]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_smfmac_f32_16x16x128_bf8_fp8 v[10:13], v[2:5], v[6:13], v2 cbsz:3 abid:1

// GFX950: v_smfmac_f32_16x16x128_bf8_fp8 a[10:13], a[2:5], a[6:13], v2 cbsz:3 abid:1 ; encoding: [0x0a,0x8b,0xbc,0xd3,0x02,0x0d,0x0a,0x1c]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_smfmac_f32_16x16x128_bf8_fp8 a[10:13], a[2:5], a[6:13], v2 cbsz:3 abid:1

// GFX950: v_smfmac_f32_16x16x128_bf8_fp8 a[10:13], a[2:5], a[6:13], v3 cbsz:1 abid:3 ; encoding: [0x0a,0x99,0xbc,0xd3,0x02,0x0d,0x0e,0x1c]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_smfmac_f32_16x16x128_bf8_fp8 a[10:13], a[2:5], a[6:13], v3 cbsz:1 abid:3

//===----------------------------------------------------------------------===//
// v_smfmac_f32_16x16x128_fp8_bf8
//===----------------------------------------------------------------------===//

// GFX950: v_smfmac_f32_16x16x128_fp8_bf8 v[10:13], a[2:5], v[4:11], v3 cbsz:3 abid:1 ; encoding: [0x0a,0x0b,0xbd,0xd3,0x02,0x09,0x0e,0x0c]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_smfmac_f32_16x16x128_fp8_bf8 v[10:13], a[2:5], v[4:11], v3 cbsz:3 abid:1

// GFX950: v_smfmac_f32_16x16x128_fp8_bf8 v[10:13], a[2:5], v[4:11], v3 cbsz:3 abid:1 ; encoding: [0x0a,0x0b,0xbd,0xd3,0x02,0x09,0x0e,0x0c]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_smfmac_f32_16x16x128fp8bf8 v[10:13], a[2:5], v[4:11], v3 cbsz:3 abid:1

// GFX950: v_smfmac_f32_16x16x128_fp8_bf8 a[10:13], v[2:5], a[4:11], v1 ; encoding: [0x0a,0x80,0xbd,0xd3,0x02,0x09,0x06,0x14]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_smfmac_f32_16x16x128_fp8_bf8 a[10:13], v[2:5], a[4:11], v1

// GFX950: v_smfmac_f32_16x16x128_fp8_bf8 v[10:13], a[2:5], v[4:11], v2 cbsz:3 abid:1 ; encoding: [0x0a,0x0b,0xbd,0xd3,0x02,0x09,0x0a,0x0c]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_smfmac_f32_16x16x128_fp8_bf8 v[10:13], a[2:5], v[4:11], v2 cbsz:3 abid:1

// GFX950: v_smfmac_f32_16x16x128_fp8_bf8 a[10:13], v[2:5], a[4:11], v3 ; encoding: [0x0a,0x80,0xbd,0xd3,0x02,0x09,0x0e,0x14]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_smfmac_f32_16x16x128_fp8_bf8 a[10:13], v[2:5], a[4:11], v3

// GFX950: v_smfmac_f32_16x16x128_fp8_bf8 v[10:13], v[2:5], v[6:13], v2 cbsz:3 abid:1 ; encoding: [0x0a,0x0b,0xbd,0xd3,0x02,0x0d,0x0a,0x04]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_smfmac_f32_16x16x128_fp8_bf8 v[10:13], v[2:5], v[6:13], v2 cbsz:3 abid:1

// GFX950: v_smfmac_f32_16x16x128_fp8_bf8 a[10:13], a[2:5], a[6:13], v2 cbsz:3 abid:1 ; encoding: [0x0a,0x8b,0xbd,0xd3,0x02,0x0d,0x0a,0x1c]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_smfmac_f32_16x16x128_fp8_bf8 a[10:13], a[2:5], a[6:13], v2 cbsz:3 abid:1

// GFX950: v_smfmac_f32_16x16x128_fp8_bf8 a[10:13], a[2:5], a[6:13], v3 cbsz:1 abid:3 ; encoding: [0x0a,0x99,0xbd,0xd3,0x02,0x0d,0x0e,0x1c]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_smfmac_f32_16x16x128_fp8_bf8 a[10:13], a[2:5], a[6:13], v3 cbsz:1 abid:3

//===----------------------------------------------------------------------===//
// v_smfmac_f32_16x16x128_fp8_fp8
//===----------------------------------------------------------------------===//

// GFX950: v_smfmac_f32_16x16x128_fp8_fp8 v[10:13], a[2:5], v[4:11], v3 cbsz:3 abid:1 ; encoding: [0x0a,0x0b,0xc3,0xd3,0x02,0x09,0x0e,0x0c]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_smfmac_f32_16x16x128_fp8_fp8 v[10:13], a[2:5], v[4:11], v3 cbsz:3 abid:1

// GFX950: v_smfmac_f32_16x16x128_fp8_fp8 v[10:13], a[2:5], v[4:11], v3 cbsz:3 abid:1 ; encoding: [0x0a,0x0b,0xc3,0xd3,0x02,0x09,0x0e,0x0c]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_smfmac_f32_16x16x128fp8fp8 v[10:13], a[2:5], v[4:11], v3 cbsz:3 abid:1

// GFX950: v_smfmac_f32_16x16x128_fp8_fp8 a[10:13], v[2:5], a[4:11], v1 ; encoding: [0x0a,0x80,0xc3,0xd3,0x02,0x09,0x06,0x14]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_smfmac_f32_16x16x128_fp8_fp8 a[10:13], v[2:5], a[4:11], v1

// GFX950: v_smfmac_f32_16x16x128_fp8_fp8 v[10:13], a[2:5], v[4:11], v2 cbsz:3 abid:1 ; encoding: [0x0a,0x0b,0xc3,0xd3,0x02,0x09,0x0a,0x0c]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_smfmac_f32_16x16x128_fp8_fp8 v[10:13], a[2:5], v[4:11], v2 cbsz:3 abid:1

// GFX950: v_smfmac_f32_16x16x128_fp8_fp8 a[10:13], v[2:5], a[4:11], v3 ; encoding: [0x0a,0x80,0xc3,0xd3,0x02,0x09,0x0e,0x14]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_smfmac_f32_16x16x128_fp8_fp8 a[10:13], v[2:5], a[4:11], v3

// GFX950: v_smfmac_f32_16x16x128_fp8_fp8 v[10:13], v[2:5], v[6:13], v2 cbsz:3 abid:1 ; encoding: [0x0a,0x0b,0xc3,0xd3,0x02,0x0d,0x0a,0x04]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_smfmac_f32_16x16x128_fp8_fp8 v[10:13], v[2:5], v[6:13], v2 cbsz:3 abid:1

// GFX950: v_smfmac_f32_16x16x128_fp8_fp8 a[10:13], a[2:5], a[6:13], v2 cbsz:3 abid:1 ; encoding: [0x0a,0x8b,0xc3,0xd3,0x02,0x0d,0x0a,0x1c]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_smfmac_f32_16x16x128_fp8_fp8 a[10:13], a[2:5], a[6:13], v2 cbsz:3 abid:1

// GFX950: v_smfmac_f32_16x16x128_fp8_fp8 a[10:13], a[2:5], a[6:13], v3 cbsz:1 abid:3 ; encoding: [0x0a,0x99,0xc3,0xd3,0x02,0x0d,0x0e,0x1c]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_smfmac_f32_16x16x128_fp8_fp8 a[10:13], a[2:5], a[6:13], v3 cbsz:1 abid:3

//===----------------------------------------------------------------------===//
// v_smfmac_f32_32x32x64_bf8_bf8
//===----------------------------------------------------------------------===//

// GFX950: v_smfmac_f32_32x32x64_bf8_bf8 v[10:25], a[2:5], v[4:11], v3 cbsz:3 abid:1 ; encoding: [0x0a,0x0b,0xcb,0xd3,0x02,0x09,0x0e,0x0c]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_smfmac_f32_32x32x64_bf8_bf8 v[10:25], a[2:5], v[4:11], v3 cbsz:3 abid:1

// GFX950: v_smfmac_f32_32x32x64_bf8_bf8 v[10:25], a[2:5], v[4:11], v3 cbsz:3 abid:1 ; encoding: [0x0a,0x0b,0xcb,0xd3,0x02,0x09,0x0e,0x0c]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_smfmac_f32_32x32x64bf8bf8 v[10:25], a[2:5], v[4:11], v3 cbsz:3 abid:1

// GFX950: v_smfmac_f32_32x32x64_bf8_bf8 a[10:25], v[2:5], a[4:11], v1 ; encoding: [0x0a,0x80,0xcb,0xd3,0x02,0x09,0x06,0x14]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_smfmac_f32_32x32x64_bf8_bf8 a[10:25], v[2:5], a[4:11], v1

// GFX950: v_smfmac_f32_32x32x64_bf8_bf8 v[10:25], a[2:5], v[4:11], v2 cbsz:3 abid:1 ; encoding: [0x0a,0x0b,0xcb,0xd3,0x02,0x09,0x0a,0x0c]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_smfmac_f32_32x32x64_bf8_bf8 v[10:25], a[2:5], v[4:11], v2 cbsz:3 abid:1

// GFX950: v_smfmac_f32_32x32x64_bf8_bf8 a[10:25], v[2:5], a[4:11], v3 ; encoding: [0x0a,0x80,0xcb,0xd3,0x02,0x09,0x0e,0x14]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_smfmac_f32_32x32x64_bf8_bf8 a[10:25], v[2:5], a[4:11], v3

// GFX950: v_smfmac_f32_32x32x64_bf8_bf8 v[10:25], v[2:5], v[6:13], v2 cbsz:3 abid:1 ; encoding: [0x0a,0x0b,0xcb,0xd3,0x02,0x0d,0x0a,0x04]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_smfmac_f32_32x32x64_bf8_bf8 v[10:25], v[2:5], v[6:13], v2 cbsz:3 abid:1

// GFX950: v_smfmac_f32_32x32x64_bf8_bf8 a[10:25], a[2:5], a[6:13], v2 cbsz:3 abid:1 ; encoding: [0x0a,0x8b,0xcb,0xd3,0x02,0x0d,0x0a,0x1c]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_smfmac_f32_32x32x64_bf8_bf8 a[10:25], a[2:5], a[6:13], v2 cbsz:3 abid:1

// GFX950: v_smfmac_f32_32x32x64_bf8_bf8 a[10:25], a[2:5], a[6:13], v3 cbsz:1 abid:3 ; encoding: [0x0a,0x99,0xcb,0xd3,0x02,0x0d,0x0e,0x1c]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_smfmac_f32_32x32x64_bf8_bf8 a[10:25], a[2:5], a[6:13], v3 cbsz:1 abid:3

//===----------------------------------------------------------------------===//
// v_smfmac_f32_32x32x64_bf8_fp8
//===----------------------------------------------------------------------===//

// GFX950: v_smfmac_f32_32x32x64_bf8_fp8 v[10:25], a[2:5], v[4:11], v3 cbsz:3 abid:1 ; encoding: [0x0a,0x0b,0xce,0xd3,0x02,0x09,0x0e,0x0c]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_smfmac_f32_32x32x64_bf8_fp8 v[10:25], a[2:5], v[4:11], v3 cbsz:3 abid:1

// GFX950: v_smfmac_f32_32x32x64_bf8_fp8 v[10:25], a[2:5], v[4:11], v3 cbsz:3 abid:1 ; encoding: [0x0a,0x0b,0xce,0xd3,0x02,0x09,0x0e,0x0c]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_smfmac_f32_32x32x64bf8fp8 v[10:25], a[2:5], v[4:11], v3 cbsz:3 abid:1

// GFX950: v_smfmac_f32_32x32x64_bf8_fp8 a[10:25], v[2:5], a[4:11], v1 ; encoding: [0x0a,0x80,0xce,0xd3,0x02,0x09,0x06,0x14]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_smfmac_f32_32x32x64_bf8_fp8 a[10:25], v[2:5], a[4:11], v1

// GFX950: v_smfmac_f32_32x32x64_bf8_fp8 v[10:25], a[2:5], v[4:11], v2 cbsz:3 abid:1 ; encoding: [0x0a,0x0b,0xce,0xd3,0x02,0x09,0x0a,0x0c]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_smfmac_f32_32x32x64_bf8_fp8 v[10:25], a[2:5], v[4:11], v2 cbsz:3 abid:1

// GFX950: v_smfmac_f32_32x32x64_bf8_fp8 a[10:25], v[2:5], a[4:11], v3 ; encoding: [0x0a,0x80,0xce,0xd3,0x02,0x09,0x0e,0x14]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_smfmac_f32_32x32x64_bf8_fp8 a[10:25], v[2:5], a[4:11], v3

// GFX950: v_smfmac_f32_32x32x64_bf8_fp8 v[10:25], v[2:5], v[6:13], v2 cbsz:3 abid:1 ; encoding: [0x0a,0x0b,0xce,0xd3,0x02,0x0d,0x0a,0x04]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_smfmac_f32_32x32x64_bf8_fp8 v[10:25], v[2:5], v[6:13], v2 cbsz:3 abid:1

// GFX950: v_smfmac_f32_32x32x64_bf8_fp8 a[10:25], a[2:5], a[6:13], v2 cbsz:3 abid:1 ; encoding: [0x0a,0x8b,0xce,0xd3,0x02,0x0d,0x0a,0x1c]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_smfmac_f32_32x32x64_bf8_fp8 a[10:25], a[2:5], a[6:13], v2 cbsz:3 abid:1

// GFX950: v_smfmac_f32_32x32x64_bf8_fp8 a[10:25], a[2:5], a[6:13], v3 cbsz:1 abid:3 ; encoding: [0x0a,0x99,0xce,0xd3,0x02,0x0d,0x0e,0x1c]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_smfmac_f32_32x32x64_bf8_fp8 a[10:25], a[2:5], a[6:13], v3 cbsz:1 abid:3

//===----------------------------------------------------------------------===//
// v_smfmac_f32_32x32x64_fp8_bf8
//===----------------------------------------------------------------------===//

// GFX950: v_smfmac_f32_32x32x64_fp8_bf8 v[10:25], a[2:5], v[4:11], v3 cbsz:3 abid:1 ; encoding: [0x0a,0x0b,0xcf,0xd3,0x02,0x09,0x0e,0x0c]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_smfmac_f32_32x32x64_fp8_bf8 v[10:25], a[2:5], v[4:11], v3 cbsz:3 abid:1

// GFX950: v_smfmac_f32_32x32x64_fp8_bf8 v[10:25], a[2:5], v[4:11], v3 cbsz:3 abid:1 ; encoding: [0x0a,0x0b,0xcf,0xd3,0x02,0x09,0x0e,0x0c]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_smfmac_f32_32x32x64fp8bf8 v[10:25], a[2:5], v[4:11], v3 cbsz:3 abid:1

// GFX950: v_smfmac_f32_32x32x64_fp8_bf8 a[10:25], v[2:5], a[4:11], v1 ; encoding: [0x0a,0x80,0xcf,0xd3,0x02,0x09,0x06,0x14]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_smfmac_f32_32x32x64_fp8_bf8 a[10:25], v[2:5], a[4:11], v1

// GFX950: v_smfmac_f32_32x32x64_fp8_bf8 v[10:25], a[2:5], v[4:11], v2 cbsz:3 abid:1 ; encoding: [0x0a,0x0b,0xcf,0xd3,0x02,0x09,0x0a,0x0c]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_smfmac_f32_32x32x64_fp8_bf8 v[10:25], a[2:5], v[4:11], v2 cbsz:3 abid:1

// GFX950: v_smfmac_f32_32x32x64_fp8_bf8 a[10:25], v[2:5], a[4:11], v3 ; encoding: [0x0a,0x80,0xcf,0xd3,0x02,0x09,0x0e,0x14]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_smfmac_f32_32x32x64_fp8_bf8 a[10:25], v[2:5], a[4:11], v3

// GFX950: v_smfmac_f32_32x32x64_fp8_bf8 v[10:25], v[2:5], v[6:13], v2 cbsz:3 abid:1 ; encoding: [0x0a,0x0b,0xcf,0xd3,0x02,0x0d,0x0a,0x04]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_smfmac_f32_32x32x64_fp8_bf8 v[10:25], v[2:5], v[6:13], v2 cbsz:3 abid:1

// GFX950: v_smfmac_f32_32x32x64_fp8_bf8 a[10:25], a[2:5], a[6:13], v2 cbsz:3 abid:1 ; encoding: [0x0a,0x8b,0xcf,0xd3,0x02,0x0d,0x0a,0x1c]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_smfmac_f32_32x32x64_fp8_bf8 a[10:25], a[2:5], a[6:13], v2 cbsz:3 abid:1

// GFX950: v_smfmac_f32_32x32x64_fp8_bf8 a[10:25], a[2:5], a[6:13], v3 cbsz:1 abid:3 ; encoding: [0x0a,0x99,0xcf,0xd3,0x02,0x0d,0x0e,0x1c]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_smfmac_f32_32x32x64_fp8_bf8 a[10:25], a[2:5], a[6:13], v3 cbsz:1 abid:3

//===----------------------------------------------------------------------===//
// v_smfmac_f32_32x32x64_fp8_fp8
//===----------------------------------------------------------------------===//

// GFX950: v_smfmac_f32_32x32x64_fp8_fp8 v[10:25], a[2:5], v[4:11], v3 cbsz:3 abid:1 ; encoding: [0x0a,0x0b,0xd3,0xd3,0x02,0x09,0x0e,0x0c]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_smfmac_f32_32x32x64_fp8_fp8 v[10:25], a[2:5], v[4:11], v3 cbsz:3 abid:1

// GFX950: v_smfmac_f32_32x32x64_fp8_fp8 v[10:25], a[2:5], v[4:11], v3 cbsz:3 abid:1 ; encoding: [0x0a,0x0b,0xd3,0xd3,0x02,0x09,0x0e,0x0c]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_smfmac_f32_32x32x64fp8fp8 v[10:25], a[2:5], v[4:11], v3 cbsz:3 abid:1

// GFX950: v_smfmac_f32_32x32x64_fp8_fp8 a[10:25], v[2:5], a[4:11], v1 ; encoding: [0x0a,0x80,0xd3,0xd3,0x02,0x09,0x06,0x14]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_smfmac_f32_32x32x64_fp8_fp8 a[10:25], v[2:5], a[4:11], v1

// GFX950: v_smfmac_f32_32x32x64_fp8_fp8 v[10:25], a[2:5], v[4:11], v2 cbsz:3 abid:1 ; encoding: [0x0a,0x0b,0xd3,0xd3,0x02,0x09,0x0a,0x0c]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_smfmac_f32_32x32x64_fp8_fp8 v[10:25], a[2:5], v[4:11], v2 cbsz:3 abid:1

// GFX950: v_smfmac_f32_32x32x64_fp8_fp8 a[10:25], v[2:5], a[4:11], v3 ; encoding: [0x0a,0x80,0xd3,0xd3,0x02,0x09,0x0e,0x14]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_smfmac_f32_32x32x64_fp8_fp8 a[10:25], v[2:5], a[4:11], v3

// GFX950: v_smfmac_f32_32x32x64_fp8_fp8 v[10:25], v[2:5], v[6:13], v2 cbsz:3 abid:1 ; encoding: [0x0a,0x0b,0xd3,0xd3,0x02,0x0d,0x0a,0x04]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_smfmac_f32_32x32x64_fp8_fp8 v[10:25], v[2:5], v[6:13], v2 cbsz:3 abid:1

// GFX950: v_smfmac_f32_32x32x64_fp8_fp8 a[10:25], a[2:5], a[6:13], v2 cbsz:3 abid:1 ; encoding: [0x0a,0x8b,0xd3,0xd3,0x02,0x0d,0x0a,0x1c]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_smfmac_f32_32x32x64_fp8_fp8 a[10:25], a[2:5], a[6:13], v2 cbsz:3 abid:1

// GFX950: v_smfmac_f32_32x32x64_fp8_fp8 a[10:25], a[2:5], a[6:13], v3 cbsz:1 abid:3 ; encoding: [0x0a,0x99,0xd3,0xd3,0x02,0x0d,0x0e,0x1c]
// ERR: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_smfmac_f32_32x32x64_fp8_fp8 a[10:25], a[2:5], a[6:13], v3 cbsz:1 abid:3
