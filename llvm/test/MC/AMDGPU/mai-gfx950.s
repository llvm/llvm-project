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
