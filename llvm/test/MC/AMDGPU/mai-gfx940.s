// RUN: llvm-mc -arch=amdgcn -mcpu=gfx940 -show-encoding %s | FileCheck -check-prefix=GFX940 %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx90a %s 2>&1 | FileCheck -check-prefix=GFX90A %s

//===----------------------------------------------------------------------===//
// Misc opcodes.
//===----------------------------------------------------------------------===//

v_accvgpr_write_b32 a10, s20
// GFX940: v_accvgpr_write_b32 a10, s20    ; encoding: [0x0a,0x40,0xd9,0xd3,0x14,0x00,0x00,0x18]

//===----------------------------------------------------------------------===//
// MFMA opcodes.
//===----------------------------------------------------------------------===//

v_mfma_f64_4x4x4_4b_f64 a[0:1], v[0:1], a[2:3], a[2:3]
// GFX940: v_mfma_f64_4x4x4_4b_f64 a[0:1], v[0:1], a[2:3], a[2:3] ; encoding: [0x00,0x80,0xef,0xd3,0x00,0x05,0x0a,0x14]
// GFX90A: error: instruction not supported on this GPU

v_mfma_f64_4x4x4_4b_f64 v[0:1], v[0:1], a[2:3], v[2:3]
// GFX940: v_mfma_f64_4x4x4_4b_f64 v[0:1], v[0:1], a[2:3], v[2:3] ; encoding: [0x00,0x00,0xef,0xd3,0x00,0x05,0x0a,0x14]
// GFX90A: error: instruction not supported on this GPU

v_mfma_f64_4x4x4f64 a[0:1], v[0:1], a[2:3], a[2:3]
// GFX940: v_mfma_f64_4x4x4_4b_f64 a[0:1], v[0:1], a[2:3], a[2:3] ; encoding: [0x00,0x80,0xef,0xd3,0x00,0x05,0x0a,0x14]

v_mfma_f64_4x4x4f64 v[0:1], v[0:1], a[2:3], v[2:3]
// GFX940: v_mfma_f64_4x4x4_4b_f64 v[0:1], v[0:1], a[2:3], v[2:3] ; encoding: [0x00,0x00,0xef,0xd3,0x00,0x05,0x0a,0x14]

v_mfma_f64_4x4x4_4b_f64 a[0:1], v[0:1], a[2:3], a[2:3] neg:[1,0,0]
// GFX940: v_mfma_f64_4x4x4_4b_f64 a[0:1], v[0:1], a[2:3], a[2:3] neg:[1,0,0] ; encoding: [0x00,0x80,0xef,0xd3,0x00,0x05,0x0a,0x34]
// GFX90A: error: instruction not supported on this GPU

v_mfma_f64_4x4x4_4b_f64 a[0:1], v[0:1], a[2:3], a[2:3] neg:[0,1,0]
// GFX940: v_mfma_f64_4x4x4_4b_f64 a[0:1], v[0:1], a[2:3], a[2:3] neg:[0,1,0] ; encoding: [0x00,0x80,0xef,0xd3,0x00,0x05,0x0a,0x54]
// GFX90A: error: instruction not supported on this GPU

v_mfma_f64_4x4x4_4b_f64 a[0:1], v[0:1], a[2:3], a[2:3] neg:[0,0,1]
// GFX940: v_mfma_f64_4x4x4_4b_f64 a[0:1], v[0:1], a[2:3], a[2:3] neg:[0,0,1] ; encoding: [0x00,0x80,0xef,0xd3,0x00,0x05,0x0a,0x94]
// GFX90A: error: instruction not supported on this GPU

v_mfma_f64_4x4x4_4b_f64 v[0:1], v[0:1], a[2:3], v[2:3] neg:[1,1,1]
// GFX940: v_mfma_f64_4x4x4_4b_f64 v[0:1], v[0:1], a[2:3], v[2:3] neg:[1,1,1] ; encoding: [0x00,0x00,0xef,0xd3,0x00,0x05,0x0a,0xf4]
// GFX90A: error: instruction not supported on this GPU

v_mfma_f64_4x4x4f64 a[0:1], v[0:1], a[2:3], a[2:3] neg:[1,0,0]
// GFX940: v_mfma_f64_4x4x4_4b_f64 a[0:1], v[0:1], a[2:3], a[2:3] neg:[1,0,0] ; encoding: [0x00,0x80,0xef,0xd3,0x00,0x05,0x0a,0x34]
// GFX90A: error: invalid modifier: neg is not supported

v_mfma_f64_4x4x4f64 v[0:1], v[0:1], a[2:3], v[2:3] neg:[1,0,0]
// GFX940: v_mfma_f64_4x4x4_4b_f64 v[0:1], v[0:1], a[2:3], v[2:3] neg:[1,0,0] ; encoding: [0x00,0x00,0xef,0xd3,0x00,0x05,0x0a,0x34]
// GFX90A: error: invalid modifier: neg is not supported

v_mfma_f64_16x16x4_f64 a[0:7], v[0:1], v[2:3], a[0:7]
// GFX940: v_mfma_f64_16x16x4_f64 a[0:7], v[0:1], v[2:3], a[0:7] ; encoding: [0x00,0x80,0xee,0xd3,0x00,0x05,0x02,0x04]
// GFX90A: error: instruction not supported on this GPU

v_mfma_f64_16x16x4_f64 v[0:7], v[0:1], v[2:3], v[0:7]
// GFX940: v_mfma_f64_16x16x4_f64 v[0:7], v[0:1], v[2:3], v[0:7] ; encoding: [0x00,0x00,0xee,0xd3,0x00,0x05,0x02,0x04]
// GFX90A: error: instruction not supported on this GPU

v_mfma_f64_16x16x4f64 a[0:7], v[0:1], v[2:3], a[0:7]
// GFX940: v_mfma_f64_16x16x4_f64 a[0:7], v[0:1], v[2:3], a[0:7] ; encoding: [0x00,0x80,0xee,0xd3,0x00,0x05,0x02,0x04]

v_mfma_f64_16x16x4f64 v[0:7], v[0:1], v[2:3], v[0:7]
// GFX940: v_mfma_f64_16x16x4_f64 v[0:7], v[0:1], v[2:3], v[0:7] ; encoding: [0x00,0x00,0xee,0xd3,0x00,0x05,0x02,0x04]

v_mfma_f64_16x16x4_f64 a[0:7], v[0:1], v[2:3], a[0:7] neg:[1,1,1]
// GFX940: v_mfma_f64_16x16x4_f64 a[0:7], v[0:1], v[2:3], a[0:7] neg:[1,1,1] ; encoding: [0x00,0x80,0xee,0xd3,0x00,0x05,0x02,0xe4]
// GFX90A: error: instruction not supported on this GPU

v_mfma_f64_16x16x4_f64 v[0:7], v[0:1], v[2:3], v[0:7] neg:[1,1,1]
// GFX940: v_mfma_f64_16x16x4_f64 v[0:7], v[0:1], v[2:3], v[0:7] neg:[1,1,1] ; encoding: [0x00,0x00,0xee,0xd3,0x00,0x05,0x02,0xe4]
// GFX90A: error: instruction not supported on this GPU

v_mfma_f64_16x16x4f64 a[0:7], v[0:1], v[2:3], a[0:7] neg:[1,0,0]
// GFX940: v_mfma_f64_16x16x4_f64 a[0:7], v[0:1], v[2:3], a[0:7] neg:[1,0,0] ; encoding: [0x00,0x80,0xee,0xd3,0x00,0x05,0x02,0x24]
// GFX90A: error: invalid modifier: neg is not supported

v_mfma_f64_16x16x4f64 v[0:7], v[0:1], v[2:3], v[0:7] neg:[1,0,0]
// GFX940: v_mfma_f64_16x16x4_f64 v[0:7], v[0:1], v[2:3], v[0:7] neg:[1,0,0] ; encoding: [0x00,0x00,0xee,0xd3,0x00,0x05,0x02,0x24]
// GFX90A: error: invalid modifier: neg is not supported

v_mfma_f32_16x16x1_4b_f32 a[0:15], v0, v1, a[18:33]
// GFX940: v_mfma_f32_16x16x1_4b_f32 a[0:15], v0, v1, a[18:33] ; encoding: [0x00,0x80,0xc1,0xd3,0x00,0x03,0x4a,0x04]
// GFX90A: error: instruction not supported on this GPU

v_mfma_f32_16x16x1_4b_f32 v[0:15], v0, v1, v[18:33]
// GFX940: v_mfma_f32_16x16x1_4b_f32 v[0:15], v0, v1, v[18:33] ; encoding: [0x00,0x00,0xc1,0xd3,0x00,0x03,0x4a,0x04]
// GFX90A: error: instruction not supported on this GPU

v_mfma_f32_16x16x1f32 a[0:15], v0, v1, a[18:33]
// GFX940: v_mfma_f32_16x16x1_4b_f32 a[0:15], v0, v1, a[18:33] ; encoding: [0x00,0x80,0xc1,0xd3,0x00,0x03,0x4a,0x04]

v_mfma_f32_16x16x1f32 v[0:15], v0, v1, v[18:33]
// GFX940: v_mfma_f32_16x16x1_4b_f32 v[0:15], v0, v1, v[18:33] ; encoding: [0x00,0x00,0xc1,0xd3,0x00,0x03,0x4a,0x04]

v_mfma_f32_4x4x1_16b_f32 a[0:3], v0, v1, a[2:5]
// GFX940: v_mfma_f32_4x4x1_16b_f32 a[0:3], v0, v1, a[2:5] ; encoding: [0x00,0x80,0xc2,0xd3,0x00,0x03,0x0a,0x04]
// GFX90A: error: instruction not supported on this GPU

v_mfma_f32_4x4x1_16b_f32 v[0:3], v0, v1, v[2:5]
// GFX940: v_mfma_f32_4x4x1_16b_f32 v[0:3], v0, v1, v[2:5] ; encoding: [0x00,0x00,0xc2,0xd3,0x00,0x03,0x0a,0x04]
// GFX90A: error: instruction not supported on this GPU

v_mfma_f32_4x4x1f32 a[0:3], v0, v1, a[2:5]
// GFX940: v_mfma_f32_4x4x1_16b_f32 a[0:3], v0, v1, a[2:5] ; encoding: [0x00,0x80,0xc2,0xd3,0x00,0x03,0x0a,0x04]

v_mfma_f32_4x4x1f32 v[0:3], v0, v1, v[2:5]
// GFX940: v_mfma_f32_4x4x1_16b_f32 v[0:3], v0, v1, v[2:5] ; encoding: [0x00,0x00,0xc2,0xd3,0x00,0x03,0x0a,0x04]

v_mfma_f32_32x32x2_f32 a[0:15], v0, v1, a[18:33]
// GFX940: v_mfma_f32_32x32x2_f32 a[0:15], v0, v1, a[18:33] ; encoding: [0x00,0x80,0xc4,0xd3,0x00,0x03,0x4a,0x04]
// GFX90A: error: instruction not supported on this GPU

v_mfma_f32_32x32x2_f32 v[0:15], v0, v1, v[18:33]
// GFX940: v_mfma_f32_32x32x2_f32 v[0:15], v0, v1, v[18:33] ; encoding: [0x00,0x00,0xc4,0xd3,0x00,0x03,0x4a,0x04]
// GFX90A: error: instruction not supported on this GPU

v_mfma_f32_32x32x2f32 a[0:15], v0, v1, a[18:33]
// GFX940: v_mfma_f32_32x32x2_f32 a[0:15], v0, v1, a[18:33] ; encoding: [0x00,0x80,0xc4,0xd3,0x00,0x03,0x4a,0x04]

v_mfma_f32_32x32x2f32 v[0:15], v0, v1, v[18:33]
// GFX940: v_mfma_f32_32x32x2_f32 v[0:15], v0, v1, v[18:33] ; encoding: [0x00,0x00,0xc4,0xd3,0x00,0x03,0x4a,0x04]

v_mfma_f32_16x16x4_f32 a[0:3], v0, v1, a[2:5]
// GFX940: v_mfma_f32_16x16x4_f32 a[0:3], v0, v1, a[2:5] ; encoding: [0x00,0x80,0xc5,0xd3,0x00,0x03,0x0a,0x04]
// GFX90A: error: instruction not supported on this GPU

v_mfma_f32_16x16x4_f32 v[0:3], v0, v1, v[2:5]
// GFX940: v_mfma_f32_16x16x4_f32 v[0:3], v0, v1, v[2:5] ; encoding: [0x00,0x00,0xc5,0xd3,0x00,0x03,0x0a,0x04]
// GFX90A: error: instruction not supported on this GPU

v_mfma_f32_16x16x4f32 a[0:3], v0, v1, a[2:5]
// GFX940: v_mfma_f32_16x16x4_f32 a[0:3], v0, v1, a[2:5] ; encoding: [0x00,0x80,0xc5,0xd3,0x00,0x03,0x0a,0x04]

v_mfma_f32_16x16x4f32 v[0:3], v0, v1, v[2:5]
// GFX940: v_mfma_f32_16x16x4_f32 v[0:3], v0, v1, v[2:5] ; encoding: [0x00,0x00,0xc5,0xd3,0x00,0x03,0x0a,0x04]

v_mfma_f32_32x32x4_2b_f16 a[0:31], v[0:1], v[2:3], a[34:65]
// GFX940: v_mfma_f32_32x32x4_2b_f16 a[0:31], v[0:1], v[2:3], a[34:65] ; encoding: [0x00,0x80,0xc8,0xd3,0x00,0x05,0x8a,0x04]
// GFX90A: error: instruction not supported on this GPU

v_mfma_f32_32x32x4_2b_f16 v[0:31], v[0:1], v[2:3], v[34:65]
// GFX940: v_mfma_f32_32x32x4_2b_f16 v[0:31], v[0:1], v[2:3], v[34:65] ; encoding: [0x00,0x00,0xc8,0xd3,0x00,0x05,0x8a,0x04]
// GFX90A: error: instruction not supported on this GPU

v_mfma_f32_32x32x4f16 a[0:31], v[0:1], v[2:3], a[34:65]
// GFX940: v_mfma_f32_32x32x4_2b_f16 a[0:31], v[0:1], v[2:3], a[34:65] ; encoding: [0x00,0x80,0xc8,0xd3,0x00,0x05,0x8a,0x04]

v_mfma_f32_32x32x4f16 v[0:31], v[0:1], v[2:3], v[34:65]
// GFX940: v_mfma_f32_32x32x4_2b_f16 v[0:31], v[0:1], v[2:3], v[34:65] ; encoding: [0x00,0x00,0xc8,0xd3,0x00,0x05,0x8a,0x04]

v_mfma_f32_16x16x4_4b_f16 a[0:15], v[0:1], v[2:3], a[18:33]
// GFX940: v_mfma_f32_16x16x4_4b_f16 a[0:15], v[0:1], v[2:3], a[18:33] ; encoding: [0x00,0x80,0xc9,0xd3,0x00,0x05,0x4a,0x04]
// GFX90A: error: instruction not supported on this GPU

v_mfma_f32_16x16x4_4b_f16 v[0:15], v[0:1], v[2:3], v[18:33]
// GFX940: v_mfma_f32_16x16x4_4b_f16 v[0:15], v[0:1], v[2:3], v[18:33] ; encoding: [0x00,0x00,0xc9,0xd3,0x00,0x05,0x4a,0x04]
// GFX90A: error: instruction not supported on this GPU

v_mfma_f32_16x16x4f16 a[0:15], v[0:1], v[2:3], a[18:33]
// GFX940: v_mfma_f32_16x16x4_4b_f16 a[0:15], v[0:1], v[2:3], a[18:33] ; encoding: [0x00,0x80,0xc9,0xd3,0x00,0x05,0x4a,0x04]

v_mfma_f32_16x16x4f16 v[0:15], v[0:1], v[2:3], v[18:33]
// GFX940: v_mfma_f32_16x16x4_4b_f16 v[0:15], v[0:1], v[2:3], v[18:33] ; encoding: [0x00,0x00,0xc9,0xd3,0x00,0x05,0x4a,0x04]

v_mfma_f32_4x4x4_16b_f16 a[0:3], v[0:1], v[2:3], a[2:5]
// GFX940: v_mfma_f32_4x4x4_16b_f16 a[0:3], v[0:1], v[2:3], a[2:5] ; encoding: [0x00,0x80,0xca,0xd3,0x00,0x05,0x0a,0x04]
// GFX90A: error: instruction not supported on this GPU

v_mfma_f32_4x4x4_16b_f16 v[0:3], v[0:1], v[2:3], v[2:5]
// GFX940: v_mfma_f32_4x4x4_16b_f16 v[0:3], v[0:1], v[2:3], v[2:5] ; encoding: [0x00,0x00,0xca,0xd3,0x00,0x05,0x0a,0x04]
// GFX90A: error: instruction not supported on this GPU

v_mfma_f32_4x4x4f16 a[0:3], v[0:1], v[2:3], a[2:5]
// GFX940: v_mfma_f32_4x4x4_16b_f16 a[0:3], v[0:1], v[2:3], a[2:5] ; encoding: [0x00,0x80,0xca,0xd3,0x00,0x05,0x0a,0x04]

v_mfma_f32_4x4x4f16 v[0:3], v[0:1], v[2:3], v[2:5]
// GFX940: v_mfma_f32_4x4x4_16b_f16 v[0:3], v[0:1], v[2:3], v[2:5] ; encoding: [0x00,0x00,0xca,0xd3,0x00,0x05,0x0a,0x04]

v_mfma_f32_32x32x8_f16 a[0:15], v[0:1], v[2:3], a[18:33]
// GFX940: v_mfma_f32_32x32x8_f16 a[0:15], v[0:1], v[2:3], a[18:33] ; encoding: [0x00,0x80,0xcc,0xd3,0x00,0x05,0x4a,0x04]
// GFX90A: error: instruction not supported on this GPU

v_mfma_f32_32x32x8_f16 v[0:15], v[0:1], v[2:3], v[18:33]
// GFX940: v_mfma_f32_32x32x8_f16 v[0:15], v[0:1], v[2:3], v[18:33] ; encoding: [0x00,0x00,0xcc,0xd3,0x00,0x05,0x4a,0x04]
// GFX90A: error: instruction not supported on this GPU

v_mfma_f32_32x32x8f16 a[0:15], v[0:1], v[2:3], a[18:33]
// GFX940: v_mfma_f32_32x32x8_f16 a[0:15], v[0:1], v[2:3], a[18:33] ; encoding: [0x00,0x80,0xcc,0xd3,0x00,0x05,0x4a,0x04]

v_mfma_f32_32x32x8f16 v[0:15], v[0:1], v[2:3], v[18:33]
// GFX940: v_mfma_f32_32x32x8_f16 v[0:15], v[0:1], v[2:3], v[18:33] ; encoding: [0x00,0x00,0xcc,0xd3,0x00,0x05,0x4a,0x04]

v_mfma_f32_16x16x16_f16 a[0:3], v[0:1], v[2:3], a[2:5]
// GFX940: v_mfma_f32_16x16x16_f16 a[0:3], v[0:1], v[2:3], a[2:5] ; encoding: [0x00,0x80,0xcd,0xd3,0x00,0x05,0x0a,0x04]
// GFX90A: error: instruction not supported on this GPU

v_mfma_f32_16x16x16_f16 v[0:3], v[0:1], v[2:3], v[2:5]
// GFX940: v_mfma_f32_16x16x16_f16 v[0:3], v[0:1], v[2:3], v[2:5] ; encoding: [0x00,0x00,0xcd,0xd3,0x00,0x05,0x0a,0x04]
// GFX90A: error: instruction not supported on this GPU

v_mfma_f32_16x16x16f16 a[0:3], v[0:1], v[2:3], a[2:5]
// GFX940: v_mfma_f32_16x16x16_f16 a[0:3], v[0:1], v[2:3], a[2:5] ; encoding: [0x00,0x80,0xcd,0xd3,0x00,0x05,0x0a,0x04]

v_mfma_f32_16x16x16f16 v[0:3], v[0:1], v[2:3], v[2:5]
// GFX940: v_mfma_f32_16x16x16_f16 v[0:3], v[0:1], v[2:3], v[2:5] ; encoding: [0x00,0x00,0xcd,0xd3,0x00,0x05,0x0a,0x04]

v_mfma_i32_32x32x4_2b_i8 a[0:31], v0, v1, a[34:65]
// GFX940: v_mfma_i32_32x32x4_2b_i8 a[0:31], v0, v1, a[34:65] ; encoding: [0x00,0x80,0xd0,0xd3,0x00,0x03,0x8a,0x04]
// GFX90A: error: instruction not supported on this GPU

v_mfma_i32_32x32x4_2b_i8 v[0:31], v0, a1, v[34:65]
// GFX940: v_mfma_i32_32x32x4_2b_i8 v[0:31], v0, a1, v[34:65] ; encoding: [0x00,0x00,0xd0,0xd3,0x00,0x03,0x8a,0x14]
// GFX90A: error: instruction not supported on this GPU

v_mfma_i32_32x32x4i8 a[0:31], v0, v1, a[34:65]
// GFX940: v_mfma_i32_32x32x4_2b_i8 a[0:31], v0, v1, a[34:65] ; encoding: [0x00,0x80,0xd0,0xd3,0x00,0x03,0x8a,0x04]

v_mfma_i32_32x32x4i8 v[0:31], v0, a1, v[34:65]
// GFX940: v_mfma_i32_32x32x4_2b_i8 v[0:31], v0, a1, v[34:65] ; encoding: [0x00,0x00,0xd0,0xd3,0x00,0x03,0x8a,0x14]

v_mfma_i32_16x16x4_4b_i8 a[0:15], v0, v1, a[18:33]
// GFX940: v_mfma_i32_16x16x4_4b_i8 a[0:15], v0, v1, a[18:33] ; encoding: [0x00,0x80,0xd1,0xd3,0x00,0x03,0x4a,0x04]
// GFX90A: error: instruction not supported on this GPU

v_mfma_i32_16x16x4_4b_i8 v[0:15], v0, v1, v[18:33]
// GFX940: v_mfma_i32_16x16x4_4b_i8 v[0:15], v0, v1, v[18:33] ; encoding: [0x00,0x00,0xd1,0xd3,0x00,0x03,0x4a,0x04]
// GFX90A: error: instruction not supported on this GPU

v_mfma_i32_16x16x4i8 a[0:15], v0, v1, a[18:33]
// GFX940: v_mfma_i32_16x16x4_4b_i8 a[0:15], v0, v1, a[18:33] ; encoding: [0x00,0x80,0xd1,0xd3,0x00,0x03,0x4a,0x04]

v_mfma_i32_16x16x4i8 v[0:15], v0, v1, v[18:33]
// GFX940: v_mfma_i32_16x16x4_4b_i8 v[0:15], v0, v1, v[18:33] ; encoding: [0x00,0x00,0xd1,0xd3,0x00,0x03,0x4a,0x04]

v_mfma_i32_4x4x4_16b_i8 a[0:3], v0, v1, a[2:5]
// GFX940: v_mfma_i32_4x4x4_16b_i8 a[0:3], v0, v1, a[2:5] ; encoding: [0x00,0x80,0xd2,0xd3,0x00,0x03,0x0a,0x04]
// GFX90A: error: instruction not supported on this GPU

v_mfma_i32_4x4x4_16b_i8 v[0:3], v0, v1, v[2:5]
// GFX940: v_mfma_i32_4x4x4_16b_i8 v[0:3], v0, v1, v[2:5] ; encoding: [0x00,0x00,0xd2,0xd3,0x00,0x03,0x0a,0x04]
// GFX90A: error: instruction not supported on this GPU

v_mfma_i32_4x4x4i8 a[0:3], v0, v1, a[2:5]
// GFX940: v_mfma_i32_4x4x4_16b_i8 a[0:3], v0, v1, a[2:5] ; encoding: [0x00,0x80,0xd2,0xd3,0x00,0x03,0x0a,0x04]

v_mfma_i32_4x4x4i8 v[0:3], v0, v1, v[2:5]
// GFX940: v_mfma_i32_4x4x4_16b_i8 v[0:3], v0, v1, v[2:5] ; encoding: [0x00,0x00,0xd2,0xd3,0x00,0x03,0x0a,0x04]

v_mfma_f32_32x32x1_2b_f32 a[0:31], v0, v1, a[34:65] blgp:7
// GFX940: v_mfma_f32_32x32x1_2b_f32 a[0:31], v0, v1, a[34:65] blgp:7 ; encoding: [0x00,0x80,0xc0,0xd3,0x00,0x03,0x8a,0xe4]
// GFX90A: error: instruction not supported on this GPU

v_mfma_f32_32x32x1_2b_f32 v[0:31], v0, v1, v[34:65] blgp:7
// GFX940: v_mfma_f32_32x32x1_2b_f32 v[0:31], v0, v1, v[34:65] blgp:7 ; encoding: [0x00,0x00,0xc0,0xd3,0x00,0x03,0x8a,0xe4]
// GFX90A: error: instruction not supported on this GPU

v_mfma_f32_32x32x1_2b_f32 a[0:31], v0, v1, a[34:65] blgp:7
// GFX940: v_mfma_f32_32x32x1_2b_f32 a[0:31], v0, v1, a[34:65] blgp:7 ; encoding: [0x00,0x80,0xc0,0xd3,0x00,0x03,0x8a,0xe4]
// GFX90A: error: instruction not supported on this GPU

v_mfma_f32_32x32x1_2b_f32 v[0:31], v0, v1, v[34:65] blgp:7
// GFX940: v_mfma_f32_32x32x1_2b_f32 v[0:31], v0, v1, v[34:65] blgp:7 ; encoding: [0x00,0x00,0xc0,0xd3,0x00,0x03,0x8a,0xe4]
// GFX90A: error: instruction not supported on this GPU

v_mfma_f32_32x32x1f32 a[0:31], v0, v1, a[34:65] blgp:7
// GFX940: v_mfma_f32_32x32x1_2b_f32 a[0:31], v0, v1, a[34:65] blgp:7 ; encoding: [0x00,0x80,0xc0,0xd3,0x00,0x03,0x8a,0xe4]

v_mfma_f32_32x32x1f32 v[0:31], v0, v1, v[34:65] blgp:7
// GFX940: v_mfma_f32_32x32x1_2b_f32 v[0:31], v0, v1, v[34:65] blgp:7 ; encoding: [0x00,0x00,0xc0,0xd3,0x00,0x03,0x8a,0xe4]

v_mfma_i32_32x32x16_i8 v[0:15], v[2:3], v[4:5], v[0:15]
// GFX940: v_mfma_i32_32x32x16_i8 v[0:15], v[2:3], v[4:5], v[0:15] ; encoding: [0x00,0x00,0xd6,0xd3,0x02,0x09,0x02,0x04]
// GFX90A: error: instruction not supported on this GPU

v_mfma_i32_32x32x16_i8 a[0:15], v[2:3], v[4:5], a[0:15]
// GFX940: v_mfma_i32_32x32x16_i8 a[0:15], v[2:3], v[4:5], a[0:15] ; encoding: [0x00,0x80,0xd6,0xd3,0x02,0x09,0x02,0x04]
// GFX90A: error: instruction not supported on this GPU

v_mfma_i32_32x32x16_i8 v[0:15], v[2:3], v[4:5], v[0:15]
// GFX940: v_mfma_i32_32x32x16_i8 v[0:15], v[2:3], v[4:5], v[0:15] ; encoding: [0x00,0x00,0xd6,0xd3,0x02,0x09,0x02,0x04]
// GFX90A: error: instruction not supported on this GPU

v_mfma_i32_32x32x16_i8 a[0:15], v[2:3], v[4:5], a[0:15]
// GFX940: v_mfma_i32_32x32x16_i8 a[0:15], v[2:3], v[4:5], a[0:15] ; encoding: [0x00,0x80,0xd6,0xd3,0x02,0x09,0x02,0x04]
// GFX90A: error: instruction not supported on this GPU

v_mfma_i32_32x32x16_i8 a[0:15], v[2:3], v[4:5], a[0:15] blgp:5
// GFX940: v_mfma_i32_32x32x16_i8 a[0:15], v[2:3], v[4:5], a[0:15] blgp:5 ; encoding: [0x00,0x80,0xd6,0xd3,0x02,0x09,0x02,0xa4]
// GFX90A: error: instruction not supported on this GPU

v_mfma_i32_32x32x16i8 a[0:15], v[2:3], v[4:5], a[0:15] blgp:5
// GFX940: v_mfma_i32_32x32x16_i8 a[0:15], v[2:3], v[4:5], a[0:15] blgp:5 ; encoding: [0x00,0x80,0xd6,0xd3,0x02,0x09,0x02,0xa4]
// GFX90A: error: instruction not supported on this GPU

v_mfma_i32_32x32x16i8 v[0:15], v[2:3], v[4:5], v[0:15] blgp:5
// GFX940: v_mfma_i32_32x32x16_i8 v[0:15], v[2:3], v[4:5], v[0:15] blgp:5 ; encoding: [0x00,0x00,0xd6,0xd3,0x02,0x09,0x02,0xa4]
// GFX90A: error: instruction not supported on this GPU

v_mfma_i32_16x16x32_i8 v[0:3], v[2:3], v[4:5], v[0:3]
// GFX940: v_mfma_i32_16x16x32_i8 v[0:3], v[2:3], v[4:5], v[0:3] ; encoding: [0x00,0x00,0xd7,0xd3,0x02,0x09,0x02,0x04]
// GFX90A: error: instruction not supported on this GPU

v_mfma_i32_16x16x32_i8 a[0:3], v[2:3], v[4:5], a[0:3]
// GFX940: v_mfma_i32_16x16x32_i8 a[0:3], v[2:3], v[4:5], a[0:3] ; encoding: [0x00,0x80,0xd7,0xd3,0x02,0x09,0x02,0x04]
// GFX90A: error: instruction not supported on this GPU

v_mfma_i32_16x16x32_i8 a[0:3], v[2:3], v[4:5], a[0:3] blgp:5
// GFX940: v_mfma_i32_16x16x32_i8 a[0:3], v[2:3], v[4:5], a[0:3] blgp:5 ; encoding: [0x00,0x80,0xd7,0xd3,0x02,0x09,0x02,0xa4]
// GFX90A: error: instruction not supported on this GPU

v_mfma_i32_16x16x32i8 v[0:3], v[2:3], v[4:5], v[0:3] blgp:5
// GFX940: v_mfma_i32_16x16x32_i8 v[0:3], v[2:3], v[4:5], v[0:3] blgp:5 ; encoding: [0x00,0x00,0xd7,0xd3,0x02,0x09,0x02,0xa4]
// GFX90A: error: instruction not supported on this GPU

v_mfma_i32_16x16x32i8 a[0:3], v[2:3], v[4:5], a[0:3] blgp:5
// GFX940: v_mfma_i32_16x16x32_i8 a[0:3], v[2:3], v[4:5], a[0:3] blgp:5 ; encoding: [0x00,0x80,0xd7,0xd3,0x02,0x09,0x02,0xa4]
// GFX90A: error: instruction not supported on this GPU

v_mfma_f32_32x32x4_2b_bf16 v[0:31], v[2:3], v[4:5], v[34:65]
// GFX940: v_mfma_f32_32x32x4_2b_bf16 v[0:31], v[2:3], v[4:5], v[34:65] ; encoding: [0x00,0x00,0xdd,0xd3,0x02,0x09,0x8a,0x04]
// GFX90A: error: instruction not supported on this GPU

v_mfma_f32_32x32x4_2b_bf16 a[0:31], v[2:3], v[4:5], a[34:65]
// GFX940: v_mfma_f32_32x32x4_2b_bf16 a[0:31], v[2:3], v[4:5], a[34:65] ; encoding: [0x00,0x80,0xdd,0xd3,0x02,0x09,0x8a,0x04]
// GFX90A: error: instruction not supported on this GPU

v_mfma_f32_32x32x4_2b_bf16 v[0:31], v[2:3], v[4:5], v[34:65]
// GFX940: v_mfma_f32_32x32x4_2b_bf16 v[0:31], v[2:3], v[4:5], v[34:65] ; encoding: [0x00,0x00,0xdd,0xd3,0x02,0x09,0x8a,0x04]
// GFX90A: error: instruction not supported on this GPU

v_mfma_f32_32x32x4_2b_bf16 a[0:31], v[2:3], v[4:5], a[34:65]
// GFX940: v_mfma_f32_32x32x4_2b_bf16 a[0:31], v[2:3], v[4:5], a[34:65] ; encoding: [0x00,0x80,0xdd,0xd3,0x02,0x09,0x8a,0x04]
// GFX90A: error: instruction not supported on this GPU

v_mfma_f32_32x32x4bf16 v[0:31], v[2:3], v[4:5], v[34:65] blgp:5
// GFX940: v_mfma_f32_32x32x4_2b_bf16 v[0:31], v[2:3], v[4:5], v[34:65] blgp:5 ; encoding: [0x00,0x00,0xdd,0xd3,0x02,0x09,0x8a,0xa4]
// GFX90A: error: operands are not valid for this GPU or mode

v_mfma_f32_32x32x4bf16 a[0:31], v[2:3], v[4:5], a[34:65] blgp:5
// GFX940: v_mfma_f32_32x32x4_2b_bf16 a[0:31], v[2:3], v[4:5], a[34:65] blgp:5 ; encoding: [0x00,0x80,0xdd,0xd3,0x02,0x09,0x8a,0xa4]
// GFX90A: error: operands are not valid for this GPU or mode

v_mfma_f32_32x32x4bf16_1k v[0:31], v[2:3], v[4:5], v[34:65] blgp:5
// GFX940: v_mfma_f32_32x32x4_2b_bf16 v[0:31], v[2:3], v[4:5], v[34:65] blgp:5 ; encoding: [0x00,0x00,0xdd,0xd3,0x02,0x09,0x8a,0xa4]

v_mfma_f32_32x32x4bf16_1k a[0:31], v[2:3], v[4:5], a[34:65] blgp:5
// GFX940: v_mfma_f32_32x32x4_2b_bf16 a[0:31], v[2:3], v[4:5], a[34:65] blgp:5 ; encoding: [0x00,0x80,0xdd,0xd3,0x02,0x09,0x8a,0xa4]

v_mfma_f32_16x16x4_4b_bf16 v[0:15], v[2:3], v[4:5], v[18:33]
// GFX940: v_mfma_f32_16x16x4_4b_bf16 v[0:15], v[2:3], v[4:5], v[18:33] ; encoding: [0x00,0x00,0xde,0xd3,0x02,0x09,0x4a,0x04]
// GFX90A: error: instruction not supported on this GPU

v_mfma_f32_16x16x4_4b_bf16 a[0:15], v[2:3], v[4:5], a[18:33]
// GFX940: v_mfma_f32_16x16x4_4b_bf16 a[0:15], v[2:3], v[4:5], a[18:33] ; encoding: [0x00,0x80,0xde,0xd3,0x02,0x09,0x4a,0x04]
// GFX90A: error: instruction not supported on this GPU

v_mfma_f32_16x16x4bf16 v[0:15], v[2:3], v[4:5], v[18:33] blgp:5
// GFX940: v_mfma_f32_16x16x4_4b_bf16 v[0:15], v[2:3], v[4:5], v[18:33] blgp:5 ; encoding: [0x00,0x00,0xde,0xd3,0x02,0x09,0x4a,0xa4]
// GFX90A: error: instruction not supported on this GPU

v_mfma_f32_16x16x4bf16 a[0:15], v[2:3], v[4:5], a[18:33] blgp:5
// GFX940: v_mfma_f32_16x16x4_4b_bf16 a[0:15], v[2:3], v[4:5], a[18:33] blgp:5 ; encoding: [0x00,0x80,0xde,0xd3,0x02,0x09,0x4a,0xa4]
// GFX90A: error: instruction not supported on this GPU

v_mfma_f32_16x16x4bf16_1k v[0:15], v[2:3], v[4:5], v[18:33] blgp:5
// GFX940: v_mfma_f32_16x16x4_4b_bf16 v[0:15], v[2:3], v[4:5], v[18:33] blgp:5 ; encoding: [0x00,0x00,0xde,0xd3,0x02,0x09,0x4a,0xa4]

v_mfma_f32_16x16x4bf16_1k a[0:15], v[2:3], v[4:5], a[18:33] blgp:5
// GFX940: v_mfma_f32_16x16x4_4b_bf16 a[0:15], v[2:3], v[4:5], a[18:33] blgp:5 ; encoding: [0x00,0x80,0xde,0xd3,0x02,0x09,0x4a,0xa4]

v_mfma_f32_4x4x4_16b_bf16 v[0:3], v[2:3], v[4:5], v[2:5]
// GFX940: v_mfma_f32_4x4x4_16b_bf16 v[0:3], v[2:3], v[4:5], v[2:5] ; encoding: [0x00,0x00,0xdf,0xd3,0x02,0x09,0x0a,0x04]
// GFX90A: error: instruction not supported on this GPU

v_mfma_f32_4x4x4_16b_bf16 a[0:3], v[2:3], v[4:5], a[2:5]
// GFX940: v_mfma_f32_4x4x4_16b_bf16 a[0:3], v[2:3], v[4:5], a[2:5] ; encoding: [0x00,0x80,0xdf,0xd3,0x02,0x09,0x0a,0x04]
// GFX90A: error: instruction not supported on this GPU

v_mfma_f32_4x4x4bf16 v[0:3], v[2:3], v[4:5], v[2:5]
// GFX940: v_mfma_f32_4x4x4_16b_bf16 v[0:3], v[2:3], v[4:5], v[2:5] ; encoding: [0x00,0x00,0xdf,0xd3,0x02,0x09,0x0a,0x04]
// GFX90A: error: instruction not supported on this GPU

v_mfma_f32_4x4x4bf16 a[0:3], v[2:3], v[4:5], a[2:5]
// GFX940: v_mfma_f32_4x4x4_16b_bf16 a[0:3], v[2:3], v[4:5], a[2:5] ; encoding: [0x00,0x80,0xdf,0xd3,0x02,0x09,0x0a,0x04]
// GFX90A: error: instruction not supported on this GPU

v_mfma_f32_4x4x4bf16_1k v[0:3], v[2:3], v[4:5], v[2:5]
// GFX940: v_mfma_f32_4x4x4_16b_bf16 v[0:3], v[2:3], v[4:5], v[2:5] ; encoding: [0x00,0x00,0xdf,0xd3,0x02,0x09,0x0a,0x04]

v_mfma_f32_4x4x4bf16_1k a[0:3], v[2:3], v[4:5], a[2:5]
// GFX940: v_mfma_f32_4x4x4_16b_bf16 a[0:3], v[2:3], v[4:5], a[2:5] ; encoding: [0x00,0x80,0xdf,0xd3,0x02,0x09,0x0a,0x04]

v_mfma_f32_32x32x8_bf16 v[0:15], v[2:3], v[4:5], v[18:33]
// GFX940: v_mfma_f32_32x32x8_bf16 v[0:15], v[2:3], v[4:5], v[18:33] ; encoding: [0x00,0x00,0xe0,0xd3,0x02,0x09,0x4a,0x04]
// GFX90A: error: instruction not supported on this GPU

v_mfma_f32_32x32x8_bf16 a[0:15], v[2:3], v[4:5], a[18:33]
// GFX940: v_mfma_f32_32x32x8_bf16 a[0:15], v[2:3], v[4:5], a[18:33] ; encoding: [0x00,0x80,0xe0,0xd3,0x02,0x09,0x4a,0x04]
// GFX90A: error: instruction not supported on this GPU

v_mfma_f32_32x32x8bf16 v[0:15], v[2:3], v[4:5], v[18:33]
// GFX940: v_mfma_f32_32x32x8_bf16 v[0:15], v[2:3], v[4:5], v[18:33] ; encoding: [0x00,0x00,0xe0,0xd3,0x02,0x09,0x4a,0x04]
// GFX90A: error: instruction not supported on this GPU

v_mfma_f32_32x32x8bf16 a[0:15], v[2:3], v[4:5], a[18:33]
// GFX940: v_mfma_f32_32x32x8_bf16 a[0:15], v[2:3], v[4:5], a[18:33] ; encoding: [0x00,0x80,0xe0,0xd3,0x02,0x09,0x4a,0x04]
// GFX90A: error: instruction not supported on this GPU

v_mfma_f32_32x32x8bf16_1k v[0:15], v[2:3], v[4:5], v[18:33]
// GFX940: v_mfma_f32_32x32x8_bf16 v[0:15], v[2:3], v[4:5], v[18:33] ; encoding: [0x00,0x00,0xe0,0xd3,0x02,0x09,0x4a,0x04]

v_mfma_f32_32x32x8bf16_1k a[0:15], v[2:3], v[4:5], a[18:33]
// GFX940: v_mfma_f32_32x32x8_bf16 a[0:15], v[2:3], v[4:5], a[18:33] ; encoding: [0x00,0x80,0xe0,0xd3,0x02,0x09,0x4a,0x04]

v_mfma_f32_16x16x16_bf16 v[0:3], v[2:3], v[4:5], v[2:5]
// GFX940: v_mfma_f32_16x16x16_bf16 v[0:3], v[2:3], v[4:5], v[2:5] ; encoding: [0x00,0x00,0xe1,0xd3,0x02,0x09,0x0a,0x04]
// GFX90A: error: instruction not supported on this GPU

v_mfma_f32_16x16x16_bf16 a[0:3], v[2:3], v[4:5], a[2:5]
// GFX940: v_mfma_f32_16x16x16_bf16 a[0:3], v[2:3], v[4:5], a[2:5] ; encoding: [0x00,0x80,0xe1,0xd3,0x02,0x09,0x0a,0x04]
// GFX90A: error: instruction not supported on this GPU

v_mfma_f32_16x16x16bf16 v[0:3], v[2:3], v[4:5], v[2:5]
// GFX940: v_mfma_f32_16x16x16_bf16 v[0:3], v[2:3], v[4:5], v[2:5] ; encoding: [0x00,0x00,0xe1,0xd3,0x02,0x09,0x0a,0x04]
// GFX90A: error: instruction not supported on this GPU

v_mfma_f32_16x16x16bf16 a[0:3], v[2:3], v[4:5], a[2:5]
// GFX940: v_mfma_f32_16x16x16_bf16 a[0:3], v[2:3], v[4:5], a[2:5] ; encoding: [0x00,0x80,0xe1,0xd3,0x02,0x09,0x0a,0x04]
// GFX90A: error: instruction not supported on this GPU

v_mfma_f32_16x16x16bf16_1k v[0:3], v[2:3], v[4:5], v[2:5]
// GFX940: v_mfma_f32_16x16x16_bf16 v[0:3], v[2:3], v[4:5], v[2:5] ; encoding: [0x00,0x00,0xe1,0xd3,0x02,0x09,0x0a,0x04]

v_mfma_f32_16x16x16bf16_1k a[0:3], v[2:3], v[4:5], a[2:5]
// GFX940: v_mfma_f32_16x16x16_bf16 a[0:3], v[2:3], v[4:5], a[2:5] ; encoding: [0x00,0x80,0xe1,0xd3,0x02,0x09,0x0a,0x04]

v_mfma_f32_16x16x8_xf32 a[0:3], v[2:3], v[4:5], a[2:5]
// GFX940: v_mfma_f32_16x16x8_xf32 a[0:3], v[2:3], v[4:5], a[2:5] ; encoding: [0x00,0x80,0xbe,0xd3,0x02,0x09,0x0a,0x04]
// GFX90A: error: instruction not supported on this GPU

v_mfma_f32_16x16x8_xf32 v[0:3], v[2:3], v[4:5], v[2:5]
// GFX940: v_mfma_f32_16x16x8_xf32 v[0:3], v[2:3], v[4:5], v[2:5] ; encoding: [0x00,0x00,0xbe,0xd3,0x02,0x09,0x0a,0x04]
// GFX90A: error: instruction not supported on this GPU

v_mfma_f32_16x16x8xf32 a[0:3], v[2:3], v[4:5], a[2:5]
// GFX940: v_mfma_f32_16x16x8_xf32 a[0:3], v[2:3], v[4:5], a[2:5] ; encoding: [0x00,0x80,0xbe,0xd3,0x02,0x09,0x0a,0x04]
// GFX90A: error: instruction not supported on this GPU

v_mfma_f32_16x16x8xf32 v[0:3], v[2:3], v[4:5], v[2:5]
// GFX940: v_mfma_f32_16x16x8_xf32 v[0:3], v[2:3], v[4:5], v[2:5] ; encoding: [0x00,0x00,0xbe,0xd3,0x02,0x09,0x0a,0x04]
// GFX90A: error: instruction not supported on this GPU

v_mfma_f32_32x32x4_xf32 v[0:15], v[2:3], v[4:5], v[18:33]
// GFX940: v_mfma_f32_32x32x4_xf32 v[0:15], v[2:3], v[4:5], v[18:33] ; encoding: [0x00,0x00,0xbf,0xd3,0x02,0x09,0x4a,0x04]
// GFX90A: error: instruction not supported on this GPU

v_mfma_f32_32x32x4_xf32 a[0:15], v[2:3], v[4:5], a[18:33]
// GFX940: v_mfma_f32_32x32x4_xf32 a[0:15], v[2:3], v[4:5], a[18:33] ; encoding: [0x00,0x80,0xbf,0xd3,0x02,0x09,0x4a,0x04]
// GFX90A: error: instruction not supported on this GPU

v_mfma_f32_32x32x4xf32 v[0:15], v[2:3], v[4:5], v[18:33]
// GFX940: v_mfma_f32_32x32x4_xf32 v[0:15], v[2:3], v[4:5], v[18:33] ; encoding: [0x00,0x00,0xbf,0xd3,0x02,0x09,0x4a,0x04]
// GFX90A: error: instruction not supported on this GPU

v_mfma_f32_32x32x4xf32 a[0:15], v[2:3], v[4:5], a[18:33]
// GFX940: v_mfma_f32_32x32x4_xf32 a[0:15], v[2:3], v[4:5], a[18:33] ; encoding: [0x00,0x80,0xbf,0xd3,0x02,0x09,0x4a,0x04]
// GFX90A: error: instruction not supported on this GPU

v_mfma_f32_16x16x32_bf8_bf8 v[0:3], v[2:3], v[4:5], v[0:3]
// GFX940: v_mfma_f32_16x16x32_bf8_bf8 v[0:3], v[2:3], v[4:5], v[0:3] ; encoding: [0x00,0x00,0xf0,0xd3,0x02,0x09,0x02,0x04]
// GFX90A: error: instruction not supported on this GPU

v_mfma_f32_16x16x32_bf8_bf8 a[0:3], v[2:3], v[4:5], a[0:3]
// GFX940: v_mfma_f32_16x16x32_bf8_bf8 a[0:3], v[2:3], v[4:5], a[0:3] ; encoding: [0x00,0x80,0xf0,0xd3,0x02,0x09,0x02,0x04]
// GFX90A: error: instruction not supported on this GPU

v_mfma_f32_16x16x32_bf8_bf8 a[0:3], v[2:3], v[4:5], a[0:3] blgp:5
// GFX940: v_mfma_f32_16x16x32_bf8_bf8 a[0:3], v[2:3], v[4:5], a[0:3] blgp:5 ; encoding: [0x00,0x80,0xf0,0xd3,0x02,0x09,0x02,0xa4]
// GFX90A: error: instruction not supported on this GPU

v_mfma_f32_16x16x32_bf8_fp8 v[0:3], v[2:3], v[4:5], v[0:3]
// GFX940: v_mfma_f32_16x16x32_bf8_fp8 v[0:3], v[2:3], v[4:5], v[0:3] ; encoding: [0x00,0x00,0xf1,0xd3,0x02,0x09,0x02,0x04]
// GFX90A: error: instruction not supported on this GPU

v_mfma_f32_16x16x32_bf8_fp8 a[0:3], v[2:3], v[4:5], a[0:3]
// GFX940: v_mfma_f32_16x16x32_bf8_fp8 a[0:3], v[2:3], v[4:5], a[0:3] ; encoding: [0x00,0x80,0xf1,0xd3,0x02,0x09,0x02,0x04]
// GFX90A: error: instruction not supported on this GPU

v_mfma_f32_16x16x32_bf8_fp8 a[0:3], v[2:3], v[4:5], a[0:3] blgp:5
// GFX940: v_mfma_f32_16x16x32_bf8_fp8 a[0:3], v[2:3], v[4:5], a[0:3] blgp:5 ; encoding: [0x00,0x80,0xf1,0xd3,0x02,0x09,0x02,0xa4]
// GFX90A: error: instruction not supported on this GPU

v_mfma_f32_16x16x32_fp8_bf8 v[0:3], v[2:3], v[4:5], v[0:3]
// GFX940: v_mfma_f32_16x16x32_fp8_bf8 v[0:3], v[2:3], v[4:5], v[0:3] ; encoding: [0x00,0x00,0xf2,0xd3,0x02,0x09,0x02,0x04]
// GFX90A: error: instruction not supported on this GPU

v_mfma_f32_16x16x32_fp8_bf8 a[0:3], v[2:3], v[4:5], a[0:3]
// GFX940: v_mfma_f32_16x16x32_fp8_bf8 a[0:3], v[2:3], v[4:5], a[0:3] ; encoding: [0x00,0x80,0xf2,0xd3,0x02,0x09,0x02,0x04]
// GFX90A: error: instruction not supported on this GPU

v_mfma_f32_16x16x32_fp8_bf8 a[0:3], v[2:3], v[4:5], a[0:3] blgp:5
// GFX940: v_mfma_f32_16x16x32_fp8_bf8 a[0:3], v[2:3], v[4:5], a[0:3] blgp:5 ; encoding: [0x00,0x80,0xf2,0xd3,0x02,0x09,0x02,0xa4]
// GFX90A: error: instruction not supported on this GPU

v_mfma_f32_16x16x32_fp8_fp8 v[0:3], v[2:3], v[4:5], v[0:3]
// GFX940: v_mfma_f32_16x16x32_fp8_fp8 v[0:3], v[2:3], v[4:5], v[0:3] ; encoding: [0x00,0x00,0xf3,0xd3,0x02,0x09,0x02,0x04]
// GFX90A: error: instruction not supported on this GPU

v_mfma_f32_16x16x32_fp8_fp8 a[0:3], v[2:3], v[4:5], a[0:3]
// GFX940: v_mfma_f32_16x16x32_fp8_fp8 a[0:3], v[2:3], v[4:5], a[0:3] ; encoding: [0x00,0x80,0xf3,0xd3,0x02,0x09,0x02,0x04]
// GFX90A: error: instruction not supported on this GPU

v_mfma_f32_16x16x32_fp8_fp8 a[0:3], v[2:3], v[4:5], a[0:3] blgp:5
// GFX940: v_mfma_f32_16x16x32_fp8_fp8 a[0:3], v[2:3], v[4:5], a[0:3] blgp:5 ; encoding: [0x00,0x80,0xf3,0xd3,0x02,0x09,0x02,0xa4]
// GFX90A: error: instruction not supported on this GPU

v_mfma_f32_32x32x16_bf8_bf8 v[0:15], v[2:3], v[4:5], v[0:15]
// GFX940: v_mfma_f32_32x32x16_bf8_bf8 v[0:15], v[2:3], v[4:5], v[0:15] ; encoding: [0x00,0x00,0xf4,0xd3,0x02,0x09,0x02,0x04]
// GFX90A: error: instruction not supported on this GPU

v_mfma_f32_32x32x16_bf8_bf8 a[0:15], v[2:3], v[4:5], a[0:15]
// GFX940: v_mfma_f32_32x32x16_bf8_bf8 a[0:15], v[2:3], v[4:5], a[0:15] ; encoding: [0x00,0x80,0xf4,0xd3,0x02,0x09,0x02,0x04]
// GFX90A: error: instruction not supported on this GPU

v_mfma_f32_32x32x16_bf8_bf8 a[0:15], v[2:3], v[4:5], a[0:15] blgp:5
// GFX940: v_mfma_f32_32x32x16_bf8_bf8 a[0:15], v[2:3], v[4:5], a[0:15] blgp:5 ; encoding: [0x00,0x80,0xf4,0xd3,0x02,0x09,0x02,0xa4]
// GFX90A: error: instruction not supported on this GPU

v_mfma_f32_32x32x16_bf8_fp8 v[0:15], v[2:3], v[4:5], v[0:15]
// GFX940: v_mfma_f32_32x32x16_bf8_fp8 v[0:15], v[2:3], v[4:5], v[0:15] ; encoding: [0x00,0x00,0xf5,0xd3,0x02,0x09,0x02,0x04]
// GFX90A: error: instruction not supported on this GPU

v_mfma_f32_32x32x16_bf8_fp8 a[0:15], v[2:3], v[4:5], a[0:15]
// GFX940: v_mfma_f32_32x32x16_bf8_fp8 a[0:15], v[2:3], v[4:5], a[0:15] ; encoding: [0x00,0x80,0xf5,0xd3,0x02,0x09,0x02,0x04]
// GFX90A: error: instruction not supported on this GPU

v_mfma_f32_32x32x16_bf8_fp8 a[0:15], v[2:3], v[4:5], a[0:15] blgp:5
// GFX940: v_mfma_f32_32x32x16_bf8_fp8 a[0:15], v[2:3], v[4:5], a[0:15] blgp:5 ; encoding: [0x00,0x80,0xf5,0xd3,0x02,0x09,0x02,0xa4]
// GFX90A: error: instruction not supported on this GPU

v_mfma_f32_32x32x16_fp8_bf8 v[0:15], v[2:3], v[4:5], v[0:15]
// GFX940: v_mfma_f32_32x32x16_fp8_bf8 v[0:15], v[2:3], v[4:5], v[0:15] ; encoding: [0x00,0x00,0xf6,0xd3,0x02,0x09,0x02,0x04]
// GFX90A: error: instruction not supported on this GPU

v_mfma_f32_32x32x16_fp8_bf8 a[0:15], v[2:3], v[4:5], a[0:15]
// GFX940: v_mfma_f32_32x32x16_fp8_bf8 a[0:15], v[2:3], v[4:5], a[0:15] ; encoding: [0x00,0x80,0xf6,0xd3,0x02,0x09,0x02,0x04]
// GFX90A: error: instruction not supported on this GPU

v_mfma_f32_32x32x16_fp8_bf8 a[0:15], v[2:3], v[4:5], a[0:15] blgp:5
// GFX940: v_mfma_f32_32x32x16_fp8_bf8 a[0:15], v[2:3], v[4:5], a[0:15] blgp:5 ; encoding: [0x00,0x80,0xf6,0xd3,0x02,0x09,0x02,0xa4]
// GFX90A: error: instruction not supported on this GPU

v_mfma_f32_32x32x16_fp8_fp8 v[0:15], v[2:3], v[4:5], v[0:15]
// GFX940: v_mfma_f32_32x32x16_fp8_fp8 v[0:15], v[2:3], v[4:5], v[0:15] ; encoding: [0x00,0x00,0xf7,0xd3,0x02,0x09,0x02,0x04]
// GFX90A: error: instruction not supported on this GPU

v_mfma_f32_32x32x16_fp8_fp8 a[0:15], v[2:3], v[4:5], a[0:15]
// GFX940: v_mfma_f32_32x32x16_fp8_fp8 a[0:15], v[2:3], v[4:5], a[0:15] ; encoding: [0x00,0x80,0xf7,0xd3,0x02,0x09,0x02,0x04]
// GFX90A: error: instruction not supported on this GPU

v_mfma_f32_32x32x16_fp8_fp8 a[0:15], v[2:3], v[4:5], a[0:15] blgp:5
// GFX940: v_mfma_f32_32x32x16_fp8_fp8 a[0:15], v[2:3], v[4:5], a[0:15] blgp:5 ; encoding: [0x00,0x80,0xf7,0xd3,0x02,0x09,0x02,0xa4]
// GFX90A: error: instruction not supported on this GPU

//===----------------------------------------------------------------------===//
// SMFMAC opcodes.
//===----------------------------------------------------------------------===//

v_smfmac_f32_16x16x32_f16 v[10:13], a[2:3], v[4:7], v0 cbsz:3 abid:1
// GFX940: v_smfmac_f32_16x16x32_f16 v[10:13], a[2:3], v[4:7], v0 cbsz:3 abid:1 ; encoding: [0x0a,0x0b,0xe2,0xd3,0x02,0x09,0x02,0x0c]
// GFX90A: error: instruction not supported on this GPU

v_smfmac_f32_16x16x32_f16 a[10:13], v[2:3], a[4:7], v1
// GFX940: v_smfmac_f32_16x16x32_f16 a[10:13], v[2:3], a[4:7], v1 ; encoding: [0x0a,0x80,0xe2,0xd3,0x02,0x09,0x06,0x14]
// GFX90A: error: instruction not supported on this GPU

v_smfmac_f32_32x32x16_f16 v[10:25], a[2:3], v[4:7], v2 cbsz:3 abid:1
// GFX940: v_smfmac_f32_32x32x16_f16 v[10:25], a[2:3], v[4:7], v2 cbsz:3 abid:1 ; encoding: [0x0a,0x0b,0xe4,0xd3,0x02,0x09,0x0a,0x0c]
// GFX90A: error: instruction not supported on this GPU

v_smfmac_f32_32x32x16_f16 a[10:25], v[2:3], a[4:7], v3
// GFX940: v_smfmac_f32_32x32x16_f16 a[10:25], v[2:3], a[4:7], v3 ; encoding: [0x0a,0x80,0xe4,0xd3,0x02,0x09,0x0e,0x14]
// GFX90A: error: instruction not supported on this GPU

v_smfmac_f32_16x16x32_bf16 v[10:13], a[2:3], v[4:7], v4 cbsz:3 abid:1
// GFX940: v_smfmac_f32_16x16x32_bf16 v[10:13], a[2:3], v[4:7], v4 cbsz:3 abid:1 ; encoding: [0x0a,0x0b,0xe6,0xd3,0x02,0x09,0x12,0x0c]
// GFX90A: error: instruction not supported on this GPU

v_smfmac_f32_16x16x32_bf16 a[10:13], v[2:3], a[4:7], v5
// GFX940: v_smfmac_f32_16x16x32_bf16 a[10:13], v[2:3], a[4:7], v5 ; encoding: [0x0a,0x80,0xe6,0xd3,0x02,0x09,0x16,0x14]
// GFX90A: error: instruction not supported on this GPU

v_smfmac_f32_32x32x16_bf16 v[10:25], a[2:3], v[4:7], v6 cbsz:3 abid:1
// GFX940: v_smfmac_f32_32x32x16_bf16 v[10:25], a[2:3], v[4:7], v6 cbsz:3 abid:1 ; encoding: [0x0a,0x0b,0xe8,0xd3,0x02,0x09,0x1a,0x0c]
// GFX90A: error: instruction not supported on this GPU

v_smfmac_f32_32x32x16_bf16 a[10:25], v[2:3], a[4:7], v7
// GFX940: v_smfmac_f32_32x32x16_bf16 a[10:25], v[2:3], a[4:7], v7 ; encoding: [0x0a,0x80,0xe8,0xd3,0x02,0x09,0x1e,0x14]
// GFX90A: error: instruction not supported on this GPU

v_smfmac_i32_16x16x64_i8 v[10:13], a[2:3], v[4:7], v8 cbsz:3 abid:1
// GFX940: v_smfmac_i32_16x16x64_i8 v[10:13], a[2:3], v[4:7], v8 cbsz:3 abid:1 ; encoding: [0x0a,0x0b,0xea,0xd3,0x02,0x09,0x22,0x0c]
// GFX90A: error: instruction not supported on this GPU

v_smfmac_i32_16x16x64_i8 a[10:13], v[2:3], a[4:7], v9
// GFX940: v_smfmac_i32_16x16x64_i8 a[10:13], v[2:3], a[4:7], v9 ; encoding: [0x0a,0x80,0xea,0xd3,0x02,0x09,0x26,0x14]
// GFX90A: error: instruction not supported on this GPU

v_smfmac_i32_32x32x32_i8 v[10:25], a[2:3], v[4:7], v10 cbsz:3 abid:1
// GFX940: v_smfmac_i32_32x32x32_i8 v[10:25], a[2:3], v[4:7], v10 cbsz:3 abid:1 ; encoding: [0x0a,0x0b,0xec,0xd3,0x02,0x09,0x2a,0x0c]
// GFX90A: error: instruction not supported on this GPU

v_smfmac_i32_32x32x32_i8 a[10:25], v[2:3], a[4:7], v11
// GFX940: v_smfmac_i32_32x32x32_i8 a[10:25], v[2:3], a[4:7], v11 ; encoding: [0x0a,0x80,0xec,0xd3,0x02,0x09,0x2e,0x14]
// GFX90A: error: instruction not supported on this GPU

//===----------------------------------------------------------------------===//
// SMFMAC aliases.
//===----------------------------------------------------------------------===//

v_smfmac_f32_16x16x32f16 v[10:13], a[2:3], v[4:7], v0 cbsz:3 abid:1
// GFX940: v_smfmac_f32_16x16x32_f16 v[10:13], a[2:3], v[4:7], v0 cbsz:3 abid:1 ; encoding: [0x0a,0x0b,0xe2,0xd3,0x02,0x09,0x02,0x0c]
// GFX90A: error: instruction not supported on this GPU

v_smfmac_f32_32x32x16f16 v[10:25], a[2:3], v[4:7], v2 cbsz:3 abid:1
// GFX940: v_smfmac_f32_32x32x16_f16 v[10:25], a[2:3], v[4:7], v2 cbsz:3 abid:1 ; encoding: [0x0a,0x0b,0xe4,0xd3,0x02,0x09,0x0a,0x0c]
// GFX90A: error: instruction not supported on this GPU

v_smfmac_f32_16x16x32bf16 v[10:13], a[2:3], v[4:7], v4 cbsz:3 abid:1
// GFX940: v_smfmac_f32_16x16x32_bf16 v[10:13], a[2:3], v[4:7], v4 cbsz:3 abid:1 ; encoding: [0x0a,0x0b,0xe6,0xd3,0x02,0x09,0x12,0x0c]
// GFX90A: error: instruction not supported on this GPU

v_smfmac_f32_32x32x16bf16 v[10:25], a[2:3], v[4:7], v6 cbsz:3 abid:1
// GFX940: v_smfmac_f32_32x32x16_bf16 v[10:25], a[2:3], v[4:7], v6 cbsz:3 abid:1 ; encoding: [0x0a,0x0b,0xe8,0xd3,0x02,0x09,0x1a,0x0c]
// GFX90A: error: instruction not supported on this GPU

v_smfmac_i32_16x16x64i8 v[10:13], a[2:3], v[4:7], v8 cbsz:3 abid:1
// GFX940: v_smfmac_i32_16x16x64_i8 v[10:13], a[2:3], v[4:7], v8 cbsz:3 abid:1 ; encoding: [0x0a,0x0b,0xea,0xd3,0x02,0x09,0x22,0x0c]
// GFX90A: error: instruction not supported on this GPU

v_smfmac_i32_32x32x32i8 a[10:25], v[2:3], a[4:7], v11
// GFX940: v_smfmac_i32_32x32x32_i8 a[10:25], v[2:3], a[4:7], v11 ; encoding: [0x0a,0x80,0xec,0xd3,0x02,0x09,0x2e,0x14]
// GFX90A: error: instruction not supported on this GPU
