// RUN: llvm-mc -arch=amdgcn -mcpu=gfx940 -show-encoding %s | FileCheck -check-prefix=GFX940 %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx90a %s 2>&1 | FileCheck -check-prefix=GFX90A %s

v_accvgpr_write_b32 a10, s20
// GFX940: v_accvgpr_write_b32 a10, s20    ; encoding: [0x0a,0x40,0xd9,0xd3,0x14,0x00,0x00,0x18]
// GFX90A: error: source operand must be either a VGPR or an inline constant

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
