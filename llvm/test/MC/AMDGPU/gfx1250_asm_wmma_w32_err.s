// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx1250 %s 2>&1 | FileCheck --check-prefix=GFX1250-ERR --implicit-check-not=error: %s

v_wmma_f32_16x16x4_f32 v[4:11], v[0:1], v[2:3], s[4:11]
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_wmma_f32_16x16x4_f32 v[4:11], v[0:1], v[2:3], 3.0
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_wmma_f32_16x16x32_bf16 v[16:23], v[0:7], v[8:15], s[16:23]
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_wmma_f32_16x16x32_bf16 v[16:23], v[0:7], v[8:15], 3.0
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_wmma_bf16_16x16x32_bf16 v[16:19], v[0:7], v[8:15], s[16:19]
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_wmma_bf16_16x16x32_bf16 v[16:19], v[0:7], v[8:15], 3.0
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_wmma_bf16f32_16x16x32_bf16 v[26:29], v[0:7], v[8:15], s[16:23]
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_wmma_bf16f32_16x16x32_bf16 v[26:29], v[0:7], v[8:15], 3.0
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_wmma_f32_16x16x64_fp8_fp8 v[16:23], v[0:7], v[8:15], s[16:23]
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_wmma_f32_16x16x64_fp8_fp8 v[16:23], v[0:7], v[8:15], 3.0
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_wmma_f32_16x16x64_fp8_fp8 v[16:23], v[0:7], v[8:15], v[16:23] neg_lo:[1,0,0]
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid neg_lo operand

v_wmma_f32_16x16x64_fp8_fp8 v[16:23], v[0:7], v[8:15], v[16:23] neg_lo:[0,1,0]
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid neg_lo operand

v_wmma_f32_16x16x64_fp8_bf8 v[16:23], v[0:7], v[8:15], s[16:23]
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_wmma_f32_16x16x64_fp8_bf8 v[16:23], v[0:7], v[8:15], 3.0
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_wmma_f32_16x16x64_fp8_bf8 v[16:23], v[0:7], v[8:15], v[16:23] neg_lo:[1,0,0]
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid neg_lo operand

v_wmma_f32_16x16x64_fp8_bf8 v[16:23], v[0:7], v[8:15], v[16:23] neg_lo:[0,1,0]
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid neg_lo operand

v_wmma_f32_16x16x64_bf8_fp8 v[16:23], v[0:7], v[8:15], s[16:23]
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_wmma_f32_16x16x64_bf8_fp8 v[16:23], v[0:7], v[8:15], 3.0
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_wmma_f32_16x16x64_bf8_fp8 v[16:23], v[0:7], v[8:15], v[16:23] neg_lo:[1,0,0]
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid neg_lo operand

v_wmma_f32_16x16x64_bf8_fp8 v[16:23], v[0:7], v[8:15], v[16:23] neg_lo:[0,1,0]
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid neg_lo operand

v_wmma_f32_16x16x64_bf8_bf8 v[16:23], v[0:7], v[8:15], s[16:23]
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_wmma_f32_16x16x64_bf8_bf8 v[16:23], v[0:7], v[8:15], 3.0
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_wmma_f32_16x16x64_bf8_bf8 v[16:23], v[0:7], v[8:15], v[16:23] neg_lo:[1,0,0]
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid neg_lo operand

v_wmma_f32_16x16x64_bf8_bf8 v[16:23], v[0:7], v[8:15], v[16:23] neg_lo:[0,1,0]
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid neg_lo operand

v_wmma_f16_16x16x64_fp8_fp8 v[16:19], v[0:7], v[8:15], s[16:19]
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_wmma_f16_16x16x64_fp8_fp8 v[16:19], v[0:7], v[8:15], 3.0
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_wmma_f16_16x16x64_fp8_fp8 v[16:19], v[0:7], v[8:15], v[16:19] neg_lo:[1,0,0]
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid neg_lo operand

v_wmma_f16_16x16x64_fp8_fp8 v[16:19], v[0:7], v[8:15], v[16:19] neg_lo:[0,1,0]
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid neg_lo operand

v_wmma_f16_16x16x64_fp8_bf8 v[16:19], v[0:7], v[8:15], s[16:19]
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_wmma_f16_16x16x64_fp8_bf8 v[16:19], v[0:7], v[8:15], 3.0
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_wmma_f16_16x16x64_fp8_bf8 v[16:19], v[0:7], v[8:15], v[16:19] neg_lo:[1,0,0]
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid neg_lo operand

v_wmma_f16_16x16x64_fp8_bf8 v[16:19], v[0:7], v[8:15], v[16:19] neg_lo:[0,1,0]
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid neg_lo operand

v_wmma_f16_16x16x64_bf8_fp8 v[16:19], v[0:7], v[8:15], s[16:19]
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_wmma_f16_16x16x64_bf8_fp8 v[16:19], v[0:7], v[8:15], 3.0
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_wmma_f16_16x16x64_bf8_fp8 v[16:19], v[0:7], v[8:15], v[16:19] neg_lo:[1,0,0]
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid neg_lo operand

v_wmma_f16_16x16x64_bf8_fp8 v[16:19], v[0:7], v[8:15], v[16:19] neg_lo:[0,1,0]
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid neg_lo operand

v_wmma_f16_16x16x64_bf8_bf8 v[16:19], v[0:7], v[8:15], s[16:19]
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_wmma_f16_16x16x64_bf8_bf8 v[16:19], v[0:7], v[8:15], 3.0
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_wmma_f16_16x16x64_bf8_bf8 v[16:19], v[0:7], v[8:15], v[16:19] neg_lo:[1,0,0]
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid neg_lo operand

v_wmma_f16_16x16x64_bf8_bf8 v[16:19], v[0:7], v[8:15], v[16:19] neg_lo:[0,1,0]
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid neg_lo operand

v_wmma_i32_16x16x64_iu8 v[16:23], v[0:7], v[8:15], s[16:23]
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_wmma_i32_16x16x64_iu8 v[16:23], v[0:7], v[8:15], 128
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_wmma_i32_16x16x64_iu8 v[16:23], v[0:7], v[8:15], v[16:23] clamp
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_wmma_i32_16x16x64_iu8 v[16:23], v[0:7], v[8:15], v[16:23] neg_lo:[0,0,1]
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid neg_lo operand

v_wmma_f32_16x16x32_f16 v[16:23], v[0:7], v[8:15], s[16:23]
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_wmma_f32_16x16x32_f16 v[16:23], v[0:7], v[8:15], 3.0
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_wmma_f16_16x16x32_f16 v[16:19], v[0:7], v[8:15], s[16:19]
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_wmma_f16_16x16x32_f16 v[16:19], v[0:7], v[8:15], 3.0
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_swmmac_f32_16x16x64_bf16 v[24:31], v[0:7], v[8:23], s32
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_swmmac_f32_16x16x64_bf16 v[24:31], v[0:7], 1.0, v32
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_swmmac_f32_16x16x64_bf16 v[24:31], v[0:7], v[8:23], v32 index_key:2
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: out of range index_key

v_swmmac_f32_16x16x64_bf16 v[24:31], v[0:7], v[8:23], v32 neg_lo:[0,0,1]
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid neg_lo operand

v_swmmac_bf16_16x16x64_bf16 v[24:27], v[0:7], v[8:23], s28
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_swmmac_bf16_16x16x64_bf16 v[24:27], v[0:7], 1.0, v28
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_swmmac_bf16_16x16x64_bf16 v[24:27], v[0:7], v[8:23], v28 index_key:2
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: out of range index_key

v_swmmac_bf16_16x16x64_bf16 v[24:27], v[0:7], v[8:23], v28 neg_lo:[0,0,1]
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid neg_lo operand

v_swmmac_bf16f32_16x16x64_bf16 v[24:31], v[0:7], v[8:23], s32
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_swmmac_bf16f32_16x16x64_bf16 v[24:31], v[0:7], 1.0, v32
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_swmmac_bf16f32_16x16x64_bf16 v[24:31], v[0:7], v[8:23], v32 index_key:2
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: out of range index_key

v_swmmac_bf16f32_16x16x64_bf16 v[24:31], v[0:7], v[8:23], v32 neg_lo:[0,0,1]
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid neg_lo operand

v_swmmac_f32_16x16x128_fp8_fp8 v[24:31], v[0:7], v[8:23], s[32:33]
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_swmmac_f32_16x16x128_fp8_fp8 v[24:31], v[0:7], 1.0, v[32:33]
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_swmmac_f32_16x16x128_fp8_fp8 v[24:31], v[0:7], v[8:23], v[32:33] index_key:2
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: out of range index_key

v_swmmac_f32_16x16x128_fp8_fp8 v[24:31], v[0:7], v[8:23], v[32:33] neg_lo:[1,0,0]
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_f32_16x16x128_fp8_fp8 v[24:31], v[0:7], v[8:23], v[32:33] neg_lo:[0,1,0]
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_f32_16x16x128_fp8_fp8 v[24:31], v[0:7], v[8:23], v[32:33] neg_lo:[0,0,1]
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_f32_16x16x128_fp8_bf8 v[24:31], v[0:7], v[8:23], s[32:33]
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_swmmac_f32_16x16x128_fp8_bf8 v[24:31], v[0:7], 1.0, v[32:33]
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_swmmac_f32_16x16x128_fp8_bf8 v[24:31], v[0:7], v[8:23], v[32:33] index_key:2
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: out of range index_key

v_swmmac_f32_16x16x128_fp8_bf8 v[24:31], v[0:7], v[8:23], v[32:33] neg_lo:[1,0,0]
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_f32_16x16x128_fp8_bf8 v[24:31], v[0:7], v[8:23], v[32:33] neg_lo:[0,1,0]
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_f32_16x16x128_fp8_bf8 v[24:31], v[0:7], v[8:23], v[32:33] neg_lo:[0,0,1]
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_f32_16x16x128_bf8_fp8 v[24:31], v[0:7], v[8:23], s[32:33]
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_swmmac_f32_16x16x128_bf8_fp8 v[24:31], v[0:7], 1.0, v[32:33]
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_swmmac_f32_16x16x128_bf8_fp8 v[24:31], v[0:7], v[8:23], v[32:33] index_key:2
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: out of range index_key

v_swmmac_f32_16x16x128_bf8_fp8 v[24:31], v[0:7], v[8:23], v[32:33] neg_lo:[1,0,0]
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_f32_16x16x128_bf8_fp8 v[24:31], v[0:7], v[8:23], v[32:33] neg_lo:[0,1,0]
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_f32_16x16x128_bf8_fp8 v[24:31], v[0:7], v[8:23], v[32:33] neg_lo:[0,0,1]
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_f32_16x16x128_bf8_bf8 v[24:31], v[0:7], v[8:23], s[32:33]
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_swmmac_f32_16x16x128_bf8_bf8 v[24:31], v[0:7], 1.0, v[32:33]
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_swmmac_f32_16x16x128_bf8_bf8 v[24:31], v[0:7], v[8:23], v[32:33] index_key:2
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: out of range index_key

v_swmmac_f32_16x16x128_bf8_bf8 v[24:31], v[0:7], v[8:23], v[32:33] neg_lo:[1,0,0]
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_f32_16x16x128_bf8_bf8 v[24:31], v[0:7], v[8:23], v[32:33] neg_lo:[0,1,0]
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_f32_16x16x128_bf8_bf8 v[24:31], v[0:7], v[8:23], v[32:33] neg_lo:[0,0,1]
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_f16_16x16x128_fp8_fp8 v[24:27], v[0:7], v[8:23], s[28:29]
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_swmmac_f16_16x16x128_fp8_fp8 v[24:27], v[0:7], 1.0, v[28:29]
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_swmmac_f16_16x16x128_fp8_fp8 v[24:27], v[0:7], v[8:23], v[28:29] index_key:2
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: out of range index_key

v_swmmac_f16_16x16x128_fp8_fp8 v[24:27], v[0:7], v[8:23], v[28:29] neg_lo:[1,0,0]
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_f16_16x16x128_fp8_fp8 v[24:27], v[0:7], v[8:23], v[28:29] neg_lo:[0,1,0]
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_f16_16x16x128_fp8_fp8 v[24:27], v[0:7], v[8:23], v[28:29] neg_lo:[0,0,1]
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_f16_16x16x128_fp8_bf8 v[24:27], v[0:7], v[8:23], s[28:29]
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_swmmac_f16_16x16x128_fp8_bf8 v[24:27], v[0:7], 1.0, v[28:29]
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_swmmac_f16_16x16x128_fp8_bf8 v[24:27], v[0:7], v[8:23], v[28:29] index_key:2
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: out of range index_key

v_swmmac_f16_16x16x128_fp8_bf8 v[24:27], v[0:7], v[8:23], v[28:29] neg_lo:[1,0,0]
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_f16_16x16x128_fp8_bf8 v[24:27], v[0:7], v[8:23], v[28:29] neg_lo:[0,1,0]
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_f16_16x16x128_fp8_bf8 v[24:27], v[0:7], v[8:23], v[28:29] neg_lo:[0,0,1]
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_f16_16x16x128_bf8_fp8 v[24:27], v[0:7], v[8:23], s[28:29]
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_swmmac_f16_16x16x128_bf8_fp8 v[24:27], v[0:7], 1.0, v[28:29]
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_swmmac_f16_16x16x128_bf8_fp8 v[24:27], v[0:7], v[8:23], v[28:29] index_key:2
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: out of range index_key

v_swmmac_f16_16x16x128_bf8_fp8 v[24:27], v[0:7], v[8:23], v[28:29] neg_lo:[1,0,0]
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_f16_16x16x128_bf8_fp8 v[24:27], v[0:7], v[8:23], v[28:29] neg_lo:[0,1,0]
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_f16_16x16x128_bf8_fp8 v[24:27], v[0:7], v[8:23], v[28:29] neg_lo:[0,0,1]
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_f16_16x16x128_bf8_bf8 v[24:27], v[0:7], v[8:23], s[28:29]
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_swmmac_f16_16x16x128_bf8_bf8 v[24:27], v[0:7], 1.0, v[28:29]
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_swmmac_f16_16x16x128_bf8_bf8 v[24:27], v[0:7], v[8:23], v[28:29] index_key:2
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: out of range index_key

v_swmmac_f16_16x16x128_bf8_bf8 v[24:27], v[0:7], v[8:23], v[28:29] neg_lo:[1,0,0]
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_f16_16x16x128_bf8_bf8 v[24:27], v[0:7], v[8:23], v[28:29] neg_lo:[0,1,0]
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_f16_16x16x128_bf8_bf8 v[24:27], v[0:7], v[8:23], v[28:29] neg_lo:[0,0,1]
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_i32_16x16x128_iu8 v[24:31], v[0:7], v[8:23], s[32:33]
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_swmmac_i32_16x16x128_iu8 v[24:31], v[0:7], 1, v[32:33]
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_swmmac_i32_16x16x128_iu8 v[24:31], v[0:7], v[8:23], v[32:33] clamp
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_swmmac_i32_16x16x128_iu8 v[24:31], v[0:7], v[8:23], v[32:33] index_key:2
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: out of range index_key

v_swmmac_i32_16x16x128_iu8 v[24:31], v[0:7], v[8:23], v[32:33] neg_lo:[0,0,1]
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid neg_lo operand

v_swmmac_f32_16x16x64_f16 v[24:31], v[0:7], v[8:23], s32
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_swmmac_f32_16x16x64_f16 v[24:31], v[0:7], 1.0, v32
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_swmmac_f32_16x16x64_f16 v[24:31], v[0:7], v[8:23], v32 index_key:2
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: out of range index_key

v_swmmac_f32_16x16x64_f16 v[24:31], v[0:7], v[8:23], v32 neg_lo:[0,0,1]
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid neg_lo operand

v_swmmac_f16_16x16x64_f16 v[24:27], v[0:7], v[8:23], s28
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_swmmac_f16_16x16x64_f16 v[24:27], v[0:7], 1.0, v28
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_swmmac_f16_16x16x64_f16 v[24:27], v[0:7], v[8:23], v28 index_key:2
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: out of range index_key

v_swmmac_f16_16x16x64_f16 v[24:27], v[0:7], v[8:23], v28 neg_lo:[0,0,1]
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid neg_lo operand

v_wmma_f32_16x16x128_f8f6f4 v[0:7], v[8:23], v[24:39], v[40:47] neg_lo:[1,0,0]
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid neg_lo operand

v_wmma_f32_16x16x128_f8f6f4 v[0:7], v[8:23], v[24:39], v[40:47] neg_lo:[0,1,0]
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid neg_lo operand

v_wmma_f32_16x16x128_f8f6f4 v[0:7], v[8:23], v[24:39], v[40:47] neg_hi:[1,0,0]
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid neg_hi operand

v_wmma_f32_16x16x128_f8f6f4 v[0:7], v[8:23], v[24:39], v[40:47] neg_hi:[0,1,0]
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid neg_hi operand

v_wmma_f32_16x16x128_f8f6f4 v[0:7], v[8:23], v[24:39], v[40:47] clamp
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_wmma_f32_16x16x128_f8f6f4 v[0:7], v[8:23], v[24:39], v[40:47] matrix_b_fmt:-1
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid matrix_b_fmt value

v_wmma_f32_16x16x128_f8f6f4 v[0:7], v[8:23], v[24:39], v[40:47] matrix_b_fmt:xxx
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid matrix_b_fmt value

v_wmma_f32_16x16x128_f8f6f4 v[0:7], v[0:7], v[20:35], v[40:47]
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: wrong register tuple size for MATRIX_FMT_FP8
// GFX1250-ERR-NEXT: {{^}}v_wmma_f32_16x16x128_f8f6f4 v[0:7], v[0:7], v[20:35], v[40:47]
// GFX1250-ERR-NEXT: {{^}}                                    ^

v_wmma_f32_16x16x128_f8f6f4 v[0:7], v[0:7], v[20:35], v[40:47] matrix_a_fmt:MATRIX_FMT_FP8
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: wrong register tuple size for MATRIX_FMT_FP8
// GFX1250-ERR-NEXT: {{^}}v_wmma_f32_16x16x128_f8f6f4 v[0:7], v[0:7], v[20:35], v[40:47] matrix_a_fmt:MATRIX_FMT_FP8
// GFX1250-ERR-NEXT: {{^}}                                    ^

v_wmma_f32_16x16x128_f8f6f4 v[0:7], v[0:7], v[20:35], v[40:47] matrix_a_fmt:MATRIX_FMT_BF8
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: wrong register tuple size for MATRIX_FMT_BF8
// GFX1250-ERR-NEXT: {{^}}v_wmma_f32_16x16x128_f8f6f4 v[0:7], v[0:7], v[20:35], v[40:47] matrix_a_fmt:MATRIX_FMT_BF8
// GFX1250-ERR-NEXT: {{^}}                                    ^

v_wmma_f32_16x16x128_f8f6f4 v[0:7], v[0:7], v[20:35], v[40:47] matrix_a_fmt:MATRIX_FMT_FP6
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: wrong register tuple size for MATRIX_FMT_FP6
// GFX1250-ERR-NEXT: {{^}}v_wmma_f32_16x16x128_f8f6f4 v[0:7], v[0:7], v[20:35], v[40:47] matrix_a_fmt:MATRIX_FMT_FP6
// GFX1250-ERR-NEXT: {{^}}                                    ^

v_wmma_f32_16x16x128_f8f6f4 v[0:7], v[0:7], v[20:35], v[40:47] matrix_a_fmt:MATRIX_FMT_BF6
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: wrong register tuple size for MATRIX_FMT_BF6
// GFX1250-ERR-NEXT: {{^}}v_wmma_f32_16x16x128_f8f6f4 v[0:7], v[0:7], v[20:35], v[40:47] matrix_a_fmt:MATRIX_FMT_BF6
// GFX1250-ERR-NEXT: {{^}}                                    ^

v_wmma_f32_16x16x128_f8f6f4 v[0:7], v[0:15], v[20:35], v[40:47] matrix_a_fmt:MATRIX_FMT_FP4
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: wrong register tuple size for MATRIX_FMT_FP4
// GFX1250-ERR-NEXT: {{^}}v_wmma_f32_16x16x128_f8f6f4 v[0:7], v[0:15], v[20:35], v[40:47] matrix_a_fmt:MATRIX_FMT_FP4
// GFX1250-ERR-NEXT: {{^}}                                    ^

v_wmma_f32_16x16x128_f8f6f4 v[0:7], v[0:15], v[20:27], v[40:47] matrix_b_fmt:MATRIX_FMT_FP8
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: wrong register tuple size for MATRIX_FMT_FP8
// GFX1250-ERR-NEXT: {{^}}v_wmma_f32_16x16x128_f8f6f4 v[0:7], v[0:15], v[20:27], v[40:47] matrix_b_fmt:MATRIX_FMT_FP8
// GFX1250-ERR-NEXT: {{^}}                                             ^

v_wmma_f32_16x16x128_f8f6f4 v[0:7], v[0:15], v[20:27], v[40:47] matrix_b_fmt:MATRIX_FMT_BF8
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: wrong register tuple size for MATRIX_FMT_BF8
// GFX1250-ERR-NEXT: {{^}}v_wmma_f32_16x16x128_f8f6f4 v[0:7], v[0:15], v[20:27], v[40:47] matrix_b_fmt:MATRIX_FMT_BF8
// GFX1250-ERR-NEXT: {{^}}                                             ^

v_wmma_f32_16x16x128_f8f6f4 v[0:7], v[0:15], v[20:27], v[40:47] matrix_b_fmt:MATRIX_FMT_FP6
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: wrong register tuple size for MATRIX_FMT_FP6
// GFX1250-ERR-NEXT: {{^}}v_wmma_f32_16x16x128_f8f6f4 v[0:7], v[0:15], v[20:27], v[40:47] matrix_b_fmt:MATRIX_FMT_FP6
// GFX1250-ERR-NEXT: {{^}}                                             ^

v_wmma_f32_16x16x128_f8f6f4 v[0:7], v[0:15], v[20:27], v[40:47] matrix_b_fmt:MATRIX_FMT_BF6
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: wrong register tuple size for MATRIX_FMT_BF6
// GFX1250-ERR-NEXT: {{^}}v_wmma_f32_16x16x128_f8f6f4 v[0:7], v[0:15], v[20:27], v[40:47] matrix_b_fmt:MATRIX_FMT_BF6
// GFX1250-ERR-NEXT: {{^}}                                             ^

v_wmma_f32_16x16x128_f8f6f4 v[0:7], v[0:15], v[20:35], v[40:47] matrix_b_fmt:MATRIX_FMT_FP4
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: wrong register tuple size for MATRIX_FMT_FP4
// GFX1250-ERR-NEXT: {{^}}v_wmma_f32_16x16x128_f8f6f4 v[0:7], v[0:15], v[20:35], v[40:47] matrix_b_fmt:MATRIX_FMT_FP4
// GFX1250-ERR-NEXT: {{^}}                                             ^

v_wmma_f32_32x16x128_f4 v[4:19], v[0:15], v[2:9], v[4:19] neg_lo:[1,0,0]
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid neg_lo operand
// GFX1250-ERR-NEXT: {{^}}v_wmma_f32_32x16x128_f4 v[4:19], v[0:15], v[2:9], v[4:19] neg_lo:[1,0,0]
// GFX1250-ERR-NEXT: {{^}}                                                          ^

v_wmma_f32_32x16x128_f4 v[4:19], v[0:15], v[2:9], v[4:19] neg_lo:[0,1,0]
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid neg_lo operand
// GFX1250-ERR-NEXT: {{^}}v_wmma_f32_32x16x128_f4 v[4:19], v[0:15], v[2:9], v[4:19] neg_lo:[0,1,0]
// GFX1250-ERR-NEXT: {{^}}                                                          ^

v_wmma_f32_32x16x128_f4 v[4:19], v[0:15], v[2:9], v[4:19] neg_hi:[1,0,0]
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid neg_hi operand
// GFX1250-ERR-NEXT: {{^}}v_wmma_f32_32x16x128_f4 v[4:19], v[0:15], v[2:9], v[4:19] neg_hi:[1,0,0]
// GFX1250-ERR-NEXT: {{^}}                                                          ^

v_wmma_f32_32x16x128_f4 v[4:19], v[0:15], v[2:9], v[4:19] neg_hi:[0,1,0]
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid neg_hi operand
// GFX1250-ERR-NEXT: {{^}}v_wmma_f32_32x16x128_f4 v[4:19], v[0:15], v[2:9], v[4:19] neg_hi:[0,1,0]
// GFX1250-ERR-NEXT: {{^}}                                                          ^
