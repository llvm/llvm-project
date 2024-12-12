// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx1251 %s 2>&1 | FileCheck --check-prefix=GFX1251-ERR --implicit-check-not=error: %s

v_wmma_f64_16x16x4_f64 v[8:23], v[0:3], v[4:7], s[8:23]
// GFX1251-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_wmma_f64_16x16x4_f64 v[8:23], v[0:3], v[4:7], 3.0
// GFX1251-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_wmma_f64_16x16x8_f64 v[16:31], v[0:7], v[8:15], s[16:31]
// GFX1251-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_wmma_f64_16x16x8_f64 v[16:31], v[0:7], v[8:15], 3.0
// GFX1251-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_wmma_f32_16x16x32_f16 v[16:23], v[0:7], v[8:15], s[16:23]
// GFX1251-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_wmma_f32_16x16x32_f16 v[16:23], v[0:7], v[8:15], 3.0
// GFX1251-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_wmma_f16_16x16x32_f16 v[16:19], v[0:7], v[8:15], s[16:19]
// GFX1251-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_wmma_f16_16x16x32_f16 v[16:19], v[0:7], v[8:15], 3.0
// GFX1251-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_swmmac_f32_16x16x64_f16 v[24:31], v[0:7], v[8:23], s32
// GFX1251-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_swmmac_f32_16x16x64_f16 v[24:31], v[0:7], 1.0, v32
// GFX1251-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_swmmac_f32_16x16x64_f16 v[24:31], v[0:7], v[8:23], v32 index_key:2
// GFX1251-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: out of range index_key

v_swmmac_f32_16x16x64_f16 v[24:31], v[0:7], v[8:23], v32 neg_lo:[0,0,1]
// GFX1251-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid neg_lo operand

v_swmmac_f16_16x16x64_f16 v[24:27], v[0:7], v[8:23], s28
// GFX1251-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_swmmac_f16_16x16x64_f16 v[24:27], v[0:7], 1.0, v28
// GFX1251-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_swmmac_f16_16x16x64_f16 v[24:27], v[0:7], v[8:23], v28 index_key:2
// GFX1251-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: out of range index_key

v_swmmac_f16_16x16x64_f16 v[24:27], v[0:7], v[8:23], v28 neg_lo:[0,0,1]
// GFX1251-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid neg_lo operand
