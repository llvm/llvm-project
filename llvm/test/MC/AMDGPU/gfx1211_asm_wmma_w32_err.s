// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx1211 %s 2>&1 | FileCheck --check-prefix=GFX1211-ERR --implicit-check-not=error: %s

v_wmma_f64_16x16x4_f64 v[8:23], v[0:3], v[4:7], s[8:23]
// GFX1211-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_wmma_f64_16x16x4_f64 v[8:23], v[0:3], v[4:7], 3.0
// GFX1211-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_wmma_f64_16x16x8_f64 v[16:31], v[0:7], v[8:15], s[16:31]
// GFX1211-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_wmma_f64_16x16x8_f64 v[16:31], v[0:7], v[8:15], 3.0
// GFX1211-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
