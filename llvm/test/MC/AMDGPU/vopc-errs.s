// RUN: not llvm-mc -arch=amdgcn -show-encoding %s 2>&1 | FileCheck %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=tahiti -show-encoding %s 2>&1 | FileCheck %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=tonga -show-encoding %s 2>&1 | FileCheck %s

// Force 32-bit encoding with non-vcc result

v_cmp_lt_f32_e32 s[0:1], v2, v4
// CHECK: 18: error: invalid operand for instruction
