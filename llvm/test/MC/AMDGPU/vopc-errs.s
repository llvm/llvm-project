// RUN: not llvm-mc -arch=amdgcn %s 2>&1 | FileCheck --implicit-check-not=error: %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=tahiti %s 2>&1 | FileCheck --implicit-check-not=error: %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=tonga %s 2>&1 | FileCheck --implicit-check-not=error: %s

// Force 32-bit encoding with non-vcc result

v_cmp_lt_f32_e32 s[0:1], v2, v4
// CHECK: 18: error: invalid operand for instruction
