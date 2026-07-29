// RUN: not llvm-mc -triple=amdgpu8.02 %s 2>&1 | FileCheck %s --check-prefix=NOVI --implicit-check-not=error:
// RUN: not llvm-mc -triple=amdgcn %s 2>&1 | FileCheck %s --check-prefix=NOSICI --implicit-check-not=error:
// RUN: not llvm-mc -triple=amdgpu6.00 %s 2>&1 | FileCheck %s --check-prefix=NOSICI --implicit-check-not=error:
// RUN: not llvm-mc -triple=amdgpu7.04 %s 2>&1 | FileCheck %s --check-prefix=NOSICI --implicit-check-not=error:

// NOSICI: :[[@LINE+2]]:{{[0-9]+}}: error: not a valid operand.
// NOVI: :[[@LINE+1]]:{{[0-9]+}}: error: invalid row_bcast value
v_mov_b32 v0, v0 row_bcast:0

// NOSICI: :[[@LINE+2]]:{{[0-9]+}}: error: not a valid operand.
// NOVI: :[[@LINE+1]]:{{[0-9]+}}: error: invalid row_bcast value
v_mov_b32 v0, v0 row_bcast:13
