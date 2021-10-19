// RUN: not llvm-mc -arch=amdgcn -mcpu=tonga %s 2>&1 | FileCheck %s --check-prefix=NOVI --implicit-check-not=error:
// RUN: not llvm-mc -arch=amdgcn %s 2>&1 | FileCheck %s --check-prefix=NOSICI --implicit-check-not=error:
// RUN: not llvm-mc -arch=amdgcn -mcpu=tahiti %s 2>&1 | FileCheck %s --check-prefix=NOSICI --implicit-check-not=error:
// RUN: not llvm-mc -arch=amdgcn -mcpu=bonaire %s 2>&1 | FileCheck %s --check-prefix=NOSICI --implicit-check-not=error:

// NOSICI: error: not a valid operand.
// NOVI: error: invalid row_bcast value
v_mov_b32 v0, v0 row_bcast:0

// NOSICI: error: not a valid operand.
// NOVI: error: invalid row_bcast value
v_mov_b32 v0, v0 row_bcast:13
