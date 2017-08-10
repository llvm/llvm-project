// RUN: not llvm-mc -arch=amdgcn -mcpu=tonga -show-encoding %s 2>&1 | FileCheck %s --check-prefix=NOCIVI --check-prefix=NOVI
// RUN: not llvm-mc -arch=amdgcn -show-encoding %s 2>&1 | FileCheck %s --check-prefix=NOSI --check-prefix=NOSICI
// RUN: not llvm-mc -arch=amdgcn -mcpu=tahiti -show-encoding %s 2>&1 | FileCheck %s --check-prefix=NOSI --check-prefix=NOSICI
// RUN: not llvm-mc -arch=amdgcn -mcpu=bonaire -show-encoding %s 2>&1 | FileCheck %s --check-prefix=NOSICI

// NOSICI: error:
// NOVI: error: failed parsing operand
v_mov_b32 v0, v0 row_bcast:0

// NOSICI: error:
// NOVI: error: failed parsing operand
v_mov_b32 v0, v0 row_bcast:13
