// RUN: not llvm-mc -triple x86_64-unknown-unknown -mattr=+acev1 %s -o /dev/null 2>&1 | FileCheck %s

// Test missing explicit BSR operand
// CHECK: error: BSR instruction requires explicit %bsr0 operand
bsrmovf %zmm1, %zmm2

// CHECK: error:
tilemovcol %tmm1, %xmm2, $5

// CHECK: error:
top4busd %tmm1, %ymm2, %zmm3

// CHECK: error:
tilemovrow $256, %zmm2, %tmm1

// Test feature gating - ACEV1 instructions require ACEV1 feature
// RUN: not llvm-mc -triple x86_64-unknown-unknown -mattr=-acev1 %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=NOACEV1

// NOACEV1: error:
bsrinit %bsr0

// NOACEV1: error:
top2bf16ps %zmm3, %zmm2, %tmm1

// Test that ACEV1 requires AMX-TILE (dependency check)
// RUN: not llvm-mc -triple x86_64-unknown-unknown -mattr=+acev1,-amx-tile %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=NOAMX

// NOAMX: error:
tilemovcol $5, %zmm2, %tmm1

// Test invalid BSR register (only bsr0 exists)
// CHECK: error:
bsrmovf %zmm1, %zmm2, %bsr1

// Test 32-bit mode rejection (ACEV1 is 64-bit only)
// RUN: not llvm-mc -triple i386-unknown-unknown -mattr=+acev1 %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=NO32BIT

// NO32BIT: error:
bsrinit %bsr0
