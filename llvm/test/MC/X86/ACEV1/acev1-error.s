// RUN: not llvm-mc -triple x86_64-unknown-unknown -mattr=+acev1 %s -o /dev/null 2>&1 | FileCheck %s

// Test missing explicit BSR operand
// CHECK: [[#@LINE+1]]:1: error: too few operands for instruction
bsrmovf %zmm1, %zmm2

// CHECK: [[#@LINE+1]]:12: error: invalid operand for instruction
tilemovcol %tmm1, %xmm2, $5

// CHECK: [[#@LINE+1]]:10: error: invalid operand for instruction
top4busd %tmm1, %ymm2, %zmm3

// CHECK: [[#@LINE+1]]:12: error: invalid operand for instruction
tilemovrow $256, %zmm2, %tmm1

// Test invalid BSR register (only bsr0 exists)
// CHECK: [[#@LINE+1]]:23: error: invalid register name
bsrmovf %zmm1, %zmm2, %bsr1

// Test 32-bit mode rejection (ACEV1 is 64-bit only)
// RUN: not llvm-mc -triple i386-unknown-unknown -mattr=+acev1 %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=NO32BIT

// NO32BIT: [[#@LINE+1]]:1: error: instruction requires: 64-bit mode
bsrinit %bsr0
