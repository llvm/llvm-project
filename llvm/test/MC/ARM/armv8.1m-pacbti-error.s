// RUN: not llvm-mc -triple thumbv8.1m.main-arm-none-eabi -mattr=+pacbti %s -show-encoding -o - 2>&1 | FileCheck %s

// CHECK: error: invalid operand for instruction
pac r0, lr, sp
// CHECK: error: invalid operand for instruction
pac r12, r1, sp
// CHECK: error: operand must be a register sp
pac r12, lr, r2

// CHECK: error: invalid operand for instruction
aut r0, lr, sp
// CHECK: error: invalid operand for instruction
aut r12, r1, sp
// CHECK: error: operand must be a register sp
aut r12, lr, r2

// CHECK: operand must be a register in range [r0, r12] or LR or PC
autg sp, r1, r2
// CHECK: operand must be a register in range [r0, r14]
autg r0, pc, r2
// CHECK: operand must be a register in range [r0, r14]
autg r0, r1, pc

// CHECK: operand must be a register in range [r0, r12] or r14
pacg sp, r1, r2
// CHECK: operand must be a register in range [r0, r12] or r14
pacg pc, r1, r2
// CHECK: operand must be a register in range [r0, r14]
pacg r0, pc, r2
// CHECK: operand must be a register in range [r0, r14]
pacg r0, r1, pc

// CHECK: operand must be a register in range [r0, r12] or LR or PC
bxaut sp, r1, r2
// CHECK: operand must be a register in range [r0, r12] or r14
bxaut r0, sp, r2
// CHECK: operand must be a register in range [r0, r12] or r14
bxaut r0, pc, r2
// CHECK: operand must be a register in range [r0, r14]
bxaut r0, r1, pc
