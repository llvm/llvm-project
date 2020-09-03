// RUN: not llvm-mc -triple x86_64-unknown-unknown %s -o /dev/null 2>&1 | FileCheck %s

// CHECK: invalid operand for instruction
tileloadd (%rip), %tmm0

// CHECK: invalid operand for instruction
tileloaddt1 1(%rip), %tmm1

// CHECK: invalid operand for instruction
tilestored %tmm2, (%rip)
