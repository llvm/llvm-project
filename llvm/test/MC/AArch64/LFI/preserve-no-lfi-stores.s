// RUN: llvm-mc -triple aarch64_lfi -mattr=+no-lfi-stores %s | FileCheck %s

// .arch and .cpu replace the subtarget feature set. They must preserve the
// +no-lfi-stores assembler policy supplied through -mattr.

str x0, [x1]
// CHECK: str x0, [x1]

.arch armv8-a

str x2, [x3]
// CHECK: str x2, [x3]

.cpu cortex-a53

str x4, [x5]
// CHECK: str x4, [x5]
