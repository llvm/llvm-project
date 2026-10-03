// RUN: llvm-mc -triple aarch64_lfi -mattr=+no-lfi-loads %s | FileCheck %s

// .arch and .cpu replace the subtarget feature set. They must preserve the
// +no-lfi-loads assembler policy supplied through -mattr.

ldr x0, [x1]
// CHECK: ldr x0, [x1]

.arch armv8-a

ldr x2, [x3]
// CHECK: ldr x2, [x3]

.cpu cortex-a53

ldr x4, [x5]
// CHECK: ldr x4, [x5]
