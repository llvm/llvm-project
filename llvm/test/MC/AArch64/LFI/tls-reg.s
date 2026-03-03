// RUN: llvm-mc -triple aarch64_lfi %s | FileCheck %s

mrs x0, tpidr_el0
// CHECK: ldr x0, [x25, #32]

mrs x1, tpidr_el0
// CHECK: ldr x1, [x25, #32]

msr tpidr_el0, x0
// CHECK: str x0, [x25, #32]

msr tpidr_el0, x1
// CHECK: str x1, [x25, #32]
