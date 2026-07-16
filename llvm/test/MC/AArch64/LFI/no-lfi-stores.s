// RUN: llvm-mc -triple aarch64_lfi -mattr=+no-lfi-stores %s | FileCheck %s

// Loads-only mode: stores pass through, loads are sandboxed. This
// configuration is not very useful in practice.

ldr x0, [x1]
// CHECK: ldr x0, [x27, w1, uxtw]

ldr x0, [x1, #8]
// CHECK:      add x28, x27, w1, uxtw
// CHECK-NEXT: ldr x0, [x28, #8]

ldp x0, x1, [x2]
// CHECK:      add x28, x27, w2, uxtw
// CHECK-NEXT: ldp x0, x1, [x28]

str x0, [x1]
// CHECK: str x0, [x1]

stp x0, x1, [x2]
// CHECK: stp x0, x1, [x2]

ldr x0, [sp, #8]
// CHECK: ldr x0, [sp, #8]

str x0, [sp, #8]
// CHECK: str x0, [sp, #8]

add sp, sp, #8
// CHECK:      add x26, sp, #8
// CHECK-NEXT: add sp, x27, w26, uxtw

br x0
// CHECK:      add x28, x27, w0, uxtw
// CHECK-NEXT: br x28
