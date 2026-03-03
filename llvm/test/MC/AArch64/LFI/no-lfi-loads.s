// RUN: llvm-mc -triple aarch64_lfi -mattr=+no-lfi-loads %s | FileCheck %s

// Stores-only mode: loads pass through, stores are sandboxed.

ldr x0, [x1]
// CHECK: ldr x0, [x1]

ldr x0, [x1, #8]
// CHECK: ldr x0, [x1, #8]

ldp x0, x1, [x2]
// CHECK: ldp x0, x1, [x2]

str x0, [x1]
// CHECK: str x0, [x27, w1, uxtw]

stp x0, x1, [x2]
// CHECK:      add x28, x27, w2, uxtw
// CHECK-NEXT: stp x0, x1, [x28]

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
