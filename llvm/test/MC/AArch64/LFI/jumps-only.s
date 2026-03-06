// RUN: llvm-mc -triple aarch64_lfi -mattr=+no-lfi-loads,+no-lfi-stores %s | FileCheck %s

// Jumps-only mode: only branches are sandboxed.

ldr x0, [x1]
// CHECK: ldr x0, [x1]

ldr x0, [x1, #8]
// CHECK: ldr x0, [x1, #8]

str x0, [x1]
// CHECK: str x0, [x1]

stp x0, x1, [x2, #16]
// CHECK: stp x0, x1, [x2, #16]

add sp, sp, #8
// CHECK: add sp, sp, #8

sub sp, sp, #8
// CHECK: sub sp, sp, #8

mov sp, x0
// CHECK: mov sp, x0

br x0
// CHECK:      add x28, x27, w0, uxtw
// CHECK-NEXT: br x28

ret
// CHECK: ret

blr x1
// CHECK:      add x28, x27, w1, uxtw
// CHECK-NEXT: blr x28

bl some_func
// CHECK: bl some_func

b some_func
// CHECK: b some_func
