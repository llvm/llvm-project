// RUN: llvm-mc -triple aarch64_lfi %s | FileCheck %s

ldr x0, [sp, #16]!
// CHECK: ldr x0, [sp, #16]!

ldr x0, [sp], #16
// CHECK: ldr x0, [sp], #16

str x0, [sp, #16]!
// CHECK: str x0, [sp, #16]!

str x0, [sp], #16
// CHECK: str x0, [sp], #16

mov sp, x0
// CHECK:      add x26, x0, #0
// CHECK-NEXT: add sp, x27, w26, uxtw

add sp, sp, #8
// CHECK:      add x26, sp, #8
// CHECK-NEXT: add sp, x27, w26, uxtw

sub sp, sp, #8
// CHECK:      sub x26, sp, #8
// CHECK-NEXT: add sp, x27, w26, uxtw

add sp, sp, x0
// CHECK:      add x26, sp, x0
// CHECK-NEXT: add sp, x27, w26, uxtw

sub sp, sp, x0
// CHECK:      sub x26, sp, x0
// CHECK-NEXT: add sp, x27, w26, uxtw

sub sp, sp, #1, lsl #12
// CHECK:      sub x26, sp, #1, lsl #12
// CHECK-NEXT: add sp, x27, w26, uxtw
