// RUN: llvm-mc -triple aarch64_lfi %s | FileCheck %s

mov x30, x0
ret
// CHECK:      mov x30, x0
// CHECK-NEXT: add x30, x27, w30, uxtw
// CHECK-NEXT: ret

ldr x30, [sp]
ret
// CHECK:      ldr x30, [sp]
// CHECK-NEXT: add x30, x27, w30, uxtw
// CHECK-NEXT: ret

ldp x29, x30, [sp]
ret
// CHECK:      ldp x29, x30, [sp]
// CHECK-NEXT: add x30, x27, w30, uxtw
// CHECK-NEXT: ret

ldp x30, x29, [sp]
ret
// CHECK:      ldp x30, x29, [sp]
// CHECK-NEXT: add x30, x27, w30, uxtw
// CHECK-NEXT: ret
