// RUN: llvm-mc -triple aarch64_lfi %s | FileCheck %s

.arch_extension pauth

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

mov x30, x0
next_func:
nop
// CHECK:      mov x30, x0
// CHECK-NEXT: add x30, x27, w30, uxtw
// CHECK:      nop

autiasp
ret
// CHECK:      autiasp
// CHECK-NEXT: add x30, x27, w30, uxtw
// CHECK-NEXT: ret

paciasp
nop
// CHECK:      paciasp
// CHECK-NEXT: nop

mov x30, x0
bl some_func
// CHECK:      mov x30, x0
// CHECK-NEXT: add x30, x27, w30, uxtw
// CHECK-NEXT: bl some_func

mov x30, x0
blr x1
// CHECK:      mov x30, x0
// CHECK-NEXT: add x30, x27, w30, uxtw
// CHECK-NEXT: add x28, x27, w1, uxtw
// CHECK-NEXT: blr x28

mov x30, x0
b some_func
// CHECK:      mov x30, x0
// CHECK-NEXT: add x30, x27, w30, uxtw
// CHECK-NEXT: b some_func

mov x30, x0
// CHECK:      mov x30, x0
// CHECK-NEXT: add x30, x27, w30, uxtw
