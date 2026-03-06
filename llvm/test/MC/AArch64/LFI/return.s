// RUN: llvm-mc -triple aarch64_lfi %s | FileCheck %s

// LR guard is deferred until the next control flow instruction for PAC
// compatibility.

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

// Deferred guard flushed before a label.
mov x30, x0
next_func:
nop
// CHECK:      mov x30, x0
// CHECK-NEXT: add x30, x27, w30, uxtw
// CHECK:      nop

// AUTIASP strips PAC, so a deferred guard is set.
autiasp
ret
// CHECK:      autiasp
// CHECK-NEXT: add x30, x27, w30, uxtw
// CHECK-NEXT: ret

// PACIASP just signs LR and doesn't need a guard.
paciasp
nop
// CHECK:      paciasp
// CHECK-NEXT: nop

// Deferred guard flushed before bl.
mov x30, x0
bl some_func
// CHECK:      mov x30, x0
// CHECK-NEXT: add x30, x27, w30, uxtw
// CHECK-NEXT: bl some_func

// Deferred guard flushed before blr.
mov x30, x0
blr x1
// CHECK:      mov x30, x0
// CHECK-NEXT: add x30, x27, w30, uxtw
// CHECK-NEXT: add x28, x27, w1, uxtw
// CHECK-NEXT: blr x28

// Deferred guard flushed before branch.
mov x30, x0
b some_func
// CHECK:      mov x30, x0
// CHECK-NEXT: add x30, x27, w30, uxtw
// CHECK-NEXT: b some_func

// Deferred guard flushed at end of stream.
mov x30, x0
// CHECK:      mov x30, x0
// CHECK-NEXT: add x30, x27, w30, uxtw
