// RUN: llvm-mc -triple aarch64_lfi --aarch64-lfi-guard-elim=false %s | FileCheck %s

// Memory accesses that define LR (x30) must sandbox the base register
// in addition to masking LR after the access.

ldr x30, [x0, #0x100]
// CHECK:      add x28, x27, w0, uxtw
// CHECK-NEXT: ldr x30, [x28, #256]
// CHECK-NEXT: add x30, x27, w30, uxtw

ldr x30, [x0]
// CHECK:      ldr x30, [x27, w0, uxtw]
// CHECK-NEXT: add x30, x27, w30, uxtw

ldr x30, [x0, #8]!
// CHECK:      add x0, x0, #8
// CHECK-NEXT: ldr x30, [x27, w0, uxtw]
// CHECK-NEXT: add x30, x27, w30, uxtw

ldr x30, [x0, #-8]!
// CHECK:      sub x0, x0, #8
// CHECK-NEXT: ldr x30, [x27, w0, uxtw]
// CHECK-NEXT: add x30, x27, w30, uxtw

ldr x30, [x0], #8
// CHECK:      ldr x30, [x27, w0, uxtw]
// CHECK-NEXT: add x0, x0, #8
// CHECK-NEXT: add x30, x27, w30, uxtw

ldr x30, [x0, x1]
// CHECK:      add x26, x0, x1
// CHECK-NEXT: ldr x30, [x27, w26, uxtw]
// CHECK-NEXT: add x30, x27, w30, uxtw

ldr x30, [x0, x1, lsl #3]
// CHECK:      add x26, x0, x1, lsl #3
// CHECK-NEXT: ldr x30, [x27, w26, uxtw]
// CHECK-NEXT: add x30, x27, w30, uxtw

ldur x30, [x0, #4]
// CHECK:      add x28, x27, w0, uxtw
// CHECK-NEXT: ldur x30, [x28, #4]
// CHECK-NEXT: add x30, x27, w30, uxtw

ldp x29, x30, [x0]
// CHECK:      add x28, x27, w0, uxtw
// CHECK-NEXT: ldp x29, x30, [x28]
// CHECK-NEXT: add x30, x27, w30, uxtw

ldp x29, x30, [x0, #16]
// CHECK:      add x28, x27, w0, uxtw
// CHECK-NEXT: ldp x29, x30, [x28, #16]
// CHECK-NEXT: add x30, x27, w30, uxtw

ldp x29, x30, [x0, #16]!
// CHECK:      add x28, x27, w0, uxtw
// CHECK-NEXT: ldp x29, x30, [x28, #16]
// CHECK-NEXT: add x0, x0, #16
// CHECK-NEXT: add x30, x27, w30, uxtw

ldp x29, x30, [x0], #16
// CHECK:      add x28, x27, w0, uxtw
// CHECK-NEXT: ldp x29, x30, [x28]
// CHECK-NEXT: add x0, x0, #16
// CHECK-NEXT: add x30, x27, w30, uxtw

// SP-based LR loads are safe without base sandboxing.

ldr x30, [sp, #16]
// CHECK:      ldr x30, [sp, #16]
// CHECK-NEXT: add x30, x27, w30, uxtw
