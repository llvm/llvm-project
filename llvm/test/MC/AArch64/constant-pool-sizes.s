// RUN: llvm-mc -triple aarch64-none-linux-gnu %s | FileCheck %s

  ldr w0, =symbol
  ldr x1, =symbol

  ldr w2, =1234567890
  ldr x3, =1234567890

// CHECK:             ldr     w0, .Ltmp0
// CHECK:             ldr     x1, .Ltmp1
// CHECK:             ldr     w2, .Ltmp2
// CHECK:             ldr     x3, .Ltmp3

// CHECK:             .p2align        2, 0x0
// CHECK-NEXT:.Ltmp0:
// CHECK-NEXT:        .word   symbol
// CHECK:             .p2align        3, 0x0
// CHECK-NEXT:.Ltmp1:
// CHECK-NEXT:        .xword  symbol
// CHECK:             .p2align        2, 0x0
// CHECK-NEXT:.Ltmp2:
// CHECK-NEXT:        .word   1234567890
// CHECK:             .p2align        3, 0x0
// CHECK-NEXT:.Ltmp3:
// CHECK-NEXT:        .xword  1234567890
