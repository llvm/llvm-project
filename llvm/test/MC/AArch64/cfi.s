// RUN: llvm-mc -triple aarch64-- -o - %s | FileCheck %s

fun:
// CHECK:      .cfi_startproc
// CHECK-NEXT: stp
  .cfi_startproc
  stp  x29, x30, [sp, #-16]!
// CHECK:      .cfi_offset w29, -16
// CHECK-NEXT: .cfi_offset w30, -8
  .cfi_offset w29, -16
  .cfi_offset w30, -8
  mov   x29, sp
// CHECK:      .cfi_def_cfa w29, 16
// CHECK-NEXT: .cfi_restore w30
// CHECK-NEXT: ldr
// CHECK-NEXT: .cfi_restore w29
  .cfi_def_cfa w29, 16
  .cfi_restore w30
  ldr  x29, [sp], #16
  .cfi_restore w29
  ret
  .cfi_endproc
// CHECK: .cfi_endproc
