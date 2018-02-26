// RUN: llvm-mc -triple aarch64-- -o - %s | FileCheck %s

fun:
  .cfi_startproc
// CHECK: .cfi_startproc
  stp  x29, x30, [sp, #-16]!
.Lcfi0:
  .cfi_offset w29, -16
// CHECK: .cfi_offset w29, -16
.Lcfi1:
  .cfi_offset w30, -8
// CHECK: .cfi_offset w30, -8
  mov   x29, sp
.Lcfi2:
  .cfi_def_cfa w29, 16
// CHECK: .cfi_def_cfa w29, 16
.Lcfi3:
  .cfi_restore w30
// CHECK: .cfi_restore w30
  ldr  x29, [sp], #16
.Lcfi4:
  .cfi_restore w29
// CHECK: .cfi_restore w29
  ret
  .cfi_endproc
// CHECK: .cfi_endproc
