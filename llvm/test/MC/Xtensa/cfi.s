# RUN: llvm-mc -triple xtensa -mattr=+fp -o - %s | FileCheck %s

#CHECK:      .cfi_startproc
#CHECK-NEXT: .cfi_def_cfa_offset 16
#CHECK-NEXT: .cfi_offset a0, -4
#CHECK-NEXT: .cfi_offset a1, -8
#CHECK-NEXT: .cfi_offset a15, -12
#CHECK-NEXT: .cfi_offset f0, -16
#CHECK-NEXT: .cfi_def_cfa_register a15
#CHECK-NEXT: .cfi_endproc
  .cfi_startproc
  .cfi_def_cfa_offset 16
  .cfi_offset a0, -4
  .cfi_offset sp, -8
  .cfi_offset a15, -12
  .cfi_offset f0, -16
  .cfi_def_cfa_register a15
  .cfi_endproc
