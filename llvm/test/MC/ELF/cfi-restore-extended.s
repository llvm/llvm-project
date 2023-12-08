// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o - | llvm-dwarfdump -debug-frame - | FileCheck %s

// Check that register numbers greater than 63 can be used in .cfi_restore directives
f:
  .cfi_startproc
  nop
// CHECK: DW_CFA_advance_loc: 1
  .cfi_restore %rbp
// CHECK-NEXT: DW_CFA_restore: RBP
  nop
// CHECK-NEXT: DW_CFA_advance_loc: 1
  .cfi_restore 89
// CHECK-NEXT: DW_CFA_restore_extended: reg89
// CHECK-NEXT: DW_CFA_nop:
  nop
  .cfi_endproc
