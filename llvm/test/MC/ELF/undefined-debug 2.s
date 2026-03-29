// RUN: not llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o %t 2>&1 | FileCheck %s
// CHECK: error: Undefined section reference: .debug_pubnames

.section .foo,"",@progbits
  .long  .debug_pubnames
