// Check that llvm-mc doesn't crash on division by zero.
// RUN: not llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s 2> %t
// RUN: FileCheck -input-file %t %s

// CHECK: expected relocatable expression
.int 1/0

// CHECK: expected relocatable expression
.int 2%0
