// RUN: llvm-mc -triple aarch64-none-linux-gnu %s -filetype=obj -o %t
// RUN: llvm-objdump -d %t | FileCheck %s

0:
.skip 4
1:
mov x0, 1b - 0b
// CHECK: mov x0, #4
mov x0, 0b - 1b
// CHECK: mov x0, #-4
