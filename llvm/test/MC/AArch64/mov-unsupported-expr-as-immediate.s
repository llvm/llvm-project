// RUN: not llvm-mc -triple aarch64-none-linux-gnu %s -filetype=obj -o /dev/null 2>&1 | FileCheck %s

0:
.skip 0x10000
1:
mov x0, 1b - 0b
// CHECK: error: fixup value out of range
// CHECK: mov x0, 1b - 0b
// CHECK: ^
mov x0, 0b - 1b
// CHECK: error: fixup value out of range
// CHECK: mov x0, 0b - 1b
// CHECK: ^
mov x0, 1b
// CHECK: error: invalid fixup for movz/movk instruction
// CHECK: mov x0, 1b
// CHECK: ^
