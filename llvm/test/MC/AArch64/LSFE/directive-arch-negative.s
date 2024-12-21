// RUN: not llvm-mc -triple aarch64 -filetype asm -o - %s 2>&1 | FileCheck %s

.arch armv9.6-a+lsfe
.arch armv9.6-a+nolsfe
ldfadd h0, h1, [x2]
// CHECK: error: instruction requires: lsfe
// CHECK: ldfadd h0, h1, [x2]