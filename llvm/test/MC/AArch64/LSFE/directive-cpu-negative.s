// RUN: not llvm-mc -triple aarch64 -filetype asm -o - %s 2>&1 | FileCheck %s

.cpu generic+lsfe
.cpu generic+nolsfe
ldfadd h0, h1, [x2]
// CHECK: error: instruction requires: lsfe
// CHECK-NEXT: ldfadd h0, h1, [x2]