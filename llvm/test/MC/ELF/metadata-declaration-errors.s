// RUN: not llvm-mc -triple x86_64-pc-linux-gnu %s \
// RUN:   -filetype=obj -o %t.o 2>&1 | FileCheck %s

// Check we do not silently ignore invalid metadata symbol (123).
// CHECK: error: invalid metadata symbol

.section .foo,"a"
.quad 0

.section bar,"ao",@progbits,123
