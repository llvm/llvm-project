// RUN: not llvm-mc -triple x86_64-unknown-unknown %s 2> %t
// RUN: FileCheck < %t %s

.rept 3
.long

// CHECK: no matching '.endr' in definition
