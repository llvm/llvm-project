// RUN: not llvm-mc %s 2> %t
// RUN: FileCheck < %t %s
// REQUIRES: default_triple

.macro M a a
.endm

// CHECK: macro 'M' has multiple parameters named 'a'
