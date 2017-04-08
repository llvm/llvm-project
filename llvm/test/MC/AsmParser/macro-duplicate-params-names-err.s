// RUN: not llvm-mc %s 2> %t
// RUN: FileCheck < %t %s

.macro M a a
.endm

// CHECK: macro 'M' has multiple parameters named 'a'
