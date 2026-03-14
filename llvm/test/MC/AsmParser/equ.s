// RUN: not llvm-mc -n -triple i386-unknown-unknown %s 2> %t
// RUN: FileCheck < %t %s

.equ	a, 0
.set	a, 1
.equ	a, 2
// CHECK: :[[#@LINE+1]]:11: error: redefinition of 'a'
.equiv	a, 3
// CHECK: :[[#@LINE+1]]:12: error: missing expression
.set  b, ""
