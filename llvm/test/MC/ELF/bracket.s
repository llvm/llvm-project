// RUN: not llvm-mc -triple i386-unknown-unknown %s 2> %t1 > %t2
// RUN: FileCheck < %t1 %s

// CHECK: error: expected ']' in brackets expression
.size	x, [.-x)

// CHECK: error: expected ')' in parentheses expression
.size	y, (.-y]
