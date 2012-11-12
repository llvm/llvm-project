// RUN: not llvm-mc -triple i386-apple-darwin10 %s 2> %t.err | FileCheck %s

.macro test1
.globl "$0 $1 $2 $$3 $n"
.endmacro

// CHECK: .globl "1 23  $3 2"
test1 1, 2 3

