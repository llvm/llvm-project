# RUN: llvm-mc -triple i386-unknown-unknown %s > %t
# RUN: FileCheck < %t %s

// CHECK: .globl $foo
.globl $foo
// CHECK: .long ($foo)
.long ($foo)
