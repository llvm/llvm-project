// RUN: not llvm-mc -n -triple i386-unknown-unknown %s 2> %t
// RUN: FileCheck < %t %s

.globl a
// CHECK: error: expected section directive before assembly directive
.long 0
        
