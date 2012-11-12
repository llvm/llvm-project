// RUN: not llvm-mc -triple x86_64-unknown-unknown %s 2> %t
// RUN: FileCheck < %t %s

.macro foo bar
        .long \bar
.endm

foo 42,  42

// CHECK: Too many arguments
