// RUN: llvm-mc -triple x86_64-unknown-unknown %s | FileCheck %s

.irpc foo,"123"
        .long \foo
.endr
.irpc foo,ab
        .long 0x\foo
.endr
.irpc foo,""
.endr

// CHECK: long 1
// CHECK: long 2
// CHECK: long 3
// CHECK: long 10
// CHECK: long 11
// CHECK-NOT: long

.irpc foo,123
.irpc bar,45
        addl %eax, \foo\bar
.endr
.endr

// CHECK: addl %eax, 14
// CHECK: addl %eax, 15
// CHECK: addl %eax, 24
// CHECK: addl %eax, 25
// CHECK: addl %eax, 34
// CHECK: addl %eax, 35
