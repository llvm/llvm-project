// RUN: llvm-mc -triple i386-unknown-unknown %s | FileCheck %s

.irp reg,%eax,%ebx
        pushl \reg
.endr

// CHECK: pushl %eax
// CHECK: pushl %ebx

.irp reg,%eax,%ebx
.irp imm,4,3,5
        addl \reg, \imm
.endr
.endr

// CHECK: addl %eax, 4
// CHECK: addl %eax, 3
// CHECK: addl %eax, 5
// CHECK: addl %ebx, 4
// CHECK: addl %ebx, 3
// CHECK: addl %ebx, 5
