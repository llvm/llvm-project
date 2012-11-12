// RUN: llvm-mc -triple x86_64-unknown-unknown %s | FileCheck %s

.irp reg,%eax,%ebx
        pushl \reg
.endr

// CHECK: pushl %eax
// CHECK: pushl %ebx
