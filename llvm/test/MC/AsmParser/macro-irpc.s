// RUN: llvm-mc -triple x86_64-unknown-unknown %s | FileCheck %s

.irpc foo,123
        .long \foo
.endr

// CHECK: long 1
// CHECK: long 2
// CHECK: long 3
