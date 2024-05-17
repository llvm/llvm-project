// RUN: llvm-mc -triple x86_64-unknown-unknown %s | FileCheck %s

.rept 2
    .long 1
.endr
# 3 "a.s"
/// Test line marker after .endr \n.

.rept 3
.rept 2
    .long 0
.endr
.endr

// CHECK: .long	1
// CHECK: .long	1

// CHECK: .long	0
// CHECK: .long	0
// CHECK: .long	0

// CHECK: .long	0
// CHECK: .long	0
// CHECK: .long	0
