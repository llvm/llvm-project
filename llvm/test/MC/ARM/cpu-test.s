// RUN: not llvm-mc -o - -triple arm-gnueabi-freebsd11.0 < %s > %t 2> %t2
// RUN: FileCheck %s < %t
// RUN: FileCheck %s --check-prefix=CHECK-ERROR < %t2

// CHECK: .cpu cortex-a8
.cpu cortex-a8
// CHECK: dsb     sy
dsb
.cpu arm9       
// CHECK-ERROR: error: instruction requires: data-barriers
dsb
// CHECK-ERROR: error: Unknown CPU name
.cpu foobar
// CHECK: .cpu cortex-m3
.cpu cortex-m3
// CHECK: sub sp, #16
sub sp,#16
