// RUN: llvm-mc -triple=aarch64-none-linux-gnu -filetype=obj -o %t.obj %s
// RUN: llvm-readobj -S --sd %t.obj | FileCheck %s

.section sec00, "ax"
.byte 1
.section sec01, "ax"
nop
nop
.section sec02, "ax"
.balign 4
nop
nop
.section sec03, "ax"
.byte 0
.section sec04, "aw"
nop
nop

// CHECK: Name: sec00
// CHECK: AddressAlignment: 4
// CHECK: Name: sec01
// CHECK: AddressAlignment: 4
// CHECK: Name: sec02
// CHECK: AddressAlignment: 4
// CHECK: Name: sec03
// CHECK: AddressAlignment: 4
// CHECK: Name: sec04
// CHECK: AddressAlignment: 1