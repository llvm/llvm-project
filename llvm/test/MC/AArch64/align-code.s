// RUN: llvm-mc -triple=aarch64 -filetype=obj -o %t.o %s
// RUN: llvm-readobj -S --sd %t.o | FileCheck %s
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
.balign 8
nop
nop
.section sec04, "ax"
.byte 0
.section sec05, "aw"
nop
nop
// CHECK: Name: sec00
// CHECK: AddressAlignment: 4
// CHECK: Name: sec01
// CHECK: AddressAlignment: 4
// CHECK: Name: sec02
// CHECK: AddressAlignment: 4
// CHECK: Name: sec03
// CHECK: AddressAlignment: 8
// CHECK: Name: sec04
// CHECK: AddressAlignment: 4
// CHECK: Name: sec05
// CHECK: AddressAlignment: 1