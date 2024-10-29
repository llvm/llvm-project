// RUN: llvm-mc -triple aarch64-- -o - %s | FileCheck %s

// CHECK: .section sec00
// CHECK-NEXT: .p2align 2
// CHECK-NEXT: nop
.section sec00, "ax"
nop
nop
// CHECK: .section sec01
// CHECK-NEXT: .p2align 2
// CHECK-NEXT: .p2align 2
// CHECK-NEXT: nop
.section sec01, "ax"
.balign 4
nop
// CHECK: .section sec02
// CHECK-NEXT: .p2align 2
// CHECK-NEXT: .byte 1
.section sec02, "ax"
// CHECK-NEXT: nop
.byte 1
nop
