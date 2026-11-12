// RUN: llvm-mc -triple=arm64-eabi < %s | FileCheck %s
_foo:
        OBJECT .req x2
        mov x4, OBJECT
        mov x4, oBjEcT
        .unreq oBJECT

_foo2:
        OBJECT .req w5
        mov w4, OBJECT
        .unreq OBJECT

// CHECK-LABEL: _foo:
// CHECK: mov x4, x2
// CHECK: mov x4, x2

// CHECK-LABEL: _foo2:
// CHECK: mov w4, w5
