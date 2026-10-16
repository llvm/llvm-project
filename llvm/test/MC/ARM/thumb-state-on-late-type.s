// RUN: llvm-mc --triple=thumbv7-none-eabi -filetype=obj %s -o - | llvm-readelf -s - | FileCheck %s

// A function's ARM or Thumb state is determined by the state in which its
// label is defined, not the state in which a later .type directive is emitted.

        .arm
arm_func:
        nop

        .thumb
        .type arm_func, %function
thumb_func:
        nop

        .arm
        .type thumb_func, %function

// CHECK: 00000000 0 FUNC LOCAL DEFAULT {{[0-9]+}} arm_func
// CHECK: 00000005 0 FUNC LOCAL DEFAULT {{[0-9]+}} thumb_func
