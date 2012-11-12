; RUN: llc < %s -march=thumb -mattr=+thumb2 | FileCheck %s 

define i32 @t2ADDrc_255(i32 %lhs) {
; CHECK: t2ADDrc_255:
; CHECK-NOT: bx lr
; CHECK: add{{.*}} #255
; CHECK: bx lr

    %Rd = add i32 %lhs, 255
    ret i32 %Rd
}

define i32 @t2ADDrc_256(i32 %lhs) {
; CHECK: t2ADDrc_256:
; CHECK-NOT: bx lr
; CHECK: add{{.*}} #256
; CHECK: bx lr

    %Rd = add i32 %lhs, 256
    ret i32 %Rd
}

define i32 @t2ADDrc_257(i32 %lhs) {
; CHECK: t2ADDrc_257:
; CHECK-NOT: bx lr
; CHECK: add{{.*}} #257
; CHECK: bx lr

    %Rd = add i32 %lhs, 257
    ret i32 %Rd
}

define i32 @t2ADDrc_4094(i32 %lhs) {
; CHECK: t2ADDrc_4094:
; CHECK-NOT: bx lr
; CHECK: add{{.*}} #4094
; CHECK: bx lr

    %Rd = add i32 %lhs, 4094
    ret i32 %Rd
}

define i32 @t2ADDrc_4095(i32 %lhs) {
; CHECK: t2ADDrc_4095:
; CHECK-NOT: bx lr
; CHECK: add{{.*}} #4095
; CHECK: bx lr

    %Rd = add i32 %lhs, 4095
    ret i32 %Rd
}

define i32 @t2ADDrc_4096(i32 %lhs) {
; CHECK: t2ADDrc_4096:
; CHECK-NOT: bx lr
; CHECK: add{{.*}} #4096
; CHECK: bx lr

    %Rd = add i32 %lhs, 4096
    ret i32 %Rd
}

define i32 @t2ADDrr(i32 %lhs, i32 %rhs) {
; CHECK: t2ADDrr:
; CHECK-NOT: bx lr
; CHECK: add
; CHECK: bx lr

    %Rd = add i32 %lhs, %rhs
    ret i32 %Rd
}

define i32 @t2ADDrs(i32 %lhs, i32 %rhs) {
; CHECK: t2ADDrs:
; CHECK-NOT: bx lr
; CHECK: add{{.*}} lsl #8
; CHECK: bx lr

    %tmp = shl i32 %rhs, 8
    %Rd = add i32 %lhs, %tmp
    ret i32 %Rd
}

