; RUN: llc -mtriple=thumb-eabi -mcpu=arm1156t2-s -mattr=+thumb2 %s -o - | FileCheck %s

define i32 @f1(i32 %a) {
; CHECK-LABEL: f1:
; CHECK: asrs r0, r0, #17
    %tmp = ashr i32 %a, 17
    ret i32 %tmp
}
