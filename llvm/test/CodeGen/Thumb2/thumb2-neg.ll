; RUN: llc -mtriple=thumb-eabi -mcpu=arm1156t2-s -mattr=+thumb2 %s -o - | FileCheck %s

define i32 @f1(i32 %a) {
; CHECK-LABEL: f1:
; CHECK: rsbs r0, r0, #0
    %tmp = sub i32 0, %a
    ret i32 %tmp
}
