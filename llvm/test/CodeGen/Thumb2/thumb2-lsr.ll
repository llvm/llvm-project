; RUN: llc -mtriple=thumb-eabi -mcpu=arm1156t2-s -mattr=+thumb2 %s -o - | FileCheck %s

define i32 @f1(i32 %a) {
; CHECK-LABEL: f1:
; CHECK: lsrs r0, r0, #13
    %tmp = lshr i32 %a, 13
    ret i32 %tmp
}
