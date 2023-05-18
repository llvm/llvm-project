; RUN: llc -mtriple=thumb-eabi -mcpu=arm1156t2-s -mattr=+thumb2 %s -o - | FileCheck %s

define i64 @f1(i64 %a, i64 %b) {
; CHECK-LABEL: f1:
; CHECK: adds r0, r0, r2
; CHECK: adcs r1, r3
    %tmp = add i64 %a, %b
    ret i64 %tmp
}
