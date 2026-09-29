; RUN: llc -mtriple=thumb-eabi -mcpu=arm1156t2-s -mattr=+thumb2 %s -o - | FileCheck %s

define i32 @f1(i32 %a, i32 %b, i32 %c) {
    %tmp1 = mul i32 %a, %b
    %tmp2 = sub i32 %c, %tmp1
    ret i32 %tmp2
}
; CHECK-LABEL: f1:
; CHECK: 	mls	r0, r0, r1, r2

; sub doesn't commute, so no mls for this one
define i32 @f2(i32 %a, i32 %b, i32 %c) {
    %tmp1 = mul i32 %a, %b
    %tmp2 = sub i32 %tmp1, %c
    ret i32 %tmp2
}
; CHECK-LABEL: f2:
; CHECK: 	muls	r0, r1, r0

