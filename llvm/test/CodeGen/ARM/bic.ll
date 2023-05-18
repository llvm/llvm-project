; RUN: llc -mtriple=arm-eabi %s -o - | FileCheck %s

define i32 @f1(i32 %a, i32 %b) {
; CHECK-LABEL: f1:
; CHECK: bic	r0, r0, r1
    %tmp = xor i32 %b, 4294967295
    %tmp1 = and i32 %a, %tmp
    ret i32 %tmp1
}

define i32 @f2(i32 %a, i32 %b) {
; CHECK-LABEL: f2:
; CHECK: bic	r0, r0, r1
    %tmp = xor i32 %b, 4294967295
    %tmp1 = and i32 %tmp, %a
    ret i32 %tmp1
}

define i32 @f3(i32 %a) {
; CHECK-LABEL: f3:
; CHECK: bic r0, r0, #255
    %tmp = and i32 %a, -256
    ret i32 %tmp
}
