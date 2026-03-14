; RUN: llc -mtriple=thumb-eabi -mcpu=arm1156t2-s -mattr=+thumb2 %s -o - | FileCheck %s

define i32 @f1(i32 %a, i32 %b) {
; CHECK-LABEL: f1:
; CHECK: eors r0, r1
    %tmp = xor i32 %a, %b
    ret i32 %tmp
}

define i32 @f2(i32 %a, i32 %b) {
; CHECK-LABEL: f2:
; CHECK: eors r0, r1
    %tmp = xor i32 %b, %a
    ret i32 %tmp
}

define i32 @f2b(i32 %a, i32 %b, i32 %c) {
; CHECK-LABEL: f2b:
; CHECK: eor.w r0, r1, r2
    %tmp = xor i32 %b, %c
    ret i32 %tmp
}

define i32 @f3(i32 %a, i32 %b) {
; CHECK-LABEL: f3:
; CHECK: eor.w r0, r0, r1, lsl #5
    %tmp = shl i32 %b, 5
    %tmp1 = xor i32 %a, %tmp
    ret i32 %tmp1
}

define i32 @f4(i32 %a, i32 %b) {
; CHECK-LABEL: f4:
; CHECK: eor.w r0, r0, r1, lsr #6
    %tmp = lshr i32 %b, 6
    %tmp1 = xor i32 %tmp, %a
    ret i32 %tmp1
}

define i32 @f5(i32 %a, i32 %b) {
; CHECK-LABEL: f5:
; CHECK: eor.w r0, r0, r1, asr #7
    %tmp = ashr i32 %b, 7
    %tmp1 = xor i32 %a, %tmp
    ret i32 %tmp1
}

define i32 @f6(i32 %a, i32 %b) {
; CHECK-LABEL: f6:
; CHECK: eor.w r0, r0, r0, ror #8
    %l8 = shl i32 %a, 24
    %r8 = lshr i32 %a, 8
    %tmp = or i32 %l8, %r8
    %tmp1 = xor i32 %tmp, %a
    ret i32 %tmp1
}
