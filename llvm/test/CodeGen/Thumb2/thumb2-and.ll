; RUN: llc < %s -march=thumb -mattr=+thumb2 | FileCheck %s

define i32 @f1(i32 %a, i32 %b) {
; CHECK: f1:
; CHECK: ands r0, r1
    %tmp = and i32 %a, %b
    ret i32 %tmp
}

define i32 @f2(i32 %a, i32 %b) {
; CHECK: f2:
; CHECK: and.w r0, r0, r1, lsl #5
    %tmp = shl i32 %b, 5
    %tmp1 = and i32 %a, %tmp
    ret i32 %tmp1
}

define i32 @f3(i32 %a, i32 %b) {
; CHECK: f3:
; CHECK: and.w r0, r0, r1, lsr #6
    %tmp = lshr i32 %b, 6
    %tmp1 = and i32 %a, %tmp
    ret i32 %tmp1
}

define i32 @f4(i32 %a, i32 %b) {
; CHECK: f4:
; CHECK: and.w r0, r0, r1, asr #7
    %tmp = ashr i32 %b, 7
    %tmp1 = and i32 %a, %tmp
    ret i32 %tmp1
}

define i32 @f5(i32 %a, i32 %b) {
; CHECK: f5:
; CHECK: and.w r0, r0, r0, ror #8
    %l8 = shl i32 %a, 24
    %r8 = lshr i32 %a, 8
    %tmp = or i32 %l8, %r8
    %tmp1 = and i32 %a, %tmp
    ret i32 %tmp1
}
