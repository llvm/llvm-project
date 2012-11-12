; RUN: llc < %s -march=thumb -mattr=+thumb2 | FileCheck %s

define i32 @f1(i32 %a) {
; CHECK: f1:
; CHECK: mvns r0, r0
    %tmp = xor i32 4294967295, %a
    ret i32 %tmp
}

define i32 @f2(i32 %a) {
; CHECK: f2:
; CHECK: mvns r0, r0
    %tmp = xor i32 %a, 4294967295
    ret i32 %tmp
}

define i32 @f5(i32 %a) {
; CHECK: f5:
; CHECK: mvn.w r0, r0, lsl #5
    %tmp = shl i32 %a, 5
    %tmp1 = xor i32 %tmp, 4294967295
    ret i32 %tmp1
}

define i32 @f6(i32 %a) {
; CHECK: f6:
; CHECK: mvn.w r0, r0, lsr #6
    %tmp = lshr i32 %a, 6
    %tmp1 = xor i32 %tmp, 4294967295
    ret i32 %tmp1
}

define i32 @f7(i32 %a) {
; CHECK: f7:
; CHECK: mvn.w r0, r0, asr #7
    %tmp = ashr i32 %a, 7
    %tmp1 = xor i32 %tmp, 4294967295
    ret i32 %tmp1
}

define i32 @f8(i32 %a) {
; CHECK: f8:
; CHECK: mvn.w r0, r0, ror #8
    %l8 = shl i32 %a, 24
    %r8 = lshr i32 %a, 8
    %tmp = or i32 %l8, %r8
    %tmp1 = xor i32 %tmp, 4294967295
    ret i32 %tmp1
}
