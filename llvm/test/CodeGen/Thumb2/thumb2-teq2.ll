; RUN: llc -mtriple=thumb-eabi -mcpu=arm1156t2-s -mattr=+thumb2 %s -o - | FileCheck %s

; These tests would be improved by 'movs r0, #0' being rematerialized below the
; tst as 'mov.w r0, #0'.

define i32 @f2(i32 %a, i32 %b) {
; CHECK: f2
; CHECK: eors {{.*}}, r1
    %tmp = xor i32 %a, %b
    %tmp1 = icmp eq i32 %tmp, 0
    %ret = select i1 %tmp1, i32 42, i32 24
    ret i32 %ret
}

define i32 @f4(i32 %a, i32 %b) {
; CHECK: f4
; CHECK: eors  {{.*}}, r1
    %tmp = xor i32 %a, %b
    %tmp1 = icmp eq i32 0, %tmp
    %ret = select i1 %tmp1, i32 42, i32 24
    ret i32 %ret
}

define i32 @f6(i32 %a, i32 %b) {
; CHECK: f6
; CHECK: teq.w  {{.*}}, r1, lsl #5
    %tmp = shl i32 %b, 5
    %tmp1 = xor i32 %a, %tmp
    %tmp2 = icmp eq i32 %tmp1, 0
    %ret = select i1 %tmp2, i32 42, i32 24
    ret i32 %ret
}

define i32 @f7(i32 %a, i32 %b) {
; CHECK: f7
; CHECK: teq.w  {{.*}}, r1, lsr #6
    %tmp = lshr i32 %b, 6
    %tmp1 = xor i32 %a, %tmp
    %tmp2 = icmp eq i32 %tmp1, 0
    %ret = select i1 %tmp2, i32 42, i32 24
    ret i32 %ret
}

define i32 @f8(i32 %a, i32 %b) {
; CHECK: f8
; CHECK: teq.w  {{.*}}, r1, asr #7
    %tmp = ashr i32 %b, 7
    %tmp1 = xor i32 %a, %tmp
    %tmp2 = icmp eq i32 %tmp1, 0
    %ret = select i1 %tmp2, i32 42, i32 24
    ret i32 %ret
}

define i32 @f9(i32 %a, i32 %b) {
; CHECK: f9
; CHECK: teq.w  {{.*}}, {{.*}}, ror #8
    %l8 = shl i32 %a, 24
    %r8 = lshr i32 %a, 8
    %tmp = or i32 %l8, %r8
    %tmp1 = xor i32 %a, %tmp
    %tmp2 = icmp eq i32 %tmp1, 0
    %ret = select i1 %tmp2, i32 42, i32 24
    ret i32 %ret
}
