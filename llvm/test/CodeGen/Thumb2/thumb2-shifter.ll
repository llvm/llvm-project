; RUN: llc < %s -march=thumb -mattr=+thumb2,+t2xtpk | FileCheck %s

define i32 @t2ADDrs_lsl(i32 %X, i32 %Y) {
; CHECK: t2ADDrs_lsl
; CHECK: add.w  r0, r0, r1, lsl #16
        %A = shl i32 %Y, 16
        %B = add i32 %X, %A
        ret i32 %B
}

define i32 @t2ADDrs_lsr(i32 %X, i32 %Y) {
; CHECK: t2ADDrs_lsr
; CHECK: add.w  r0, r0, r1, lsr #16
        %A = lshr i32 %Y, 16
        %B = add i32 %X, %A
        ret i32 %B
}

define i32 @t2ADDrs_asr(i32 %X, i32 %Y) {
; CHECK: t2ADDrs_asr
; CHECK: add.w  r0, r0, r1, asr #16
        %A = ashr i32 %Y, 16
        %B = add i32 %X, %A
        ret i32 %B
}

; i32 ror(n) = (x >> n) | (x << (32 - n))
define i32 @t2ADDrs_ror(i32 %X, i32 %Y) {
; CHECK: t2ADDrs_ror
; CHECK: add.w  r0, r0, r1, ror #16
        %A = lshr i32 %Y, 16
        %B = shl  i32 %Y, 16
        %C = or   i32 %B, %A
        %R = add  i32 %X, %C
        ret i32 %R
}

define i32 @t2ADDrs_noRegShift(i32 %X, i32 %Y, i8 %sh) {
; CHECK: t2ADDrs_noRegShift
; CHECK: uxtb r2, r2
; CHECK: lsls r1, r2
; CHECK: add  r0, r1
        %shift.upgrd.1 = zext i8 %sh to i32
        %A = shl i32 %Y, %shift.upgrd.1
        %B = add i32 %X, %A
        ret i32 %B
}

