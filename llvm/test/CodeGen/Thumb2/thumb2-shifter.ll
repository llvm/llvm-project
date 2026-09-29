; RUN: llc -mtriple=thumb-eabi -mcpu=cortex-a8 %s -o - | FileCheck %s --check-prefix=A8
; RUN: llc -mtriple=thumb-eabi -mcpu=swift %s -o - | FileCheck %s --check-prefix=SWIFT

; rdar://12892707

define i32 @t2ADDrs_lsl(i32 %X, i32 %Y) {
; A8: t2ADDrs_lsl
; A8: add.w  r0, r0, r1, lsl #16
        %A = shl i32 %Y, 16
        %B = add i32 %X, %A
        ret i32 %B
}

define i32 @t2ADDrs_lsr(i32 %X, i32 %Y) {
; A8: t2ADDrs_lsr
; A8: add.w  r0, r0, r1, lsr #16
        %A = lshr i32 %Y, 16
        %B = add i32 %X, %A
        ret i32 %B
}

define i32 @t2ADDrs_asr(i32 %X, i32 %Y) {
; A8: t2ADDrs_asr
; A8: add.w  r0, r0, r1, asr #16
        %A = ashr i32 %Y, 16
        %B = add i32 %X, %A
        ret i32 %B
}

; i32 ror(n) = (x >> n) | (x << (32 - n))
define i32 @t2ADDrs_ror(i32 %X, i32 %Y) {
; A8: t2ADDrs_ror
; A8: add.w  r0, r0, r1, ror #16
        %A = lshr i32 %Y, 16
        %B = shl  i32 %Y, 16
        %C = or   i32 %B, %A
        %R = add  i32 %X, %C
        ret i32 %R
}

define i32 @t2ADDrs_noRegShift(i32 %X, i32 %Y, i8 %sh) {
; A8: t2ADDrs_noRegShift
; A8: uxtb r2, r2
; A8: lsls r1, r2
; A8: add  r0, r1

; SWIFT: t2ADDrs_noRegShift
; SWIFT-NOT: lsls
; SWIFT: lsl.w
        %shift.upgrd.1 = zext i8 %sh to i32
        %A = shl i32 %Y, %shift.upgrd.1
        %B = add i32 %X, %A
        ret i32 %B
}

define i32 @t2ADDrs_noRegShift2(i32 %X, i32 %Y, i8 %sh) {
; A8: t2ADDrs_noRegShift2
; A8: uxtb r2, r2
; A8: lsrs r1, r2
; A8: add  r0, r1

; SWIFT: t2ADDrs_noRegShift2
; SWIFT-NOT: lsrs
; SWIFT: lsr.w
        %shift.upgrd.1 = zext i8 %sh to i32
        %A = lshr i32 %Y, %shift.upgrd.1
        %B = add i32 %X, %A
        ret i32 %B
}

define i32 @t2ADDrs_noRegShift3(i32 %X, i32 %Y, i8 %sh) {
; A8: t2ADDrs_noRegShift3
; A8: uxtb r2, r2
; A8: asrs r1, r2
; A8: add  r0, r1

; SWIFT: t2ADDrs_noRegShift3
; SWIFT-NOT: asrs
; SWIFT: asr.w
        %shift.upgrd.1 = zext i8 %sh to i32
        %A = ashr i32 %Y, %shift.upgrd.1
        %B = add i32 %X, %A
        ret i32 %B
}

define i32 @t2ADDrs_optsize(i32 %X, i32 %Y, i8 %sh) optsize {
; SWIFT: t2ADDrs_optsize
; SWIFT-NOT: lsl.w
; SWIFT: lsls
        %shift.upgrd.1 = zext i8 %sh to i32
        %A = shl i32 %Y, %shift.upgrd.1
        %B = add i32 %X, %A
        ret i32 %B
}

define i32 @t2ADDrs_minsize(i32 %X, i32 %Y, i8 %sh) minsize {
; SWIFT: t2ADDrs_minsize
; SWIFT-NOT: lsr.w
; SWIFT: lsrs
        %shift.upgrd.1 = zext i8 %sh to i32
        %A = lshr i32 %Y, %shift.upgrd.1
        %B = add i32 %X, %A
        ret i32 %B
}
