; RUN: llc < %s -march=arm | FileCheck %s

define i64 @f1() {
; CHECK: f1:
entry:
        ret i64 0
}

define i64 @f2() {
; CHECK: f2:
entry:
        ret i64 1
}

define i64 @f3() {
; CHECK: f3:
; CHECK: mvn r0, #-2147483648
entry:
        ret i64 2147483647
}

define i64 @f4() {
; CHECK: f4:
; CHECK: mov r0, #-2147483648
entry:
        ret i64 2147483648
}

define i64 @f5() {
; CHECK: f5:
; CHECK: mvn r0, #0
; CHECK: mvn r1, #-2147483648
entry:
        ret i64 9223372036854775807
}

define i64 @f6(i64 %x, i64 %y) {
; CHECK: f6:
; CHECK: adds
; CHECK: adc
entry:
        %tmp1 = add i64 %y, 1           ; <i64> [#uses=1]
        ret i64 %tmp1
}

define void @f7() {
; CHECK: f7:
entry:
        %tmp = call i64 @f8( )          ; <i64> [#uses=0]
        ret void
}

declare i64 @f8()

define i64 @f9(i64 %a, i64 %b) {
; CHECK: f9:
; CHECK: subs r
; CHECK: sbc
entry:
        %tmp = sub i64 %a, %b           ; <i64> [#uses=1]
        ret i64 %tmp
}

define i64 @f(i32 %a, i32 %b) {
; CHECK: f:
; CHECK: smull
entry:
        %tmp = sext i32 %a to i64               ; <i64> [#uses=1]
        %tmp1 = sext i32 %b to i64              ; <i64> [#uses=1]
        %tmp2 = mul i64 %tmp1, %tmp             ; <i64> [#uses=1]
        ret i64 %tmp2
}

define i64 @g(i32 %a, i32 %b) {
; CHECK: g:
; CHECK: umull
entry:
        %tmp = zext i32 %a to i64               ; <i64> [#uses=1]
        %tmp1 = zext i32 %b to i64              ; <i64> [#uses=1]
        %tmp2 = mul i64 %tmp1, %tmp             ; <i64> [#uses=1]
        ret i64 %tmp2
}

define i64 @f10() {
; CHECK: f10:
entry:
        %a = alloca i64, align 8                ; <i64*> [#uses=1]
        %retval = load i64* %a          ; <i64> [#uses=1]
        ret i64 %retval
}
