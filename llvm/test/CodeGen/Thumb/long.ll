; RUN: llc -mtriple=thumb-eabi %s -verify-machineinstrs -o - | FileCheck %s
; RUN: llc -mtriple=thumb-apple-darwin %s -verify-machineinstrs -o - | \
; RUN:    FileCheck %s -check-prefix CHECK -check-prefix CHECK-DARWIN

define i64 @f1() {
entry:
        ret i64 0
; CHECK-LABEL: f1:
; CHECK: movs r0, #0
; CHECK: movs r1, r0
}

define i64 @f2() {
entry:
        ret i64 1
; CHECK-LABEL: f2:
; CHECK: movs r0, #1
; CHECK: movs r1, #0
}

define i64 @f3() {
entry:
        ret i64 2147483647
; CHECK-LABEL: f3:
; CHECK: ldr r0,
; CHECK: movs r1, #0
}

define i64 @f4() {
entry:
        ret i64 2147483648
; CHECK-LABEL: f4:
; CHECK: movs r0, #1
; CHECK: lsls r0, r0, #31
; CHECK: movs r1, #0
}

define i64 @f5() {
entry:
        ret i64 9223372036854775807
; CHECK-LABEL: f5:
; CHECK: movs r0, #0
; CHECK: mvns r0, r0
; CHECK: ldr r1,
}

define i64 @f6(i64 %x, i64 %y) {
entry:
        %tmp1 = add i64 %y, 1           ; <i64> [#uses=1]
        ret i64 %tmp1
; CHECK-LABEL: f6:
; CHECK: movs r1, #0
; CHECK: adds r0, r2, #1
; CHECK: adcs r1, r3
}

define i64 @f6a(i64 %x, i64 %y) {
entry:
        %tmp1 = add i64 %y, 10
        ret i64 %tmp1
; CHECK-LABEL: f6a:
; CHECK: movs r1, #0
; CHECK: adds r2, #10
; CHECK: adcs r1, r3
; CHECK: movs r0, r2
}

define i64 @f6b(i64 %x, i64 %y) {
entry:
        %tmp1 = add i64 %y, 1000
        ret i64 %tmp1
; CHECK-LABEL: f6b:
; CHECK: movs r0, #125
; CHECK: lsls r0, r0, #3
; CHECK: movs r1, #0
; CHECK: adds r0, r2, r0
; CHECK: adcs r1, r3
}

define void @f7() {
entry:
        %tmp = call i64 @f8( )          ; <i64> [#uses=0]
        ret void
; CHECK-LABEL: f7:
; CHECK: bl
}

declare i64 @f8()

define i64 @f9(i64 %a, i64 %b) {
entry:
        %tmp = sub i64 %a, %b           ; <i64> [#uses=1]
        ret i64 %tmp
; CHECK-LABEL: f9:
; CHECK: subs r0, r0, r2
; CHECK: sbcs r1, r3
}

define i64 @f9a(i64 %x, i64 %y) { ; ADDC with small negative imm => SUBS imm
entry:
        %tmp1 = sub i64 %y, 10
        ret i64 %tmp1
; CHECK-LABEL: f9a:
; CHECK: movs r0, #0
; CHECK: subs r2, #10
; CHECK: sbcs r3, r0
; CHECK: movs r0, r2
; CHECK: movs r1, r3
}

define i64 @f9b(i64 %x, i64 %y) { ; ADDC with big negative imm => SUBS reg
entry:
        %tmp1 = sub i64 1000, %y
        ret i64 %tmp1
; CHECK-LABEL: f9b:
; CHECK: movs r0, #125
; CHECK: lsls r0, r0, #3
; CHECK: movs r1, #0
; CHECK: subs r0, r0, r2
; CHECK: sbcs r1, r3
}

define i64 @f9c(i64 %x, i32 %y) { ; SUBS with small positive imm => SUBS imm
entry:
        %conv = sext i32 %y to i64
        %shl = shl i64 %conv, 32
        %or = or i64 %shl, 1
        %sub = sub nsw i64 %x, %or
        ret i64 %sub
; CHECK-LABEL: f9c:
; CHECK: subs r0, r0, #1
; CHECK: sbcs r1, r2
}

define i64 @f9d(i64 %x, i32 %y) { ; SUBS with small negative imm => ADDS imm
entry:
        %conv = sext i32 %y to i64
        %shl = shl i64 %conv, 32
        %or = or i64 %shl, 4294967295
        %sub = sub nsw i64 %x, %or
        ret i64 %sub
; CHECK-LABEL: f9d:
; CHECK: adds r0, r0, #1
; CHECK: sbcs r1, r2
}

define i64 @f(i32 %a, i32 %b) {
entry:
        %tmp = sext i32 %a to i64               ; <i64> [#uses=1]
        %tmp1 = sext i32 %b to i64              ; <i64> [#uses=1]
        %tmp2 = mul i64 %tmp1, %tmp             ; <i64> [#uses=1]
        ret i64 %tmp2
; CHECK-LABEL: f:
; CHECK-V6: bl __aeabi_lmul
; CHECK-DARWIN: __muldi3
}

define i64 @g(i32 %a, i32 %b) {
entry:
        %tmp = zext i32 %a to i64               ; <i64> [#uses=1]
        %tmp1 = zext i32 %b to i64              ; <i64> [#uses=1]
        %tmp2 = mul i64 %tmp1, %tmp             ; <i64> [#uses=1]
        ret i64 %tmp2
; CHECK-LABEL: g:
; CHECK-V6: bl __aeabi_lmul
; CHECK-DARWIN: __muldi3
}

define i64 @f10() {
entry:
        %a = alloca i64, align 8                ; <i64*> [#uses=1]
        %retval = load i64, i64* %a          ; <i64> [#uses=1]
        ret i64 %retval
; CHECK-LABEL: f10:
; CHECK: sub sp, #8
; CHECK: ldr r0, [sp]
; CHECK: ldr r1, [sp, #4]
; CHECK: add sp, #8
}

define i64 @f11(i64 %x, i64 %y) {
entry:
        %tmp1 = add i64 -1000, %y
        %tmp2 = add i64 %tmp1, -1000
        ret i64 %tmp2
; CHECK-LABEL: f11:
; CHECK: movs r0, #125
; CHECK: lsls r0, r0, #3
; CHECK: movs r1, #0
; CHECK: subs r2, r2, r0
; CHECK: sbcs r3, r1
; CHECK: subs r0, r2, r0
; CHECK: sbcs r3, r1
; CHECK: movs r1, r3
}

