; RUN: llc < %s -mtriple=arm-none-eabi -mcpu=cortex-m23 -march=thumb | \
; RUN:     FileCheck %s -check-prefix=CHECK

define i32 @f1(i32 %a, i32 %b) {
entry:
; CHECK-LABEL: f1

; CHECK: sdiv
        %tmp1 = sdiv i32 %a, %b         ; <i32> [#uses=1]
        ret i32 %tmp1
}

define i32 @f2(i32 %a, i32 %b) {
entry:
; CHECK-LABEL: f2
; CHECK: udiv
        %tmp1 = udiv i32 %a, %b         ; <i32> [#uses=1]
        ret i32 %tmp1
}

define i32 @f3(i32 %a, i32 %b) {
entry:
; CHECK-LABEL: f3


        %tmp1 = srem i32 %a, %b         ; <i32> [#uses=1]
        ret i32 %tmp1
; CHECK: sdiv
; CHECK-NEXT: muls
; CHECK-NEXT: subs
}

define i32 @f4(i32 %a, i32 %b) {
entry:
; CHECK-LABEL: f4

; CHECK: udiv
; CHECK-NEXT: muls
; CHECK-NEXT: subs
        %tmp1 = urem i32 %a, %b         ; <i32> [#uses=1]
        ret i32 %tmp1
}


define i64 @f5(i64 %a, i64 %b) {
entry:
; CHECK-LABEL: f5

; EABI MODE = Remainder in R2-R3, quotient in R0-R1
; CHECK: __aeabi_ldivmod
; CHECK-NEXT: mov r0, r2
; CHECK-NEXT: mov r1, r3
        %tmp1 = srem i64 %a, %b         ; <i64> [#uses=1]
        ret i64 %tmp1
}

define i64 @f6(i64 %a, i64 %b) {
entry:
; CHECK-LABEL: f6

; EABI MODE = Remainder in R2-R3, quotient in R0-R1
; CHECK: __aeabi_uldivmod
; CHECK: mov r0, r2
; CHECK: mov r1, r3
        %tmp1 = urem i64 %a, %b         ; <i64> [#uses=1]
        ret i64 %tmp1
}
