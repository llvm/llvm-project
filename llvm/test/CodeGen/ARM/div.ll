; RUN: llc < %s -mtriple=arm-apple-ios -mcpu=cortex-a8    | \
; RUN:     FileCheck %s -check-prefix=CHECK -check-prefix=CHECK-SWDIV
; RUN: llc < %s -mtriple=arm-apple-ios -mcpu=swift        | \
; RUN:     FileCheck %s -check-prefix=CHECK -check-prefix=CHECK-HWDIV
; RUN: llc < %s -mtriple=arm-apple-ios -mcpu=cortex-r4    | \
; RUN:     FileCheck %s -check-prefix=CHECK -check-prefix=CHECK-SWDIV
; RUN: llc < %s -mtriple=arm-apple-ios -mcpu=cortex-r4f   | \
; RUN:     FileCheck %s -check-prefix=CHECK -check-prefix=CHECK-SWDIV
; RUN: llc < %s -mtriple=arm-apple-ios -mcpu=cortex-r5    | \
; RUN:     FileCheck %s -check-prefix=CHECK -check-prefix=CHECK-HWDIV
; RUN: llc < %s -mtriple=arm-none-eabi -mcpu=cortex-a8    | \
; RUN:     FileCheck %s -check-prefix=CHECK -check-prefix=CHECK-EABI
; RUN: llc < %s -mtriple=armv7ve-none-linux-gnu           | \
; RUN:     FileCheck %s -check-prefix=CHECK -check-prefix=CHECK-HWDIV
; RUN: llc < %s -mtriple=thumbv7ve-none-linux-gnu         | \
; RUN:     FileCheck %s -check-prefix=CHECK -check-prefix=CHECK-HWDIV \
; RUN:                  -check-prefix=CHECK-THUMB

define i32 @f1(i32 %a, i32 %b) {
entry:
; CHECK-LABEL: f1
; CHECK-SWDIV: __divsi3

; CHECK-THUMB: .thumb_func
; CHECK-HWDIV: sdiv

; CHECK-EABI: __aeabi_idiv
        %tmp1 = sdiv i32 %a, %b         ; <i32> [#uses=1]
        ret i32 %tmp1
}

define i32 @f2(i32 %a, i32 %b) {
entry:
; CHECK-LABEL: f2
; CHECK-SWDIV: __udivsi3

; CHECK-THUMB: .thumb_func
; CHECK-HWDIV: udiv

; CHECK-EABI: __aeabi_uidiv
        %tmp1 = udiv i32 %a, %b         ; <i32> [#uses=1]
        ret i32 %tmp1
}

define i32 @f3(i32 %a, i32 %b) {
entry:
; CHECK-LABEL: f3
; CHECK-SWDIV: __modsi3

; CHECK-THUMB: .thumb_func
; CHECK-HWDIV: sdiv
; CHECK-HWDIV: mls

; EABI MODE = Remainder in R1, quotient in R0
; CHECK-EABI: __aeabi_idivmod
; CHECK-EABI-NEXT: mov r0, r1
        %tmp1 = srem i32 %a, %b         ; <i32> [#uses=1]
        ret i32 %tmp1
}

define i32 @f4(i32 %a, i32 %b) {
entry:
; CHECK-LABEL: f4
; CHECK-SWDIV: __umodsi3

; CHECK-THUMB: .thumb_func
; CHECK-HWDIV: udiv
; CHECK-HWDIV: mls

; EABI MODE = Remainder in R1, quotient in R0
; CHECK-EABI: __aeabi_uidivmod
; CHECK-EABI-NEXT: mov r0, r1
        %tmp1 = urem i32 %a, %b         ; <i32> [#uses=1]
        ret i32 %tmp1
}


define i64 @f5(i64 %a, i64 %b) {
entry:
; CHECK-LABEL: f5
; CHECK-SWDIV: __moddi3

; CHECK-HWDIV: __moddi3

; EABI MODE = Remainder in R2-R3, quotient in R0-R1
; CHECK-EABI: __aeabi_ldivmod
; CHECK-EABI-NEXT: mov r0, r2
; CHECK-EABI-NEXT: mov r1, r3
        %tmp1 = srem i64 %a, %b         ; <i64> [#uses=1]
        ret i64 %tmp1
}

define i64 @f6(i64 %a, i64 %b) {
entry:
; CHECK-LABEL: f6
; CHECK-SWDIV: __umoddi3

; CHECK-HWDIV: __umoddi3

; EABI MODE = Remainder in R2-R3, quotient in R0-R1
; CHECK-EABI: __aeabi_uldivmod
; CHECK-EABI-NEXT: mov r0, r2
; CHECK-EABI-NEXT: mov r1, r3
        %tmp1 = urem i64 %a, %b         ; <i64> [#uses=1]
        ret i64 %tmp1
}
