; RUN: llc -mtriple=thumb-apple-darwin -mcpu=arm1156t2-s -mattr=+thumb2 %s -o - \
; RUN:    | FileCheck %s -check-prefix=CHECK-THUMB
; RUN: llc -mtriple=thumb-apple-darwin -mcpu=cortex-m3 -mattr=+thumb2 %s -o - \
; RUN:    | FileCheck %s -check-prefix=CHECK-THUMBV7M
; RUN: llc -mtriple=thumb-apple-darwin -mcpu=swift %s -o - \
; RUN:    | FileCheck %s -check-prefix=CHECK-HWDIV
; RUN: llc -mtriple=thumb-apple-darwin -mcpu=cortex-r4 %s -o - \
; RUN:    | FileCheck %s -check-prefix=CHECK-HWDIV
; RUN: llc -mtriple=thumb-apple-darwin -mcpu=cortex-r4f %s -o - \
; RUN:    | FileCheck %s -check-prefix=CHECK-HWDIV
; RUN: llc -mtriple=thumb-apple-darwin -mcpu=cortex-r5 %s -o - \
; RUN:    | FileCheck %s -check-prefix=CHECK-HWDIV

define i32 @f1(i32 %a, i32 %b) {
entry:
; CHECK-THUMB: f1
; CHECK-THUMB: __divsi3
; CHECK-THUMBV7M: f1
; CHECK-THUMBV7M: sdiv
; CHECK-HWDIV: f1
; CHECK-HWDIV: sdiv
        %tmp1 = sdiv i32 %a, %b         ; <i32> [#uses=1]
        ret i32 %tmp1
}

define i32 @f2(i32 %a, i32 %b) {
entry:
; CHECK-THUMB: f2
; CHECK-THUMB: __udivsi3
; CHECK-THUMBV7M: f2
; CHECK-THUMBV7M: udiv
; CHECK-HWDIV: f2
; CHECK-HWDIV: udiv
        %tmp1 = udiv i32 %a, %b         ; <i32> [#uses=1]
        ret i32 %tmp1
}

define i32 @f3(i32 %a, i32 %b) {
entry:
; CHECK-THUMB: f3
; CHECK-THUMB: __modsi3
; CHECK-THUMBV7M: f3
; CHECK-THUMBV7M: sdiv
; CHECK-HWDIV: f3
; CHECK-HWDIV: sdiv
        %tmp1 = srem i32 %a, %b         ; <i32> [#uses=1]
        ret i32 %tmp1
}

define i32 @f4(i32 %a, i32 %b) {
entry:
; CHECK-THUMB: f4
; CHECK-THUMB: __umodsi3
; CHECK-THUMBV7M: f4
; CHECK-THUMBV7M: udiv
; CHECK-HWDIV: f4
; CHECK-HWDIV: udiv
        %tmp1 = urem i32 %a, %b         ; <i32> [#uses=1]
        ret i32 %tmp1
}

