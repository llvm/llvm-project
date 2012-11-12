; RUN: llc < %s -mtriple=thumb-apple-darwin -mattr=+thumb2 \
; RUN:    | FileCheck %s -check-prefix=CHECK-THUMB
; RUN: llc < %s -march=thumb -mcpu=cortex-m3 -mattr=+thumb2 \
; RUN:    | FileCheck %s -check-prefix=CHECK-THUMBV7M
; RUN: llc < %s -march=thumb -mcpu=swift \
; RUN:    | FileCheck %s -check-prefix=CHECK-SWIFT-T2

define i32 @f1(i32 %a, i32 %b) {
entry:
; CHECK-THUMB: f1
; CHECK-THUMB: __divsi3
; CHECK-THUMBV7M: f1
; CHECK-THUMBV7M: sdiv
; CHECK-SWIFT-T2: f1
; CHECK-SWIFT-T2: sdiv
        %tmp1 = sdiv i32 %a, %b         ; <i32> [#uses=1]
        ret i32 %tmp1
}

define i32 @f2(i32 %a, i32 %b) {
entry:
; CHECK-THUMB: f2
; CHECK-THUMB: __udivsi3
; CHECK-THUMBV7M: f2
; CHECK-THUMBV7M: udiv
; CHECK-SWIFT-T2: f2
; CHECK-SWIFT-T2: udiv
        %tmp1 = udiv i32 %a, %b         ; <i32> [#uses=1]
        ret i32 %tmp1
}

define i32 @f3(i32 %a, i32 %b) {
entry:
; CHECK-THUMB: f3
; CHECK-THUMB: __modsi3
; CHECK-THUMBV7M: f3
; CHECK-THUMBV7M: sdiv
; CHECK-SWIFT-T2: f3
; CHECK-SWIFT-T2: sdiv
        %tmp1 = srem i32 %a, %b         ; <i32> [#uses=1]
        ret i32 %tmp1
}

define i32 @f4(i32 %a, i32 %b) {
entry:
; CHECK-THUMB: f4
; CHECK-THUMB: __umodsi3
; CHECK-THUMBV7M: f4
; CHECK-THUMBV7M: udiv
; CHECK-SWIFT-T2: f4
; CHECK-SWIFT-T2: udiv
        %tmp1 = urem i32 %a, %b         ; <i32> [#uses=1]
        ret i32 %tmp1
}

