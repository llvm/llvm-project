; RUN: opt < %s -dfsan -verify -dfsan-args-abi -S | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; CHECK: @__dfsan_shadow_width_bits = weak_odr constant i32 [[#SBITS:]]
; CHECK: @__dfsan_shadow_width_bytes = weak_odr constant i32 [[#SBYTES:]]

; CHECK-LABEL: @unreachable_bb1.dfsan
define i8 @unreachable_bb1() {
  ; CHECK: ret { i8, i[[#SBITS]] } { i8 1, i[[#SBITS]] 0 }
  ; CHECK-NOT: bb2:
  ; CHECK-NOT: bb3:
  ; CHECK-NOT: bb4:
  ret i8 1

bb2:
  ret i8 2

bb3:
  br label %bb4

bb4:
  br label %bb3
}

declare void @abort() noreturn

; CHECK-LABEL: @unreachable_bb2.dfsan
define i8 @unreachable_bb2() {
  call void @abort() noreturn
  ; CHECK-NOT: i8 12
  ; CHECK: unreachable
  ret i8 12
}
