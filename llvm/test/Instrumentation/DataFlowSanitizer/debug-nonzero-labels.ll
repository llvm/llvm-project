; RUN: opt < %s -dfsan -dfsan-args-abi -dfsan-debug-nonzero-labels -S | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; CHECK: @__dfsan_shadow_width_bits = weak_odr constant i32 [[#SBITS:]]
; CHECK: @__dfsan_shadow_width_bytes = weak_odr constant i32 [[#SBYTES:]]

declare i32 @g()

; CHECK: define { i32, i[[#SBITS]] } @f.dfsan(i32 %0, i32 %1, i[[#SBITS]] %2, i[[#SBITS]] %3)
define i32 @f(i32, i32) {
  ; CHECK: [[LOCALLABELALLOCA:%.*]] = alloca i[[#SBITS]]
  %i = alloca i32
  ; CHECK: [[ARGCMP1:%.*]] = icmp ne i[[#SBITS]] %3, 0
  ; CHECK: br i1 [[ARGCMP1]]
  ; CHECK: [[ARGCMP2:%.*]] = icmp ne i[[#SBITS]] %2, 0
  ; CHECK: br i1 [[ARGCMP2]]
  %x = add i32 %0, %1
  store i32 %x, i32* %i
  ; CHECK: [[CALL:%.*]] = call { i32, i[[#SBITS]] } @g.dfsan()
  ; CHECK: [[CALLLABEL:%.*]] = extractvalue { i32, i[[#SBITS]] } [[CALL]], 1
  ; CHECK: [[CALLCMP:%.*]] = icmp ne i[[#SBITS]] [[CALLLABEL]], 0
  ; CHECK: br i1 [[CALLCMP]]
  %call = call i32 @g()
  ; CHECK: [[LOCALLABEL:%.*]] = load i[[#SBITS]], i[[#SBITS]]* [[LOCALLABELALLOCA]]
  ; CHECK: [[LOCALCMP:%.*]] = icmp ne i[[#SBITS]] [[LOCALLABEL]], 0
  ; CHECK: br i1 [[LOCALCMP]]
  %load = load i32, i32* %i
  ret i32 %load
}
