; RUN: opt < %s -dfsan -dfsan-track-origins=1 -dfsan-fast-16-labels=true -dfsan-instrument-with-call-threshold=0 -S | FileCheck %s --check-prefix=CHECK
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; CHECK: @__dfsan_shadow_width_bits = weak_odr constant i32 [[#SBITS:]]
; CHECK: @__dfsan_shadow_width_bytes = weak_odr constant i32 [[#SBYTES:]]

define void @store_threshold([2 x i64]* %p, [2 x i64] %a) {
  ; CHECK: @"dfs$store_threshold"
  ; CHECK: [[AO:%.*]] = load i32, i32* getelementptr inbounds ([200 x i32], [200 x i32]* @__dfsan_arg_origin_tls, i64 0, i64 1), align 4
  ; CHECK: [[AS:%.*]] = load [2 x i[[#SBITS]]], [2 x i[[#SBITS]]]* inttoptr (i64 add (i64 ptrtoint ([100 x i64]* @__dfsan_arg_tls to i64), i64 2) to [2 x i[[#SBITS]]]*), align 2
  ; CHECK: [[AS0:%.*]] = extractvalue [2 x i[[#SBITS]]] [[AS]], 0
  ; CHECK: [[AS1:%.*]] = extractvalue [2 x i[[#SBITS]]] [[AS]], 1
  ; CHECK: [[AS01:%.*]] = or i[[#SBITS]] [[AS0]], [[AS1]]
  ; CHECK: [[ADDR:%.*]] = bitcast [2 x i64]* %p to i8*
  ; CHECK: call void @__dfsan_maybe_store_origin(i[[#SBITS]] [[AS01]], i8* [[ADDR]], i64 16, i32 [[AO]])
  ; CHECK: store [2 x i64] %a, [2 x i64]* %p, align 8

  store [2 x i64] %a, [2 x i64]* %p
  ret void
}
