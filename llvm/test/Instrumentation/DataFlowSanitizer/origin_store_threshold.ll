; RUN: opt < %s -passes=dfsan -dfsan-track-origins=1  -dfsan-instrument-with-call-threshold=0 -S | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @store_threshold(ptr %p, [2 x i64] %a) {
  ; CHECK: @store_threshold.dfsan
  ; CHECK: [[AO:%.*]] = load i32, ptr getelementptr inbounds ([200 x i32], ptr @__dfsan_arg_origin_tls, i64 0, i64 1), align 4
  ; CHECK: [[AS:%.*]] = load [2 x i8], ptr inttoptr (i64 add (i64 ptrtoint (ptr @__dfsan_arg_tls to i64), i64 2) to ptr), align 2
  ; CHECK: [[AS0:%.*]] = extractvalue [2 x i8] [[AS]], 0
  ; CHECK: [[AS1:%.*]] = extractvalue [2 x i8] [[AS]], 1
  ; CHECK: [[AS01:%.*]] = or i8 [[AS0]], [[AS1]]
  ; CHECK: call void @__dfsan_maybe_store_origin(i8 [[AS01]], ptr %p, i64 16, i32 [[AO]])
  ; CHECK: store [2 x i64] %a, ptr %p, align 8

  store [2 x i64] %a, ptr %p
  ret void
}
