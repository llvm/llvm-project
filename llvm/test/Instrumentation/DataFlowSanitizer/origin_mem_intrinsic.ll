; RUN: opt < %s -dfsan -dfsan-track-origins=1 -dfsan-fast-16-labels=true -S | FileCheck %s --check-prefix=CHECK
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare void @llvm.memset.p0i8.i64(i8* nocapture, i8, i64, i1)

define void @memset(i8* %p, i8 %v) {
  ; CHECK: @"dfs$memset"
  ; CHECK: [[O:%.*]] = load i32, i32* getelementptr inbounds ([200 x i32], [200 x i32]* @__dfsan_arg_origin_tls, i64 0, i64 1), align 4
  ; CHECK: [[S:%.*]] = load i16, i16* inttoptr (i64 add (i64 ptrtoint ([100 x i64]* @__dfsan_arg_tls to i64), i64 2) to i16*), align 2
  ; CHECK: call void @__dfsan_set_label(i16 [[S]], i32 [[O]], i8* %p, i64 1)
  call void @llvm.memset.p0i8.i64(i8* %p, i8 %v, i64 1, i1 1)
  ret void
}