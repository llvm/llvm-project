; RUN: opt < %s -dfsan -dfsan-args-abi -S | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare void @llvm.memset.p0i8.i64(i8* nocapture, i8, i64, i1)

define void @ms(i8* %p, i8 %v) {
  ; CHECK-LABEL: @"dfs$ms"(i8* %0, i8 %1, i16 %2, i16 %3)
  ; CHECK: call void @__dfsan_set_label(i16 %3, i32 0, i8* %0, i64 1)
  call void @llvm.memset.p0i8.i64(i8* %p, i8 %v, i64 1, i1 1)
  ret void
}
