; RUN: opt < %s -dfsan -S | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; CHECK: @__dfsan_arg_tls = external thread_local(initialexec) global [64 x i16]
; CHECK: @__dfsan_retval_tls = external thread_local(initialexec) global i16

declare i32 @f(i32)
declare float @llvm.sqrt.f32(float)

; CHECK: @"dfs$call"
define i32 @call() {
  ; CHECK: store{{.*}}__dfsan_arg_tls
  ; CHECK: call{{.*}}@"dfs$f"
  ; CHECK: load{{.*}}__dfsan_retval_tls
  %r = call i32 @f(i32 0)

  ; CHECK-NOT: store{{.*}}__dfsan_arg_tls
  %i = call float @llvm.sqrt.f32(float -1.0)

  ; CHECK: store{{.*}}__dfsan_retval_tls
  ; CHECK: ret i32
  ret i32 %r
}
