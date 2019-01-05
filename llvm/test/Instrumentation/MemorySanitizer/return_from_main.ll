; RUN: opt < %s -msan-check-access-address=0 -S -passes=msan 2>&1 | FileCheck  \
; RUN: %s
; RUN: opt < %s -msan -msan-check-access-address=0 -S | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define i32 @main() sanitize_memory {
entry:
  %call = tail call i32 @f()
  ret i32 %call
}

declare i32 @f() sanitize_memory

; CHECK-LABEL: @main
; CHECK: call i32 @f()
; CHECK: store i32 0, {{.*}} @__msan_retval_tls
; CHECK: br i1
; CHECK: call void @__msan_warning_noreturn()
; CHECK: ret i32
