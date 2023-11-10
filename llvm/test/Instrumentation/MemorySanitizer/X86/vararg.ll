; RUN: opt < %s -msan-check-access-address=0 -S -passes=msan 2>&1
; RUN: opt < %s -msan-check-access-address=0 -S -passes=msan -msan-kernel=1 2>&1
; Test that code using va_start can be compiled on i386.

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:32:32-n8:16:32-S128"
target triple = "i386-unknown-linux-gnu"

define void @VaStart(ptr %s, ...) {
entry:
  %vl = alloca ptr, align 4
  call void @llvm.va_start(ptr %vl)
  ret void
}

declare void @llvm.va_start(ptr)
