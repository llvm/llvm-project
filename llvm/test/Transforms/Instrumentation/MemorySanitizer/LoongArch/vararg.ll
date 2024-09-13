; RUN: opt < %s -msan-check-access-address=0 -S -passes=msan 2>&1
; Test that code using va_start can be compiled on LoongArch.

target datalayout = "e-m:e-p:64:64-i64:64-i128:128-n64-S128"
target triple = "loongarch64-unknown-linux-gnu"

define void @VaStart(ptr %s, ...) {
entry:
  %vl = alloca ptr, align 4
  call void @llvm.va_start(ptr %vl)
  ret void
}

declare void @llvm.va_start(ptr)
