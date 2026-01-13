; RUN: opt -S -passes='separate-const-offset-from-gep' %s | FileCheck %s
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

define ptr @src(i32 %0) {
; CHECK-LABEL: @src(
; CHECK-NEXT: %base = alloca [4 x i32], align 16
; CHECK-NEXT: %2 = xor i64 0, 3
; CHECK-NEXT: %gep = getelementptr [4 x i32], ptr %base, i64 0, i64 %2
; CHECK-NEXT: ret ptr %gep

; CHECK-NOT: getelementptr i8, ptr %gep, i64 12
  %base = alloca [4 x i32], align 16
  %2 = xor i64 0, 3
  %gep = getelementptr [4 x i32], ptr %base, i64 0, i64 %2
  ret ptr %gep
}
