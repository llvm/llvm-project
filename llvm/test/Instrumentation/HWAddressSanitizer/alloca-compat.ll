; Test that alloca instrumentation with old API levels does not use short granules.
;
; RUN: opt < %s -passes=hwasan -S | FileCheck %s

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64--linux-android"

declare void @use32(ptr)

define void @test_alloca() sanitize_hwaddress {
; CHECK-LABEL: @test_alloca(
; CHECK: %[[X_TAG:[^ ]*]] = trunc i64 {{.*}} to i8
; CHECK: call void @llvm.memset.p0.i64(ptr align 1 {{.*}}, i8 %[[X_TAG]], i64 1, i1 false)
  %x = alloca i32, align 4
  call void @use32(ptr nonnull %x)
  ret void
}
