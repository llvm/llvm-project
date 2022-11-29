; Test that the old outlined check is used with old API levels.

; RUN: opt < %s -passes=hwasan -S | FileCheck %s

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64--linux-android"

define i8 @test_load8(ptr %a) sanitize_hwaddress {
; CHECK-LABEL: @test_load8(
; CHECK: call void @llvm.hwasan.check.memaccess(ptr {{.*}}, ptr {{.*}}, i32 0)
  %b = load i8, ptr %a, align 4
  ret i8 %b
}
