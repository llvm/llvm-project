; RUN: opt < %s -asan -asan-module -asan-instrument-dynamic-allocas -S | FileCheck %s

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.10.0"

define i32 @test_promotable_allocas() sanitize_address {
entry:
; CHECK: %0 = alloca i32, align 4
; CHECK: store i32 0, i32* %0, align 4
; CHECK: %1 = load i32, i32* %0, align 4
; CHECK: ret i32 %1

; CHECK-NOT: __asan_stack_malloc_0
; CHECK-NOT: icmp
; CHECK-NOT: call void @__asan_report_store4

  %0 = alloca i32, align 4
  store i32 0, i32* %0, align 4
  %1 = load i32, i32* %0, align 4
  ret i32 %1
}
