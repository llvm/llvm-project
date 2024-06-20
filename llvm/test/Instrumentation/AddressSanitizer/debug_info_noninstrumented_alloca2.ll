; Make sure we don't break the IR when moving non-instrumented allocas

; RUN: opt < %s -passes=asan -asan-use-stack-safety=1 -S | FileCheck %s
; RUN: opt < %s -passes=asan -asan-use-stack-safety=1 -asan-instrument-dynamic-allocas -S | FileCheck %s

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.10.0"

define i32 @foo(i64 %i) sanitize_address {
entry:
  %non_instrumented1 = alloca i32, align 4
  %t = load i32, ptr %non_instrumented1, align 4
  %instrumented = alloca [2 x i32], align 4
  %ai = getelementptr inbounds [2 x i32], ptr %instrumented, i64 0, i64 %i
  store volatile i8 0, ptr %ai, align 4
  ret i32 %t
}

; CHECK: entry:
; CHECK: %non_instrumented1 = alloca i32, align 4
; CHECK: load i32, ptr %non_instrumented1
; CHECK: load i32, ptr @__asan_option_detect_stack_use_after_return
