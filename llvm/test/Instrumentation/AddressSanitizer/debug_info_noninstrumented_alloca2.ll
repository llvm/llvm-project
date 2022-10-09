; Make sure we don't break the IR when moving non-instrumented allocas

; RUN: opt < %s -passes=asan -S | FileCheck %s
; RUN: opt < %s -passes=asan -asan-instrument-dynamic-allocas -S | FileCheck %s

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.10.0"

define i32 @foo() sanitize_address {
entry:
  %non_instrumented1 = alloca i32, align 4
  %t = load i32, i32* %non_instrumented1, align 4
  %instrumented = alloca i32, align 4
  %ptr = ptrtoint i32* %instrumented to i32
  ret i32 %t
}

; CHECK: entry:
; CHECK: %non_instrumented1 = alloca i32, align 4
; CHECK: load i32, i32* %non_instrumented1
; CHECK: load i32, i32* @__asan_option_detect_stack_use_after_return
