; This test checks that non-instrumented allocas stay in the first basic block.
; Only first-basic-block allocas are considered stack slots, and moving them
; breaks debug info.

; RUN: opt < %s -asan -asan-module -enable-new-pm=0 -S | FileCheck %s
; RUN: opt < %s -passes='asan-pipeline' -S | FileCheck %s
; RUN: opt < %s -asan -asan-module -enable-new-pm=0 -asan-instrument-dynamic-allocas -S | FileCheck %s
; RUN: opt < %s -passes='asan-pipeline' -asan-instrument-dynamic-allocas -S | FileCheck %s

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.10.0"

define i32 @foo() sanitize_address {
entry:
  ; Won't be instrumented because of asan-skip-promotable-allocas.
  %non_instrumented1 = alloca i32, align 4

  ; Regular alloca, will get instrumented (forced by the ptrtoint below).
  %instrumented = alloca i32, align 4

  ; Won't be instrumented because of asan-skip-promotable-allocas.
  %non_instrumented2 = alloca i32, align 4

  br label %bb0

bb0:
  ; Won't be instrumented because of asan-skip-promotable-allocas.
  %non_instrumented3 = alloca i32, align 4

  %ptr = ptrtoint i32* %instrumented to i32
  br label %bb1

bb1:
  ret i32 %ptr
}

; CHECK: entry:
; CHECK: %non_instrumented1 = alloca i32, align 4
; CHECK: %non_instrumented2 = alloca i32, align 4
; CHECK: load i32, i32* @__asan_option_detect_stack_use_after_return
; CHECK: bb0:
; CHECK: %non_instrumented3 = alloca i32, align 4
