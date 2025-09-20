; RUN: opt < %s -passes=inferattrs,alloc-token -alloc-token-mode=increment -S | FileCheck %s

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; CHECK-LABEL: @test_invoke_malloc
define ptr @test_invoke_malloc() sanitize_alloc_token personality ptr @__gxx_personality_v0 {
entry:
  ; CHECK: invoke ptr @__alloc_token_malloc(i64 64, i64 0)
  ; CHECK-NEXT: to label %normal unwind label %cleanup
  ; CHECK-NOT: call ptr @__alloc_token_malloc
  ; CHECK-NOT: call ptr @malloc
  %ptr = invoke ptr @malloc(i64 64) to label %normal unwind label %cleanup

normal:
  ret ptr %ptr

cleanup:
  %lp = landingpad { ptr, i32 } cleanup
  ret ptr null
}

; CHECK-LABEL: @test_invoke_operator_new
define ptr @test_invoke_operator_new() sanitize_alloc_token personality ptr @__gxx_personality_v0 {
entry:
  ; CHECK: invoke ptr @__alloc_token_Znwm(i64 32, i64 1)
  ; CHECK-NEXT: to label %normal unwind label %cleanup
  ; CHECK-NOT: call ptr @__alloc_token_Znwm
  ; CHECK-NOT: call ptr @_Znwm
  %ptr = invoke ptr @_Znwm(i64 32) to label %normal unwind label %cleanup

normal:
  ret ptr %ptr

cleanup:
  %lp = landingpad { ptr, i32 } cleanup
  ret ptr null
}

; Test complex exception flow with multiple invoke allocations
; CHECK-LABEL: @test_complex_invoke_flow
define ptr @test_complex_invoke_flow() sanitize_alloc_token personality ptr @__gxx_personality_v0 {
entry:
  ; CHECK: invoke ptr @__alloc_token_malloc(i64 16, i64 2)
  ; CHECK-NEXT: to label %first_ok unwind label %cleanup1
  %ptr1 = invoke ptr @malloc(i64 16) to label %first_ok unwind label %cleanup1

first_ok:
  ; CHECK: invoke ptr @__alloc_token_Znwm(i64 32, i64 3)
  ; CHECK-NEXT: to label %second_ok unwind label %cleanup2
  %ptr2 = invoke ptr @_Znwm(i64 32) to label %second_ok unwind label %cleanup2

second_ok:
  ret ptr %ptr1

cleanup1:
  %lp1 = landingpad { ptr, i32 } cleanup
  ret ptr null

cleanup2:
  %lp2 = landingpad { ptr, i32 } cleanup
  ret ptr null
}

; Test mixed call/invoke
; CHECK-LABEL: @test_mixed_call_invoke
define ptr @test_mixed_call_invoke() sanitize_alloc_token personality ptr @__gxx_personality_v0 {
entry:
  ; CHECK: call ptr @__alloc_token_malloc(i64 8, i64 4)
  %ptr1 = call ptr @malloc(i64 8)

  ; CHECK: invoke ptr @__alloc_token_malloc(i64 16, i64 5)
  ; CHECK-NEXT: to label %normal unwind label %cleanup
  %ptr2 = invoke ptr @malloc(i64 16) to label %normal unwind label %cleanup

normal:
  ret ptr %ptr1

cleanup:
  %lp = landingpad { ptr, i32 } cleanup
  ret ptr null
}

declare ptr @malloc(i64)
declare ptr @_Znwm(i64)
declare i32 @__gxx_personality_v0(...)
