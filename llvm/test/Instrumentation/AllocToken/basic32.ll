; RUN: opt < %s -passes=inferattrs,alloc-token -alloc-token-mode=increment -S | FileCheck %s

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:32:32-n8:16:32-S128"
target triple = "i386-pc-linux-gnu"

declare ptr @malloc(i32)
declare ptr @_Znwm(i32)

; CHECK-LABEL: @test_basic_rewriting
define ptr @test_basic_rewriting() sanitize_alloc_token {
entry:
  ; CHECK: [[PTR1:%[0-9]]] = call ptr @__alloc_token_malloc(i32 64, i32 0)
  ; CHECK-NOT: call ptr @malloc(
  %ptr1 = call ptr @malloc(i32 64)
  ret ptr %ptr1
}

; CHECK-LABEL: @test_cpp_operators
define ptr @test_cpp_operators() sanitize_alloc_token {
entry:
  ; CHECK: call ptr @__alloc_token_Znwm(i32 32, i32 1)
  ; CHECK-NOT: call ptr @_Znwm(
  %ptr1 = call ptr @_Znwm(i32 32)
  ret ptr %ptr1
}
