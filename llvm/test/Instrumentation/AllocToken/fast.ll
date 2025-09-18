; RUN: opt < %s -passes=inferattrs,alloc-token -alloc-token-mode=increment -alloc-token-fast-abi -alloc-token-max=3 -S | FileCheck %s

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare ptr @malloc(i64)
declare ptr @calloc(i64, i64)
declare ptr @realloc(ptr, i64)
declare ptr @_Znwm(i64)
declare ptr @_Znam(i64)

; Test basic allocation call rewriting
; CHECK-LABEL: @test_basic_rewriting
define ptr @test_basic_rewriting() sanitize_alloc_token {
entry:
  ; CHECK: [[PTR1:%ptr[0-9]]] = call ptr @__alloc_token_0_malloc(i64 64)
  ; CHECK: call ptr @__alloc_token_1_calloc(i64 8, i64 8)
  ; CHECK: call ptr @__alloc_token_2_realloc(ptr [[PTR1]], i64 128)
  ; CHECK-NOT: call ptr @malloc(
  ; CHECK-NOT: call ptr @calloc(
  ; CHECK-NOT: call ptr @realloc(
  %ptr1 = call ptr @malloc(i64 64)
  %ptr2 = call ptr @calloc(i64 8, i64 8)
  %ptr3 = call ptr @realloc(ptr %ptr1, i64 128)
  ret ptr %ptr3
}

; Test C++ operator rewriting
; CHECK-LABEL: @test_cpp_operators
define ptr @test_cpp_operators() sanitize_alloc_token {
entry:
  ; CHECK: call ptr @__alloc_token_0_Znwm(i64 32)
  ; CHECK: call ptr @__alloc_token_1_Znam(i64 64)
  ; CHECK-NOT: call ptr @_Znwm(
  ; CHECK-NOT: call ptr @_Znam(
  %ptr1 = call ptr @_Znwm(i64 32)
  %ptr2 = call ptr @_Znam(i64 64)
  ret ptr %ptr1
}
