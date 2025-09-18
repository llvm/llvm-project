; RUN: opt < %s -passes=inferattrs,alloc-token -alloc-token-mode=increment -S | FileCheck %s

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare ptr @malloc(i64)
declare ptr @calloc(i64, i64)
declare ptr @realloc(ptr, i64)
declare ptr @_Znwm(i64)
declare ptr @_Znam(i64)
declare void @free(ptr)
declare void @_ZdlPv(ptr)
declare i32 @foobar(i64)

; Test basic allocation call rewriting
; CHECK-LABEL: @test_basic_rewriting
define ptr @test_basic_rewriting() sanitize_alloc_token {
entry:
  ; CHECK: [[PTR1:%[0-9]]] = call ptr @__alloc_token_malloc(i64 64, i64 0)
  ; CHECK: call ptr @__alloc_token_calloc(i64 8, i64 8, i64 1)
  ; CHECK: call ptr @__alloc_token_realloc(ptr [[PTR1]], i64 128, i64 2)
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
  ; CHECK: call ptr @__alloc_token_Znwm(i64 32, i64 3)
  ; CHECK: call ptr @__alloc_token_Znam(i64 64, i64 4)
  ; CHECK-NOT: call ptr @_Znwm(
  ; CHECK-NOT: call ptr @_Znam(
  %ptr1 = call ptr @_Znwm(i64 32)
  %ptr2 = call ptr @_Znam(i64 64)
  ret ptr %ptr1
}

; Functions without sanitize_alloc_token do not get instrumented
; CHECK-LABEL: @without_attribute
define ptr @without_attribute() {
entry:
  ; CHECK: call ptr @malloc(i64 16)
  ; CHECK-NOT: call ptr @__alloc_token_malloc
  %ptr = call ptr @malloc(i64 16)
  ret ptr %ptr
}

; Test that free/delete are untouched
; CHECK-LABEL: @test_free_untouched
define void @test_free_untouched(ptr %ptr) sanitize_alloc_token {
entry:
  ; CHECK: call void @free(ptr %ptr)
  ; CHECK: call void @_ZdlPv(ptr %ptr)
  ; CHECK-NOT: call ptr @__alloc_token_
  call void @free(ptr %ptr)
  call void @_ZdlPv(ptr %ptr)
  ret void
}

; Non-allocation functions are untouched
; CHECK-LABEL: @no_allocations
define i32 @no_allocations(i32 %x) sanitize_alloc_token {
entry:
  ; CHECK: call i32 @foobar
  ; CHECK-NOT: call i32 @__alloc_token_
  %result = call i32 @foobar(i64 42)
  ret i32 %result
}

; Test that tail calls are preserved
; CHECK-LABEL: @test_tail_call_preserved
define ptr @test_tail_call_preserved() sanitize_alloc_token {
entry:
  ; CHECK: tail call ptr @__alloc_token_malloc(i64 42, i64 5)
  ; CHECK-NOT: tail call ptr @malloc(
  %result = tail call ptr @malloc(i64 42)
  ret ptr %result
}
