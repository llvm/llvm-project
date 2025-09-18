; RUN: opt < %s -passes=inferattrs,alloc-token -alloc-token-mode=increment -alloc-token-extended -S | FileCheck %s

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare ptr @malloc(i64)
declare ptr @custom_malloc(i64)
declare ptr @kmalloc(i64, i64)

; CHECK-LABEL: @test_libcall
define ptr @test_libcall() sanitize_alloc_token {
entry:
  ; CHECK: call ptr @__alloc_token_malloc(i64 64, i64 0)
  %ptr1 = call ptr @malloc(i64 64)
  ret ptr %ptr1
}

; CHECK-LABEL: @test_libcall_hint
define ptr @test_libcall_hint() sanitize_alloc_token {
entry:
  ; CHECK: call ptr @__alloc_token_malloc(i64 64, i64 1)
  %ptr1 = call ptr @malloc(i64 64), !alloc_token !0
  ret ptr %ptr1
}

; CHECK-LABEL: @test_nonlibcall_nohint
define ptr @test_nonlibcall_nohint() sanitize_alloc_token {
entry:
  ; CHECK: call ptr @custom_malloc(i64 8)
  ; CHECK: call ptr @kmalloc(i64 32, i64 0)
  %ptr1 = call ptr @custom_malloc(i64 8)
  %ptr2 = call ptr @kmalloc(i64 32, i64 0)
  ret ptr %ptr1
}

; CHECK-LABEL: @test_nonlibcall_hint
define ptr @test_nonlibcall_hint() sanitize_alloc_token {
entry:
  ; CHECK: call ptr @__alloc_token_custom_malloc(i64 8, i64 2)
  ; CHECK: call ptr @__alloc_token_kmalloc(i64 32, i64 0, i64 3)
  ; CHECK: call ptr @__alloc_token_custom_malloc(i64 64, i64 4)
  ; CHECK: call ptr @__alloc_token_kmalloc(i64 128, i64 2, i64 5)
  %ptr1 = call ptr @custom_malloc(i64 8), !alloc_token !0
  %ptr2 = call ptr @kmalloc(i64 32, i64 0), !alloc_token !0
  %ptr3 = call ptr @custom_malloc(i64 64), !alloc_token !0
  %ptr4 = call ptr @kmalloc(i64 128, i64 2), !alloc_token !0
  ret ptr %ptr1
}

; Functions without sanitize_alloc_token do not get instrumented
; CHECK-LABEL: @without_attribute
define ptr @without_attribute() {
entry:
  ; CHECK: call ptr @malloc(i64 64)
  ; CHECK: call ptr @custom_malloc(i64 8)
  ; CHECK: call ptr @kmalloc(i64 32, i64 0)
  %ptr1 = call ptr @malloc(i64 64), !alloc_token !0
  %ptr2 = call ptr @custom_malloc(i64 8), !alloc_token !0
  %ptr3 = call ptr @kmalloc(i64 32, i64 0), !alloc_token !0
  ret ptr %ptr1
}

!0 = !{!"int", i1 0}
