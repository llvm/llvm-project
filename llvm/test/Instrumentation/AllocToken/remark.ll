; RUN: opt < %s -passes=inferattrs,alloc-token -pass-remarks=alloc-token -S 2>&1 | FileCheck %s

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare ptr @malloc(i64)

; CHECK-NOT: remark: <unknown>:0:0: Call to 'malloc' in 'test_has_metadata' without source-level type token
; CHECK: remark: <unknown>:0:0: Call to 'malloc' in 'test_no_metadata' without source-level type token

; CHECK-LABEL: @test_has_metadata
define ptr @test_has_metadata() sanitize_alloc_token {
entry:
  ; CHECK: call ptr @__alloc_token_malloc(
  %ptr1 = call ptr @malloc(i64 64), !alloc_token !0
  ret ptr %ptr1
}

; CHECK-LABEL: @test_no_metadata
define ptr @test_no_metadata() sanitize_alloc_token {
entry:
  ; CHECK: call ptr @__alloc_token_malloc(
  %ptr1 = call ptr @malloc(i64 32)
  ret ptr %ptr1
}

!0 = !{!"int", i1 0}
