; Test that all supported module flags are retrieved correctly.
;
; RUN: opt < %s -passes='inferattrs,alloc-token' -S | FileCheck %s --check-prefixes=CHECK,DEFAULT
; RUN: opt < %s -passes='inferattrs,alloc-token' -alloc-token-max=2 -alloc-token-fast-abi=0 -alloc-token-extended=0 -S | FileCheck %s --check-prefixes=CHECK,OVERRIDE

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"

declare ptr @_Znwm(i64)
declare ptr @malloc(i64)
declare ptr @my_malloc(i64)

define void @test() sanitize_alloc_token {
; CHECK-LABEL: define void @test(
; DEFAULT: call ptr @__alloc_token_0_malloc(i64 8)
; DEFAULT: call ptr @__alloc_token_1__Znwm(i64 8)
; DEFAULT: call ptr @__alloc_token_2_malloc(i64 8)
; DEFAULT: call ptr @__alloc_token_0_my_malloc(i64 8)
; OVERRIDE: call ptr @__alloc_token_malloc(i64 8, i64 0)
; OVERRIDE: call ptr @__alloc_token__Znwm(i64 8, i64 1)
; OVERRIDE: call ptr @__alloc_token_malloc(i64 8, i64 0)
; OVERRIDE: call ptr @my_malloc(i64 8)
  %1 = call ptr @malloc(i64 8)
  %2 = call ptr @_Znwm(i64 8)
  %3 = call ptr @malloc(i64 8)
  %4 = call ptr @my_malloc(i64 8), !alloc_token !0
  ret void
}

!0 = !{!"int", i1 0}

!llvm.module.flags = !{!1, !2, !3, !4}
!1 = !{i32 1, !"alloc-token-mode", !"increment"}
!2 = !{i32 1, !"alloc-token-max", i64 3}
!3 = !{i32 1, !"alloc-token-fast-abi", i64 1}
!4 = !{i32 1, !"alloc-token-extended", i64 1}
