; Test for all allocation functions that should be ignored by default.
;
; RUN: opt < %s -passes=inferattrs,alloc-token -S | FileCheck %s

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare ptr @strdup(ptr)
declare ptr @__strdup(ptr)
declare ptr @strndup(ptr, i64)
declare ptr @__strndup(ptr, i64)

; CHECK-LABEL: @test_ignorable_allocation_functions
define ptr @test_ignorable_allocation_functions(ptr %ptr) sanitize_alloc_token {
entry:
  ; CHECK: call ptr @strdup(
  %ptr1 = call ptr @strdup(ptr %ptr)
  ; CHECK: call ptr @__strdup(
  %ptr2 = call ptr @__strdup(ptr %ptr)
  ; CHECK: call ptr @strndup(
  %ptr3 = call ptr @strndup(ptr %ptr, i64 42)
  ; CHECK: call ptr @__strndup(
  %ptr4 = call ptr @__strndup(ptr %ptr, i64 42)
  ret ptr %ptr1
}
