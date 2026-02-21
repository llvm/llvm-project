; REQUIRES: x86-registered-target
;
; Test that allocas whose addresses escape to another function are still
; instrumented by ASan, even if all accesses are in-bounds.
;
; This is a regression test for https://github.com/llvm/llvm-project/issues/178576
; where passing an alloca's address to a function that performs a load on it
; would cause a false positive. The callee's load would be instrumented by ASan,
; but the caller's alloca would not be (because it was considered "safe"),
; leaving the shadow memory uninitialized.

; RUN: opt < %s -S -passes=asan -asan-use-stack-safety=1 -o - | FileCheck %s

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function that loads from a pointer parameter - ASan will instrument this load.
; If the caller's alloca is not instrumented, the shadow memory will be uninitialized.
declare void @func(ptr %ptr)

; CHECK-LABEL: define void @caller
define void @caller() sanitize_address {
entry:
  ; The alloca's address escapes via the call to @func.
  ; Even though all direct accesses are in-bounds, we must instrument this
  ; alloca so that shadow memory is properly initialized for the callee.
  ; CHECK: call i64 @__asan_stack_malloc
  %ptr = alloca ptr, align 8
  store ptr null, ptr %ptr, align 8
  call void @func(ptr %ptr)
  ret void
}

; Test case where the alloca's address does NOT escape - should still be optimized.
; CHECK-LABEL: define i32 @no_escape
define i32 @no_escape() sanitize_address {
entry:
  ; No call passes the address, so this alloca can remain uninstrumented.
  ; CHECK-NOT: call i64 @__asan_stack_malloc
  %buf = alloca [10 x i8], align 1
  %0 = load volatile i8, ptr %buf, align 1
  ret i32 0
}
