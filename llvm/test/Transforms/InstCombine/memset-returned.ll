; Test that the memset library call simplifier handles the 'returned'
; attribute correctly when converting memset to llvm.memset.
;
; RUN: opt < %s -passes=instcombine -S | FileCheck %s

declare ptr @memset(ptr, i32, i64)

define ptr @test_memset_returned(ptr %p, i32 %v, i64 %n) {
entry:
  %r = call ptr @memset(ptr returned %p, i32 %v, i64 %n)
  ret ptr %r
}

; CHECK-LABEL: define ptr @test_memset_returned(
; CHECK: call void @llvm.memset.p0.i64(ptr align 1 %p,
; CHECK-NOT: returned
; CHECK: ret ptr %p