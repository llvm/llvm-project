; Test that the memcmp library call simplifier works correctly.
;
; RUN: opt < %s -passes=instcombine -S | FileCheck %s

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"

declare ptr @memcmp(ptr, ptr, i32)

; Check that memcmp functions with the wrong prototype aren't simplified.

define ptr @test_no_simplify1(ptr %mem, i32 %size) {
; CHECK-LABEL: @test_no_simplify1(
  %ret = call ptr @memcmp(ptr %mem, ptr %mem, i32 %size)
; CHECK-NEXT: call ptr @memcmp
  ret ptr %ret
; CHECK-NEXT: ret ptr %ret
}
