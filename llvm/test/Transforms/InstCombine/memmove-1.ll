; Test that the memmove library call simplifier works correctly.
;
; RUN: opt < %s -passes=instcombine -S | FileCheck %s

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"

declare ptr @memmove(ptr, ptr, i32)

; Check memmove(mem1, mem2, size) -> llvm.memmove(mem1, mem2, size, 1).

define ptr @test_simplify1(ptr %mem1, ptr %mem2, i32 %size) {
; CHECK-LABEL: @test_simplify1(
  %ret = call ptr @memmove(ptr %mem1, ptr %mem2, i32 %size)
; CHECK: call void @llvm.memmove
  ret ptr %ret
; CHECK: ret ptr %mem1
}

define ptr @test_simplify2(ptr %mem1, ptr %mem2, i32 %size) {
; CHECK-LABEL: @test_simplify2(
; CHECK-NEXT:   tail call void @llvm.memmove
; CHECK-NEXT:   ret ptr %mem1
  %ret = tail call ptr @memmove(ptr %mem1, ptr %mem2, i32 %size)
  ret ptr %ret
}

define ptr @test_no_simplify1(ptr %mem1, ptr %mem2, i32 %size) {
; CHECK-LABEL: @test_no_simplify1(
; CHECK-NEXT:   %ret = musttail call ptr @memmove(ptr %mem1, ptr %mem2, i32 %size)
; CHECK-NEXT:   ret ptr %ret
  %ret = musttail call ptr @memmove(ptr %mem1, ptr %mem2, i32 %size)
  ret ptr %ret
}

define ptr @test_no_incompatible_attr(ptr %mem1, ptr %mem2, i32 %size) {
; CHECK-LABEL: @test_no_incompatible_attr(
  %ret = call dereferenceable(1) ptr @memmove(ptr %mem1, ptr %mem2, i32 %size)
; CHECK: call void @llvm.memmove
  ret ptr %ret
; CHECK: ret ptr %mem1
}
