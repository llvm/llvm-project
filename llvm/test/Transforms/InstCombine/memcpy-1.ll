; Test that the memcpy library call simplifier works correctly.
;
; RUN: opt < %s -instcombine -S | FileCheck %s

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"

declare i8* @memcpy(i8*, i8*, i32)

; Check memcpy(mem1, mem2, size) -> llvm.memcpy(mem1, mem2, size, 1).

define i8* @test_simplify1(i8* %mem1, i8* %mem2, i32 %size) {
; CHECK-LABEL: @test_simplify1(
; CHECK-NEXT:    call void @llvm.memcpy.p0i8.p0i8.i32(i8* %mem1, i8* %mem2, i32 %size, i32 1, i1 false)
; CHECK-NEXT:    ret i8* %mem1
;
  %ret = call i8* @memcpy(i8* %mem1, i8* %mem2, i32 %size)
  ret i8* %ret
}

; Verify that the strictfp attr doesn't block this optimization.

define i8* @test_simplify2(i8* %mem1, i8* %mem2, i32 %size) {
; CHECK-LABEL: @test_simplify2(
  %ret = call i8* @memcpy(i8* %mem1, i8* %mem2, i32 %size) strictfp
; CHECK: call void @llvm.memcpy
  ret i8* %ret
; CHECK: ret i8* %mem1
}
