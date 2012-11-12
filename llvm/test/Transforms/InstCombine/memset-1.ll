; Test that the memset library call simplifier works correctly.
;
; RUN: opt < %s -instcombine -S | FileCheck %s

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"

declare i8* @memset(i8*, i32, i32)

; Check memset(mem1, val, size) -> llvm.memset(mem1, val, size, 1).

define i8* @test_simplify1(i8* %mem, i32 %val, i32 %size) {
; CHECK: @test_simplify1
  %ret = call i8* @memset(i8* %mem, i32 %val, i32 %size)
; CHECK: call void @llvm.memset
  ret i8* %ret
; CHECK: ret i8* %mem
}
