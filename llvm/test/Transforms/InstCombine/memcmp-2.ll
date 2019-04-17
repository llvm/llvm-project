; Test that the memcmp library call simplifier works correctly.
;
; RUN: opt < %s -instcombine -S | FileCheck %s

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"

declare i32* @memcmp(i8*, i8*, i32)

; Check that memcmp functions with the wrong prototype aren't simplified.

define i32* @test_no_simplify1(i8* %mem, i32 %size) {
; CHECK-LABEL: @test_no_simplify1(
  %ret = call i32* @memcmp(i8* %mem, i8* %mem, i32 %size)
; CHECK-NEXT: call i32* @memcmp
  ret i32* %ret
; CHECK-NEXT: ret i32* %ret
}
