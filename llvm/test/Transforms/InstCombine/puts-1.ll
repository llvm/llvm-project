; Test that the puts library call simplifier works correctly.
;
; RUN: opt < %s -instcombine -S | FileCheck %s

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"

@empty = constant [1 x i8] zeroinitializer

declare i32 @puts(i8*)

; Check puts("") -> putchar('\n').

define void @test_simplify1() {
; CHECK-LABEL: @test_simplify1(
  %str = getelementptr [1 x i8], [1 x i8]* @empty, i32 0, i32 0
  call i32 @puts(i8* %str)
; CHECK-NEXT: call i32 @putchar(i32 10)
  ret void
; CHECK-NEXT: ret void
}

; Don't simplify if the return value is used.

define i32 @test_no_simplify1() {
; CHECK-LABEL: @test_no_simplify1(
  %str = getelementptr [1 x i8], [1 x i8]* @empty, i32 0, i32 0
  %ret = call i32 @puts(i8* %str)
; CHECK-NEXT: call i32 @puts(i8* getelementptr inbounds ([1 x i8], [1 x i8]* @empty, i32 0, i32 0))
  ret i32 %ret
; CHECK-NEXT: ret i32 %ret
}
