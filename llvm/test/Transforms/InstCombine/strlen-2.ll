; Test that the strlen library call simplifier works correctly.
;
; RUN: opt < %s -instcombine -S | FileCheck %s

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"

@hello = constant [6 x i8] c"hello\00"

declare i32 @strlen(i8*, i32)

define i32 @test_no_simplify1() {
; CHECK-LABEL: @test_no_simplify1(
  %hello_p = getelementptr [6 x i8], [6 x i8]* @hello, i32 0, i32 0
  %hello_l = call i32 @strlen(i8* %hello_p, i32 187)
; CHECK-NEXT: %hello_l = call i32 @strlen
  ret i32 %hello_l
; CHECK-NEXT: ret i32 %hello_l
}
