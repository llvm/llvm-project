; Test that the printf library call simplifier works correctly.
;
; RUN: opt < %s -instcombine -S | FileCheck %s

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"

@hello_world = constant [13 x i8] c"hello world\0A\00"
@h = constant [2 x i8] c"h\00"
@percent_s = constant [4 x i8] c"%s\0A\00"
@format_str = constant [3 x i8] c"%s\00"
@charstr = constant [2 x i8] c"a\00"

declare void @printf(i8*, ...)

; Check simplification of printf with void return type.

define void @test_simplify1() {
; CHECK-LABEL: @test_simplify1(
  %fmt = getelementptr [2 x i8], [2 x i8]* @h, i32 0, i32 0
  call void (i8*, ...) @printf(i8* %fmt)
; CHECK-NEXT: call i32 @putchar(i32 104)
  ret void
; CHECK-NEXT: ret void
}

define void @test_simplify2() {
; CHECK-LABEL: @test_simplify2(
  %fmt = getelementptr [13 x i8], [13 x i8]* @hello_world, i32 0, i32 0
  call void (i8*, ...) @printf(i8* %fmt)
; CHECK-NEXT: call i32 @puts(i8* getelementptr inbounds ([12 x i8], [12 x i8]* @str, i32 0, i32 0))
  ret void
; CHECK-NEXT: ret void
}

define void @test_simplify6() {
; CHECK-LABEL: @test_simplify6(
  %fmt = getelementptr [4 x i8], [4 x i8]* @percent_s, i32 0, i32 0
  %str = getelementptr [13 x i8], [13 x i8]* @hello_world, i32 0, i32 0
  call void (i8*, ...) @printf(i8* %fmt, i8* %str)
; CHECK-NEXT: call i32 @puts(i8* getelementptr inbounds ([13 x i8], [13 x i8]* @hello_world, i32 0, i32 0))
  ret void
; CHECK-NEXT: ret void
}

define void @test_simplify7() {
; CHECK-LABEL: @test_simplify7(
  %fmt = getelementptr [3 x i8], [3 x i8]* @format_str, i32 0, i32 0
  %str = getelementptr [2 x i8], [2 x i8]* @charstr, i32 0, i32 0
  call void (i8*, ...) @printf(i8* %fmt, i8* %str)
; CHECK-NEXT: call i32 @putchar(i32 97)
  ret void
; CHECK-NEXT: ret void
}
