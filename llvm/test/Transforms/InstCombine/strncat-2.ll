; Test that the strncat libcall simplifier works correctly.
;
; RUN: opt < %s -instcombine -S | FileCheck %s

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"

@hello = constant [6 x i8] c"hello\00"
@empty = constant [1 x i8] c"\00"
@a = common global [32 x i8] zeroinitializer, align 1

declare i8* @strncat(i8*, i8*, i32)

define void @test_simplify1() {
; CHECK-LABEL: @test_simplify1(
; CHECK-NOT: call i8* @strncat
; CHECK: ret void

  %dst = getelementptr [32 x i8], [32 x i8]* @a, i32 0, i32 0
  %src = getelementptr [6 x i8], [6 x i8]* @hello, i32 0, i32 0
  call i8* @strncat(i8* %dst, i8* %src, i32 13)
  ret void
}

define void @test_simplify2() {
; CHECK-LABEL: @test_simplify2(
; CHECK-NEXT: ret void

  %dst = getelementptr [32 x i8], [32 x i8]* @a, i32 0, i32 0
  %src = getelementptr [1 x i8], [1 x i8]* @empty, i32 0, i32 0
  call i8* @strncat(i8* %dst, i8* %src, i32 13)
  ret void
}

define void @test_simplify3() {
; CHECK-LABEL: @test_simplify3(
; CHECK-NEXT: ret void

  %dst = getelementptr [32 x i8], [32 x i8]* @a, i32 0, i32 0
  %src = getelementptr [6 x i8], [6 x i8]* @hello, i32 0, i32 0
  call i8* @strncat(i8* %dst, i8* %src, i32 0)
  ret void
}

define void @test_nosimplify1() {
; CHECK-LABEL: @test_nosimplify1(
; CHECK: call i8* @strncat
; CHECK: ret void

  %dst = getelementptr [32 x i8], [32 x i8]* @a, i32 0, i32 0
  %src = getelementptr [6 x i8], [6 x i8]* @hello, i32 0, i32 0
  call i8* @strncat(i8* %dst, i8* %src, i32 1)
  ret void
}
