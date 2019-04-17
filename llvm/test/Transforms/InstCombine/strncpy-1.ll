; Test that the strncpy library call simplifier works correctly.
;
; RUN: opt < %s -instcombine -S | FileCheck %s

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"

@hello = constant [6 x i8] c"hello\00"
@null = constant [1 x i8] zeroinitializer
@null_hello = constant [7 x i8] c"\00hello\00"
@a = common global [32 x i8] zeroinitializer, align 1
@b = common global [32 x i8] zeroinitializer, align 1

declare i8* @strncpy(i8*, i8*, i32)
declare i32 @puts(i8*)

; Check a bunch of strncpy invocations together.

define i32 @test_simplify1() {
; CHECK-LABEL: @test_simplify1(
; CHECK-NOT: call i8* @strncpy
; CHECK: call i32 @puts
  %target = alloca [1024 x i8]
  %arg1 = getelementptr [1024 x i8], [1024 x i8]* %target, i32 0, i32 0
  store i8 0, i8* %arg1

  %arg2 = getelementptr [6 x i8], [6 x i8]* @hello, i32 0, i32 0
  %rslt1 = call i8* @strncpy(i8* %arg1, i8* %arg2, i32 6)

  %arg3 = getelementptr [1 x i8], [1 x i8]* @null, i32 0, i32 0
  %rslt2 = call i8* @strncpy(i8* %rslt1, i8* %arg3, i32 42)

  %arg4 = getelementptr [7 x i8], [7 x i8]* @null_hello, i32 0, i32 0
  %rslt3 = call i8* @strncpy(i8* %rslt2, i8* %arg4, i32 42)

  call i32 @puts( i8* %rslt3 )
  ret i32 0
}

; Check strncpy(x, "", y) -> memset(x, '\0', y, 1).

define void @test_simplify2() {
; CHECK-LABEL: @test_simplify2(
  %dst = getelementptr [32 x i8], [32 x i8]* @a, i32 0, i32 0
  %src = getelementptr [1 x i8], [1 x i8]* @null, i32 0, i32 0

  call i8* @strncpy(i8* %dst, i8* %src, i32 32)
; CHECK: call void @llvm.memset.p0i8.i32
  ret void
}

; Check strncpy(x, y, 0) -> x.

define i8* @test_simplify3() {
; CHECK-LABEL: @test_simplify3(
  %dst = getelementptr [32 x i8], [32 x i8]* @a, i32 0, i32 0
  %src = getelementptr [6 x i8], [6 x i8]* @hello, i32 0, i32 0

  %ret = call i8* @strncpy(i8* %dst, i8* %src, i32 0)
  ret i8* %ret
; CHECK: ret i8* getelementptr inbounds ([32 x i8], [32 x i8]* @a, i32 0, i32 0)
}

; Check  strncpy(x, s, c) -> memcpy(x, s, c, 1) [s and c are constant].

define void @test_simplify4() {
; CHECK-LABEL: @test_simplify4(
  %dst = getelementptr [32 x i8], [32 x i8]* @a, i32 0, i32 0
  %src = getelementptr [6 x i8], [6 x i8]* @hello, i32 0, i32 0

  call i8* @strncpy(i8* %dst, i8* %src, i32 6)
; CHECK: call void @llvm.memcpy.p0i8.p0i8.i32
  ret void
}

; Check cases that shouldn't be simplified.

define void @test_no_simplify1() {
; CHECK-LABEL: @test_no_simplify1(
  %dst = getelementptr [32 x i8], [32 x i8]* @a, i32 0, i32 0
  %src = getelementptr [32 x i8], [32 x i8]* @b, i32 0, i32 0

  call i8* @strncpy(i8* %dst, i8* %src, i32 32)
; CHECK: call i8* @strncpy
  ret void
}

define void @test_no_simplify2() {
; CHECK-LABEL: @test_no_simplify2(
  %dst = getelementptr [32 x i8], [32 x i8]* @a, i32 0, i32 0
  %src = getelementptr [6 x i8], [6 x i8]* @hello, i32 0, i32 0

  call i8* @strncpy(i8* %dst, i8* %src, i32 8)
; CHECK: call i8* @strncpy
  ret void
}
