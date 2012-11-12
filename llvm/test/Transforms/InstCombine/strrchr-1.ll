; Test that the strrchr library call simplifier works correctly.
; RUN: opt < %s -instcombine -S | FileCheck %s

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"

@hello = constant [14 x i8] c"hello world\5Cn\00"
@null = constant [1 x i8] zeroinitializer
@chp = global i8* zeroinitializer

declare i8* @strrchr(i8*, i32)

define void @test_simplify1() {
; CHECK: store i8* getelementptr inbounds ([14 x i8]* @hello, i32 0, i32 6)
; CHECK-NOT: call i8* @strrchr
; CHECK: ret void

  %str = getelementptr [14 x i8]* @hello, i32 0, i32 0
  %dst = call i8* @strrchr(i8* %str, i32 119)
  store i8* %dst, i8** @chp
  ret void
}

define void @test_simplify2() {
; CHECK: store i8* null, i8** @chp, align 4
; CHECK-NOT: call i8* @strrchr
; CHECK: ret void

  %str = getelementptr [1 x i8]* @null, i32 0, i32 0
  %dst = call i8* @strrchr(i8* %str, i32 119)
  store i8* %dst, i8** @chp
  ret void
}

define void @test_simplify3() {
; CHECK: store i8* getelementptr inbounds ([14 x i8]* @hello, i32 0, i32 13)
; CHECK-NOT: call i8* @strrchr
; CHECK: ret void

  %src = getelementptr [14 x i8]* @hello, i32 0, i32 0
  %dst = call i8* @strrchr(i8* %src, i32 0)
  store i8* %dst, i8** @chp
  ret void
}

define void @test_nosimplify1(i32 %chr) {
; CHECK: @test_nosimplify1
; CHECK: call i8* @strrchr
; CHECK: ret void

  %src = getelementptr [14 x i8]* @hello, i32 0, i32 0
  %dst = call i8* @strrchr(i8* %src, i32 %chr)
  store i8* %dst, i8** @chp
  ret void
}
