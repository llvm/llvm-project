; Test that the strchr library call simplifier works correctly.
; RUN: opt < %s -instcombine -S | FileCheck %s

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128-n8:16:32"

@hello = constant [14 x i8] c"hello world\5Cn\00"
@null = constant [1 x i8] zeroinitializer
@newlines = constant [3 x i8] c"\0D\0A\00"
@chp = global i8* zeroinitializer

declare i8* @strchr(i8*, i32)

define void @test_simplify1() {
; CHECK: store i8* getelementptr inbounds ([14 x i8], [14 x i8]* @hello, i32 0, i32 6)
; CHECK-NOT: call i8* @strchr
; CHECK: ret void

  %str = getelementptr [14 x i8], [14 x i8]* @hello, i32 0, i32 0
  %dst = call i8* @strchr(i8* %str, i32 119)
  store i8* %dst, i8** @chp
  ret void
}

define void @test_simplify2() {
; CHECK: store i8* null, i8** @chp, align 4
; CHECK-NOT: call i8* @strchr
; CHECK: ret void

  %str = getelementptr [1 x i8], [1 x i8]* @null, i32 0, i32 0
  %dst = call i8* @strchr(i8* %str, i32 119)
  store i8* %dst, i8** @chp
  ret void
}

define void @test_simplify3() {
; CHECK: store i8* getelementptr inbounds ([14 x i8], [14 x i8]* @hello, i32 0, i32 13)
; CHECK-NOT: call i8* @strchr
; CHECK: ret void

  %src = getelementptr [14 x i8], [14 x i8]* @hello, i32 0, i32 0
  %dst = call i8* @strchr(i8* %src, i32 0)
  store i8* %dst, i8** @chp
  ret void
}

define void @test_simplify4(i32 %chr) {
; CHECK: call i8* @memchr
; CHECK-NOT: call i8* @strchr
; CHECK: ret void

  %src = getelementptr [14 x i8], [14 x i8]* @hello, i32 0, i32 0
  %dst = call i8* @strchr(i8* %src, i32 %chr)
  store i8* %dst, i8** @chp
  ret void
}

define void @test_simplify5() {
; CHECK: store i8* getelementptr inbounds ([14 x i8], [14 x i8]* @hello, i32 0, i32 13)
; CHECK-NOT: call i8* @strchr
; CHECK: ret void

  %src = getelementptr [14 x i8], [14 x i8]* @hello, i32 0, i32 0
  %dst = call i8* @strchr(i8* %src, i32 65280)
  store i8* %dst, i8** @chp
  ret void
}

; Check transformation strchr(p, 0) -> p + strlen(p)
define void @test_simplify6(i8* %str) {
; CHECK: %strlen = call i32 @strlen(i8* %str)
; CHECK-NOT: call i8* @strchr
; CHECK: %strchr = getelementptr i8, i8* %str, i32 %strlen
; CHECK: store i8* %strchr, i8** @chp, align 4
; CHECK: ret void

  %dst = call i8* @strchr(i8* %str, i32 0)
  store i8* %dst, i8** @chp
  ret void
}

; Check transformation strchr("\r\n", C) != nullptr -> (C & 9217) != 0
define i1 @test_simplify7(i32 %C) {
; CHECK-LABEL: @test_simplify7
; CHECK-NEXT: [[TRUNC:%.*]] = trunc i32 %C to i16
; CHECK-NEXT: [[TRUNC_AND:%.*]] = and i16 [[TRUNC]], 255
; CHECK-NEXT: %memchr.bounds = icmp ult i16 [[TRUNC_AND]], 16
; CHECK-NEXT: [[SHL:%.*]] = shl i16 1, [[TRUNC_AND]]
; CHECK-NEXT: [[AND:%.*]] = and i16 [[SHL]], 9217
; CHECK-NEXT: %memchr.bits = icmp ne i16 [[AND]], 0
; CHECK-NEXT: %memchr1 = and i1 %memchr.bounds, %memchr.bits
; CHECK-NEXT: ret i1 %memchr1

  %dst = call i8* @strchr(i8* getelementptr inbounds ([3 x i8], [3 x i8]* @newlines, i64 0, i64 0), i32 %C)
  %cmp = icmp ne i8* %dst, null
  ret i1 %cmp
}
