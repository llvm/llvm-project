; Test lib call simplification of __strncpy_chk calls with various values
; for len and dstlen.
;
; RUN: opt < %s -instcombine -S | FileCheck %s

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"

@a = common global [60 x i8] zeroinitializer, align 1
@b = common global [60 x i8] zeroinitializer, align 1
@.str = private constant [12 x i8] c"abcdefghijk\00"

; Check cases where dstlen >= len

define void @test_simplify1() {
; CHECK: @test_simplify1
  %dst = getelementptr inbounds [60 x i8]* @a, i32 0, i32 0
  %src = getelementptr inbounds [12 x i8]* @.str, i32 0, i32 0

; CHECK-NEXT: call void @llvm.memcpy.p0i8.p0i8.i32
  call i8* @__strncpy_chk(i8* %dst, i8* %src, i32 12, i32 60)
  ret void
}

define void @test_simplify2() {
; CHECK: @test_simplify2
  %dst = getelementptr inbounds [60 x i8]* @a, i32 0, i32 0
  %src = getelementptr inbounds [12 x i8]* @.str, i32 0, i32 0

; CHECK-NEXT: call void @llvm.memcpy.p0i8.p0i8.i32
  call i8* @__strncpy_chk(i8* %dst, i8* %src, i32 12, i32 12)
  ret void
}

define void @test_simplify3() {
; CHECK: @test_simplify3
  %dst = getelementptr inbounds [60 x i8]* @a, i32 0, i32 0
  %src = getelementptr inbounds [60 x i8]* @b, i32 0, i32 0

; CHECK-NEXT: call i8* @strncpy
  call i8* @__strncpy_chk(i8* %dst, i8* %src, i32 12, i32 60)
  ret void
}

; Check cases where dstlen < len

define void @test_no_simplify1() {
; CHECK: @test_no_simplify1
  %dst = getelementptr inbounds [60 x i8]* @a, i32 0, i32 0
  %src = getelementptr inbounds [12 x i8]* @.str, i32 0, i32 0

; CHECK-NEXT: call i8* @__strncpy_chk
  call i8* @__strncpy_chk(i8* %dst, i8* %src, i32 8, i32 4)
  ret void
}

define void @test_no_simplify2() {
; CHECK: @test_no_simplify2
  %dst = getelementptr inbounds [60 x i8]* @a, i32 0, i32 0
  %src = getelementptr inbounds [60 x i8]* @b, i32 0, i32 0

; CHECK-NEXT: call i8* @__strncpy_chk
  call i8* @__strncpy_chk(i8* %dst, i8* %src, i32 8, i32 0)
  ret void
}

declare i8* @__strncpy_chk(i8*, i8*, i32, i32)
