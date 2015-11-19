; Test lib call simplification of __strcpy_chk calls with various values
; for src, dst, and slen.
;
; RUN: opt < %s -instcombine -S | FileCheck %s

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"

@a = common global [60 x i8] zeroinitializer, align 1
@b = common global [60 x i8] zeroinitializer, align 1
@.str = private constant [12 x i8] c"abcdefghijk\00"

; Check cases where slen >= strlen (src).

define i8* @test_simplify1() {
; CHECK-LABEL: @test_simplify1(
  %dst = getelementptr inbounds [60 x i8], [60 x i8]* @a, i32 0, i32 0
  %src = getelementptr inbounds [12 x i8], [12 x i8]* @.str, i32 0, i32 0

; CHECK-NEXT: call void @llvm.memcpy.p0i8.p0i8.i32(i8* align 1 getelementptr inbounds ([60 x i8], [60 x i8]* @a, i32 0, i32 0), i8* align 1 getelementptr inbounds ([12 x i8], [12 x i8]* @.str, i32 0, i32 0), i32 12, i1 false)
; CHECK-NEXT: ret i8* getelementptr inbounds ([60 x i8], [60 x i8]* @a, i32 0, i32 0)
  %ret = call i8* @__strcpy_chk(i8* %dst, i8* %src, i32 60)
  ret i8* %ret
}

define i8* @test_simplify2() {
; CHECK-LABEL: @test_simplify2(
  %dst = getelementptr inbounds [60 x i8], [60 x i8]* @a, i32 0, i32 0
  %src = getelementptr inbounds [12 x i8], [12 x i8]* @.str, i32 0, i32 0

; CHECK-NEXT: call void @llvm.memcpy.p0i8.p0i8.i32(i8* align 1 getelementptr inbounds ([60 x i8], [60 x i8]* @a, i32 0, i32 0), i8* align 1 getelementptr inbounds ([12 x i8], [12 x i8]* @.str, i32 0, i32 0), i32 12, i1 false)
; CHECK-NEXT: ret i8* getelementptr inbounds ([60 x i8], [60 x i8]* @a, i32 0, i32 0)
  %ret = call i8* @__strcpy_chk(i8* %dst, i8* %src, i32 12)
  ret i8* %ret
}

define i8* @test_simplify3() {
; CHECK-LABEL: @test_simplify3(
  %dst = getelementptr inbounds [60 x i8], [60 x i8]* @a, i32 0, i32 0
  %src = getelementptr inbounds [12 x i8], [12 x i8]* @.str, i32 0, i32 0

; CHECK-NEXT: call void @llvm.memcpy.p0i8.p0i8.i32(i8* align 1 getelementptr inbounds ([60 x i8], [60 x i8]* @a, i32 0, i32 0), i8* align 1 getelementptr inbounds ([12 x i8], [12 x i8]* @.str, i32 0, i32 0), i32 12, i1 false)
; CHECK-NEXT: ret i8* getelementptr inbounds ([60 x i8], [60 x i8]* @a, i32 0, i32 0)
  %ret = call i8* @__strcpy_chk(i8* %dst, i8* %src, i32 -1)
  ret i8* %ret
}

; Check cases where there are no string constants.

define i8* @test_simplify4() {
; CHECK-LABEL: @test_simplify4(
  %dst = getelementptr inbounds [60 x i8], [60 x i8]* @a, i32 0, i32 0
  %src = getelementptr inbounds [60 x i8], [60 x i8]* @b, i32 0, i32 0

; CHECK-NEXT: %strcpy = call i8* @strcpy(i8* getelementptr inbounds ([60 x i8], [60 x i8]* @a, i32 0, i32 0), i8* getelementptr inbounds ([60 x i8], [60 x i8]* @b, i32 0, i32 0))
; CHECK-NEXT: ret i8* %strcpy
  %ret = call i8* @__strcpy_chk(i8* %dst, i8* %src, i32 -1)
  ret i8* %ret
}

; Check case where the string length is not constant.

define i8* @test_simplify5() {
; CHECK-LABEL: @test_simplify5(
  %dst = getelementptr inbounds [60 x i8], [60 x i8]* @a, i32 0, i32 0
  %src = getelementptr inbounds [12 x i8], [12 x i8]* @.str, i32 0, i32 0

; CHECK-NEXT: %len = call i32 @llvm.objectsize.i32.p0i8(i8* getelementptr inbounds ([60 x i8], [60 x i8]* @a, i32 0, i32 0), i1 false)
; CHECK-NEXT: %1 = call i8* @__memcpy_chk(i8* getelementptr inbounds ([60 x i8], [60 x i8]* @a, i32 0, i32 0), i8* getelementptr inbounds ([12 x i8], [12 x i8]* @.str, i32 0, i32 0), i32 12, i32 %len)
; CHECK-NEXT: ret i8* %1
  %len = call i32 @llvm.objectsize.i32.p0i8(i8* %dst, i1 false)
  %ret = call i8* @__strcpy_chk(i8* %dst, i8* %src, i32 %len)
  ret i8* %ret
}

; Check case where the source and destination are the same.

define i8* @test_simplify6() {
; CHECK-LABEL: @test_simplify6(
  %dst = getelementptr inbounds [60 x i8], [60 x i8]* @a, i32 0, i32 0

; CHECK-NEXT: %len = call i32 @llvm.objectsize.i32.p0i8(i8* getelementptr inbounds ([60 x i8], [60 x i8]* @a, i32 0, i32 0), i1 false)
; CHECK-NEXT: %ret = call i8* @__strcpy_chk(i8* getelementptr inbounds ([60 x i8], [60 x i8]* @a, i32 0, i32 0), i8* getelementptr inbounds ([60 x i8], [60 x i8]* @a, i32 0, i32 0), i32 %len)
; CHECK-NEXT: ret i8* %ret
  %len = call i32 @llvm.objectsize.i32.p0i8(i8* %dst, i1 false)
  %ret = call i8* @__strcpy_chk(i8* %dst, i8* %dst, i32 %len)
  ret i8* %ret
}

; Check case where slen < strlen (src).

define i8* @test_no_simplify1() {
; CHECK-LABEL: @test_no_simplify1(
  %dst = getelementptr inbounds [60 x i8], [60 x i8]* @a, i32 0, i32 0
  %src = getelementptr inbounds [60 x i8], [60 x i8]* @b, i32 0, i32 0

; CHECK-NEXT: %ret = call i8* @__strcpy_chk(i8* getelementptr inbounds ([60 x i8], [60 x i8]* @a, i32 0, i32 0), i8* getelementptr inbounds ([60 x i8], [60 x i8]* @b, i32 0, i32 0), i32 8)
; CHECK-NEXT: ret i8* %ret
  %ret = call i8* @__strcpy_chk(i8* %dst, i8* %src, i32 8)
  ret i8* %ret
}

declare i8* @__strcpy_chk(i8*, i8*, i32) nounwind
declare i32 @llvm.objectsize.i32.p0i8(i8*, i1) nounwind readonly
