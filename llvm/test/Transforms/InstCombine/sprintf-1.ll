; Test that the sprintf library call simplifier works correctly.
;
; RUN: opt < %s -instcombine -S | FileCheck %s
; RUN: opt < %s -mtriple xcore-xmos-elf -instcombine -S | FileCheck %s -check-prefix=CHECK-IPRINTF

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"

@hello_world = constant [13 x i8] c"hello world\0A\00"
@null = constant [1 x i8] zeroinitializer
@null_hello = constant [7 x i8] c"\00hello\00"
@h = constant [2 x i8] c"h\00"
@percent_c = constant [3 x i8] c"%c\00"
@percent_d = constant [3 x i8] c"%d\00"
@percent_f = constant [3 x i8] c"%f\00"
@percent_s = constant [3 x i8] c"%s\00"

declare i32 @sprintf(i8*, i8*, ...)

; Check sprintf(dst, fmt) -> llvm.memcpy(str, fmt, strlen(fmt) + 1, 1).

define void @test_simplify1(i8* %dst) {
; CHECK-LABEL: @test_simplify1(
  %fmt = getelementptr [13 x i8], [13 x i8]* @hello_world, i32 0, i32 0
  call i32 (i8*, i8*, ...) @sprintf(i8* %dst, i8* %fmt)
; CHECK-NEXT: call void @llvm.memcpy.p0i8.p0i8.i32(i8* align 1 %dst, i8* align 1 getelementptr inbounds ([13 x i8], [13 x i8]* @hello_world, i32 0, i32 0), i32 13, i1 false)
  ret void
; CHECK-NEXT: ret void
}

define void @test_simplify2(i8* %dst) {
; CHECK-LABEL: @test_simplify2(
  %fmt = getelementptr [1 x i8], [1 x i8]* @null, i32 0, i32 0
  call i32 (i8*, i8*, ...) @sprintf(i8* %dst, i8* %fmt)
; CHECK-NEXT: store i8 0, i8* %dst, align 1
  ret void
; CHECK-NEXT: ret void
}

define void @test_simplify3(i8* %dst) {
; CHECK-LABEL: @test_simplify3(
  %fmt = getelementptr [7 x i8], [7 x i8]* @null_hello, i32 0, i32 0
  call i32 (i8*, i8*, ...) @sprintf(i8* %dst, i8* %fmt)
; CHECK-NEXT: store i8 0, i8* %dst, align 1
  ret void
; CHECK-NEXT: ret void
}

; Check sprintf(dst, "%c", chr) -> *(i8*)dst = chr; *((i8*)dst + 1) = 0.

define void @test_simplify4(i8* %dst) {
; CHECK-LABEL: @test_simplify4(
  %fmt = getelementptr [3 x i8], [3 x i8]* @percent_c, i32 0, i32 0
  call i32 (i8*, i8*, ...) @sprintf(i8* %dst, i8* %fmt, i8 104)
; CHECK-NEXT: store i8 104, i8* %dst, align 1
; CHECK-NEXT: [[NUL:%[a-z0-9]+]] = getelementptr i8, i8* %dst, i32 1
; CHECK-NEXT: store i8 0, i8* [[NUL]], align 1
  ret void
; CHECK-NEXT: ret void
}

; Check sprintf(dst, "%s", str) -> llvm.memcpy(dest, str, strlen(str) + 1, 1).

define void @test_simplify5(i8* %dst, i8* %str) {
; CHECK-LABEL: @test_simplify5(
  %fmt = getelementptr [3 x i8], [3 x i8]* @percent_s, i32 0, i32 0
  call i32 (i8*, i8*, ...) @sprintf(i8* %dst, i8* %fmt, i8* %str)
; CHECK-NEXT: [[STRLEN:%[a-z0-9]+]] = call i32 @strlen(i8* %str)
; CHECK-NEXT: [[LENINC:%[a-z0-9]+]] = add i32 [[STRLEN]], 1
; CHECK-NEXT: call void @llvm.memcpy.p0i8.p0i8.i32(i8* align 1 %dst, i8* align 1 %str, i32 [[LENINC]], i1 false)
  ret void
; CHECK-NEXT: ret void
}

; Check sprintf(dst, format, ...) -> siprintf(str, format, ...) if no floating.

define void @test_simplify6(i8* %dst) {
; CHECK-IPRINTF-LABEL: @test_simplify6(
  %fmt = getelementptr [3 x i8], [3 x i8]* @percent_d, i32 0, i32 0
  call i32 (i8*, i8*, ...) @sprintf(i8* %dst, i8* %fmt, i32 187)
; CHECK-IPRINTF-NEXT: call i32 (i8*, i8*, ...) @siprintf(i8* %dst, i8* getelementptr inbounds ([3 x i8], [3 x i8]* @percent_d, i32 0, i32 0), i32 187)
  ret void
; CHECK-IPRINTF-NEXT: ret void
}

define void @test_no_simplify1(i8* %dst) {
; CHECK-IPRINTF-LABEL: @test_no_simplify1(
  %fmt = getelementptr [3 x i8], [3 x i8]* @percent_f, i32 0, i32 0
  call i32 (i8*, i8*, ...) @sprintf(i8* %dst, i8* %fmt, double 1.87)
; CHECK-IPRINTF-NEXT: call i32 (i8*, i8*, ...) @sprintf(i8* %dst, i8* getelementptr inbounds ([3 x i8], [3 x i8]* @percent_f, i32 0, i32 0), double 1.870000e+00)
  ret void
; CHECK-IPRINTF-NEXT: ret void
}

define void @test_no_simplify2(i8* %dst, i8* %fmt, double %d) {
; CHECK-LABEL: @test_no_simplify2(
  call i32 (i8*, i8*, ...) @sprintf(i8* %dst, i8* %fmt, double %d)
; CHECK-NEXT: call i32 (i8*, i8*, ...) @sprintf(i8* %dst, i8* %fmt, double %d)
  ret void
; CHECK-NEXT: ret void
}
