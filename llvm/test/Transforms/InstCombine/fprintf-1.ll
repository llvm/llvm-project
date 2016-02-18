; Test that the fprintf library call simplifier works correctly.
;
; RUN: opt < %s -instcombine -S | FileCheck %s
; RUN: opt < %s -mtriple xcore-xmos-elf -instcombine -S | FileCheck %s -check-prefix=CHECK-IPRINTF

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"

%FILE = type { }

@hello_world = constant [13 x i8] c"hello world\0A\00"
@percent_c = constant [3 x i8] c"%c\00"
@percent_d = constant [3 x i8] c"%d\00"
@percent_f = constant [3 x i8] c"%f\00"
@percent_s = constant [3 x i8] c"%s\00"

declare i32 @fprintf(%FILE*, i8*, ...)

; Check fprintf(fp, "foo") -> fwrite("foo", 3, 1, fp).

define void @test_simplify1(%FILE* %fp) {
; CHECK-LABEL: @test_simplify1(
  %fmt = getelementptr [13 x i8], [13 x i8]* @hello_world, i32 0, i32 0
  call i32 (%FILE*, i8*, ...) @fprintf(%FILE* %fp, i8* %fmt)
; CHECK-NEXT: call i32 @fwrite(i8* getelementptr inbounds ([13 x i8], [13 x i8]* @hello_world, i32 0, i32 0), i32 12, i32 1, %FILE* %fp)
  ret void
; CHECK-NEXT: ret void
}

; Check fprintf(fp, "%c", chr) -> fputc(chr, fp).

define void @test_simplify2(%FILE* %fp) {
; CHECK-LABEL: @test_simplify2(
  %fmt = getelementptr [3 x i8], [3 x i8]* @percent_c, i32 0, i32 0
  call i32 (%FILE*, i8*, ...) @fprintf(%FILE* %fp, i8* %fmt, i8 104)
; CHECK-NEXT: call i32 @fputc(i32 104, %FILE* %fp)
  ret void
; CHECK-NEXT: ret void
}

; Check fprintf(fp, "%s", str) -> fputs(str, fp).
; NOTE: The fputs simplifier simplifies this further to fwrite.

define void @test_simplify3(%FILE* %fp) {
; CHECK-LABEL: @test_simplify3(
  %fmt = getelementptr [3 x i8], [3 x i8]* @percent_s, i32 0, i32 0
  %str = getelementptr [13 x i8], [13 x i8]* @hello_world, i32 0, i32 0
  call i32 (%FILE*, i8*, ...) @fprintf(%FILE* %fp, i8* %fmt, i8* %str)
; CHECK-NEXT: call i32 @fwrite(i8* getelementptr inbounds ([13 x i8], [13 x i8]* @hello_world, i32 0, i32 0), i32 12, i32 1, %FILE* %fp)
  ret void
; CHECK-NEXT: ret void
}

; Check fprintf(fp, fmt, ...) -> fiprintf(fp, fmt, ...) if no floating point.

define void @test_simplify4(%FILE* %fp) {
; CHECK-IPRINTF-LABEL: @test_simplify4(
  %fmt = getelementptr [3 x i8], [3 x i8]* @percent_d, i32 0, i32 0
  call i32 (%FILE*, i8*, ...) @fprintf(%FILE* %fp, i8* %fmt, i32 187)
; CHECK-IPRINTF-NEXT: call i32 (%FILE*, i8*, ...) @fiprintf(%FILE* %fp, i8* getelementptr inbounds ([3 x i8], [3 x i8]* @percent_d, i32 0, i32 0), i32 187)
  ret void
; CHECK-IPRINTF-NEXT: ret void
}

define void @test_simplify5(%FILE* %fp) {
; CHECK-LABEL: @test_simplify5(
  %fmt = getelementptr [13 x i8], [13 x i8]* @hello_world, i32 0, i32 0
  call i32 (%FILE*, i8*, ...) @fprintf(%FILE* %fp, i8* %fmt) [ "deopt"() ]
; CHECK-NEXT: call i32 @fwrite(i8* getelementptr inbounds ([13 x i8], [13 x i8]* @hello_world, i32 0, i32 0), i32 12, i32 1, %FILE* %fp) [ "deopt"() ]
  ret void
; CHECK-NEXT: ret void
}

define void @test_no_simplify1(%FILE* %fp) {
; CHECK-IPRINTF-LABEL: @test_no_simplify1(
  %fmt = getelementptr [3 x i8], [3 x i8]* @percent_f, i32 0, i32 0
  call i32 (%FILE*, i8*, ...) @fprintf(%FILE* %fp, i8* %fmt, double 1.87)
; CHECK-IPRINTF-NEXT: call i32 (%FILE*, i8*, ...) @fprintf(%FILE* %fp, i8* getelementptr inbounds ([3 x i8], [3 x i8]* @percent_f, i32 0, i32 0), double 1.870000e+00)
  ret void
; CHECK-IPRINTF-NEXT: ret void
}

define void @test_no_simplify2(%FILE* %fp, double %d) {
; CHECK-LABEL: @test_no_simplify2(
  %fmt = getelementptr [3 x i8], [3 x i8]* @percent_f, i32 0, i32 0
  call i32 (%FILE*, i8*, ...) @fprintf(%FILE* %fp, i8* %fmt, double %d)
; CHECK-NEXT: call i32 (%FILE*, i8*, ...) @fprintf(%FILE* %fp, i8* getelementptr inbounds ([3 x i8], [3 x i8]* @percent_f, i32 0, i32 0), double %d)
  ret void
; CHECK-NEXT: ret void
}

define i32 @test_no_simplify3(%FILE* %fp) {
; CHECK-LABEL: @test_no_simplify3(
  %fmt = getelementptr [13 x i8], [13 x i8]* @hello_world, i32 0, i32 0
  %1 = call i32 (%FILE*, i8*, ...) @fprintf(%FILE* %fp, i8* %fmt)
; CHECK-NEXT: call i32 (%FILE*, i8*, ...) @fprintf(%FILE* %fp, i8* getelementptr inbounds ([13 x i8], [13 x i8]* @hello_world, i32 0, i32 0))
  ret i32 %1
; CHECK-NEXT: ret i32 %1
}
