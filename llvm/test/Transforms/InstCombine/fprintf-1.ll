; Test that the fprintf library call simplifier works correctly.
;
; RUN: opt < %s -passes=instcombine -S | FileCheck %s
; RUN: opt < %s -mtriple xcore-xmos-elf -passes=instcombine -S | FileCheck %s -check-prefix=CHECK-IPRINTF

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"

%FILE = type { }

@hello_world = constant [13 x i8] c"hello world\0A\00"
@percent_c = constant [3 x i8] c"%c\00"
@percent_d = constant [3 x i8] c"%d\00"
@percent_f = constant [3 x i8] c"%f\00"
@percent_s = constant [3 x i8] c"%s\00"
@percent_m = constant [3 x i8] c"%m\00"

declare i32 @fprintf(ptr, ptr, ...)

; Check fprintf(fp, "foo") -> fwrite("foo", 3, 1, fp).

define void @test_simplify1(ptr %fp) {
; CHECK-LABEL: @test_simplify1(
  call i32 (ptr, ptr, ...) @fprintf(ptr %fp, ptr @hello_world)
; CHECK-NEXT: call i32 @fwrite(ptr nonnull @hello_world, i32 12, i32 1, ptr %fp)
  ret void
; CHECK-NEXT: ret void
}

; Check fprintf(fp, "%c", chr) -> fputc(chr, fp).

define void @test_simplify2(ptr %fp) {
; CHECK-LABEL: @test_simplify2(
  call i32 (ptr, ptr, ...) @fprintf(ptr %fp, ptr @percent_c, i8 104)
; CHECK-NEXT: call i32 @fputc(i32 104, ptr %fp)
  ret void
; CHECK-NEXT: ret void
}

; Check fprintf(fp, "%s", str) -> fputs(str, fp).
; NOTE: The fputs simplifier simplifies this further to fwrite.

define void @test_simplify3(ptr %fp) {
; CHECK-LABEL: @test_simplify3(
  call i32 (ptr, ptr, ...) @fprintf(ptr %fp, ptr @percent_s, ptr @hello_world)
; CHECK-NEXT: call i32 @fwrite(ptr nonnull @hello_world, i32 12, i32 1, ptr %fp)
  ret void
; CHECK-NEXT: ret void
}

; Check fprintf(fp, fmt, ...) -> fiprintf(fp, fmt, ...) if no floating point.

define void @test_simplify4(ptr %fp) {
; CHECK-IPRINTF-LABEL: @test_simplify4(
  call i32 (ptr, ptr, ...) @fprintf(ptr %fp, ptr @percent_d, i32 187)
; CHECK-IPRINTF-NEXT: call i32 (ptr, ptr, ...) @fiprintf(ptr %fp, ptr nonnull @percent_d, i32 187)
  ret void
; CHECK-IPRINTF-NEXT: ret void
}

define void @test_simplify5(ptr %fp) {
; CHECK-LABEL: @test_simplify5(
  call i32 (ptr, ptr, ...) @fprintf(ptr %fp, ptr @hello_world) [ "deopt"() ]
; CHECK-NEXT: call i32 @fwrite(ptr nonnull @hello_world, i32 12, i32 1, ptr %fp) [ "deopt"() ]
  ret void
; CHECK-NEXT: ret void
}

define void @test_no_simplify1(ptr %fp) {
; CHECK-IPRINTF-LABEL: @test_no_simplify1(
  call i32 (ptr, ptr, ...) @fprintf(ptr %fp, ptr @percent_f, double 1.87)
; CHECK-IPRINTF-NEXT: call i32 (ptr, ptr, ...) @fprintf(ptr %fp, ptr nonnull @percent_f, double 1.870000e+00)
  ret void
; CHECK-IPRINTF-NEXT: ret void
}

define void @test_no_simplify2(ptr %fp, double %d) {
; CHECK-LABEL: @test_no_simplify2(
  call i32 (ptr, ptr, ...) @fprintf(ptr %fp, ptr @percent_f, double %d)
; CHECK-NEXT: call i32 (ptr, ptr, ...) @fprintf(ptr %fp, ptr nonnull @percent_f, double %d)
  ret void
; CHECK-NEXT: ret void
}

define i32 @test_no_simplify3(ptr %fp) {
; CHECK-LABEL: @test_no_simplify3(
  %1 = call i32 (ptr, ptr, ...) @fprintf(ptr %fp, ptr @hello_world)
; CHECK-NEXT: call i32 (ptr, ptr, ...) @fprintf(ptr %fp, ptr nonnull @hello_world)
  ret i32 %1
; CHECK-NEXT: ret i32 %1
}

; Verify that a call with a format string containing just the %m directive
; and no arguments is not simplified.

define void @test_no_simplify4(ptr %fp) {
; CHECK-LABEL: @test_no_simplify4(
  call i32 (ptr, ptr, ...) @fprintf(ptr %fp, ptr @percent_m)
; CHECK-NEXT: call i32 (ptr, ptr, ...) @fprintf(ptr %fp, ptr nonnull @percent_m)
  ret void
; CHECK-NEXT: ret void
}
