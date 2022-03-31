; Test that the fputs library call simplifier works correctly.
;
; RUN: opt < %s -passes=instcombine -S | FileCheck %s

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"

%FILE = type { }

@empty = constant [1 x i8] zeroinitializer
@A = constant [2 x i8] c"A\00"
@hello = constant [7 x i8] c"hello\0A\00"

declare i32 @fputs(i8*, %FILE*)

; Check fputs(str, fp) --> fwrite(str, strlen(s), 1, fp).

define void @test_simplify1(%FILE* %fp) {
; CHECK-LABEL: @test_simplify1(
  %str = getelementptr [1 x i8], [1 x i8]* @empty, i32 0, i32 0
  call i32 @fputs(i8* %str, %FILE* %fp)
  ret void
; CHECK-NEXT: ret void
}

; NOTE: The fwrite simplifier simplifies this further to fputc.

define void @test_simplify2(%FILE* %fp) {
; CHECK-LABEL: @test_simplify2(
  %str = getelementptr [2 x i8], [2 x i8]* @A, i32 0, i32 0
  call i32 @fputs(i8* %str, %FILE* %fp)
; CHECK-NEXT: call i32 @fputc(i32 65, %FILE* %fp)
  ret void
; CHECK-NEXT: ret void
}

define void @test_simplify3(%FILE* %fp) {
; CHECK-LABEL: @test_simplify3(
  %str = getelementptr [7 x i8], [7 x i8]* @hello, i32 0, i32 0
  call i32 @fputs(i8* %str, %FILE* %fp)
; CHECK-NEXT: call i32 @fwrite(i8* getelementptr inbounds ([7 x i8], [7 x i8]* @hello, i32 0, i32 0), i32 6, i32 1, %FILE* %fp)
  ret void
; CHECK-NEXT: ret void
}
