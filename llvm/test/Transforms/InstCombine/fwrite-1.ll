; Test that the fwrite library call simplifier works correctly.
;
; RUN: opt < %s -passes=instcombine -S | FileCheck %s

target datalayout = "e-p:64:64:64"

%FILE = type { }

@str = constant [1 x i8] zeroinitializer
@empty = constant [0 x i8] zeroinitializer

declare i64 @fwrite(ptr, i64, i64, ptr)

; Check fwrite(S, 1, 1, fp) -> fputc(S[0], fp).

define void @test_simplify1(ptr %fp) {
; CHECK-LABEL: @test_simplify1(
  call i64 @fwrite(ptr @str, i64 1, i64 1, ptr %fp)
; CHECK-NEXT: call i32 @fputc(i32 0, ptr %fp)
  ret void
; CHECK-NEXT: ret void
}

define void @test_simplify2(ptr %fp) {
; CHECK-LABEL: @test_simplify2(
  call i64 @fwrite(ptr @empty, i64 1, i64 0, ptr %fp)
  ret void
; CHECK-NEXT: ret void
}

define void @test_simplify3(ptr %fp) {
; CHECK-LABEL: @test_simplify3(
  call i64 @fwrite(ptr @empty, i64 0, i64 1, ptr %fp)
  ret void
; CHECK-NEXT: ret void
}

define i64 @test_no_simplify1(ptr %fp) {
; CHECK-LABEL: @test_no_simplify1(
  %ret = call i64 @fwrite(ptr @str, i64 1, i64 1, ptr %fp)
; CHECK-NEXT: call i64 @fwrite
  ret i64 %ret
; CHECK-NEXT: ret i64 %ret
}

define void @test_no_simplify2(ptr %fp, i64 %size) {
; CHECK-LABEL: @test_no_simplify2(
  call i64 @fwrite(ptr @str, i64 %size, i64 1, ptr %fp)
; CHECK-NEXT: call i64 @fwrite
  ret void
; CHECK-NEXT: ret void
}
