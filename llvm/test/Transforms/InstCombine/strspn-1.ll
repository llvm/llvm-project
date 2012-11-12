; Test that the strspn library call simplifier works correctly.
;
; RUN: opt < %s -instcombine -S | FileCheck %s

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"

@abcba = constant [6 x i8] c"abcba\00"
@abc = constant [4 x i8] c"abc\00"
@null = constant [1 x i8] zeroinitializer

declare i64 @strspn(i8*, i8*)

; Check strspn(s, "") -> 0.

define i64 @test_simplify1(i8* %str) {
; CHECK: @test_simplify1
  %pat = getelementptr [1 x i8]* @null, i32 0, i32 0

  %ret = call i64 @strspn(i8* %str, i8* %pat)
  ret i64 %ret
; CHECK-NEXT: ret i64 0
}

; Check strspn("", s) -> 0.

define i64 @test_simplify2(i8* %pat) {
; CHECK: @test_simplify2
  %str = getelementptr [1 x i8]* @null, i32 0, i32 0

  %ret = call i64 @strspn(i8* %str, i8* %pat)
  ret i64 %ret
; CHECK-NEXT: ret i64 0
}

; Check strspn(s1, s2), where s1 and s2 are constants.

define i64 @test_simplify3() {
; CHECK: @test_simplify3
  %str = getelementptr [6 x i8]* @abcba, i32 0, i32 0
  %pat = getelementptr [4 x i8]* @abc, i32 0, i32 0

  %ret = call i64 @strspn(i8* %str, i8* %pat)
  ret i64 %ret
; CHECK-NEXT: ret i64 5
}

; Check cases that shouldn't be simplified.

define i64 @test_no_simplify1(i8* %str, i8* %pat) {
; CHECK: @test_no_simplify1

  %ret = call i64 @strspn(i8* %str, i8* %pat)
; CHECK-NEXT: %ret = call i64 @strspn(i8* %str, i8* %pat)
  ret i64 %ret
; CHECK-NEXT: ret i64 %ret
}
