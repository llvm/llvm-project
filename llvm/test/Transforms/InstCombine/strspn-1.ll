; Test that the strspn library call simplifier works correctly.
;
; RUN: opt < %s -passes=instcombine -S | FileCheck %s

@abcba = constant [6 x i8] c"abcba\00"
@abc = constant [4 x i8] c"abc\00"
@null = constant [1 x i8] zeroinitializer

declare i64 @strspn(ptr, ptr)

; Check strspn(s, "") -> 0.

define i64 @test_simplify1(ptr %str) {
; CHECK-LABEL: @test_simplify1(

  %ret = call i64 @strspn(ptr %str, ptr @null)
  ret i64 %ret
; CHECK-NEXT: ret i64 0
}

; Check strspn("", s) -> 0.

define i64 @test_simplify2(ptr %pat) {
; CHECK-LABEL: @test_simplify2(

  %ret = call i64 @strspn(ptr @null, ptr %pat)
  ret i64 %ret
; CHECK-NEXT: ret i64 0
}

; Check strspn(s1, s2), where s1 and s2 are constants.

define i64 @test_simplify3() {
; CHECK-LABEL: @test_simplify3(

  %ret = call i64 @strspn(ptr @abcba, ptr @abc)
  ret i64 %ret
; CHECK-NEXT: ret i64 5
}

; Check cases that shouldn't be simplified.

define i64 @test_no_simplify1(ptr %str, ptr %pat) {
; CHECK-LABEL: @test_no_simplify1(

  %ret = call i64 @strspn(ptr %str, ptr %pat)
; CHECK-NEXT: %ret = call i64 @strspn(ptr %str, ptr %pat)
  ret i64 %ret
; CHECK-NEXT: ret i64 %ret
}
