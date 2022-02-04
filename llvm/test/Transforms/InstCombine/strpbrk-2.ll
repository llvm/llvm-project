; Test that the strpbrk library call simplifier works correctly.
;
; RUN: opt < %s -passes=instcombine -S | FileCheck %s

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"

@hello = constant [12 x i8] c"hello world\00"
@w = constant [2 x i8] c"w\00"

declare i16* @strpbrk(i8*, i8*)

; Check that 'strpbrk' functions with the wrong prototype aren't simplified.

define i16* @test_no_simplify1() {
; CHECK-LABEL: @test_no_simplify1(
  %str = getelementptr [12 x i8], [12 x i8]* @hello, i32 0, i32 0
  %pat = getelementptr [2 x i8], [2 x i8]* @w, i32 0, i32 0

  %ret = call i16* @strpbrk(i8* %str, i8* %pat)
; CHECK-NEXT: %ret = call i16* @strpbrk
  ret i16* %ret
; CHECK-NEXT: ret i16* %ret
}
