; Test that the wcslen library call simplifier works correctly.
;
; RUN: opt < %s -instcombine -S | FileCheck %s

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"

; Without the wchar_size metadata we should see no optimization happening.

@hello = constant [6 x i32] [i32 104, i32 101, i32 108, i32 108, i32 111, i32 0]

declare i64 @wcslen(i32*)

define i64 @test_no_simplify1() {
; CHECK-LABEL: @test_no_simplify1(
; CHECK-NEXT: %hello_l = call i64 @wcslen(i32* getelementptr inbounds ([6 x i32], [6 x i32]* @hello, i64 0, i64 0))
; CHECK-NEXT: ret i64 %hello_l
  %hello_p = getelementptr [6 x i32], [6 x i32]* @hello, i64 0, i64 0
  %hello_l = call i64 @wcslen(i32* %hello_p)
  ret i64 %hello_l
}
