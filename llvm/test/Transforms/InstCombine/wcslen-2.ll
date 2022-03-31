; Test that the wcslen library call simplifier works correctly.
;
; RUN: opt < %s -passes=instcombine -S | FileCheck %s

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"

!0 = !{i32 1, !"wchar_size", i32 4}
!llvm.module.flags = !{!0}

@hello = constant [6 x i32] [i32 104, i32 101, i32 108, i32 108, i32 111, i32 0]

declare i64 @wcslen(i32*, i32)

define i64 @test_no_simplify1() {
; CHECK-LABEL: @test_no_simplify1(
  %hello_p = getelementptr [6 x i32], [6 x i32]* @hello, i64 0, i64 0
  %hello_l = call i64 @wcslen(i32* %hello_p, i32 187)
; CHECK-NEXT: %hello_l = call i64 @wcslen
  ret i64 %hello_l
; CHECK-NEXT: ret i64 %hello_l
}
