; Test that the toascii library call simplifier works correctly.
;
; RUN: opt < %s -instcombine -S | FileCheck %s

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"

declare i32 @toascii(i32)

; Check isascii(c) -> c & 0x7f.

define i32 @test_simplify1() {
; CHECK-LABEL: @test_simplify1(
  %ret = call i32 @toascii(i32 0)
  ret i32 %ret
; CHECK-NEXT: ret i32 0
}

define i32 @test_simplify2() {
; CHECK-LABEL: @test_simplify2(
  %ret = call i32 @toascii(i32 1)
  ret i32 %ret
; CHECK-NEXT: ret i32 1
}

define i32 @test_simplify3() {
; CHECK-LABEL: @test_simplify3(
  %ret = call i32 @toascii(i32 127)
  ret i32 %ret
; CHECK-NEXT: ret i32 127
}

define i32 @test_simplify4() {
; CHECK-LABEL: @test_simplify4(
  %ret = call i32 @toascii(i32 128)
  ret i32 %ret
; CHECK-NEXT: ret i32 0
}

define i32 @test_simplify5() {
; CHECK-LABEL: @test_simplify5(
  %ret = call i32 @toascii(i32 255)
  ret i32 %ret
; CHECK-NEXT: ret i32 127
}

define i32 @test_simplify6() {
; CHECK-LABEL: @test_simplify6(
  %ret = call i32 @toascii(i32 256)
  ret i32 %ret
; CHECK-NEXT: ret i32 0
}

define i32 @test_simplify7(i32 %x) {
; CHECK-LABEL: @test_simplify7(
  %ret = call i32 @toascii(i32 %x)
; CHECK-NEXT: [[AND:%[a-z0-9]+]] = and i32 %x, 127
  ret i32 %ret
; CHECK-NEXT: ret i32 [[AND]]
}
