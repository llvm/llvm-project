; Test that the isdigit library call simplifier works correctly.
;
; RUN: opt < %s -passes=instcombine -S | FileCheck %s

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"

declare i32 @isdigit(i32)

; Check isdigit(c) -> (c - '0') <u 10;

define i32 @test_simplify1() {
; CHECK-LABEL: @test_simplify1(
  %ret = call i32 @isdigit(i32 47)
  ret i32 %ret
; CHECK-NEXT: ret i32 0
}

define i32 @test_simplify2() {
; CHECK-LABEL: @test_simplify2(
  %ret = call i32 @isdigit(i32 48)
  ret i32 %ret
; CHECK-NEXT: ret i32 1
}

define i32 @test_simplify3() {
; CHECK-LABEL: @test_simplify3(
  %ret = call i32 @isdigit(i32 57)
  ret i32 %ret
; CHECK-NEXT: ret i32 1
}

define i32 @test_simplify4() {
; CHECK-LABEL: @test_simplify4(
  %ret = call i32 @isdigit(i32 58)
  ret i32 %ret
; CHECK-NEXT: ret i32 0
}

define i32 @test_simplify5(i32 %x) {
; CHECK-LABEL: @test_simplify5(

  %ret = call i32 @isdigit(i32 %x)
; CHECK-NEXT: [[ADD:%[a-z0-9]+]] = add i32 %x, -48
; CHECK-NEXT: [[CMP:%[a-z0-9]+]] = icmp ult i32 [[ADD]], 10
; CHECK-NEXT: [[ZEXT:%[a-z0-9]+]] = zext i1 [[CMP]] to i32
  ret i32 %ret
; CHECK-NEXT: ret i32 [[ZEXT]]
}
