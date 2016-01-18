; RUN: llc -verify-machineinstrs -o - %s -mtriple=arm64-apple-ios7.0 | FileCheck %s

define i64 @test_free_zext(i8* %a, i16* %b) {
; CHECK-LABEL: test_free_zext
; CHECK-DAG: ldrb w[[A:[0-9]+]], [x0]
; CHECK: ldrh w[[B:[0-9]+]], [x1]
; CHECK: add x0, x[[B]], x[[A]]
  %1 = load i8, i8* %a, align 1
  %conv = zext i8 %1 to i64
  %2 = load i16, i16* %b, align 2
  %conv1 = zext i16 %2 to i64
  %add = add nsw i64 %conv1, %conv
  ret i64 %add
}
