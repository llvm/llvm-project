; RUN: opt -S -instcombine < %s | FileCheck %s

@g = external global i32

define i1 @test(i32 %other) {
; CHECK-LABEL: @test
; CHECK: %test = icmp sgt i32 %other, 0
  %positive = load i32, i32* @g, !range !{i32 1, i32 2048}
  %cmp = icmp slt i32 %positive, %other
  %sel = select i1 %cmp, i32 %positive, i32 %other
  %test = icmp sgt i32 %sel, 0
  ret i1 %test
}

define i1 @test2(i32 %other) {
; CHECK-LABEL: @test2
; CHECK: %test = icmp sgt i32 %other, 0
  %positive = load i32, i32* @g, !range !{i32 1, i32 2048}
  %cmp = icmp slt i32 %other, %positive
  %sel = select i1 %cmp, i32 %other, i32 %positive
  %test = icmp sgt i32 %sel, 0
  ret i1 %test
}

; %positive might be zero
define i1 @test3(i32 %other) {
; CHECK-LABEL: @test3
; CHECK: %test = icmp sgt i32 %sel, 0
  %positive = load i32, i32* @g, !range !{i32 0, i32 2048}
  %cmp = icmp slt i32 %positive, %other
  %sel = select i1 %cmp, i32 %positive, i32 %other
  %test = icmp sgt i32 %sel, 0
  ret i1 %test
}
