; RUN: opt < %s -instcombine -S | FileCheck %s
;
; This test makes sure that InstCombine does not replace the sequence of
; xor/sub instruction followed by cmp instruction into a single cmp instruction
; if there is more than one use of xor/sub.

define zeroext i1 @test1(i32 %lhs, i32 %rhs) {
; CHECK-LABEL: @test1(
; CHECK-NEXT: %xor = xor i32 %lhs, 5
; CHECK-NEXT: %cmp1 = icmp eq i32 %xor, 10

  %xor = xor i32 %lhs, 5
  %cmp1 = icmp eq i32 %xor, 10
  %cmp2 = icmp eq i32 %xor, %rhs
  %sel = or i1 %cmp1, %cmp2
  ret i1 %sel
}

define zeroext i1 @test2(i32 %lhs, i32 %rhs) {
; CHECK-LABEL: @test2(
; CHECK-NEXT: %xor = xor i32 %lhs, %rhs
; CHECK-NEXT: %cmp1 = icmp eq i32 %xor, 0

  %xor = xor i32 %lhs, %rhs
  %cmp1 = icmp eq i32 %xor, 0
  %cmp2 = icmp eq i32 %xor, 32
  %sel = xor i1 %cmp1, %cmp2
  ret i1 %sel
}

define zeroext i1 @test3(i32 %lhs, i32 %rhs) {
; CHECK-LABEL: @test3(
; CHECK-NEXT: %sub = sub nsw i32 %lhs, %rhs
; CHECK-NEXT: %cmp1 = icmp eq i32 %sub, 0

  %sub = sub nsw i32 %lhs, %rhs
  %cmp1 = icmp eq i32 %sub, 0
  %cmp2 = icmp eq i32 %sub, 31
  %sel = or i1 %cmp1, %cmp2
  ret i1 %sel
}
