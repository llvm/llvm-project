; RUN: opt < %s -instcombine -S | FileCheck %s

; Result of left shifting a non-negative integer 
; with nsw flag should also be non-negative
define i1 @test_shift_nonnegative(i32 %a) {
; CHECK-LABEL: @test_shift_nonnegative(
; CHECK: ret i1 true
  %b = lshr i32 %a, 2
  %shift = shl nsw i32 %b, 3
  %cmp = icmp sge i32 %shift, 0
  ret i1 %cmp
}

; Result of left shifting a negative integer with
; nsw flag should also be negative
define i1 @test_shift_negative(i32 %a, i32 %b) {
; CHECK-LABEL: @test_shift_negative(
; CHECK: ret i1 true
  %c = or i32 %a, -2147483648
  %d = and i32 %b, 7
  %shift = shl nsw i32 %c, %d
  %cmp = icmp slt i32 %shift, 0
  ret i1 %cmp
}
