; RUN: opt < %s -instcombine -S | FileCheck %s

; With left shift, the comparison should not be modified.
; CHECK-LABEL: @test_shift_and_cmp_not_changed1(
; CHECK: icmp slt i8 %andp, 32
define i1 @test_shift_and_cmp_not_changed1(i8 %p) #0 {
entry:
  %shlp = shl i8 %p, 5
  %andp = and i8 %shlp, -64
  %cmp = icmp slt i8 %andp, 32
  ret i1 %cmp
}

; With arithmetic right shift, the comparison should not be modified.
; CHECK-LABEL: @test_shift_and_cmp_not_changed2(
; CHECK: icmp slt i8 %andp, 32
define i1 @test_shift_and_cmp_not_changed2(i8 %p) #0 {
entry:
  %shlp = ashr i8 %p, 5
  %andp = and i8 %shlp, -64
  %cmp = icmp slt i8 %andp, 32
  ret i1 %cmp
}

; This should simplify functionally to the left shift case.
; The extra input parameter should be optimized away.
; CHECK-LABEL: @test_shift_and_cmp_changed1(
; CHECK:  %andp = shl i8 %p, 5
; CHECK-NEXT: %shl = and i8 %andp, -64
; CHECK-NEXT:  %cmp = icmp slt i8 %shl, 32
define i1 @test_shift_and_cmp_changed1(i8 %p, i8 %q) #0 {
entry:
  %andp = and i8 %p, 6
  %andq = and i8 %q, 8
  %or = or i8 %andq, %andp
  %shl = shl i8 %or, 5
  %ashr = ashr i8 %shl, 5
  %cmp = icmp slt i8 %ashr, 1
  ret i1 %cmp
}

; Unsigned compare allows a transformation to compare against 0.
; CHECK-LABEL: @test_shift_and_cmp_changed2(
; CHECK: icmp eq i8 %andp, 0
define i1 @test_shift_and_cmp_changed2(i8 %p) #0 {
entry:
  %shlp = shl i8 %p, 5
  %andp = and i8 %shlp, -64
  %cmp = icmp ult i8 %andp, 32
  ret i1 %cmp
}

; nsw on the shift should not affect the comparison.
; CHECK-LABEL: @test_shift_and_cmp_changed3(
; CHECK: icmp slt i8 %andp, 32
define i1 @test_shift_and_cmp_changed3(i8 %p) #0 {
entry:
  %shlp = shl nsw i8 %p, 5
  %andp = and i8 %shlp, -64
  %cmp = icmp slt i8 %andp, 32
  ret i1 %cmp
}

; Logical shift right allows a return true because the 'and' guarantees no bits are set.
; CHECK-LABEL: @test_shift_and_cmp_changed4(
; CHECK: ret i1 true
define i1 @test_shift_and_cmp_changed4(i8 %p) #0 {
entry:
  %shlp = lshr i8 %p, 5
  %andp = and i8 %shlp, -64
  %cmp = icmp slt i8 %andp, 32
  ret i1 %cmp
}

