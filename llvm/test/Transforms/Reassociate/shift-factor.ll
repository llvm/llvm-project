; RUN: opt < %s -reassociate -instcombine -S | FileCheck %s

; There should be exactly one shift and one add left.

define i32 @test1(i32 %X, i32 %Y) {
; CHECK-LABEL: @test1(
; CHECK-NEXT:    [[REASS_ADD:%.*]] = add i32 %Y, %X
; CHECK-NEXT:    [[REASS_MUL:%.*]] = shl i32 [[REASS_ADD]], 1
; CHECK-NEXT:    ret i32 [[REASS_MUL]]
;
  %t2 = shl i32 %X, 1
  %t6 = shl i32 %Y, 1
  %t4 = add i32 %t6, %t2
  ret i32 %t4
}

