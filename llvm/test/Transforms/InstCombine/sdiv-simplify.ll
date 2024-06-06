; RUN: opt < %s -passes=instcombine -S | FileCheck %s

define i32 @sdiv_to_udiv(i32 %arg0, i32 %arg1) {
; CHECK-LABEL: @sdiv_to_udiv(
; CHECK-NEXT:    [[T0:%.*]] = shl nuw nsw i32 [[ARG0:%.*]], 8
; CHECK-NEXT:    [[T2:%.*]] = add nuw nsw i32 [[T0:%.*]], 6242049
; CHECK-NEXT:    [[T3:%.*]] = udiv i32 [[T2]], 192
; CHECK-NEXT:    ret i32 [[T3]]
;
  %t0 = shl nuw nsw i32 %arg0, 8
  %t1 = or disjoint i32 %t0, 1
  %t2 = add nuw nsw i32 %t1, 6242048
  %t3 = sdiv i32 %t2, 192
  ret i32 %t3
}
