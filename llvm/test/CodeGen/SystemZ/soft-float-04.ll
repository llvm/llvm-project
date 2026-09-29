; RUN: llc -mtriple=s390x-linux-gnu -mcpu=z14 -O3 -mattr=soft-float < %s  | FileCheck %s
;
; Check that this function with soft-float does not result in a s390.tdc
; intrinsic (which cannot be handled by SoftenFloatOperand).

define void @fun(float %arg) {
; CHECK-LABEL: fun:
; CHECK: cijl
bb:
  %tmp = bitcast float %arg to i32
  br label %bb1

bb1:                                              ; preds = %bb
  %tmp2 = icmp sgt i32 %tmp, -1
  br i1 %tmp2, label %bb3, label %bb4

bb3:                                              ; preds = %bb1
  unreachable

bb4:                                              ; preds = %bb1
  unreachable
}
