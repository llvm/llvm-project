; RUN: llc < %s -O3 -mtriple=thumbv7-apple-ios -mcpu=cortex-a8 | FileCheck %s
; Formerly crashed, 3573915.

define void @RotateStarsFP_Vec() nounwind {
bb.nph372:
  br label %bb8

bb8:                                              ; preds = %bb8, %bb.nph372
  %0 = fadd <4 x float> undef, <float 0xBFEE353F80000000, float 0xBFEE353F80000000, float 0xBFEE353F80000000, float 0xBFEE353F80000000>
  %1 = fmul <4 x float> %0, undef
  %2 = fmul <4 x float> %1, undef
  %3 = fadd <4 x float> undef, %2
  store <4 x float> %3, <4 x float>* undef, align 4
  br label %bb8
; CHECK: RotateStarsFP_Vec:
; CHECK: vld1.64
}
