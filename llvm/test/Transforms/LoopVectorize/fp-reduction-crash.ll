; REQUIRES: asserts
; RUN: opt < %s -loop-vectorize -S | FileCheck %s

; CHECK-LABEL: quux
define void @quux() {
bb:
  br label %bb4

bb1:                                              ; preds = %bb4
  %tmp = phi double [ %tmp6, %bb4 ]
  br i1 undef, label %bb4, label %bb2

bb2:                                              ; preds = %bb1
  %tmp3 = phi double [ %tmp, %bb1 ]
  ret void

bb4:                                              ; preds = %bb1, %bb
  %tmp5 = phi double [ 1.300000e+01, %bb ], [ %tmp, %bb1 ]
  %tmp6 = fadd double %tmp5, 1.000000e+00
  br label %bb1
}
