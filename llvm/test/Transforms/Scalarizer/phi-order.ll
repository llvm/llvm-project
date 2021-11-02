; RUN: opt %s -passes='function(scalarizer)' -S -o - | FileCheck %s

; This input caused the scalarizer to insert non-PHI nodes
; in between PHI nodes (%1 and %2).

define <3 x float> @func(i32 %inval) {
.entry:
  br label %0

0:                                                ; preds = %3, %.entry
; CHECK: %.i01 = phi float [ 0.000000e+00, %.entry ], [ %.i01, %3 ]
; CHECK-NEXT: %.i12 = phi float [ 0.000000e+00, %.entry ], [ %.i12, %3 ]
; CHECK-NEXT: %.i23 = phi float [ 0.000000e+00, %.entry ], [ %.i23, %3 ]
; CHECK-NEXT: %1 = phi float [ 1.000000e+00, %.entry ], [ 2.000000e+00, %3 ]
; CHECK-NEXT: %.upto0 = insertelement <3 x float> poison, float %.i01, i32 0
; CHECK-NEXT: %.upto1 = insertelement <3 x float> %.upto0, float %.i12, i32 1
; CHECK-NEXT: %2 = insertelement <3 x float> %.upto1, float %.i23, i32 2
  %1 = phi <3 x float> [ <float 0.0, float 0.0, float 0.0>, %.entry], [ %1, %3 ]
  %2 = phi float [ 1.0, %.entry], [ 2.0, %3 ]
  br label %3

3:                                                ; preds = %0
  %cond = icmp eq i32 %inval, 0
  br i1 %cond, label %0, label %exit

exit:
  ret <3 x float>  %1
}

