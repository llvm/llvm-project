; RUN: opt < %s -loop-vectorize -S | FileCheck %s

; There could be more than one LCSSA PHIs in loop exit block.

; CHECK-LABEL: bb1.bb3_crit_edge:
; CHECK: %_tmp133.lcssa1 = phi i16 [ %scalar.recur, %bb2 ], [ %vector.recur.extract.for.phi, %middle.block ]
; CHECK: %_tmp133.lcssa = phi i16 [ %scalar.recur, %bb2 ], [ %vector.recur.extract.for.phi, %middle.block ]

define void @f1() {
bb2.lr.ph:
  br label %bb2

bb2:                                              ; preds = %bb2, %bb2.lr.ph
  %_tmp132 = phi i16 [ 0, %bb2.lr.ph ], [ %_tmp10, %bb2 ]
  %_tmp133 = phi i16 [ undef, %bb2.lr.ph ], [ %_tmp10, %bb2 ]
  %_tmp10 = sub nsw i16 %_tmp132, 1
  %_tmp15 = icmp ne i16 %_tmp10, 0
  br i1 %_tmp15, label %bb2, label %bb1.bb3_crit_edge

bb1.bb3_crit_edge:                                ; preds = %bb2
  %_tmp133.lcssa1 = phi i16 [ %_tmp133, %bb2 ]
  %_tmp133.lcssa = phi i16 [ %_tmp133, %bb2 ]
  ret void
}
