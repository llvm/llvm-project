; RUN: llc < %s -mtriple=ve | FileCheck %s

; Function Attrs: norecurse nounwind readnone
define signext i32 @br_jt(i32 signext %0) {
; CHECK-LABEL: br_jt:
; CHECK:       .LBB{{[0-9]+}}_11:
; CHECK-NEXT:    adds.w.sx %s0, %s0, (0)1
; CHECK-NEXT:    brlt.w 2, %s0, .LBB{{[0-9]+}}_3
; CHECK-NEXT:  # %bb.1:
; CHECK-NEXT:    breq.w 1, %s0, .LBB{{[0-9]+}}_2
; CHECK-NEXT:  # %bb.8:
; CHECK-NEXT:    brne.w 2, %s0, .LBB{{[0-9]+}}_9
; CHECK-NEXT:  # %bb.6:
; CHECK-NEXT:    or %s0, 0, (0)1
; CHECK-NEXT:    br.l.t .LBB{{[0-9]+}}_9
; CHECK-NEXT:  .LBB{{[0-9]+}}_3:
; CHECK-NEXT:    breq.w 3, %s0, .LBB{{[0-9]+}}_7
; CHECK-NEXT:  # %bb.4:
; CHECK-NEXT:    brne.w 4, %s0, .LBB{{[0-9]+}}_9
; CHECK-NEXT:  # %bb.5:
; CHECK-NEXT:    or %s0, 7, (0)1
; CHECK-NEXT:    br.l.t .LBB{{[0-9]+}}_9
; CHECK-NEXT:  .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    or %s0, 3, (0)1
; CHECK-NEXT:    br.l.t .LBB{{[0-9]+}}_9
; CHECK-NEXT:  .LBB{{[0-9]+}}_7:
; CHECK-NEXT:    or %s0, 4, (0)1
; CHECK-NEXT:  .LBB{{[0-9]+}}_9:
; CHECK-NEXT:    adds.w.sx %s0, %s0, (0)1
; CHECK-NEXT:    or %s11, 0, %s9
  switch i32 %0, label %5 [
    i32 1, label %6
    i32 2, label %2
    i32 3, label %3
    i32 4, label %4
  ]

2:                                                ; preds = %1
  br label %6

3:                                                ; preds = %1
  br label %6

4:                                                ; preds = %1
  br label %6

5:                                                ; preds = %1
  br label %6

6:                                                ; preds = %1, %5, %4, %3, %2
  %7 = phi i32 [ %0, %5 ], [ 7, %4 ], [ 4, %3 ], [ 0, %2 ], [ 3, %1 ]
  ret i32 %7
}
