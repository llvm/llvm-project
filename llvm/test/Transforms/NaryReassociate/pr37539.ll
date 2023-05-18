; RUN: opt < %s -passes=nary-reassociate -S -o - | FileCheck %s

; The test check that compilation does not segv (see pr37539).

define void @f1() {
; CHECK-LABEL: @f1(
; CHECK-NEXT:    br label %[[BB1:.*]]
; CHECK:         [[BB1]]
; CHECK-NEXT:    [[P1:%.*]] = phi i16 [ 0, [[TMP0:%.*]] ], [ [[A1:%.*]], %[[BB1]] ]
; CHECK-NEXT:    [[SCEVGEP_OFFS:%.*]] = add i16 2, 0
; CHECK-NEXT:    [[A1]] = add i16 [[P1]], [[SCEVGEP_OFFS]]
; CHECK-NEXT:    br i1 false, label %[[BB1]], label %[[BB7:.*]]
; CHECK:         [[BB7]]
; CHECK-NEXT:    ret void
;
  br label %bb1

bb1:
  %p1 = phi i16 [ 0, %0 ], [ %a1, %bb1 ]
  %p2 = phi i16 [ 0, %0 ], [ %a2, %bb1 ]
  %scevgep.offs = add i16 2, 0
  %a1 = add i16 %p1, %scevgep.offs
  %scevgep.offs5 = add i16 2, 0
  %a2 = add i16 %p2, %scevgep.offs5
  br i1 false, label %bb1, label %bb7

bb7:
  ret void
}
