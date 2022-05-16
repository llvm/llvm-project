; Tests loop-simplify does not move the loop metadata, because
; the loopexit block is not the latch of the loop _bb6.

; FIXME(#55416): The metadata should not move.

; RUN: opt < %s -passes=loop-simplify -S | FileCheck %s
; CHECK-LABEL: loop.header.loopexit:
; CHECK: br label %loop.header, !llvm.loop !0
; CHECK-LABEL: loop.latch:
; CHECK-NOT: br i1 %p, label %loop.latch, label %loop.header.loopexit, !llvm.loop !0

define void @func(i1 %p) {
entry:
  br label %loop.header

loop.header:
  br i1 %p, label %bb1, label %exit

bb1:
  br i1 %p, label %bb2, label %bb3

bb2:
  br label %bb3

bb3:
  br label %loop.latch

loop.latch:
  br i1 %p, label %loop.latch, label %loop.header, !llvm.loop !0

exit:
  ret void
}

!0 = distinct !{!0, !1}
!1 = !{!"llvm.loop.mustprogress"}
