; RUN: opt -loop-simplify -S < %s | FileCheck %s

; This will test whether or not the metadata from the current loop 1 latch
; is removed, and applied to the new latch after running the loop-simplify
; pass on this function. The loop simplify pass ensures that each loop has exit
; blocks which only have predecessors that are inside of the loop. This
; guarantees that the loop preheader/header will dominate the exit blocks. For
; this function currently loop 2 does not have a dedicated exit block.

; CHECK: loop_1_loopHeader.loopexit:
; CHECK: br label %loop_1_loopHeader, !llvm.loop [[LOOP_1_LATCH_MD:![0-9]+]]
; CHECK: loop_2_loopHeader
; CHECK: br i1 %grt_B, label %loop_1_loopHeader.loopexit, label %loop_2_do
; CHECK-NOT:  br i1 %grt_B, label %loop_1_loopHeader, label %loop_2_do, !llvm.loop{{.*}}

define void @function(i32 %A) {
entry:
  %B = add i32 %A, 45
  %C = add i32 %A, 22
  br label %loop_1_loopHeader

loop_1_loopHeader:                              ; preds = %loop_2_loopHeader, %entry
  %loop_1_idx = phi i32 [ 1, %entry], [ %loop_1_update_idx, %loop_2_loopHeader ]
  %grt_C = icmp slt i32 %loop_1_idx, %C
  br i1 %grt_C, label %exit, label %loop_1_do

loop_1_do:                                      ; preds = %loop_1_loopHeader
  %loop_1_update_idx = add nuw nsw i32 %loop_1_idx, 1
  br label %loop_2_loopHeader

loop_2_loopHeader:                              ; preds = %loop_2_do, %_loop_1_do
  %loop_2_idx = phi i32 [ 1, %loop_1_do ], [ %loop_2_update_idx, %loop_2_do ]
  %grt_B = icmp slt i32 %loop_2_idx, %B
  br i1 %grt_B, label %loop_1_loopHeader, label %loop_2_do, !llvm.loop !0

loop_2_do:                                      ; preds = %loop_2_loopHeader
  %loop_2_update_idx = add nuw nsw i32 %loop_2_idx, 1
  br label %loop_2_loopHeader, !llvm.loop !2

exit:                                       ; preds = %loop_1_loopHeader
  ret void
}

!0 = distinct !{!0, !1}
!1 = !{!"llvm.loop.unroll.disable"}
!2 = distinct !{!2, !1}

