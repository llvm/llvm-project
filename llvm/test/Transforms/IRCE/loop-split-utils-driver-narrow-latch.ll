; RUN: opt -passes=irce -irce-use-loop-split-utils=true -irce-allow-narrow-latch=true \
; RUN:     -irce-skip-profitability-checks -S < %s 2>&1 | FileCheck %s
;
; Narrow-latch shape: a wide i64 induction whose counted exit compares a
; truncation of it ("icmp slt i32 (trunc iv.next), 100"). With
; -irce-allow-narrow-latch, IRCE drives it through LoopSplitUtils'
; AllowTruncatedLatchCompare path. The wide i64 induction drives the partition
; boundaries and clamps (emitted in the wide type), the main loop folds the
; range check to true, and the guarded post-loop keeps the original check.

; CHECK-LABEL: @irce_driver_narrow_latch(
; The main loop folds the range check and clamps the wide induction inclusively:
; CHECK:       loop:
; CHECK:         br i1 true, label %backedge, label %check_failed
; CHECK:       backedge:
; CHECK:         icmp sle i64 %iv.next, 98
; A guarded post-loop keeps the original wide range check:
; CHECK:       ls.guard1:
; CHECK:       loop.ls1:
; CHECK:         %[[RC:.*]] = icmp slt i64 %iv.ls1, 99
; CHECK:         br i1 %[[RC]], label %backedge.ls1, label %check_failed

define i32 @irce_driver_narrow_latch() {
entry:
  br label %loop
loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %backedge ]
  %rc = icmp slt i64 %iv, 99
  br i1 %rc, label %backedge, label %check_failed
backedge:
  %iv.next = add i64 %iv, 1
  %narrow.iv = trunc i64 %iv.next to i32
  %latch.cond = icmp slt i32 %narrow.iv, 100
  br i1 %latch.cond, label %loop, label %exit
exit:
  ret i32 %narrow.iv
check_failed:
  ret i32 -1
}
