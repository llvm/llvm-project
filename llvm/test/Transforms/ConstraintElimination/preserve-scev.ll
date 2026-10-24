; RUN: opt -passes='constraint-elimination,print<scalar-evolution>' -disable-output %s 2>&1 | FileCheck %s

; Make sure ScalarEvolution's cached trip counts are dropped when
; constraint-elimination folds the condition of a loop-exiting branch. The
; preserved analysis would otherwise keep a stale, less precise backedge-taken
; count (umin of both original exits), blocking later trip-count based folds,
; e.g. in indvars.
; See https://github.com/llvm/llvm-project/issues/213872.

; CHECK-LABEL: Classifying expressions for: @multiple_pow2
; CHECK:       Loop %loop: <multiple exits> backedge-taken count is ((4 * %count) /u 4)
; CHECK-NEXT:    exit count for loop: ((4 * %count) /u 4)
; CHECK-NEXT:    exit count for loop.latch: ***COULDNOTCOMPUTE***

define void @multiple_pow2(i64 %count) {
entry:
  %end = shl i64 %count, 2
  br label %loop

loop:                                             ; preds = %loop.latch, %entry
  %iv = phi i64 [ %iv.next, %loop.latch ], [ 0, %entry ]
  %cmp.i.not = icmp eq i64 %iv, %end
  br i1 %cmp.i.not, label %exit, label %loop.latch

loop.latch:                                       ; preds = %loop
  %iv.next = add i64 %iv, 4
  %cmp2.i.i = icmp ult i64 %iv, %end
  br i1 %cmp2.i.i, label %loop, label %exit

exit:                                             ; preds = %loop.latch, %loop
  ret void
}
