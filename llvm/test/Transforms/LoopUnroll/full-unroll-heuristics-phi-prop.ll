; RUN: opt < %s -S -loop-unroll -unroll-max-iteration-count-to-analyze=100 -unroll-threshold=10 -unroll-max-percent-threshold-boost=200 | FileCheck %s
; RUN: opt < %s -S -passes='require<opt-remark-emit>,loop(unroll-full)' -unroll-max-iteration-count-to-analyze=100 -unroll-threshold=10 -unroll-max-percent-threshold-boost=200 | FileCheck %s
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"

define i64 @propagate_loop_phis() {
; CHECK-LABEL: @propagate_loop_phis(
; CHECK-NOT: br i1
; CHECK: ret i64 3
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %inc, %loop ]
  %x0 = phi i64 [ 0, %entry ], [ %x2, %loop ]
  %x1 = or i64 %x0, 1
  %x2 = or i64 %x1, 2
  %inc = add nuw nsw i64 %iv, 1
  %cond = icmp sge i64 %inc, 10
  br i1 %cond, label %loop.end, label %loop

loop.end:
  %x.lcssa = phi i64 [ %x2, %loop ]
  ret i64 %x.lcssa
}
