; RUN: opt -passes=irce -irce-use-loop-split-utils=true \
; RUN:     -irce-skip-profitability-checks -S < %s 2>&1 | FileCheck %s
;
; With -irce-use-loop-split-utils, IRCE restructures the loop through the
; generic LoopSplitUtils primitive instead of LoopConstrainer. The result is
; behaviorally identical -- the main range runs with the bounds check folded to
; true, and a post range keeps the original check -- but the IR shape (block
; names, block count) differs. This test pins those semantic invariants.

; CHECK-LABEL: @irce_driver_upper(
; The main loop runs with the range check folded to a constant true:
; CHECK:       loop:
; CHECK:         br i1 true, label %in.bounds, label %out.of.bounds
; A guarded post range keeps the original (unfolded) range check:
; CHECK:       ls.guard1:
; CHECK:       loop.ls1:
; CHECK:         %[[UC:.*]] = icmp slt i32 %i.ls1, 30
; CHECK:         br i1 %[[UC]], label %in.bounds.ls1, label %out.of.bounds

define i32 @irce_driver_upper(i32 %n) {
entry:
  %g = icmp sgt i32 %n, 0
  br i1 %g, label %loop, label %early
early:
  ret i32 0
loop:
  %i = phi i32 [ 0, %entry ], [ %i.next, %in.bounds ]
  %acc = phi i32 [ 0, %entry ], [ %acc.next, %in.bounds ]
  %i.next = add nsw i32 %i, 1
  %uc = icmp slt i32 %i, 30
  br i1 %uc, label %in.bounds, label %out.of.bounds
in.bounds:
  %m = mul i32 %i, 3
  %acc.next = add i32 %acc, %m
  %next = icmp slt i32 %i.next, %n
  br i1 %next, label %loop, label %exit
out.of.bounds:
  br label %exit
exit:
  %r = phi i32 [ %acc.next, %in.bounds ], [ %acc, %out.of.bounds ]
  ret i32 %r
}

; A decreasing loop (step -1) with a lower+upper range check: only the upper
; limit is unsafe, so a pre-loop keeps the real check and the main partition
; (the clone) folds it. The decreasing clamp uses an sge test.
; CHECK-LABEL: @irce_driver_decreasing(
; CHECK:       ls.guard0:
; CHECK:         icmp sge i32 %{{.*}}, %smax
; The main partition folds the range check to constant true:
; CHECK:       loop.ls1:
; CHECK:         and i1 true, true
; CHECK:       ls.final.exit:

define void @irce_driver_decreasing(ptr %arr, ptr %a_len_ptr, i32 %n) {
entry:
  %len = load i32, ptr %a_len_ptr, !range !0
  %first.itr.check = icmp sgt i32 %n, 0
  %start = sub i32 %n, 1
  br i1 %first.itr.check, label %loop, label %exit
loop:
  %idx = phi i32 [ %start, %entry ], [ %idx.dec, %in.bounds ]
  %idx.dec = sub i32 %idx, 1
  %abc.high = icmp slt i32 %idx, %len
  %abc.low = icmp sge i32 %idx, 0
  %abc = and i1 %abc.low, %abc.high
  br i1 %abc, label %in.bounds, label %out.of.bounds
in.bounds:
  %addr = getelementptr i32, ptr %arr, i32 %idx
  store i32 0, ptr %addr
  %next = icmp sgt i32 %idx.dec, -1
  br i1 %next, label %loop, label %exit
out.of.bounds:
  ret void
exit:
  ret void
}

; A non-unit stride (IV += 7) has no SCEV-computable trip count, so IRCE drives
; it through LoopSplitUtils' uncomputable-trip-count fallback: the main loop
; folds the check to true and clamps inclusively against a grid-aligned bound,
; while the post-loop keeps the original check and the original latch.
; CHECK-LABEL: @irce_driver_stride(
; CHECK:       loop:
; CHECK:         %idx.next = add nsw i32 %idx, 7
; CHECK:         br i1 true, label %in.bounds, label %out.of.bounds
; CHECK:         icmp sle i32 %idx.next, %{{.*}}
; The post-loop keeps the original (unfolded) range check:
; CHECK:       loop.ls1:
; CHECK:         icmp slt i32 %idx.ls1, %len

define void @irce_driver_stride(ptr %arr, ptr %a_len_ptr) {
entry:
  %len = load i32, ptr %a_len_ptr, !range !0
  br label %loop
loop:
  %idx = phi i32 [ 0, %entry ], [ %idx.next, %in.bounds ]
  %idx.next = add i32 %idx, 7
  %abc = icmp slt i32 %idx, %len
  br i1 %abc, label %in.bounds, label %out.of.bounds
in.bounds:
  %addr = getelementptr i32, ptr %arr, i32 %idx
  store i32 0, ptr %addr
  %next = icmp slt i32 %idx.next, 100
  br i1 %next, label %loop, label %exit
out.of.bounds:
  ret void
exit:
  ret void
}

!0 = !{i32 0, i32 2147483647}
