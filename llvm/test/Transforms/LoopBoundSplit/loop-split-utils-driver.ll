; RUN: opt -passes=loop-bound-split -verify-each \
; RUN:     -loop-bound-split-use-loop-split-utils=true -S < %s | FileCheck %s
;
; With -loop-bound-split-use-loop-split-utils, LoopBoundSplit performs the split
; through the generic LoopSplitUtils primitive instead of its bespoke
; clone-and-fold transform. The result is behaviorally identical -- a pre-loop
; that runs while the split condition holds (folded to true) and a guarded
; post-loop that runs the rest (folded to false) -- but emits LoopSplitUtils'
; canonical ls.guard/ls.exit block shape. This test pins those invariants.

; CHECK-LABEL: @split_bound(
; The pre-loop's split condition is folded to a constant true, and its latch is
; clamped to the split boundary (min(n-1, 511)):
; CHECK:       loop:
; CHECK:         br i1 true, label %then, label %else
; CHECK:       latch:
; CHECK:         %[[IN:.*]] = add nsw i32 %i, 1
; CHECK:         icmp sle i32 %[[IN]], %smin
; A guard decides whether the post-loop runs at all:
; CHECK:       ls.guard1:
; CHECK:         icmp sle i32 512, %{{.*}}
; The post-loop's split condition is folded to a constant false:
; CHECK:       loop.ls1:
; CHECK:         br i1 false, label %then.ls1, label %else.ls1
; CHECK:       ls.final.exit:
; CHECK:         ret void

; for (i = 0; i < n; i++) { if (i < 512) a[i] = i*3; else a[i] = 7; }
define void @split_bound(ptr %a, i32 %n) {
entry:
  br label %loop
loop:
  %i = phi i32 [ 0, %entry ], [ %i.next, %latch ]
  %c = icmp slt i32 %i, 512
  br i1 %c, label %then, label %else
then:
  %m = mul i32 %i, 3
  %p0 = getelementptr inbounds i32, ptr %a, i32 %i
  store i32 %m, ptr %p0
  br label %latch
else:
  %p1 = getelementptr inbounds i32, ptr %a, i32 %i
  store i32 7, ptr %p1
  br label %latch
latch:
  %i.next = add nsw i32 %i, 1
  %cond = icmp slt i32 %i.next, %n
  br i1 %cond, label %loop, label %exit
exit:
  ret void
}
