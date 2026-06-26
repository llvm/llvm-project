; RUN: opt -passes='loop-simplify,lcssa,loop-split-test,verify' \
; RUN:   -loop-split-points=4 -loop-split-unguarded=0 \
; RUN:   -verify-dom-info -verify-loop-info -S < %s | FileCheck %s

; Per-partition entry guards are optional. The split loop is bottom-tested, so
; by default every partition is guarded by a `Start <= End` check that skips a
; zero-iteration partition. A caller that can prove a partition runs at least
; once may omit its guard (here partition 0, via -loop-split-unguarded=0): that
; guard then enters the sub-loop unconditionally and has no skip edge to the
; next partition. Partition 1 keeps its guard. The dominator tree is maintained
; incrementally for the missing skip edge, so -verify-dom-info must pass.

define i64 @reduction(ptr %a, i64 %n) {
; CHECK-LABEL: define i64 @reduction(
; Partition 0's guard enters unconditionally - no entry icmp, no skip edge.
; CHECK:       ls.guard0:
; CHECK:         br label %[[ENTRY:.*]]
; CHECK:       [[ENTRY]]:
; CHECK-NEXT:    br label %[[LOOP:.*]]
; The first sub-loop is the original, clamped to the partition-0 end.
; CHECK:       [[LOOP]]:
; CHECK:         br i1 {{.*}}, label %[[LOOP]], label %[[EXIT:.*]]
; CHECK:       [[EXIT]]:
; CHECK-NEXT:    br label %[[GUARD1:.*]]
; Partition 1 keeps its conditional guard.
; CHECK:       [[GUARD1]]:
; CHECK-NEXT:    [[CHK1:%.*]] = icmp sle i64 4, {{.*}}
; CHECK-NEXT:    br i1 [[CHK1]], label %[[ENTRY1:.*]], label %[[FINAL:.*]]
entry:
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %loop ]
  %sum = phi i64 [ 0, %entry ], [ %sum.next, %loop ]
  %p = getelementptr i64, ptr %a, i64 %i
  %v = load i64, ptr %p
  %sum.next = add i64 %sum, %v
  %i.next = add i64 %i, 1
  %c = icmp slt i64 %i.next, %n
  br i1 %c, label %loop, label %exit

exit:
  %sum.lcssa = phi i64 [ %sum.next, %loop ]
  ret i64 %sum.lcssa
}
