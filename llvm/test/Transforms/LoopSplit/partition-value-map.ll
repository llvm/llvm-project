; RUN: opt -passes='loop-simplify,lcssa,loop-split-test' \
; RUN:   -loop-split-points=4,8 -loop-split-print-partition-map \
; RUN:   -disable-output < %s 2>&1 | FileCheck %s

; LoopSplitUtils preserves the original-to-clone value map for every partition
; and exposes it through getPartitionValue(). Partition 0 reuses the original
; loop (identity mapping); later partitions return the cloned counterpart, named
; with a `.lsN` suffix. The latch compare (%c) is rewritten rather than cloned,
; so it has no surviving counterpart and is omitted from every partition.

define i32 @sum(ptr %a, i32 %n) {
entry:
  br label %loop

loop:
  %i = phi i32 [ 0, %entry ], [ %i.next, %loop ]
  %acc = phi i32 [ 0, %entry ], [ %acc.next, %loop ]
  %p = getelementptr inbounds i32, ptr %a, i32 %i
  %v = load i32, ptr %p
  %acc.next = add i32 %acc, %v
  %i.next = add i32 %i, 1
  %c = icmp slt i32 %i.next, %n
  br i1 %c, label %loop, label %exit

exit:
  %lcssa = phi i32 [ %acc.next, %loop ]
  ret i32 %lcssa
}

; CHECK: LS-MAP partition 0:
; CHECK:   i -> i
; CHECK:   acc -> acc
; CHECK:   acc.next -> acc.next
; CHECK:   i.next -> i.next
; CHECK: LS-MAP partition 1:
; CHECK:   i -> i.ls1
; CHECK:   acc -> acc.ls1
; CHECK:   acc.next -> acc.next.ls1
; CHECK:   i.next -> i.next.ls1
; CHECK: LS-MAP partition 2:
; CHECK:   i -> i.ls2
; CHECK:   acc -> acc.ls2
; CHECK:   acc.next -> acc.next.ls2
; CHECK:   i.next -> i.next.ls2
