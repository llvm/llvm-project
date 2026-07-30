; RUN: opt -S < %s -passes='require<opt-remark-emit>,loop-unroll,require<block-freq>' | FileCheck %s
; Crasher from PR20987.

; CHECK: define void @update_loop_info_in_subloops
; CHECK: entry:
; CHECK: L:
; CHECK: L.inner:
; CHECK: L.inner.latch:
; CHECK: L.latch:
; CHECK: L.inner.1:
; CHECK: L.inner.latch.1:
; CHECK: L.latch.1:

define void @update_loop_info_in_subloops() {
entry:
  br label %L

L:
  %0 = phi i64 [ 1, %entry ], [ %1, %L.latch ]
  br label %L.inner

L.inner:
  br label %L.inner.latch

L.inner.latch:
  br i1 false, label %L.latch, label %L.inner

L.latch:
  %1 = add i64 %0, 1
  %2 = icmp eq i64 %1, 3
  br i1 %2, label %exit, label %L

exit:
  ret void
}
