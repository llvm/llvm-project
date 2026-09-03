; RUN: opt -passes=irce -irce-skip-profitability-checks \
; RUN:   -verify-analysis-invalidation -S < %s | FileCheck %s

; Parsing the loop structure may need to materialize SCEVs in the preheader.
; If a later legality check prevents cloning, those speculative instructions
; must be removed before IRCE reports that it did not change the function.

declare void @cannot_duplicate() noduplicate
declare void @convergent_call() convergent

define void @no_materialization(ptr %n.ptr) {
; CHECK-LABEL: define void @no_materialization(
entry:
; CHECK: entry:
; CHECK-NEXT: %n = load i32, ptr %n.ptr, align 4, !range !0
; CHECK-NEXT: br label %loop
  %n = load i32, ptr %n.ptr, !range !0
  br label %loop

loop:
  %idx = phi i32 [ 0, %entry ], [ %idx.next, %in.bounds ]
  %idx.next = add i32 %idx, 1
  %bound = add nsw i32 %n, 1
  %in.range = icmp slt i32 %idx, 50
  call void @cannot_duplicate()
  br i1 %in.range, label %in.bounds, label %out.of.bounds

in.bounds:
; CHECK: %bound = add nsw i32 %n, 1
  %next = icmp slt i32 %idx.next, %bound
  br i1 %next, label %loop, label %exit

out.of.bounds:
  ret void

exit:
  ret void
}

!0 = !{i32 1, i32 1000}

; A loop that does not need pre- or post-loop clones may still be constrained
; when its body is unsafe to clone.

define void @no_cloning_needed(i32 range(i32 100, 150) %len,
                               i32 range(i32 1, 50) %n) {
; CHECK-LABEL: define void @no_cloning_needed(
entry:
  br label %loop

loop:
  %idx = phi i32 [ 0, %entry ], [ %idx.next, %in.bounds ]
  %in.range = icmp slt i32 %idx, %len
  call void @convergent_call()
; CHECK: br i1 true, label %in.bounds, label %out.of.bounds
  br i1 %in.range, label %in.bounds, label %out.of.bounds

in.bounds:
  %idx.next = add nuw nsw i32 %idx, 1
  %next = icmp slt i32 %idx.next, %n
  br i1 %next, label %loop, label %exit

out.of.bounds:
  ret void

exit:
  ret void
}
