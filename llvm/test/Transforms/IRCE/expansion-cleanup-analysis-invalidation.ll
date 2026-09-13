; RUN: opt -passes=irce -irce-skip-profitability-checks \
; RUN:   -verify-analysis-invalidation -S < %s | FileCheck %s
; RUN: opt -passes=irce -irce-skip-profitability-checks \
; RUN:   -irce-allow-narrow-latch=false -verify-analysis-invalidation -S \
; RUN:   < %s | FileCheck %s --check-prefix=NARROW

; Parsing the loop structure and computing the exit limits may materialize
; SCEVs in the preheader before all bail-out points have been passed. If a
; later check fails, those speculative instructions must be removed before
; IRCE reports that it did not change the function.

declare void @sideeffect()

; The pre-loop limit can be expanded, but the post-loop limit contains a
; conditionally executed division which cannot safely be expanded in the
; preheader. Make sure the partially expanded pre-loop limit is removed and
; the name of the reused induction start is left unchanged.

define void @cleanup_after_partial_expansion(ptr %num.ptr, ptr %denom.ptr,
                                             i64 range(i64 -100, -1) %offset,
                                             i64 range(i64 0, 10) %start,
                                             i1 %maybe.exit) {
; CHECK-LABEL: define void @cleanup_after_partial_expansion(
; CHECK-SAME: i64 range(i64 0, 10) %start,
entry:
; CHECK: entry:
; CHECK-NEXT: %num = load i64, ptr %num.ptr, align 8, !range !0
; CHECK-NEXT: %denom = load i64, ptr %denom.ptr, align 8, !range !0
; CHECK-NEXT: br label %loop
  %num = load i64, ptr %num.ptr, align 8, !range !0
  %denom = load i64, ptr %denom.ptr, align 8, !range !0
  br label %loop

exit:
  ret void

loop:
  %iv = phi i64 [ %start, %entry ], [ %iv.next, %guarded ]
  %checked = phi i64 [ %offset, %entry ], [ %checked.next, %guarded ]
  %iv.next = add nuw nsw i64 %iv, 1
  br i1 %maybe.exit, label %range.check, label %exit

range.check:
  %div.result = udiv i64 %num, %denom
  %rc = icmp slt i64 %checked, %div.result
  br i1 %rc, label %guarded, label %exit

guarded:
  %checked.next = add nsw i64 %checked, 1
  call void @sideeffect()
  %loop.cond = icmp slt i64 %iv.next, 1000
  br i1 %loop.cond, label %loop, label %exit
}

; IRCE canonicalizes range-check branches before calculating the constrained
; subranges. If mismatched types make that calculation fail, the branch
; inversion must still be reported as a change.

define void @inversion_before_subrange_failure() {
; NARROW-LABEL: define void @inversion_before_subrange_failure(
; NARROW: %range.check.failed = icmp slt i64 %iv, 100
; NARROW-NEXT: br i1 %range.check.failed, label %backedge, label %check.failed
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %backedge ]
  %range.check.failed = icmp sge i64 %iv, 100
  br i1 %range.check.failed, label %check.failed, label %backedge

backedge:
  %iv.next = add i64 %iv, 1
  %narrow.iv = trunc i64 %iv.next to i32
  %latch.cond = icmp slt i32 %narrow.iv, 100
  br i1 %latch.cond, label %loop, label %exit

exit:
  ret void

check.failed:
  ret void
}

!0 = !{i64 0, i64 100}
