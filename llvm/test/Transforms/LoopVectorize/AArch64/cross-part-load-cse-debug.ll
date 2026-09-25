; REQUIRES: asserts
; RUN: split-file %s %t
;
; When cross-part analysis selects IC=2 after the ordinary heuristics decline
; interleaving, it must not emit a contradictory non-interleaving diagnostic.
; RUN: opt -passes=loop-vectorize -force-vector-width=4 \
; RUN:     -force-target-max-vector-interleave=2 -small-loop-cost=0 \
; RUN:     -enable-interleave-cse \
; RUN:     -interleave-cse-min-pct=1 \
; RUN:     -debug-only=loop-vectorize -disable-output %t/success.ll 2>&1 \
; RUN:     | FileCheck %t/success.ll --check-prefix=SUCCESS
;
; A fixed-VF tail-folded plan is ineligible for interleaving. Cross-part
; analysis must not override that policy or emit a profitability estimate.
; The same holds when partial-alias masking additionally forces IC=1, which
; requires runtime difference checks and therefore a second checked pointer.
; RUN: opt -passes=loop-vectorize -force-vector-width=4 \
; RUN:     -force-target-max-vector-interleave=2 -small-loop-cost=0 \
; RUN:     -force-target-supports-masked-memory-ops \
; RUN:     -force-tail-folding-style=data-and-control \
; RUN:     -tail-folding-policy=must-fold-tail \
; RUN:     -force-partial-aliasing-vectorization \
; RUN:     -enable-interleave-cse \
; RUN:     -interleave-cse-min-pct=1 -debug-only=loop-vectorize \
; RUN:     -disable-output %t/success.ll 2>&1 \
; RUN:     | FileCheck %t/success.ll --check-prefixes=MASKED,ALIAS
;
; When the ordinary branch-cost heuristic recommends IC=1, a successful
; cross-part selection must return before emitting its baseline diagnostic.
; RUN: opt -passes=loop-vectorize -force-vector-width=4 \
; RUN:     -force-target-max-vector-interleave=2 \
; RUN:     -force-target-instruction-cost=1 -small-loop-cost=12 \
; RUN:     -enable-loadstore-runtime-interleave=false \
; RUN:     -enable-interleave-cse \
; RUN:     -interleave-cse-min-pct=1 \
; RUN:     -debug-only=loop-vectorize -disable-output %t/success.ll 2>&1 \
; RUN:     | FileCheck %t/success.ll --check-prefix=SUCCESS-SMALL
;
; The same-part duplicate after a genuine cross-part match must report exactly
; one opportunity and must not double the saving past the 6% threshold.
; RUN: opt -passes=loop-vectorize -force-vector-width=4 \
; RUN:     -force-target-max-vector-interleave=2 -small-loop-cost=0 \
; RUN:     -force-target-instruction-cost=1 \
; RUN:     -enable-interleave-cse -interleave-cse-min-pct=6 \
; RUN:     -debug-only=loop-vectorize \
; RUN:     -disable-output %t/duplicate.ll 2>&1 \
; RUN:     | FileCheck %t/duplicate.ll
;
; A predicted saving exactly equal to the configured threshold is sufficient,
; preserving the inclusive >= comparison.
; RUN: opt -passes=loop-vectorize -force-vector-width=4 \
; RUN:     -force-target-max-vector-interleave=2 -small-loop-cost=0 \
; RUN:     -force-target-instruction-cost=1 \
; RUN:     -enable-interleave-cse -interleave-cse-min-pct=5 \
; RUN:     -debug-only=loop-vectorize \
; RUN:     -disable-output %t/threshold.ll 2>&1 \
; RUN:     | FileCheck %t/threshold.ll
;
; With cross-part analysis disabled, the ordinary branch-cost diagnostic
; remains unchanged.
; RUN: opt -passes=loop-vectorize -force-vector-width=4 \
; RUN:     -force-target-max-vector-interleave=2 \
; RUN:     -force-target-instruction-cost=1 -small-loop-cost=12 \
; RUN:     -enable-loadstore-runtime-interleave=false \
; RUN:     -debug-only=loop-vectorize -disable-output %t/success.ll 2>&1 \
; RUN:     | FileCheck %t/success.ll --check-prefix=DISABLED-SMALL
;
; A constant source offset cannot equal the scalable part offset for every
; runtime vscale, so the exact analysis reports no opportunity and keeps UF=1.
; RUN: opt -passes=loop-vectorize -mtriple=aarch64-none-linux-gnu -mattr=+sve \
; RUN:     -force-vector-width="vscale x 2" \
; RUN:     -force-target-max-vector-interleave=2 -small-loop-cost=0 \
; RUN:     -enable-interleave-cse \
; RUN:     -interleave-cse-min-pct=1 -debug-only=loop-vectorize \
; RUN:     -disable-output %t/success.ll 2>&1 \
; RUN:     | FileCheck %t/success.ll --check-prefix=SCALABLE
;
; A narrowed plan materializes VFxUF before IC selection. The redundancy
; analysis must return before emitting an estimate for that unsupported shape.
; RUN: opt -passes=loop-vectorize -mtriple=arm64-apple-macosx \
; RUN:     -force-vector-width=2 -force-target-max-vector-interleave=2 \
; RUN:     -force-target-num-vector-regs=1024 \
; RUN:     -force-target-instruction-cost=1 -small-loop-cost=0 \
; RUN:     -enable-interleave-cse -interleave-cse-min-pct=1 \
; RUN:     -debug-only=loop-vectorize -disable-output \
; RUN:     %S/cross-part-load-cse-narrowed.ll 2>&1 \
; RUN:     | FileCheck %s --check-prefix=NARROWED
;
; NARROWED-LABEL: LV: Checking a loop in 'narrowed'
; NARROWED-NOT: LV: Cross-part load redundancy estimate:
; NARROWED: Executing best plan with VF=2, UF=1

;--- success.ll
; Every prefix below closes its 'positive' block with a second -LABEL line, so
; that the checks cannot be satisfied by the 'partial_alias' log that follows.
;
; SUCCESS-LABEL: LV: Checking a loop in 'positive'
; SUCCESS: LV: Cross-part load redundancy estimate: opportunities=1, predicted-saved-cost={{[^,]+}}, loop-cost={{[^,]+}}, saving={{[0-9]+}}%, required=1%; selecting IC=2.
; SUCCESS-NEXT: LV: Exact cross-part load redundancy predicts a downstream saving; raising IC to 2.
; SUCCESS-NOT: LV: Not Interleaving.
; SUCCESS: LV: Found a vectorizable loop
; SUCCESS: Executing best plan with VF=4, UF=2
; SUCCESS-LABEL: LV: Checking a loop in 'partial_alias'
;
; MASKED-LABEL: LV: Checking a loop in 'positive'
; MASKED-NOT: LV: Cross-part load redundancy estimate:
; MASKED-NOT: Exact cross-part load redundancy predicts a downstream saving
; MASKED-NOT: LV: Not interleaving due to partial aliasing vectorization.
; MASKED: Executing best plan with VF=4, UF=1
;
; The ALIAS-LABEL line below also closes the MASKED block above, because
; FileCheck partitions the input at the -LABEL lines of every active prefix.
; ALIAS-LABEL: LV: Checking a loop in 'partial_alias'
; ALIAS-NOT: LV: Cross-part load redundancy estimate:
; ALIAS-NOT: Exact cross-part load redundancy predicts a downstream saving
; ALIAS: LV: Not interleaving due to partial aliasing vectorization.
; ALIAS: Executing best plan with VF=4, UF=1
;
; SUCCESS-SMALL-LABEL: LV: Checking a loop in 'positive'
; SUCCESS-SMALL: LV: Cross-part load redundancy estimate: opportunities=1, predicted-saved-cost={{[^,]+}}, loop-cost={{[^,]+}}, saving={{[0-9]+}}%, required=1%; selecting IC=2.
; SUCCESS-SMALL-NEXT: LV: Exact cross-part load redundancy predicts a downstream saving; raising IC to 2.
; SUCCESS-SMALL-NOT: LV: Interleaving to reduce branch cost.
; SUCCESS-SMALL: LV: Found a vectorizable loop
; SUCCESS-SMALL: Executing best plan with VF=4, UF=2
; SUCCESS-SMALL-LABEL: LV: Checking a loop in 'partial_alias'
;
; DISABLED-SMALL-LABEL: LV: Checking a loop in 'positive'
; DISABLED-SMALL-NOT: Cross-part load redundancy
; DISABLED-SMALL: LV: Interleaving to reduce branch cost.
; DISABLED-SMALL-NOT: Cross-part load redundancy
; DISABLED-SMALL: LV: Found a vectorizable loop
; DISABLED-SMALL: Executing best plan with VF=4, UF=1
; DISABLED-SMALL-LABEL: LV: Checking a loop in 'partial_alias'
;
; SCALABLE-LABEL: LV: Checking a loop in 'positive'
; SCALABLE-NOT: Exact cross-part load redundancy predicts a downstream saving
; SCALABLE: LV: VF is vscale x 2
; SCALABLE-NEXT: LV: Cross-part load redundancy estimate: opportunities=0, predicted-saved-cost=0, loop-cost={{[^,]+}}, saving=0%, required=1%; skipping.
; SCALABLE-NEXT: LV: Not Interleaving.
; SCALABLE-NOT: Exact cross-part load redundancy predicts a downstream saving
; SCALABLE: LV: Found a vectorizable loop (vscale x 2)
; SCALABLE: Executing best plan with VF=vscale x 2, UF=1
; SCALABLE-LABEL: LV: Checking a loop in 'partial_alias'

target triple = "aarch64-unknown-linux-gnu"

define void @positive(ptr noalias %a, ptr noalias %c, i64 %n) {
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %a.iv = getelementptr inbounds i32, ptr %a, i64 %iv
  %l1 = load i32, ptr %a.iv, align 4
  %iv.plus.4 = add nuw nsw i64 %iv, 4
  %a.iv.plus.4 = getelementptr inbounds i32, ptr %a, i64 %iv.plus.4
  %l2 = load i32, ptr %a.iv.plus.4, align 4
  %sum = add i32 %l1, %l2
  %c.iv = getelementptr inbounds i32, ptr %c, i64 %iv
  store i32 %sum, ptr %c.iv, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %cmp = icmp slt i64 %iv.next, %n
  br i1 %cmp, label %loop, label %exit

exit:
  ret void
}

; The %b/%c pair needs a runtime difference check, which enables partial-alias
; masking, while the cross-part reuse candidate on %a stays present.
define void @partial_alias(ptr noalias %a, ptr %b, ptr %c, i64 %n) {
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %a.iv = getelementptr inbounds i32, ptr %a, i64 %iv
  %l1 = load i32, ptr %a.iv, align 4
  %iv.plus.4 = add nuw nsw i64 %iv, 4
  %a.iv.plus.4 = getelementptr inbounds i32, ptr %a, i64 %iv.plus.4
  %l2 = load i32, ptr %a.iv.plus.4, align 4
  %b.iv = getelementptr inbounds i32, ptr %b, i64 %iv
  %l3 = load i32, ptr %b.iv, align 4
  %sum.1 = add i32 %l1, %l2
  %sum.2 = add i32 %sum.1, %l3
  %c.iv = getelementptr inbounds i32, ptr %c, i64 %iv
  store i32 %sum.2, ptr %c.iv, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %cmp = icmp slt i64 %iv.next, %n
  br i1 %cmp, label %loop, label %exit

exit:
  ret void
}

;--- threshold.ll
; CHECK-LABEL: LV: Checking a loop in 'threshold_equal'
; CHECK: LV: Cross-part load redundancy estimate: opportunities=1, predicted-saved-cost=1, loop-cost=10, saving=5%, required=5%; selecting IC=2.
; CHECK-NEXT: LV: Exact cross-part load redundancy predicts a downstream saving; raising IC to 2.
; CHECK: Executing best plan with VF=4, UF=2

target triple = "aarch64-unknown-linux-gnu"

; With -force-target-instruction-cost=1 the modeled vector body costs exactly
; 10: scalar steps, three address computations, two widened loads, one add, one
; widened store, the backedge, and the canonical IV increment. The second
; address is derived from a loop-invariant %a + 4 so that no in-loop index
; arithmetic is costed. The single redundant load saves exactly 1, so the
; comparison is 1 * 100 == 10 * 2 * 5, i.e. exact equality with the 5%
; threshold.
define void @threshold_equal(ptr noalias %a, ptr noalias %c, i64 %n) {
entry:
  %a.plus.4 = getelementptr inbounds i32, ptr %a, i64 4
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %a.iv = getelementptr inbounds i32, ptr %a, i64 %iv
  %l1 = load i32, ptr %a.iv, align 4
  %a.iv.plus.4 = getelementptr inbounds i32, ptr %a.plus.4, i64 %iv
  %l2 = load i32, ptr %a.iv.plus.4, align 4
  %sum = add i32 %l1, %l2
  %c.iv = getelementptr inbounds i32, ptr %c, i64 %iv
  store i32 %sum, ptr %c.iv, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %cmp = icmp slt i64 %iv.next, %n
  br i1 %cmp, label %loop, label %exit

exit:
  ret void
}

;--- duplicate.ll
; CHECK-LABEL: LV: Checking a loop in 'duplicate_after_cross_part'
; CHECK: LV: Cross-part load redundancy estimate: opportunities=1, predicted-saved-cost={{[^,]+}}, loop-cost={{[^,]+}}, saving={{[0-9]+}}%, required=6%; skipping.
; CHECK-NOT: Exact cross-part load redundancy predicts a downstream saving
; CHECK: LV: Found a vectorizable loop

target triple = "aarch64-unknown-linux-gnu"

define void @duplicate_after_cross_part(ptr noalias %a, ptr noalias %c, i64 %n) {
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %a.iv = getelementptr inbounds i32, ptr %a, i64 %iv
  %l1 = load i32, ptr %a.iv, align 4
  %iv.plus.4 = add nuw nsw i64 %iv, 4
  %a.iv.plus.4 = getelementptr inbounds i32, ptr %a, i64 %iv.plus.4
  %l2 = load i32, ptr %a.iv.plus.4, align 4
  %l3 = load i32, ptr %a.iv.plus.4, align 4
  %sum.1 = add i32 %l1, %l2
  %sum.2 = add i32 %sum.1, %l3
  %c.iv = getelementptr inbounds i32, ptr %c, i64 %iv
  store i32 %sum.2, ptr %c.iv, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %cmp = icmp slt i64 %iv.next, %n
  br i1 %cmp, label %loop, label %exit

exit:
  ret void
}
