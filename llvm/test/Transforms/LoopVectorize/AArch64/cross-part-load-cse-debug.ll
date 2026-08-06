; REQUIRES: asserts
; RUN: split-file %s %t
;
; When cross-part analysis selects IC=2 after the ordinary heuristics decline
; interleaving, it must not emit a contradictory non-interleaving diagnostic.
; RUN: opt -passes=loop-vectorize -force-vector-width=4 \
; RUN:     -force-target-max-vector-interleave=2 -small-loop-cost=0 \
; RUN:     -enable-interleave-cse -interleave-cse-min-ops=1 \
; RUN:     -interleave-cse-min-pct=1 \
; RUN:     -debug-only=loop-vectorize -disable-output %t/success.ll 2>&1 \
; RUN:     | FileCheck %t/success.ll --check-prefix=SUCCESS
;
; A fixed-VF, wide-lane-mask tail-folded plan reaches IC selection, but its
; masked widened loads remain outside the exact unmasked-load model.
; RUN: opt -passes=loop-vectorize -force-vector-width=4 \
; RUN:     -force-target-max-vector-interleave=2 -small-loop-cost=0 \
; RUN:     -force-target-supports-masked-memory-ops \
; RUN:     -force-tail-folding-style=data-and-control \
; RUN:     -tail-folding-policy=must-fold-tail -enable-wide-lane-mask \
; RUN:     -enable-interleave-cse -interleave-cse-min-ops=1 \
; RUN:     -interleave-cse-min-pct=1 -debug-only=loop-vectorize \
; RUN:     -disable-output %t/success.ll 2>&1 \
; RUN:     | FileCheck %t/success.ll --check-prefix=MASKED
;
; When the ordinary branch-cost heuristic recommends IC=1, a successful
; cross-part selection must return before emitting its baseline diagnostic.
; RUN: opt -passes=loop-vectorize -force-vector-width=4 \
; RUN:     -force-target-max-vector-interleave=2 \
; RUN:     -force-target-instruction-cost=1 -small-loop-cost=12 \
; RUN:     -enable-loadstore-runtime-interleave=false \
; RUN:     -enable-interleave-cse -interleave-cse-min-ops=1 \
; RUN:     -interleave-cse-min-pct=1 \
; RUN:     -debug-only=loop-vectorize -disable-output %t/success.ll 2>&1 \
; RUN:     | FileCheck %t/success.ll --check-prefix=SUCCESS-SMALL
;
; The same-part duplicate after a genuine cross-part match must report exactly
; one opportunity and fail the requested minimum of two.
; RUN: opt -passes=loop-vectorize -force-vector-width=4 \
; RUN:     -force-target-max-vector-interleave=2 -small-loop-cost=0 \
; RUN:     -enable-interleave-cse -interleave-cse-min-ops=2 \
; RUN:     -debug-only=loop-vectorize \
; RUN:     -disable-output %t/duplicate.ll 2>&1 \
; RUN:     | FileCheck %t/duplicate.ll
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
; A forced scalable VF reaches vectorization but remains outside the fixed-VF
; heuristic and executes one scalable part.
; RUN: opt -passes=loop-vectorize -mtriple=aarch64-none-linux-gnu -mattr=+sve \
; RUN:     -force-vector-width="vscale x 2" \
; RUN:     -force-target-max-vector-interleave=2 -small-loop-cost=0 \
; RUN:     -enable-interleave-cse -interleave-cse-min-ops=1 \
; RUN:     -interleave-cse-min-pct=1 -debug-only=loop-vectorize \
; RUN:     -disable-output %t/success.ll 2>&1 \
; RUN:     | FileCheck %t/success.ll --check-prefix=SCALABLE

;--- success.ll
; SUCCESS-LABEL: LV: Checking a loop in 'positive'
; SUCCESS: LV: Cross-part load overlap estimate: ops=1, required-ops=1, predicted-saved-cost={{[^,]+}}, loop-cost={{[^,]+}}, saving={{[0-9]+}}%, required=1%; selecting IC=2.
; SUCCESS-NEXT: LV: Exact cross-part load overlap predicts a downstream saving; raising IC to 2.
; SUCCESS-NOT: LV: Not Interleaving.
; SUCCESS: LV: Found a vectorizable loop
; MASKED-LABEL: LV: Checking a loop in 'positive'
; MASKED: LV: Cross-part load overlap estimate: ops=0,
; MASKED-NOT: Exact cross-part load overlap predicts a downstream saving
; MASKED: Executing best plan with VF=4, UF=1
; SUCCESS-SMALL-LABEL: LV: Checking a loop in 'positive'
; SUCCESS-SMALL: LV: Cross-part load overlap estimate: ops=1, required-ops=1, predicted-saved-cost={{[^,]+}}, loop-cost={{[^,]+}}, saving={{[0-9]+}}%, required=1%; selecting IC=2.
; SUCCESS-SMALL-NEXT: LV: Exact cross-part load overlap predicts a downstream saving; raising IC to 2.
; SUCCESS-SMALL-NOT: LV: Interleaving to reduce branch cost.
; SUCCESS-SMALL: LV: Found a vectorizable loop
; DISABLED-SMALL-LABEL: LV: Checking a loop in 'positive'
; DISABLED-SMALL-NOT: Cross-part load overlap
; DISABLED-SMALL: LV: Interleaving to reduce branch cost.
; DISABLED-SMALL-NOT: Cross-part load overlap
; DISABLED-SMALL: LV: Found a vectorizable loop
; SCALABLE-LABEL: LV: Checking a loop in 'positive'
; SCALABLE-NOT: LV: Cross-part load overlap estimate:
; SCALABLE: LV: VF is vscale x 2
; SCALABLE-NEXT: LV: Not Interleaving.
; SCALABLE-NOT: LV: Cross-part load overlap estimate:
; SCALABLE: LV: Found a vectorizable loop (vscale x 2)
; SCALABLE: Executing best plan with VF=vscale x 2, UF=1

target triple = "aarch64-unknown-linux-gnu"

define void @positive(ptr noalias %a, ptr noalias %c, i64 %n) {
entry:
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %loop ]
  %a.i = getelementptr inbounds i32, ptr %a, i64 %i
  %l1 = load i32, ptr %a.i, align 4
  %i4 = add nuw nsw i64 %i, 4
  %a.i4 = getelementptr inbounds i32, ptr %a, i64 %i4
  %l2 = load i32, ptr %a.i4, align 4
  %sum = add i32 %l1, %l2
  %c.i = getelementptr inbounds i32, ptr %c, i64 %i
  store i32 %sum, ptr %c.i, align 4
  %i.next = add nuw nsw i64 %i, 1
  %cmp = icmp slt i64 %i.next, %n
  br i1 %cmp, label %loop, label %exit

exit:
  ret void
}

;--- duplicate.ll
; CHECK-LABEL: LV: Checking a loop in 'duplicate_after_cross_part'
; CHECK: LV: Cross-part load overlap estimate: ops=1, required-ops=2, predicted-saved-cost={{[^,]+}}, loop-cost={{[^,]+}}, saving={{[0-9]+}}%, required=5%; skipping.
; CHECK-NOT: Exact cross-part load overlap predicts a downstream saving
; CHECK: LV: Found a vectorizable loop

target triple = "aarch64-unknown-linux-gnu"

define void @duplicate_after_cross_part(ptr noalias %a, ptr noalias %c, i64 %n) {
entry:
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %loop ]
  %a.i = getelementptr inbounds i32, ptr %a, i64 %i
  %l1 = load i32, ptr %a.i, align 4
  %i4 = add nuw nsw i64 %i, 4
  %a.i4 = getelementptr inbounds i32, ptr %a, i64 %i4
  %l2 = load i32, ptr %a.i4, align 4
  %l3 = load i32, ptr %a.i4, align 4
  %sum.1 = add i32 %l1, %l2
  %sum.2 = add i32 %sum.1, %l3
  %c.i = getelementptr inbounds i32, ptr %c, i64 %i
  store i32 %sum.2, ptr %c.i, align 4
  %i.next = add nuw nsw i64 %i, 1
  %cmp = icmp slt i64 %i.next, %n
  br i1 %cmp, label %loop, label %exit

exit:
  ret void
}
