; RUN: split-file %s %t
;
; The analysis only predicts downstream savings to guide IC selection. It does
; not modify VPlan or eliminate any widened loads.
;
; RUN: opt -passes=loop-vectorize -force-vector-width=4 \
; RUN:     -force-target-max-vector-interleave=2 -small-loop-cost=0 \
; RUN:     -enable-interleave-cse -interleave-cse-min-ops=1 \
; RUN:     -interleave-cse-min-pct=1 \
; RUN:     -S %t/positive.ll | FileCheck %t/positive.ll --check-prefix=FORCED
; RUN: opt -passes=loop-vectorize -force-target-max-vector-interleave=2 \
; RUN:     -small-loop-cost=0 -enable-interleave-cse \
; RUN:     -interleave-cse-min-ops=1 -interleave-cse-min-pct=1 \
; RUN:     -S %t/positive.ll \
; RUN:     | FileCheck %t/positive.ll --check-prefix=PRODUCTION
; RUN: opt -passes=loop-vectorize -force-vector-width=4 \
; RUN:     -force-target-max-vector-interleave=2 -small-loop-cost=0 \
; RUN:     -enable-interleave-cse -S %t/positive.ll \
; RUN:     | FileCheck %t/positive.ll --check-prefix=THRESHOLD
; RUN: opt -passes=loop-vectorize -force-vector-width=4 \
; RUN:     -force-target-max-vector-interleave=2 -small-loop-cost=0 \
; RUN:     -enable-interleave-cse -interleave-cse-min-ops=1 \
; RUN:     -interleave-cse-min-pct=100 -S %t/positive.ll \
; RUN:     | FileCheck %t/positive.ll --check-prefix=PERCENT
; Force deterministic costs so the ordinary branch-cost heuristic recommends
; IC=1, allowing this run to verify that cross-part analysis can raise it to 2.
; RUN: opt -passes=loop-vectorize -force-vector-width=4 \
; RUN:     -force-target-max-vector-interleave=2 \
; RUN:     -force-target-instruction-cost=1 -small-loop-cost=12 \
; RUN:     -enable-loadstore-runtime-interleave=false \
; RUN:     -enable-interleave-cse -interleave-cse-min-ops=1 \
; RUN:     -interleave-cse-min-pct=1 \
; RUN:     -S %t/positive.ll | FileCheck %t/positive.ll --check-prefix=SMALL-IC
; RUN: opt -passes=loop-vectorize -force-vector-width=4 \
; RUN:     -force-target-max-vector-interleave=2 -small-loop-cost=0 \
; RUN:     -S %t/positive.ll | FileCheck %t/positive.ll --check-prefix=DISABLED
; RUN: opt -passes=loop-vectorize -force-vector-width=4 \
; RUN:     -force-target-max-vector-interleave=2 -small-loop-cost=0 \
; RUN:     -enable-interleave-cse -force-vector-interleave=1 \
; RUN:     -S %t/positive.ll | FileCheck %t/positive.ll --check-prefix=USERIC
; RUN: opt -passes=loop-vectorize -force-vector-width=4 \
; RUN:     -force-target-max-vector-interleave=1 -small-loop-cost=0 \
; RUN:     -enable-interleave-cse -interleave-cse-min-ops=1 \
; RUN:     -interleave-cse-min-pct=1 \
; RUN:     -S %t/positive.ll | FileCheck %t/positive.ll --check-prefix=MAXIC
; RUN: opt -passes=loop-vectorize -mtriple=aarch64-none-linux-gnu -mattr=+sve \
; RUN:     -force-vector-width="vscale x 2" \
; RUN:     -force-target-max-vector-interleave=2 -small-loop-cost=0 \
; RUN:     -enable-interleave-cse -interleave-cse-min-ops=1 \
; RUN:     -interleave-cse-min-pct=1 -S %t/positive.ll \
; RUN:     | FileCheck %t/positive.ll --check-prefix=SCALABLE
;
; RUN: opt -passes=loop-vectorize -force-vector-width=4 \
; RUN:     -force-target-max-vector-interleave=2 -small-loop-cost=0 \
; RUN:     -enable-interleave-cse -interleave-cse-min-ops=2 \
; RUN:     -S %t/duplicate.ll \
; RUN:     | FileCheck %t/duplicate.ll
; RUN: opt -passes=loop-vectorize -force-vector-width=4 \
; RUN:     -force-target-max-vector-interleave=2 -small-loop-cost=0 \
; RUN:     -enable-interleave-cse -interleave-cse-min-ops=2 \
; RUN:     -interleave-cse-min-pct=6 -S %t/two-opportunities.ll \
; RUN:     | FileCheck %t/two-opportunities.ll
; RUN: opt -passes=loop-vectorize -force-vector-width=4 \
; RUN:     -force-target-max-vector-interleave=2 -small-loop-cost=0 \
; RUN:     -enable-interleave-cse -interleave-cse-min-ops=1 \
; RUN:     -interleave-cse-min-pct=1 -S %t/type-mismatch.ll \
; RUN:     | FileCheck %t/type-mismatch.ll
; RUN: opt -passes=loop-vectorize -force-vector-width=4 \
; RUN:     -force-target-max-vector-interleave=2 -small-loop-cost=0 \
; RUN:     -enable-interleave-cse -interleave-cse-min-ops=1 \
; RUN:     -interleave-cse-min-pct=1 -S %t/simple-negatives.ll \
; RUN:     | FileCheck %t/simple-negatives.ll
; RUN: opt -passes=loop-vectorize -force-vector-width=4 \
; RUN:     -force-target-max-vector-interleave=2 -small-loop-cost=0 \
; RUN:     -enable-interleave-cse -interleave-cse-min-ops=1 \
; RUN:     -interleave-cse-min-pct=1 -S %t/provenance.ll \
; RUN:     | FileCheck %t/provenance.ll
; RUN: opt -passes=loop-vectorize -force-vector-width=4 \
; RUN:     -force-target-max-vector-interleave=2 -small-loop-cost=0 \
; RUN:     -enable-interleave-cse -interleave-cse-min-ops=1 \
; RUN:     -interleave-cse-min-pct=1 -S %t/poison.ll \
; RUN:     | FileCheck %t/poison.ll
; RUN: opt -passes=loop-vectorize -force-vector-width=4 \
; RUN:     -force-target-max-vector-interleave=2 -small-loop-cost=0 \
; RUN:     -enable-interleave-cse -interleave-cse-min-ops=1 \
; RUN:     -interleave-cse-min-pct=1 -S %t/multi-block.ll \
; RUN:     | FileCheck %t/multi-block.ll
; RUN: opt -passes=loop-vectorize -force-vector-width=4 \
; RUN:     -force-target-max-vector-interleave=2 -small-loop-cost=0 \
; RUN:     -enable-interleave-cse -interleave-cse-min-ops=1 \
; RUN:     -interleave-cse-min-pct=1 -S %t/reverse.ll \
; RUN:     | FileCheck %t/reverse.ll

;--- positive.ll
target triple = "aarch64-unknown-linux-gnu"

; With VF=4, a[i+4] in part 0 has the same modeled vector address as a[i] in
; part 1. The predicted downstream saving raises IC to 2, while all four
; widened loads remain because the analysis does not realize the overlap.
;
; FORCED-LABEL: @positive(
; FORCED:       vector.body:
; FORCED-COUNT-4: load <4 x i32>
; FORCED-NOT:   load <4 x i32>
; FORCED:       add nuw i64 %index, 8
;
; PRODUCTION-LABEL: @positive(
; PRODUCTION:       vector.body:
; PRODUCTION-COUNT-4: load <4 x i32>
; PRODUCTION-NOT:   load <4 x i32>
; PRODUCTION:       add nuw i64 %index, 8
;
; THRESHOLD-LABEL: @positive(
; THRESHOLD:       add nuw i64 %index, 4
;
; PERCENT-LABEL: @positive(
; PERCENT:       add nuw i64 %index, 4
;
; SMALL-IC-LABEL: @positive(
; SMALL-IC:       add nuw i64 %index, 8
;
; DISABLED-LABEL: @positive(
; DISABLED:       add nuw i64 %index, 4
;
; USERIC-LABEL: @positive(
; USERIC:       add nuw i64 %index, 4
;
; MAXIC-LABEL: @positive(
; MAXIC:       add nuw i64 %index, 4
;
; SCALABLE-LABEL: @positive(
; SCALABLE:       call i64 @llvm.vscale.i64()
; SCALABLE:       vector.body:
; SCALABLE-COUNT-2: load <vscale x 2 x i32>
; SCALABLE-NOT:   load <vscale x 2 x i32>
; SCALABLE:       %index.next = add nuw i64 %index, %
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
target triple = "aarch64-unknown-linux-gnu"

; The first a[i+4] forms a predicted cross-part overlap with a[i] in part 1.
; The same-part duplicate must not count as a second opportunity.
;
; CHECK-LABEL: @duplicate_after_cross_part(
; CHECK:       add nuw i64 %index, 4
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

;--- two-opportunities.ll
target triple = "aarch64-unknown-linux-gnu"

; At VF=4, a[i+4] in part 0 overlaps a[i] in part 1, and a[i+8] in
; part 0 independently overlaps a[i+4] in part 1. Both opportunities are
; credited, satisfying the requested minimum of two and raising IC to 2.
;
; CHECK-LABEL: @two_opportunities(
; CHECK:       vector.body:
; CHECK-COUNT-6: load <4 x i32>
; CHECK-NOT:   load <4 x i32>
; CHECK:       add nuw i64 %index, 8
define void @two_opportunities(ptr noalias %a, ptr noalias %c, i64 %n) {
entry:
  br label %loop
loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %loop ]
  %a.i = getelementptr inbounds i32, ptr %a, i64 %i
  %l1 = load i32, ptr %a.i, align 4
  %i4 = add nuw nsw i64 %i, 4
  %a.i4 = getelementptr inbounds i32, ptr %a, i64 %i4
  %l2 = load i32, ptr %a.i4, align 4
  %i8 = add nuw nsw i64 %i, 8
  %a.i8 = getelementptr inbounds i32, ptr %a, i64 %i8
  %l3 = load i32, ptr %a.i8, align 4
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

;--- type-mismatch.ll
target triple = "aarch64-unknown-linux-gnu"

; The part-shifted addresses are equal, but loads of different value types
; cannot share a result. The type component of the key keeps IC at 1.
;
; CHECK-LABEL: @different_types(
; CHECK:       vector.body:
; CHECK-COUNT-1: load <4 x i32>
; CHECK-COUNT-1: load <4 x float>
; CHECK-NOT:   load <
; CHECK:       add nuw i64 %index, 4
define void @different_types(ptr noalias %a, ptr noalias %c, i64 %n) {
entry:
  br label %loop
loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %loop ]
  %a.i = getelementptr inbounds i32, ptr %a, i64 %i
  %l1 = load i32, ptr %a.i, align 4
  %i4 = add nuw nsw i64 %i, 4
  %a.i4 = getelementptr inbounds float, ptr %a, i64 %i4
  %l2 = load float, ptr %a.i4, align 4
  %l2.bits = bitcast float %l2 to i32
  %sum = add i32 %l1, %l2.bits
  %c.i = getelementptr inbounds i32, ptr %c, i64 %i
  store i32 %sum, ptr %c.i, align 4
  %i.next = add nuw nsw i64 %i, 1
  %cmp = icmp slt i64 %i.next, %n
  br i1 %cmp, label %loop, label %exit
exit:
  ret void
}

;--- simple-negatives.ll
target triple = "aarch64-unknown-linux-gnu"

; a[i] + a[i+3]: the offset 3 is not a multiple of VF=4, so no part-shifted
; address ever matches exactly. Interleave count stays 1.
;
; CHECK-LABEL: @inequality(
; CHECK:       vector.body:
; CHECK-COUNT-2: load <4 x i32>
; CHECK-NOT:   load <4 x i32>
; CHECK:       add nuw i64 %index, 4
define void @inequality(ptr noalias %a, ptr noalias %c, i64 %n) {
entry:
  br label %loop
loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %loop ]
  %a.i = getelementptr inbounds i32, ptr %a, i64 %i
  %l1 = load i32, ptr %a.i, align 4
  %i3 = add nuw nsw i64 %i, 3
  %a.i3 = getelementptr inbounds i32, ptr %a, i64 %i3
  %l2 = load i32, ptr %a.i3, align 4
  %sum = add i32 %l1, %l2
  %c.i = getelementptr inbounds i32, ptr %c, i64 %i
  store i32 %sum, ptr %c.i, align 4
  %i.next = add nuw nsw i64 %i, 1
  %cmp = icmp slt i64 %i.next, %n
  br i1 %cmp, label %loop, label %exit
exit:
  ret void
}

; A store occurs between the two matching logical-part loads, so the analysis
; must not credit their overlap and IC remains 1.
;
; CHECK-LABEL: @write_between(
; CHECK:       vector.body:
; CHECK-COUNT-2: load <4 x i32>
; CHECK-NOT:   load <4 x i32>
; CHECK:       add nuw i64 %index, 4
define void @write_between(ptr noalias %a, ptr noalias %b, ptr noalias %c, i64 %n) {
entry:
  br label %loop
loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %loop ]
  %a.i = getelementptr inbounds i32, ptr %a, i64 %i
  %l1 = load i32, ptr %a.i, align 4
  %b.i = getelementptr inbounds i32, ptr %b, i64 %i
  store i32 %l1, ptr %b.i, align 4
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

; Stride-2 accesses are represented by interleave recipes rather than supported
; simple consecutive widened loads, so the analysis fails closed and leaves IC
; at 1.
;
; CHECK-LABEL: @non_unit_stride(
; CHECK:       vector.body:
; CHECK-COUNT-2: load <8 x i32>
; CHECK-NOT:   load <8 x i32>
; CHECK:       add nuw i64 %index, 4
define void @non_unit_stride(ptr noalias %a, ptr noalias %c, i64 %n) {
entry:
  br label %loop
loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %loop ]
  %twice.i = shl nuw nsw i64 %i, 1
  %a.i = getelementptr inbounds i32, ptr %a, i64 %twice.i
  %l1 = load i32, ptr %a.i, align 4
  %twice.i4 = add nuw nsw i64 %twice.i, 4
  %a.i4 = getelementptr inbounds i32, ptr %a, i64 %twice.i4
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

;--- provenance.ll
target triple = "aarch64-unknown-linux-gnu"

; VPlan folds each identical-arm select to its underlying GEP. The constant
; condition also lets ScalarEvolution canonicalize the original select to that
; same GEP. Deriving the address from the folded VPValue therefore exposes the
; exact cross-part overlap and raises IC to 2.
;
; CHECK-LABEL: @folded_provenance(
; CHECK:       vector.body:
; CHECK:       add nuw i64 %index, 8
define void @folded_provenance(ptr noalias %a, ptr noalias %c, i64 %n) {
entry:
  br label %loop
loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %loop ]
  %a.i = getelementptr inbounds i32, ptr %a, i64 %i
  %p1 = select i1 true, ptr %a.i, ptr %a.i
  %l1 = load i32, ptr %p1, align 4
  %i4 = add nuw nsw i64 %i, 4
  %a.i4 = getelementptr inbounds i32, ptr %a, i64 %i4
  %p2 = select i1 true, ptr %a.i4, ptr %a.i4
  %l2 = load i32, ptr %p2, align 4
  %sum = add i32 %l1, %l2
  %c.i = getelementptr inbounds i32, ptr %c, i64 %i
  store i32 %sum, ptr %c.i, align 4
  %i.next = add nuw nsw i64 %i, 1
  %cmp = icmp slt i64 %i.next, %n
  br i1 %cmp, label %loop, label %exit
exit:
  ret void
}

;--- poison.ll
target triple = "aarch64-unknown-linux-gnu"

; The addresses overlap, but poison-generating annotations require metadata
; intersection when the overlap is realized. The analysis rejects these loads
; rather than predicting reuse it cannot preserve directly.
;
; CHECK-LABEL: @poison_annotations(
; CHECK:       vector.body:
; CHECK:       add nuw i64 %index, 4
define void @poison_annotations(ptr noalias %a, ptr noalias %c, i64 %n) {
entry:
  br label %loop
loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %loop ]
  %a.i = getelementptr inbounds i32, ptr %a, i64 %i
  %l1 = load i32, ptr %a.i, align 4, !range !0
  %i4 = add nuw nsw i64 %i, 4
  %a.i4 = getelementptr inbounds i32, ptr %a, i64 %i4
  %l2 = load i32, ptr %a.i4, align 4, !range !1
  %sum = add i32 %l1, %l2
  %c.i = getelementptr inbounds i32, ptr %c, i64 %i
  store i32 %sum, ptr %c.i, align 4
  %i.next = add nuw nsw i64 %i, 1
  %cmp = icmp slt i64 %i.next, %n
  br i1 %cmp, label %loop, label %exit
exit:
  ret void
}

!0 = !{i32 0, i32 100}
!1 = !{i32 0, i32 101}

;--- multi-block.ll
target triple = "aarch64-unknown-linux-gnu"

; The matching forward loads are present, but the separate latch places this
; loop outside the analysis's single-block scope.
;
; CHECK-LABEL: @multi_block(
; CHECK:       vector.body:
; CHECK:       add nuw i64 %index, 4
define void @multi_block(ptr noalias %a, ptr noalias %c, i64 %n) {
entry:
  br label %loop.header
loop.header:
  %i = phi i64 [ 0, %entry ], [ %i.next, %loop.latch ]
  %a.i = getelementptr inbounds i32, ptr %a, i64 %i
  %l1 = load i32, ptr %a.i, align 4
  %i4 = add nuw nsw i64 %i, 4
  %a.i4 = getelementptr inbounds i32, ptr %a, i64 %i4
  %l2 = load i32, ptr %a.i4, align 4
  %sum = add i32 %l1, %l2
  %c.i = getelementptr inbounds i32, ptr %c, i64 %i
  store i32 %sum, ptr %c.i, align 4
  br label %loop.latch
loop.latch:
  %i.next = add nuw nsw i64 %i, 1
  %cmp = icmp slt i64 %i.next, %n
  br i1 %cmp, label %loop.header, label %exit
exit:
  ret void
}

;--- reverse.ll
target triple = "aarch64-unknown-linux-gnu"

; The reverse access is physically between the two forward loads. Its load is
; not credited, while its pure end-pointer address recipe must not be treated as
; a write barrier. The one forward equality therefore raises IC to 2.
;
; CHECK-LABEL: @reverse_between(
; CHECK:       vector.body:
; CHECK-COUNT-6: load <4 x i32>
; CHECK-NOT:   load <4 x i32>
; CHECK:       add nuw i64 %index, 8
define void @reverse_between(ptr noalias %a, ptr noalias %c, i64 %n) {
entry:
  %last = add i64 %n, -1
  br label %loop
loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %loop ]
  %a.i = getelementptr inbounds i32, ptr %a, i64 %i
  %l1 = load i32, ptr %a.i, align 4
  %reverse.i = sub i64 %last, %i
  %a.reverse = getelementptr inbounds i32, ptr %a, i64 %reverse.i
  %reverse = load i32, ptr %a.reverse, align 4
  %i4 = add nuw nsw i64 %i, 4
  %a.i4 = getelementptr inbounds i32, ptr %a, i64 %i4
  %l2 = load i32, ptr %a.i4, align 4
  %sum.forward = add i32 %l1, %l2
  %sum = add i32 %sum.forward, %reverse
  %c.i = getelementptr inbounds i32, ptr %c, i64 %i
  store i32 %sum, ptr %c.i, align 4
  %i.next = add nuw nsw i64 %i, 1
  %cmp = icmp slt i64 %i.next, %n
  br i1 %cmp, label %loop, label %exit
exit:
  ret void
}
