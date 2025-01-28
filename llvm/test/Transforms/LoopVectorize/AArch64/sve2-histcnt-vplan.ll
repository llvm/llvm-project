; RUN: opt < %s -mattr=+sve2 -passes=loop-vectorize,instcombine -enable-histogram-loop-vectorization -sve-gather-overhead=2 -sve-scatter-overhead=2 -force-vector-interleave=1 -debug-only=loop-vectorize -S 2>&1 | FileCheck %s
; REQUIRES: asserts

target triple = "aarch64-unknown-linux-gnu"

;; Based on the following C code:
;;
;; void simple_histogram(int *buckets, unsigned *indices, int N) {
;;   for (int i = 0; i < N; ++i)
;;     buckets[indices[i]]++;
;; }

;; Check that the scalar plan contains the original instructions.
; CHECK: VPlan 'Initial VPlan for VF={1},UF>=1' {
; CHECK-NEXT: Live-in [[VFxUF:.*]] = VF * UF
; CHECK-NEXT: Live-in [[VTC:.*]] = vector-trip-count
; CHECK-NEXT: Live-in [[OTC:.*]] = original trip-count
; CHECK-EMPTY:
; CHECK-NEXT: ir-bb<entry>:
; CHECK-NEXT: Successor(s): vector.ph
; CHECK-EMPTY:
; CHECK-NEXT: vector.ph:
; CHECK-NEXT: Successor(s): vector loop
; CHECK-EMPTY:
; CHECK-NEXT: <x1> vector loop: {
; CHECK-NEXT:   vector.body:
; CHECK-NEXT:     EMIT [[IV:.*]] = CANONICAL-INDUCTION ir<0>, [[IV_NEXT:.*]]
; CHECK-NEXT:     [[STEPS:vp.*]] = SCALAR-STEPS [[IV]], ir<1>
; CHECK-NEXT:     CLONE [[GEP_IDX:.*]] = getelementptr inbounds ir<%indices>, [[STEPS]]
; CHECK-NEXT:     CLONE [[IDX:.*]] = load [[GEP_IDX]]
; CHECK-NEXT:     CLONE [[EXT_IDX:.*]] = zext [[IDX]]
; CHECK-NEXT:     CLONE [[GEP_BUCKET:.*]] = getelementptr inbounds ir<%buckets>, [[EXT_IDX]]
; CHECK-NEXT:     CLONE [[HISTVAL:.*]] = load [[GEP_BUCKET]]
; CHECK-NEXT:     CLONE [[UPDATE:.*]] = add nsw [[HISTVAL]], ir<1>
; CHECK-NEXT:     CLONE store [[UPDATE]], [[GEP_BUCKET]]
; CHECK-NEXT:     EMIT [[IV_NEXT]] = add nuw [[IV]], [[VFxUF]]
; CHECK-NEXT:     EMIT branch-on-count [[IV_NEXT]], [[VTC]]
; CHECK-NEXT:   No successors
; CHECK-NEXT: }
; CHECK-NEXT: Successor(s): middle.block
; CHECK-EMPTY:
; CHECK-NEXT: middle.block:
; CHECK-NEXT:   EMIT [[TC_CHECK:.*]] = icmp eq [[OTC:.*]], [[VTC]]
; CHECK-NEXT:   EMIT branch-on-cond [[TC_CHECK]]
; CHECK-NEXT: Successor(s): ir-bb<for.exit>, scalar.ph
; CHECK-EMPTY:
; CHECK-NEXT: scalar.ph:
; CHECK-NEXT:   EMIT vp<[[RESUME:%.+]]> = resume-phi [[VTC]], ir<0>
; CHECK-NEXT: Successor(s): ir-bb<for.body>
; CHECK-EMPTY:
; CHECK-NEXT: ir-bb<for.body>:
; CHECK-NEXT:   IR   %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ] (extra operand: vp<[[RESUME]]> from scalar.ph)
; CHECK:        IR   %exitcond = icmp eq i64 %iv.next, %N
; CHECK-NEXT: No successors
; CHECK-EMPTY:
; CHECK-NEXT: ir-bb<for.exit>:
; CHECK-NEXT: No successors
; CHECK-NEXT: }

;; Check that the vectorized plan contains a histogram recipe instead.
; CHECK: VPlan 'Initial VPlan for VF={vscale x 2,vscale x 4},UF>=1' {
; CHECK-NEXT: Live-in [[VFxUF:.*]] = VF * UF
; CHECK-NEXT: Live-in [[VTC:.*]] = vector-trip-count
; CHECK-NEXT: Live-in [[OTC:.*]] = original trip-count
; CHECK-EMPTY:
; CHECK-NEXT: ir-bb<entry>:
; CHECK-NEXT: Successor(s): vector.ph
; CHECK-EMPTY:
; CHECK-NEXT: vector.ph:
; CHECK-NEXT: Successor(s): vector loop
; CHECK-EMPTY:
; CHECK-NEXT: <x1> vector loop: {
; CHECK-NEXT:   vector.body:
; CHECK-NEXT:     EMIT [[IV:.*]] = CANONICAL-INDUCTION ir<0>, [[IV_NEXT:.*]]
; CHECK-NEXT:     [[STEPS:vp.*]] = SCALAR-STEPS [[IV]], ir<1>
; CHECK-NEXT:     CLONE [[GEP_IDX:.*]] = getelementptr inbounds ir<%indices>, [[STEPS]]
; CHECK-NEXT:     [[VECP_IDX:vp.*]] = vector-pointer [[GEP_IDX]]
; CHECK-NEXT:     WIDEN [[IDX:.*]] = load [[VECP_IDX]]
; CHECK-NEXT:     WIDEN-CAST [[EXT_IDX:.*]] = zext [[IDX]] to i64
; CHECK-NEXT:     WIDEN-GEP Inv[Var] [[GEP_BUCKET:.*]] = getelementptr inbounds ir<%buckets>, [[EXT_IDX]]
; CHECK-NEXT:     WIDEN-HISTOGRAM buckets: [[GEP_BUCKET]], inc: ir<1>
; CHECK-NEXT:     EMIT [[IV_NEXT]] = add nuw [[IV]], [[VFxUF]]
; CHECK-NEXT:     EMIT branch-on-count [[IV_NEXT]], [[VTC]]
; CHECK-NEXT:   No successors
; CHECK-NEXT: }
; CHECK-NEXT: Successor(s): middle.block
; CHECK-EMPTY:
; CHECK-NEXT: middle.block:
; CHECK-NEXT:   EMIT [[TC_CHECK:.*]] = icmp eq [[OTC]], [[VTC]]
; CHECK-NEXT:   EMIT branch-on-cond [[TC_CHECK]]
; CHECK-NEXT: Successor(s): ir-bb<for.exit>, scalar.ph
; CHECK-EMPTY:
; CHECK-NEXT: scalar.ph:
; CHECK-NEXT:   EMIT vp<[[RESUME:%.+]]> = resume-phi [[VTC]], ir<0>
; CHECK-NEXT: Successor(s): ir-bb<for.body>
; CHECK-EMPTY:
; CHECK-NEXT: ir-bb<for.body>:
; CHECK-NEXT:   IR   %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ] (extra operand: vp<[[RESUME]]> from scalar.ph)
; CHECK:        IR   %exitcond = icmp eq i64 %iv.next, %N
; CHECK-NEXT: No successors
; CHECK-EMPTY:
; CHECK-NEXT: ir-bb<for.exit>:
; CHECK-NEXT: No successors
; CHECK-NEXT: }

define void @simple_histogram(ptr noalias %buckets, ptr readonly %indices, i64 %N) {
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %gep.indices = getelementptr inbounds i32, ptr %indices, i64 %iv
  %l.idx = load i32, ptr %gep.indices, align 4
  %idxprom1 = zext i32 %l.idx to i64
  %gep.bucket = getelementptr inbounds i32, ptr %buckets, i64 %idxprom1
  %l.bucket = load i32, ptr %gep.bucket, align 4
  %inc = add nsw i32 %l.bucket, 1
  store i32 %inc, ptr %gep.bucket, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, %N
  br i1 %exitcond, label %for.exit, label %for.body

for.exit:
  ret void
}
