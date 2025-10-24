; RUN: opt -passes=loop-vectorize -debug-only=loop-vectorize \
; RUN:   -scalable-vectorization=on -force-target-supports-scalable-vectors \
; RUN:   -disable-output 2>&1 < %s | FileCheck %s


; This function is generated from the following C/C++ program:
; int simple_csa_int_select(int N, int *data, int a) {
;   int t = -1;
;   for (int i = 0; i < N; i++) {
;     if (a < data[i])
;       t = data[i];
;   }
;   return t; // use t
; }
define i32 @simple_csa_int_select(i32 %N, ptr %data, i64 %a) {
entry:
  %cmp9 = icmp sgt i32 %N, 0
  br i1 %cmp9, label %loop.preheader, label %exit

loop.preheader:                               ; preds = %entry
  %wide.trip.count = zext i32 %N to i64
  br label %loop

exit:                                 ; preds = %loop, %entry
  %t.0.lcssa = phi i32 [ -1, %entry ], [ %spec.select, %loop ]
  ret i32 %t.0.lcssa

loop:                                         ; preds = %loop.preheader, %loop
  %iv = phi i64 [ 0, %loop.preheader ], [ %iv.next, %loop ]
  %t.010 = phi i32 [ -1, %loop.preheader ], [ %spec.select, %loop ]
  %arrayidx = getelementptr inbounds i32, ptr %data, i64 %iv
  %0 = load i32, ptr %arrayidx, align 4
  %1 = sext i32 %0 to i64
  %cmp1 = icmp slt i64 %a, %1
  %spec.select = select i1 %cmp1, i32 %0, i32 %t.010
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond.not = icmp eq i64 %iv.next, %wide.trip.count
  br i1 %exitcond.not, label %exit, label %loop
}

; CHECK: VPlan 'Initial VPlan for VF={vscale x 1},UF>=1' {
; CHECK-NEXT: Live-in vp<%0> = VF
; CHECK-NEXT: Live-in vp<%1> = VF * UF
; CHECK-NEXT: Live-in vp<%2> = vector-trip-count
; CHECK-NEXT: vp<%3> = original trip-count
; CHECK-EMPTY:
; CHECK-NEXT: ir-bb<loop.preheader>:
; CHECK-NEXT:   IR   %wide.trip.count = zext i32 %N to i64
; CHECK-NEXT:   EMIT vp<%3> = EXPAND SCEV (zext i32 %N to i64)
; CHECK-NEXT: Successor(s): scalar.ph, vector.ph
; CHECK-EMPTY:
; CHECK-NEXT: vector.ph:
; CHECK-NEXT: Successor(s): vector loop
; CHECK-EMPTY:
; CHECK-NEXT: <x1> vector loop: {
; CHECK-NEXT:   vector.body:
; CHECK-NEXT:     EMIT vp<%4> = CANONICAL-INDUCTION ir<0>, vp<%index.next>
; CHECK-NEXT:     WIDEN-REDUCTION-PHI ir<%t.010> = phi ir<-1>, vp<%10>
; CHECK-NEXT:     ACTIVE-LANE-MASK-PHI vp<%5> = phi ir<false>, vp<%9>
; CHECK-NEXT:     vp<%6> = SCALAR-STEPS vp<%4>, ir<1>, vp<%0>
; CHECK-NEXT:     CLONE ir<%arrayidx> = getelementptr inbounds ir<%data>, vp<%6>
; CHECK-NEXT:     vp<%7> = vector-pointer ir<%arrayidx>
; CHECK-NEXT:     WIDEN ir<%0> = load vp<%7>
; CHECK-NEXT:     WIDEN-CAST ir<%1> = sext ir<%0> to i64
; CHECK-NEXT:     WIDEN ir<%cmp1> = icmp slt ir<%a>, ir<%1>
; CHECK-NEXT:     EMIT vp<%8> = any-of ir<%cmp1>
; CHECK-NEXT:     WIDEN-SELECT-VECTOR vp<%9> = select  vp<%8>, ir<%cmp1>, vp<%5>
; CHECK-NEXT:     WIDEN-SELECT-VECTOR vp<%10> = select  vp<%8>, ir<%0>, ir<%t.010>
; CHECK-NEXT:     EMIT vp<%index.next> = add nuw vp<%4>, vp<%1>
; CHECK-NEXT:     EMIT branch-on-count vp<%index.next>, vp<%2>
; CHECK-NEXT:   No successors
; CHECK-NEXT: }
; CHECK-NEXT: Successor(s): middle.block
; CHECK-EMPTY:
; CHECK-NEXT: middle.block:
; CHECK-NEXT:   EMIT vp<%12> = extract-last-active vp<%10>, vp<%9>, ir<-1>
; CHECK-NEXT:   EMIT vp<%cmp.n> = icmp eq vp<%3>, vp<%2>
; CHECK-NEXT:   EMIT branch-on-cond vp<%cmp.n>
; CHECK-NEXT: Successor(s): ir-bb<exit.loopexit>, scalar.ph
; CHECK-EMPTY:
; CHECK-NEXT: ir-bb<exit.loopexit>:
; CHECK-NEXT:   IR   %spec.select.lcssa = phi i32 [ %spec.select, %loop ] (extra operand: vp<%12> from middle.block)
; CHECK-NEXT: No successors
; CHECK-EMPTY:
; CHECK-NEXT: scalar.ph:
; CHECK-NEXT:   EMIT-SCALAR vp<%bc.resume.val> = phi [ vp<%2>, middle.block ], [ ir<0>, ir-bb<loop.preheader> ]
; CHECK-NEXT:   EMIT-SCALAR vp<%bc.merge.rdx> = phi [ vp<%12>, middle.block ], [ ir<-1>, ir-bb<loop.preheader> ]
; CHECK-NEXT: Successor(s): ir-bb<loop>
; CHECK-EMPTY:
; CHECK-NEXT: ir-bb<loop>:
; CHECK-NEXT:   IR   %iv = phi i64 [ 0, %loop.preheader ], [ %iv.next, %loop ] (extra operand: vp<%bc.resume.val> from scalar.ph)
; CHECK-NEXT:   IR   %t.010 = phi i32 [ -1, %loop.preheader ], [ %spec.select, %loop ] (extra operand: vp<%bc.merge.rdx> from scalar.ph)
; CHECK-NEXT:   IR   %arrayidx = getelementptr inbounds i32, ptr %data, i64 %iv
; CHECK-NEXT:   IR   %0 = load i32, ptr %arrayidx, align 4
; CHECK-NEXT:   IR   %1 = sext i32 %0 to i64
; CHECK-NEXT:   IR   %cmp1 = icmp slt i64 %a, %1
; CHECK-NEXT:   IR   %spec.select = select i1 %cmp1, i32 %0, i32 %t.010
; CHECK-NEXT:   IR   %iv.next = add nuw nsw i64 %iv, 1
; CHECK-NEXT:   IR   %exitcond.not = icmp eq i64 %iv.next, %wide.trip.count
; CHECK-NEXT: No successors
; CHECK-NEXT: }


; CHECK: Cost of 1 for VF vscale x 1: induction instruction   %iv.next = add nuw nsw i64 %iv, 1
; CHECK-NEXT: Cost of 1 for VF vscale x 1: induction instruction   %iv = phi i64 [ 0, %loop.preheader ], [ %iv.next, %loop ]
; CHECK-NEXT: Cost of 1 for VF vscale x 1: exit condition instruction   %exitcond.not = icmp eq i64 %iv.next, %wide.trip.count
; CHECK-NEXT: Cost of 0 for VF vscale x 1: EMIT vp<%4> = CANONICAL-INDUCTION ir<0>, vp<%index.next>
; CHECK-NEXT: Cost of 1 for VF vscale x 1: WIDEN-REDUCTION-PHI ir<%t.010> = phi ir<-1>, vp<%10>
; CHECK-NEXT: Cost of 1 for VF vscale x 1: ACTIVE-LANE-MASK-PHI vp<%5> = phi ir<false>, vp<%9>
; CHECK-NEXT: Cost of 0 for VF vscale x 1: vp<%6> = SCALAR-STEPS vp<%4>, ir<1>, vp<%0>
; CHECK-NEXT: Cost of 0 for VF vscale x 1: CLONE ir<%arrayidx> = getelementptr inbounds ir<%data>, vp<%6>
; CHECK-NEXT: Cost of 0 for VF vscale x 1: vp<%7> = vector-pointer ir<%arrayidx>
; CHECK-NEXT: Cost of 1 for VF vscale x 1: WIDEN ir<%0> = load vp<%7>
; CHECK-NEXT: Cost of 0 for VF vscale x 1: WIDEN-CAST ir<%1> = sext ir<%0> to i64
; CHECK-NEXT: Cost of 1 for VF vscale x 1: WIDEN ir<%cmp1> = icmp slt ir<%a>, ir<%1>
; CHECK-NEXT: Cost of 1 for VF vscale x 1: EMIT vp<%8> = any-of ir<%cmp1>
; CHECK-NEXT: Cost of 1 for VF vscale x 1: WIDEN-SELECT-VECTOR vp<%9> = select  vp<%8>, ir<%cmp1>, vp<%5>
; CHECK-NEXT: Cost of 1 for VF vscale x 1: WIDEN-SELECT-VECTOR vp<%10> = select  vp<%8>, ir<%0>, ir<%t.010>
; CHECK-NEXT: Cost of 0 for VF vscale x 1: EMIT vp<%index.next> = add nuw vp<%4>, vp<%1>
; CHECK-NEXT: Cost of 0 for VF vscale x 1: EMIT branch-on-count vp<%index.next>, vp<%2>
; CHECK-NEXT: Cost of 1 for VF vscale x 1: vector loop backedge
; CHECK-NEXT: Cost of 0 for VF vscale x 1: IR   %wide.trip.count = zext i32 %N to i64
; CHECK-NEXT: Cost of 0 for VF vscale x 1: EMIT vp<%3> = EXPAND SCEV (zext i32 %N to i64)
; CHECK-NEXT: Cost of 0 for VF vscale x 1: EMIT-SCALAR vp<%bc.resume.val> = phi [ vp<%2>, middle.block ], [ ir<0>, ir-bb<loop.preheader> ]
; CHECK-NEXT: Cost of 0 for VF vscale x 1: EMIT-SCALAR vp<%bc.merge.rdx> = phi [ vp<%12>, middle.block ], [ ir<-1>, ir-bb<loop.preheader> ]
; CHECK-NEXT: Cost of 0 for VF vscale x 1: IR   %iv = phi i64 [ 0, %loop.preheader ], [ %iv.next, %loop ] (extra operand: vp<%bc.resume.val> from scalar.ph)
; CHECK-NEXT: Cost of 0 for VF vscale x 1: IR   %t.010 = phi i32 [ -1, %loop.preheader ], [ %spec.select, %loop ] (extra operand: vp<%bc.merge.rdx> from scalar.ph)
; CHECK-NEXT: Cost of 0 for VF vscale x 1: IR   %arrayidx = getelementptr inbounds i32, ptr %data, i64 %iv
; CHECK-NEXT: Cost of 0 for VF vscale x 1: IR   %0 = load i32, ptr %arrayidx, align 4
; CHECK-NEXT: Cost of 0 for VF vscale x 1: IR   %1 = sext i32 %0 to i64
; CHECK-NEXT: Cost of 0 for VF vscale x 1: IR   %cmp1 = icmp slt i64 %a, %1
; CHECK-NEXT: Cost of 0 for VF vscale x 1: IR   %spec.select = select i1 %cmp1, i32 %0, i32 %t.010
; CHECK-NEXT: Cost of 0 for VF vscale x 1: IR   %iv.next = add nuw nsw i64 %iv, 1
; CHECK-NEXT: Cost of 0 for VF vscale x 1: IR   %exitcond.not = icmp eq i64 %iv.next, %wide.trip.count
; CHECK-NEXT: Cost of 1 for VF vscale x 1: EMIT vp<%12> = extract-last-active vp<%10>, vp<%9>, ir<-1>
; CHECK-NEXT: Cost of 0 for VF vscale x 1: EMIT vp<%cmp.n> = icmp eq vp<%3>, vp<%2>
; CHECK-NEXT: Cost of 0 for VF vscale x 1: EMIT branch-on-cond vp<%cmp.n>
; CHECK-NEXT: Cost of 0 for VF vscale x 1: IR   %spec.select.lcssa = phi i32 [ %spec.select, %loop ] (extra operand: vp<%12> from middle.block)
