; RUN: opt -passes=loop-vectorize -debug-only=loop-vectorize \
; RUN:   -scalable-vectorization=on -force-target-supports-scalable-vectors \
; RUN:   -force-tail-folding-style=none  -enable-csa-vectorization \
; RUN:   -prefer-predicate-over-epilogue=predicate-else-scalar-epilogue \
; RUN:   -disable-output 2>&1 < %s | FileCheck %s --check-prefix=NO-EVL
; RUN: opt -passes=loop-vectorize -debug-only=loop-vectorize \
; RUN:   -scalable-vectorization=on -force-target-supports-scalable-vectors \
; RUN:   -force-tail-folding-style=data-with-evl -enable-csa-vectorization \
; RUN:   -prefer-predicate-over-epilogue=predicate-else-scalar-epilogue \
; RUN:   -disable-output 2>&1 < %s | FileCheck %s --check-prefix=EVL


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

; NO-EVL: VPlan 'Initial VPlan for VF={vscale x 1},UF>=1' {
; NO-EVL-NEXT: Live-in vp<%0> = VF * UF
; NO-EVL-NEXT: Live-in vp<%1> = vector-trip-count
; NO-EVL-NEXT: vp<%2> = original trip-count
; NO-EVL-EMPTY:
; NO-EVL-NEXT: ir-bb<loop.preheader>:
; NO-EVL-NEXT:   IR   %wide.trip.count = zext i32 %N to i64
; NO-EVL-NEXT:   EMIT vp<%2> = EXPAND SCEV (zext i32 %N to i64)
; NO-EVL-NEXT: Successor(s): vector.ph
; NO-EVL-EMPTY:
; NO-EVL-NEXT: vector.ph:
; NO-EVL-NEXT: Successor(s): vector loop
; NO-EVL-EMPTY:
; NO-EVL-NEXT: <x1> vector loop: {
; NO-EVL-NEXT:   vector.body:
; NO-EVL-NEXT:     EMIT vp<%3> = CANONICAL-INDUCTION ir<0>, vp<%index.next>
; NO-EVL-NEXT:     EMIT ir<%t.010> = csa-data-phi ir<poison>, ir<%spec.select>
; NO-EVL-NEXT:     EMIT vp<%csa.mask.phi> = csa-mask-phi ir<false>
; NO-EVL-NEXT:     vp<%4> = SCALAR-STEPS vp<%3>, ir<1>
; NO-EVL-NEXT:     CLONE ir<%arrayidx> = getelementptr inbounds ir<%data>, vp<%4>
; NO-EVL-NEXT:     vp<%5> = vector-pointer ir<%arrayidx>
; NO-EVL-NEXT:     WIDEN ir<%0> = load vp<%5>
; NO-EVL-NEXT:     WIDEN-CAST ir<%1> = sext ir<%0> to i64
; NO-EVL-NEXT:     WIDEN ir<%cmp1> = icmp slt ir<%a>, ir<%1>
; NO-EVL-NEXT:     EMIT vp<%csa.cond.anyof> = any-of ir<%cmp1>
; NO-EVL-NEXT:     EMIT vp<%csa.mask.sel> = csa-mask-sel ir<%cmp1>, vp<%csa.mask.phi>, vp<%csa.cond.anyof>
; NO-EVL-NEXT:     EMIT ir<%spec.select> = csa-data-update ir<%t.010>, ir<%cmp1>, ir<%0>, ir<%t.010>, vp<%csa.mask.sel>, vp<%csa.cond.anyof>
; NO-EVL-NEXT:     EMIT vp<%index.next> = add nuw vp<%3>, vp<%0>
; NO-EVL-NEXT:     EMIT branch-on-count vp<%index.next>, vp<%1>
; NO-EVL-NEXT:   No successors
; NO-EVL-NEXT: }
; NO-EVL-NEXT: Successor(s): middle.block
; NO-EVL-EMPTY:
; NO-EVL-NEXT: middle.block:
; NO-EVL-NEXT:   EMIT vp<%7> = extract-from-end ir<%spec.select>, ir<1>
; NO-EVL-NEXT:   EMIT vp<%8> = CSA-EXTRACT-SCALAR ir<-1>, vp<%csa.mask.sel>, ir<%spec.select>
; NO-EVL-NEXT:   EMIT vp<%cmp.n> = icmp eq vp<%2>, vp<%1>
; NO-EVL-NEXT:   EMIT branch-on-cond vp<%cmp.n>
; NO-EVL-NEXT: Successor(s): ir-bb<exit.loopexit>, scalar.ph
; NO-EVL-EMPTY:
; NO-EVL-NEXT: scalar.ph:
; NO-EVL-NEXT:   EMIT vp<%bc.resume.val> = resume-phi vp<%1>, ir<0>
; NO-EVL-NEXT: Successor(s): ir-bb<loop>
; NO-EVL-EMPTY:
; NO-EVL-NEXT: ir-bb<loop>:
; NO-EVL-NEXT:   IR   %iv = phi i64 [ 0, %loop.preheader ], [ %iv.next, %loop ] (extra operand: vp<%bc.resume.val> from scalar.ph)
; NO-EVL-NEXT:   IR   %t.010 = phi i32 [ -1, %loop.preheader ], [ %spec.select, %loop ]
; NO-EVL-NEXT:   IR   %arrayidx = getelementptr inbounds i32, ptr %data, i64 %iv
; NO-EVL-NEXT:   IR   %0 = load i32, ptr %arrayidx, align 4
; NO-EVL-NEXT:   IR   %1 = sext i32 %0 to i64
; NO-EVL-NEXT:   IR   %cmp1 = icmp slt i64 %a, %1
; NO-EVL-NEXT:   IR   %spec.select = select i1 %cmp1, i32 %0, i32 %t.010
; NO-EVL-NEXT:   IR   %iv.next = add nuw nsw i64 %iv, 1
; NO-EVL-NEXT:   IR   %exitcond.not = icmp eq i64 %iv.next, %wide.trip.count
; NO-EVL-NEXT: No successors
; NO-EVL-EMPTY:
; NO-EVL-NEXT: ir-bb<exit.loopexit>:
; NO-EVL-NEXT:   IR   %spec.select.lcssa = phi i32 [ %spec.select, %loop ] (extra operand: vp<%7> from middle.block)
; NO-EVL-NEXT: No successors
; NO-EVL-NEXT: }

; NO-EVL: Cost of 1 for VF vscale x 1: induction instruction   %iv.next = add nuw nsw i64 %iv, 1
; NO-EVL-NEXT: Cost of 1 for VF vscale x 1: induction instruction   %iv = phi i64 [ 0, %loop.preheader ], [ %iv.next, %loop ]
; NO-EVL-NEXT: Cost of 1 for VF vscale x 1: exit condition instruction   %exitcond.not = icmp eq i64 %iv.next, %wide.trip.count
; NO-EVL-NEXT: Cost of 0 for VF vscale x 1: EMIT vp<%3> = CANONICAL-INDUCTION ir<0>, vp<%index.next>
; NO-EVL-NEXT: Cost of 2 for VF vscale x 1: EMIT ir<%t.010> = csa-data-phi ir<poison>, ir<%spec.select>
; NO-EVL-NEXT: Cost of 0 for VF vscale x 1: EMIT vp<%csa.mask.phi> = csa-mask-phi ir<false>
; NO-EVL-NEXT: Cost of 0 for VF vscale x 1: vp<%4> = SCALAR-STEPS vp<%3>, ir<1>
; NO-EVL-NEXT: Cost of 0 for VF vscale x 1: CLONE ir<%arrayidx> = getelementptr inbounds ir<%data>, vp<%4>
; NO-EVL-NEXT: Cost of 0 for VF vscale x 1: vp<%5> = vector-pointer ir<%arrayidx>
; NO-EVL-NEXT: Cost of 1 for VF vscale x 1: WIDEN ir<%0> = load vp<%5>
; NO-EVL-NEXT: Cost of 1 for VF vscale x 1: WIDEN-CAST ir<%1> = sext ir<%0> to i64
; NO-EVL-NEXT: Cost of 1 for VF vscale x 1: WIDEN ir<%cmp1> = icmp slt ir<%a>, ir<%1>
; NO-EVL-NEXT: Cost of 0 for VF vscale x 1: EMIT vp<%csa.cond.anyof> = any-of ir<%cmp1>
; NO-EVL-NEXT: Cost of 0 for VF vscale x 1: EMIT vp<%csa.mask.sel> = csa-mask-sel ir<%cmp1>, vp<%csa.mask.phi>, vp<%csa.cond.anyof>
; NO-EVL-NEXT: Cost of 4 for VF vscale x 1: EMIT ir<%spec.select> = csa-data-update ir<%t.010>, ir<%cmp1>, ir<%0>, ir<%t.010>, vp<%csa.mask.sel>, vp<%csa.cond.anyof>
; NO-EVL-NEXT: Cost of 0 for VF vscale x 1: EMIT vp<%index.next> = add nuw vp<%3>, vp<%0>
; NO-EVL-NEXT: Cost of 0 for VF vscale x 1: EMIT branch-on-count vp<%index.next>, vp<%1>
; NO-EVL-NEXT: Cost of 1 for VF vscale x 1: vector loop backedge

; EVL: VPlan 'Initial VPlan for VF={vscale x 1},UF>=1' {
; EVL-NEXT: Live-in vp<%0> = VF
; EVL-NEXT: Live-in vp<%1> = VF * UF
; EVL-NEXT: Live-in vp<%2> = vector-trip-count
; EVL-NEXT: Live-in vp<%3> = backedge-taken count
; EVL-NEXT: vp<%4> = original trip-count
; EVL-EMPTY:
; EVL-NEXT: ir-bb<loop.preheader>:
; EVL-NEXT:   IR   %wide.trip.count = zext i32 %N to i64
; EVL-NEXT:   EMIT vp<%4> = EXPAND SCEV (zext i32 %N to i64)
; EVL-NEXT: Successor(s): vector.ph
; EVL-EMPTY:
; EVL-NEXT: vector.ph:
; EVL-NEXT: Successor(s): vector loop
; EVL-EMPTY:
; EVL-NEXT: <x1> vector loop: {
; EVL-NEXT:   vector.body:
; EVL-NEXT:     EMIT vp<%5> = CANONICAL-INDUCTION ir<0>, vp<%index.next>
; EVL-NEXT:     ir<%iv> = WIDEN-INDUCTION  ir<0>, ir<1>, vp<%0>
; EVL-NEXT:     EMIT ir<%t.010> = csa-data-phi ir<poison>, ir<%spec.select>
; EVL-NEXT:     EMIT vp<%csa.mask.phi> = csa-mask-phi ir<false>
; EVL-NEXT:     EMIT vp<%6> = icmp ule ir<%iv>, vp<%3>
; EVL-NEXT:     WIDEN-GEP Inv[Var] ir<%arrayidx> = getelementptr inbounds ir<%data>, ir<%iv>
; EVL-NEXT:   Successor(s): pred.load
; EVL-EMPTY:
; EVL-NEXT:   <xVFxUF> pred.load: {
; EVL-NEXT:     pred.load.entry:
; EVL-NEXT:       BRANCH-ON-MASK vp<%6>
; EVL-NEXT:     Successor(s): pred.load.if, pred.load.continue
; EVL-EMPTY:
; EVL-NEXT:     pred.load.if:
; EVL-NEXT:       REPLICATE ir<%0> = load ir<%arrayidx> (S->V)
; EVL-NEXT:     Successor(s): pred.load.continue
; EVL-EMPTY:
; EVL-NEXT:     pred.load.continue:
; EVL-NEXT:       PHI-PREDICATED-INSTRUCTION vp<%7> = ir<%0>
; EVL-NEXT:     No successors
; EVL-NEXT:   }
; EVL-NEXT:   Successor(s): loop.0
; EVL-EMPTY:
; EVL-NEXT:   loop.0:
; EVL-NEXT:     WIDEN-CAST ir<%1> = sext vp<%7> to i64
; EVL-NEXT:     WIDEN ir<%cmp1> = icmp slt ir<%a>, ir<%1>
; EVL-NEXT:     EMIT vp<%csa.cond.anyof> = any-of ir<%cmp1>
; EVL-NEXT:     EMIT vp<%csa.mask.sel> = csa-mask-sel ir<%cmp1>, vp<%csa.mask.phi>, vp<%csa.cond.anyof>
; EVL-NEXT:     EMIT ir<%spec.select> = csa-data-update ir<%t.010>, ir<%cmp1>, vp<%7>, ir<%t.010>, vp<%csa.mask.sel>, vp<%csa.cond.anyof>
; EVL-NEXT:     EMIT vp<%index.next> = add vp<%5>, vp<%1>
; EVL-NEXT:     EMIT branch-on-count vp<%index.next>, vp<%2>
; EVL-NEXT:   No successors
; EVL-NEXT: }
; EVL-NEXT: Successor(s): middle.block
; EVL-EMPTY:
; EVL-NEXT: middle.block:
; EVL-NEXT:   EMIT vp<%9> = extract-from-end ir<%spec.select>, ir<1>
; EVL-NEXT:   EMIT vp<%10> = CSA-EXTRACT-SCALAR ir<-1>, vp<%csa.mask.sel>, ir<%spec.select>
; EVL-NEXT:   EMIT branch-on-cond ir<true>
; EVL-NEXT: Successor(s): ir-bb<exit.loopexit>, scalar.ph
; EVL-EMPTY:
; EVL-NEXT: scalar.ph:
; EVL-NEXT:   EMIT vp<%bc.resume.val> = resume-phi vp<%2>, ir<0>
; EVL-NEXT: Successor(s): ir-bb<loop>
; EVL-EMPTY:
; EVL-NEXT: ir-bb<loop>:
; EVL-NEXT:   IR   %iv = phi i64 [ 0, %loop.preheader ], [ %iv.next, %loop ] (extra operand: vp<%bc.resume.val> from scalar.ph)
; EVL-NEXT:   IR   %t.010 = phi i32 [ -1, %loop.preheader ], [ %spec.select, %loop ]
; EVL-NEXT:   IR   %arrayidx = getelementptr inbounds i32, ptr %data, i64 %iv
; EVL-NEXT:   IR   %0 = load i32, ptr %arrayidx, align 4
; EVL-NEXT:   IR   %1 = sext i32 %0 to i64
; EVL-NEXT:   IR   %cmp1 = icmp slt i64 %a, %1
; EVL-NEXT:   IR   %spec.select = select i1 %cmp1, i32 %0, i32 %t.010
; EVL-NEXT:   IR   %iv.next = add nuw nsw i64 %iv, 1
; EVL-NEXT:   IR   %exitcond.not = icmp eq i64 %iv.next, %wide.trip.count
; EVL-NEXT: No successors
; EVL-EMPTY:
; EVL-NEXT: ir-bb<exit.loopexit>:
; EVL-NEXT:   IR   %spec.select.lcssa = phi i32 [ %spec.select, %loop ] (extra operand: vp<%9> from middle.block)
; EVL-NEXT: No successors
; EVL-NEXT: }

; EVL: Cost of 1 for VF vscale x 1: induction instruction   %iv.next = add nuw nsw i64 %iv, 1
; EVL-NEXT: Cost of 1 for VF vscale x 1: induction instruction   %iv = phi i64 [ 0, %loop.preheader ], [ %iv.next, %loop ]
; EVL-NEXT: Cost of 1 for VF vscale x 1: exit condition instruction   %exitcond.not = icmp eq i64 %iv.next, %wide.trip.count
; EVL-NEXT: Cost of 0 for VF vscale x 1: EMIT vp<%5> = CANONICAL-INDUCTION ir<0>, vp<%index.next>
; EVL-NEXT: Cost of 0 for VF vscale x 1: ir<%iv> = WIDEN-INDUCTION  ir<0>, ir<1>, vp<%0>
; EVL-NEXT: Cost of 2 for VF vscale x 1: EMIT ir<%t.010> = csa-data-phi ir<poison>, ir<%spec.select>
; EVL-NEXT: Cost of 0 for VF vscale x 1: EMIT vp<%csa.mask.phi> = csa-mask-phi ir<false>
; EVL-NEXT: Cost of 0 for VF vscale x 1: EMIT vp<%6> = icmp ule ir<%iv>, vp<%3>
; EVL-NEXT: Cost of 0 for VF vscale x 1: WIDEN-GEP Inv[Var] ir<%arrayidx> = getelementptr inbounds ir<%data>, ir<%iv>
; EVL-NEXT: Cost of 1 for VF vscale x 1: WIDEN-CAST ir<%1> = sext vp<%7> to i64
; EVL-NEXT: Cost of 1 for VF vscale x 1: WIDEN ir<%cmp1> = icmp slt ir<%a>, ir<%1>
; EVL-NEXT: Cost of 0 for VF vscale x 1: EMIT vp<%csa.cond.anyof> = any-of ir<%cmp1>
; EVL-NEXT: Cost of 0 for VF vscale x 1: EMIT vp<%csa.mask.sel> = csa-mask-sel ir<%cmp1>, vp<%csa.mask.phi>, vp<%csa.cond.anyof>
; EVL-NEXT: Cost of 4 for VF vscale x 1: EMIT ir<%spec.select> = csa-data-update ir<%t.010>, ir<%cmp1>, vp<%7>, ir<%t.010>, vp<%csa.mask.sel>, vp<%csa.cond.anyof>
; EVL-NEXT: Cost of 0 for VF vscale x 1: EMIT vp<%index.next> = add vp<%5>, vp<%1>
; EVL-NEXT: Cost of 0 for VF vscale x 1: EMIT branch-on-count vp<%index.next>, vp<%2>
; EVL-NEXT: Cost of 1 for VF vscale x 1: vector loop backedge
