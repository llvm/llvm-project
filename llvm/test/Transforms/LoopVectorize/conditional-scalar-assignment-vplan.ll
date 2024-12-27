; RUN: opt -passes=loop-vectorize -debug-only=loop-vectorize \
; RUN:   -scalable-vectorization=on -force-target-supports-scalable-vectors \
; RUN:   -force-tail-folding-style=none  -enable-csa-vectorization \
; RUN:   -prefer-predicate-over-epilogue=predicate-else-scalar-epilogue \
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
; CHECK-NEXT: Live-in vp<%0> = VF * UF
; CHECK-NEXT: Live-in vp<%1> = vector-trip-count
; CHECK-NEXT: vp<%2> = original trip-count
; CHECK-EMPTY:
; CHECK-NEXT: ir-bb<loop.preheader>:
; CHECK-NEXT:   IR   %wide.trip.count = zext i32 %N to i64
; CHECK-NEXT:   EMIT vp<%2> = EXPAND SCEV (zext i32 %N to i64)
; CHECK-NEXT: Successor(s): vector.ph
; CHECK-EMPTY:
; CHECK-NEXT: vector.ph:
; CHECK-NEXT: Successor(s): vector loop
; CHECK-EMPTY:
; CHECK-NEXT: <x1> vector loop: {
; CHECK-NEXT:   vector.body:
; CHECK-NEXT:     EMIT vp<%3> = CANONICAL-INDUCTION ir<0>, vp<%index.next>
; CHECK-NEXT:     EMIT ir<%t.010> = csa-data-phi ir<poison>, ir<%spec.select>
; CHECK-NEXT:     EMIT vp<%csa.mask.phi> = csa-mask-phi ir<false>
; CHECK-NEXT:     vp<%4> = SCALAR-STEPS vp<%3>, ir<1>
; CHECK-NEXT:     CLONE ir<%arrayidx> = getelementptr inbounds ir<%data>, vp<%4>
; CHECK-NEXT:     vp<%5> = vector-pointer ir<%arrayidx>
; CHECK-NEXT:     WIDEN ir<%0> = load vp<%5>
; CHECK-NEXT:     WIDEN-CAST ir<%1> = sext ir<%0> to i64
; CHECK-NEXT:     WIDEN ir<%cmp1> = icmp slt ir<%a>, ir<%1>
; CHECK-NEXT:     EMIT vp<%csa.cond.anyof> = any-of ir<%cmp1>
; CHECK-NEXT:     EMIT vp<%csa.mask.sel> = csa-mask-sel ir<%cmp1>, vp<%csa.mask.phi>, vp<%csa.cond.anyof>
; CHECK-NEXT:     EMIT ir<%spec.select> = csa-data-update ir<%t.010>, ir<%cmp1>, ir<%0>, ir<%t.010>, vp<%csa.mask.sel>, vp<%csa.cond.anyof>
; CHECK-NEXT:     EMIT vp<%index.next> = add nuw vp<%3>, vp<%0>
; CHECK-NEXT:     EMIT branch-on-count vp<%index.next>, vp<%1>
; CHECK-NEXT:   No successors
; CHECK-NEXT: }
; CHECK-NEXT: Successor(s): middle.block
; CHECK-EMPTY:
; CHECK-NEXT: middle.block:
; CHECK-NEXT:   EMIT vp<%7> = extract-from-end ir<%spec.select>, ir<1>
; CHECK-NEXT:   EMIT vp<%8> = CSA-EXTRACT-SCALAR ir<-1>, vp<%csa.mask.sel>, ir<%spec.select>
; CHECK-NEXT:   EMIT vp<%cmp.n> = icmp eq vp<%2>, vp<%1>
; CHECK-NEXT:   EMIT branch-on-cond vp<%cmp.n>
; CHECK-NEXT: Successor(s): ir-bb<exit.loopexit>, scalar.ph
; CHECK-EMPTY:
; CHECK-NEXT: scalar.ph:
; CHECK-NEXT:   EMIT vp<%bc.resume.val> = resume-phi vp<%1>, ir<0>
; CHECK-NEXT: Successor(s): ir-bb<loop>
; CHECK-EMPTY:
; CHECK-NEXT: ir-bb<loop>:
; CHECK-NEXT:   IR   %iv = phi i64 [ 0, %loop.preheader ], [ %iv.next, %loop ] (extra operand: vp<%bc.resume.val> from scalar.ph)
; CHECK-NEXT:   IR   %t.010 = phi i32 [ -1, %loop.preheader ], [ %spec.select, %loop ]
; CHECK-NEXT:   IR   %arrayidx = getelementptr inbounds i32, ptr %data, i64 %iv
; CHECK-NEXT:   IR   %0 = load i32, ptr %arrayidx, align 4
; CHECK-NEXT:   IR   %1 = sext i32 %0 to i64
; CHECK-NEXT:   IR   %cmp1 = icmp slt i64 %a, %1
; CHECK-NEXT:   IR   %spec.select = select i1 %cmp1, i32 %0, i32 %t.010
; CHECK-NEXT:   IR   %iv.next = add nuw nsw i64 %iv, 1
; CHECK-NEXT:   IR   %exitcond.not = icmp eq i64 %iv.next, %wide.trip.count
; CHECK-NEXT: No successors
; CHECK-EMPTY:
; CHECK-NEXT: ir-bb<exit.loopexit>:
; CHECK-NEXT:   IR   %spec.select.lcssa = phi i32 [ %spec.select, %loop ] (extra operand: vp<%7> from middle.block)
; CHECK-NEXT: No successors
; CHECK-NEXT: }
