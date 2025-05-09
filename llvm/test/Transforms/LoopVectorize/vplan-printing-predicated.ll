; REQUIRES: asserts
; RUN: opt -passes=loop-vectorize -debug-only=loop-vectorize -force-vector-width=4 -prefer-predicate-over-epilogue=predicate-else-scalar-epilogue -force-tail-folding-style=data-and-control -disable-output %s 2>&1 | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"

; Tests for printing predicated VPlans.

define dso_local void @alias_mask(ptr noalias %a, ptr %b, ptr %c, i32 %n) {
; CHECK-LABEL: 'alias_mask'
; CHECK:      VPlan 'Initial VPlan for VF={4},UF>=1' {
; CHECK-NEXT: Live-in vp<%0> = VF
; CHECK-NEXT: vp<%3> = original trip-count
; CHECK-EMPTY: 
; CHECK-NEXT: ir-bb<for.body.preheader>:
; CHECK-NEXT:   IR   %wide.trip.count = zext nneg i32 %n to i64
; CHECK-NEXT:   EMIT vp<%3> = EXPAND SCEV (zext i32 %n to i64)
; CHECK-NEXT:   EMIT vp<%4> = EXPAND SCEV (ptrtoint ptr %c to i64)
; CHECK-NEXT:   EMIT vp<%5> = EXPAND SCEV (ptrtoint ptr %b to i64)
; CHECK-NEXT: Successor(s): vector.ph
; CHECK-EMPTY: 
; CHECK-NEXT: vector.ph:
; CHECK-NEXT:   EMIT vp<%6> = ALIAS-LANE-MASK vp<%5>, vp<%4> (write-after-read)
; CHECK-NEXT:   EMIT vp<%index.part.next> = VF * Part + ir<0>
; CHECK-NEXT:   EMIT vp<%active.lane.mask.entry> = active lane mask vp<%index.part.next>, vp<%3>
; CHECK-NEXT: Successor(s): vector loop
; CHECK-EMPTY: 
; CHECK-NEXT: <x1> vector loop: {
; CHECK-NEXT:   vector.body:
; CHECK-NEXT:     EMIT vp<%7> = CANONICAL-INDUCTION ir<0>, vp<%index.next>
; CHECK-NEXT:     ACTIVE-LANE-MASK-PHI vp<%8> = phi vp<%active.lane.mask.entry>, vp<%active.lane.mask.next>
; CHECK-NEXT:     EMIT vp<%9> = and vp<%8>, vp<%6>
; CHECK-NEXT:   Successor(s): pred.store
; CHECK-EMPTY: 
; CHECK-NEXT:   <xVFxUF> pred.store: {
; CHECK-NEXT:     pred.store.entry:
; CHECK-NEXT:       BRANCH-ON-MASK vp<%9>
; CHECK-NEXT:     Successor(s): pred.store.if, pred.store.continue
; CHECK-EMPTY: 
; CHECK-NEXT:     pred.store.if:
; CHECK-NEXT:       vp<%10> = SCALAR-STEPS vp<%7>, ir<1>, vp<%0>
; CHECK-NEXT:       REPLICATE ir<%arrayidx> = getelementptr inbounds ir<%a>, vp<%10>
; CHECK-NEXT:       REPLICATE ir<%0> = load ir<%arrayidx>
; CHECK-NEXT:       REPLICATE ir<%arrayidx2> = getelementptr inbounds ir<%b>, vp<%10>
; CHECK-NEXT:       REPLICATE ir<%1> = load ir<%arrayidx2>
; CHECK-NEXT:       REPLICATE ir<%arrayidx6> = getelementptr inbounds ir<%c>, vp<%10>
; CHECK-NEXT:       REPLICATE ir<%add> = add ir<%1>, ir<%0>
; CHECK-NEXT:       REPLICATE store ir<%add>, ir<%arrayidx6>
; CHECK-NEXT:     Successor(s): pred.store.continue
; CHECK-EMPTY: 
; CHECK-NEXT:     pred.store.continue:
; CHECK-NEXT:     No successors
; CHECK-NEXT:   }
; CHECK-NEXT:   Successor(s): for.body.2
; CHECK-EMPTY: 
; CHECK-NEXT:   for.body.2:
; CHECK-NEXT:     EMIT vp<%popcount> = popcount vp<%6>
; CHECK-NEXT:     EMIT vp<%index.next> = add vp<%7>, vp<%popcount>
; CHECK-NEXT:     EMIT vp<%11> = VF * Part + vp<%index.next>
; CHECK-NEXT:     EMIT vp<%active.lane.mask.next> = active lane mask vp<%11>, vp<%3>
; CHECK-NEXT:     EMIT vp<%12> = not vp<%active.lane.mask.next>
; CHECK-NEXT:     EMIT branch-on-cond vp<%12>
; CHECK-NEXT:   No successors
; CHECK-NEXT: }
; CHECK-NEXT: Successor(s): middle.block
; CHECK-EMPTY: 
; CHECK-NEXT: middle.block:
; CHECK-NEXT: Successor(s): ir-bb<for.cond.cleanup.loopexit>
; CHECK-EMPTY: 
; CHECK-NEXT: ir-bb<for.cond.cleanup.loopexit>:
; CHECK-NEXT: No successors
; CHECK-NEXT: }
entry:
  %cmp11 = icmp sgt i32 %n, 0
  br i1 %cmp11, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:                               ; preds = %entry
  %wide.trip.count = zext nneg i32 %n to i64
  br label %for.body

for.cond.cleanup.loopexit:                        ; preds = %for.body
  br label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.cond.cleanup.loopexit, %entry
  ret void

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i64 [ 0, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds i8, ptr %a, i64 %indvars.iv
  %0 = load i8, ptr %arrayidx, align 1
  %arrayidx2 = getelementptr inbounds i8, ptr %b, i64 %indvars.iv
  %1 = load i8, ptr %arrayidx2, align 1
  %add = add i8 %1, %0
  %arrayidx6 = getelementptr inbounds i8, ptr %c, i64 %indvars.iv
  store i8 %add, ptr %arrayidx6, align 1
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond.not, label %for.cond.cleanup.loopexit, label %for.body
}
