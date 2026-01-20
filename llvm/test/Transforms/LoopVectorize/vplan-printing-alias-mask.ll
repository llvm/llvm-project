; REQUIRES: asserts
; RUN: opt -S -debug-only=loop-vectorize -force-vector-width=4 -passes=loop-vectorize -force-partial-aliasing-vectorization -disable-output %s 2>&1 | FileCheck %s

define void @alias_mask(ptr noalias %a, ptr %b, ptr %c, i32 %n) {
; CHECK-LABEL: 'alias_mask'
; CHECK:      VPlan 'Initial VPlan for VF={4},UF>=1' {
; CHECK-NEXT: Live-in vp<[[VF:%.+]]> = VF
; CHECK-NEXT: Live-in vp<[[VFxUF:%.+]]> = VF * UF
; CHECK-NEXT: Live-in vp<[[VEC_TC:%.+]]> = vector-trip-count
; CHECK-NEXT: Live-in vp<[[ALIAS_MASK:%.+]]> = alias-mask
; CHECK-NEXT: vp<[[TC:%.+]]> = original trip-count
; CHECK-EMPTY:
; CHECK-NEXT: ir-bb<for.body.preheader>:
; CHECK-NEXT:   EMIT vp<[[TC]]> = EXPAND SCEV (zext i32 %n to i64)
; CHECK-NEXT:   IR   %wide.trip.count = zext nneg i32 %n to i64
; CHECK-NEXT: Successor(s): scalar.ph, vector.ph
; CHECK-EMPTY:
; CHECK-NEXT: vector.ph:
; CHECK-NEXT: Successor(s): vector loop
; CHECK-EMPTY:
; CHECK-NEXT: <x1> vector loop: {
; CHECK-NEXT:   vector.body:
; CHECK-NEXT:     EMIT vp<[[CAN_IV:%.+]]> = CANONICAL-INDUCTION ir<0>, vp<%index.next>
; CHECK-NEXT:     vp<[[STEPS:%.+]]> = SCALAR-STEPS vp<[[CAN_IV]]>, ir<1>, vp<[[VF]]>
; CHECK-NEXT:   Successor(s): pred.load
; CHECK-EMPTY:
; CHECK-NEXT:   <xVFxUF> pred.load: {
; CHECK-NEXT:     pred.load.entry:
; CHECK-NEXT:       BRANCH-ON-MASK vp<[[ALIAS_MASK]]>
; CHECK-NEXT:     Successor(s): pred.load.if, pred.load.continue
; CHECK-EMPTY:
; CHECK-NEXT:     pred.load.if:
; CHECK-NEXT:       REPLICATE ir<%arrayidx> = getelementptr inbounds ir<%a>, vp<[[STEPS]]>
; CHECK-NEXT:       REPLICATE ir<%0> = load ir<%arrayidx> (S->V)
; CHECK-NEXT:       REPLICATE ir<%arrayidx2> = getelementptr inbounds ir<%b>, vp<[[STEPS]]>
; CHECK-NEXT:       REPLICATE ir<%1> = load ir<%arrayidx2> (S->V)
; CHECK-NEXT:     Successor(s): pred.load.continue
; CHECK-EMPTY:
; CHECK-NEXT:     pred.load.continue:
; CHECK-NEXT:       PHI-PREDICATED-INSTRUCTION vp<[[VEC_A:%.+]]> = ir<%0>
; CHECK-NEXT:       PHI-PREDICATED-INSTRUCTION vp<[[VEC_B:%.+]]> = ir<%1>
; CHECK-NEXT:     No successors
; CHECK-NEXT:   }
; CHECK-NEXT:   Successor(s): for.body.1
; CHECK-EMPTY:
; CHECK-NEXT:   for.body.1:
; CHECK-NEXT:     WIDEN ir<%add> = add vp<[[VEC_B]]>, vp<[[VEC_A]]>
; CHECK-NEXT:   Successor(s): pred.store
; CHECK-EMPTY:
; CHECK-NEXT:   <xVFxUF> pred.store: {
; CHECK-NEXT:     pred.store.entry:
; CHECK-NEXT:       BRANCH-ON-MASK vp<[[ALIAS_MASK]]>
; CHECK-NEXT:     Successor(s): pred.store.if, pred.store.continue
; CHECK-EMPTY:
; CHECK-NEXT:     pred.store.if:
; CHECK-NEXT:       REPLICATE ir<%arrayidx6> = getelementptr inbounds ir<%c>, vp<[[STEPS]]>
; CHECK-NEXT:       REPLICATE store ir<%add>, ir<%arrayidx6>
; CHECK-NEXT:     Successor(s): pred.store.continue
; CHECK-EMPTY:
; CHECK-NEXT:     pred.store.continue:
; CHECK-NEXT:     No successors
; CHECK-NEXT:   }
; CHECK-NEXT:   Successor(s): for.body.2
; CHECK-EMPTY:
; CHECK-NEXT:   for.body.2:
; CHECK-NEXT:     EMIT vp<%index.next> = add nuw vp<[[CAN_IV]]>, vp<[[VFxUF]]>
; CHECK-NEXT:     EMIT branch-on-count vp<%index.next>, vp<[[VEC_TC]]>
; CHECK-NEXT:   No successors
; CHECK-NEXT: }
; CHECK-NEXT: Successor(s): middle.block

; CHECK:      VPlan 'Final VPlan for VF={4},UF={1}' {
; CHECK-NEXT: Live-in ir<%wide.trip.count> = original trip-count
; CHECK-EMPTY:
; CHECK-NEXT: ir-bb<for.body.preheader>:
; CHECK-NEXT:   IR   %wide.trip.count = zext nneg i32 %n to i64
; CHECK-NEXT:   EMIT vp<%min.iters.check> = icmp ult ir<%wide.trip.count>, ir<4>
; CHECK-NEXT:   EMIT branch-on-cond vp<%min.iters.check>
; CHECK-NEXT: Successor(s): ir-bb<scalar.ph>, vector.min.vf.check
; CHECK-EMPTY:
; CHECK-NEXT: vector.min.vf.check:
; CHECK-NEXT:   EMIT-SCALAR vp<[[PTR_B:%.+]]> = inttoptr ir<%b2> to ptr
; CHECK-NEXT:   EMIT-SCALAR vp<[[PTR_C:%.+]]> = inttoptr ir<%c1> to ptr
; CHECK-NEXT:   WIDEN-INTRINSIC vp<[[ALIAS_MASK:%.+]]> = call llvm.loop.dependence.war.mask(vp<[[PTR_B]]>, vp<[[PTR_C]]>, ir<1>)
; CHECK-NEXT:   EMIT vp<[[CLAMPED_VF:%.+]]> = num-active-lanes vp<[[ALIAS_MASK]]>
; CHECK-NEXT:   EMIT vp<%cmp.vf> = icmp ult vp<[[CLAMPED_VF]]>, ir<2>
; CHECK-NEXT:   EMIT branch-on-cond vp<%cmp.vf>
; CHECK-NEXT: Successor(s): ir-bb<scalar.ph>, vector.ph
; CHECK-EMPTY:
; CHECK-NEXT: vector.ph:
; CHECK-NEXT:   EMIT vp<%n.mod.vf> = urem ir<%wide.trip.count>, vp<[[CLAMPED_VF]]>
; CHECK-NEXT:   EMIT vp<%n.vec> = sub ir<%wide.trip.count>, vp<%n.mod.vf>
; CHECK-NEXT: Successor(s): vector.body
; CHECK-EMPTY:
; CHECK-NEXT: vector.body:
; CHECK-NEXT:   EMIT-SCALAR vp<%index> = phi [ ir<0>, vector.ph ], [ vp<%index.next>, for.body.2 ]
; CHECK-NEXT:   vp<%8> = SCALAR-STEPS vp<%index>, ir<1>, vp<[[CLAMPED_VF]]>
; CHECK-NEXT: Successor(s): pred.load
; CHECK-EMPTY:
; CHECK-NEXT: <xVFxUF> pred.load: {
; CHECK-NEXT:   pred.load.entry:
; CHECK-NEXT:     BRANCH-ON-MASK vp<[[ALIAS_MASK]]>
; CHECK-NEXT:   Successor(s): pred.load.if, pred.load.continue
; CHECK-EMPTY:
; CHECK-NEXT:   pred.load.if:
; CHECK-NEXT:     REPLICATE ir<%arrayidx> = getelementptr inbounds ir<%a>, vp<%8>
; CHECK-NEXT:     REPLICATE ir<%0> = load ir<%arrayidx> (S->V)
; CHECK-NEXT:     REPLICATE ir<%arrayidx2> = getelementptr inbounds ir<%b>, vp<%8>
; CHECK-NEXT:     REPLICATE ir<%1> = load ir<%arrayidx2> (S->V)
; CHECK-NEXT:   Successor(s): pred.load.continue
; CHECK-EMPTY:
; CHECK-NEXT:   pred.load.continue:
; CHECK-NEXT:     PHI-PREDICATED-INSTRUCTION vp<[[VEC_A:%.+]]> = ir<%0>
; CHECK-NEXT:     PHI-PREDICATED-INSTRUCTION vp<[[VEC_B:%.+]]> = ir<%1>
; CHECK-NEXT:   No successors
; CHECK-NEXT: }
; CHECK-NEXT: Successor(s): for.body.1
; CHECK-EMPTY:
; CHECK-NEXT: for.body.1:
; CHECK-NEXT:   WIDEN ir<%add> = add vp<[[VEC_B]]>, vp<[[VEC_A]]>
; CHECK-NEXT: Successor(s): pred.store
; CHECK-EMPTY:
; CHECK-NEXT: <xVFxUF> pred.store: {
; CHECK-NEXT:   pred.store.entry:
; CHECK-NEXT:     BRANCH-ON-MASK vp<[[ALIAS_MASK]]>
; CHECK-NEXT:   Successor(s): pred.store.if, pred.store.continue
; CHECK-EMPTY:
; CHECK-NEXT:   pred.store.if:
; CHECK-NEXT:     REPLICATE ir<%arrayidx6> = getelementptr inbounds ir<%c>, vp<%8>
; CHECK-NEXT:     REPLICATE store ir<%add>, ir<%arrayidx6>
; CHECK-NEXT:   Successor(s): pred.store.continue
; CHECK-EMPTY:
; CHECK-NEXT:   pred.store.continue:
; CHECK-NEXT:   No successors
; CHECK-NEXT: }
; CHECK-NEXT: Successor(s): for.body.2
; CHECK-EMPTY:
; CHECK-NEXT: for.body.2:
; CHECK-NEXT:   EMIT vp<%index.next> = add nuw vp<%index>, vp<[[CLAMPED_VF]]>
; CHECK-NEXT:   EMIT vp<[[EXIT_COND:%.+]]> = icmp eq vp<%index.next>, vp<%n.vec>
; CHECK-NEXT:   EMIT branch-on-cond vp<[[EXIT_COND]]>
; CHECK-NEXT: Successor(s): middle.block, vector.body
; CHECK-EMPTY:
; CHECK-NEXT: middle.block:

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
