; REQUIRES: asserts

; RUN: opt -passes=loop-vectorize -debug-only=loop-vectorize -force-vector-interleave=1 -force-vector-width=4 -prefer-inloop-reductions -enable-interleaved-mem-accesses=true -enable-masked-interleaved-mem-accesses -force-widen-divrem-via-safe-divisor=0 -disable-output %s 2>&1 | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"

; Tests for printing VPlans.

define void @print_call_and_memory(i64 %n, ptr noalias %y, ptr noalias %x) nounwind uwtable {
; CHECK-LABEL: Checking a loop in 'print_call_and_memory'
; CHECK:      VPlan 'Initial VPlan for VF={4},UF>=1' {
; CHECK-NEXT: Live-in vp<[[VFxUF:%.]]> = VF * UF
; CHECK-NEXT: Live-in vp<[[VTC:%.+]]> = vector-trip-count
; CHECK-NEXT: Live-in ir<%n> = original trip-count
; CHECK-EMPTY:
; CHECK-NEXT: vector.ph:
; CHECK-NEXT: Successor(s): vector loop
; CHECK-EMPTY:
; CHECK-NEXT: <x1> vector loop: {
; CHECK-NEXT: vector.body:
; CHECK-NEXT:   EMIT vp<[[CAN_IV:%.+]]> = CANONICAL-INDUCTION ir<0>, vp<[[CAN_IV_NEXT:%.+]]>
; CHECK-NEXT:   vp<[[STEPS:%.+]]> = SCALAR-STEPS vp<[[CAN_IV]]>, ir<1>
; CHECK-NEXT:   CLONE ir<%arrayidx> = getelementptr inbounds ir<%y>, vp<[[STEPS]]>
; CHECK-NEXT:   vp<[[VEC_PTR:%.+]]> = vector-pointer ir<%arrayidx>
; CHECK-NEXT:   WIDEN ir<%lv> = load vp<[[VEC_PTR]]>
; CHECK-NEXT:   WIDEN-CALL ir<%call> = call @llvm.sqrt.f32(ir<%lv>)
; CHECK-NEXT:   CLONE ir<%arrayidx2> = getelementptr inbounds ir<%x>, vp<[[STEPS]]>
; CHECK-NEXT:   vp<[[VEC_PTR2:%.+]]> = vector-pointer ir<%arrayidx2>
; CHECK-NEXT:   WIDEN store vp<[[VEC_PTR2]]>, ir<%call>
; CHECK-NEXT:   EMIT vp<[[CAN_IV_NEXT]]> = add nuw vp<[[CAN_IV]]>, vp<[[VFxUF]]>
; CHECK-NEXT:   EMIT branch-on-count vp<[[CAN_IV_NEXT]]>, vp<[[VTC]]>
; CHECK-NEXT: No successors
; CHECK-NEXT: }
; CHECK-NEXT: Successor(s): middle.block
; CHECK-EMPTY:
; CHECK-NEXT: middle.block:
; CHECK-NEXT:    EMIT vp<[[CMP:%.+]]> = icmp eq ir<%n>, vp<[[VTC]]>
; CHECK-NEXT:    EMIT branch-on-cond vp<[[CMP]]>
; CHECK-NEXT:  Successor(s): ir-bb<for.end.loopexit>, scalar.ph
; CHECK-EMPTY:
; CHECK-NEXT:  ir-bb<for.end.loopexit>
; CHECK-NEXT:  No successors
; CHECK-EMPTY:
; CHECK-NEXT:  scalar.ph
; CHECK-NEXT:  No successors
; CHECK-NEXT: }
;
entry:
  %cmp6 = icmp sgt i64 %n, 0
  br i1 %cmp6, label %for.body, label %for.end

for.body:                                         ; preds = %entry, %for.body
  %iv = phi i64 [ %iv.next, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds float, ptr %y, i64 %iv
  %lv = load float, ptr %arrayidx, align 4
  %call = tail call float @llvm.sqrt.f32(float %lv) nounwind readnone
  %arrayidx2 = getelementptr inbounds float, ptr %x, i64 %iv
  store float %call, ptr %arrayidx2, align 4
  %iv.next = add i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  ret void
}

define void @print_widen_gep_and_select(i64 %n, ptr noalias %y, ptr noalias %x, ptr %z) nounwind uwtable {
; CHECK-LABEL: Checking a loop in 'print_widen_gep_and_select'
; CHECK:      VPlan 'Initial VPlan for VF={4},UF>=1' {
; CHECK-NEXT: Live-in vp<[[VFxUF:%.]]> = VF * UF
; CHECK-NEXT: Live-in vp<[[VTC:%.+]]> = vector-trip-count
; CHECK-NEXT: Live-in ir<%n> = original trip-count
; CHECK-EMPTY:
; CHECK-NEXT: vector.ph:
; CHECK-NEXT: Successor(s): vector loop
; CHECK-EMPTY:
; CHECK-NEXT: <x1> vector loop: {
; CHECK-NEXT: vector.body:
; CHECK-NEXT:   EMIT vp<[[CAN_IV:%.+]]> = CANONICAL-INDUCTION ir<0>, vp<[[CAN_IV_NEXT:%.+]]>
; CHECK-NEXT:   WIDEN-INDUCTION %iv = phi %iv.next, 0, ir<1>
; CHECK-NEXT:   vp<[[STEPS:%.+]]> = SCALAR-STEPS vp<[[CAN_IV]]>, ir<1>
; CHECK-NEXT:   WIDEN-GEP Inv[Var] ir<%arrayidx> = getelementptr inbounds ir<%y>, ir<%iv>
; CHECK-NEXT:   vp<[[VEC_PTR:%.+]]> = vector-pointer ir<%arrayidx>
; CHECK-NEXT:   WIDEN ir<%lv> = load vp<[[VEC_PTR]]>
; CHECK-NEXT:   WIDEN ir<%cmp> = icmp eq ir<%arrayidx>, ir<%z>
; CHECK-NEXT:   WIDEN-SELECT ir<%sel> = select ir<%cmp>, ir<1.000000e+01>, ir<2.000000e+01>
; CHECK-NEXT:   WIDEN ir<%add> = fadd ir<%lv>, ir<%sel>
; CHECK-NEXT:   CLONE ir<%arrayidx2> = getelementptr inbounds ir<%x>, vp<[[STEPS]]>
; CHECK-NEXT:   vp<[[VEC_PTR2:%.+]]> = vector-pointer ir<%arrayidx2>
; CHECK-NEXT:   WIDEN store vp<[[VEC_PTR2]]>, ir<%add>
; CHECK-NEXT:   EMIT vp<[[CAN_IV_NEXT]]> = add nuw vp<[[CAN_IV]]>, vp<[[VFxUF]]>
; CHECK-NEXT:   EMIT branch-on-count vp<[[CAN_IV_NEXT]]>, vp<[[VTC]]>
; CHECK-NEXT: No successors
; CHECK-NEXT: }
; CHECK-NEXT: Successor(s): middle.block
; CHECK-EMPTY:
; CHECK-NEXT: middle.block:
; CHECK-NEXT:    EMIT vp<[[CMP:%.+]]> = icmp eq ir<%n>, vp<[[VTC]]>
; CHECK-NEXT:    EMIT branch-on-cond vp<[[CMP]]>
; CHECK-NEXT:  Successor(s): ir-bb<for.end.loopexit>, scalar.ph
; CHECK-EMPTY:
; CHECK-NEXT:  ir-bb<for.end.loopexit>
; CHECK-NEXT:  No successors
; CHECK-EMPTY:
; CHECK-NEXT:  scalar.ph
; CHECK-NEXT:  No successors
; CHECK-NEXT: }
;
entry:
  %cmp6 = icmp sgt i64 %n, 0
  br i1 %cmp6, label %for.body, label %for.end

for.body:                                         ; preds = %entry, %for.body
  %iv = phi i64 [ %iv.next, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds float, ptr %y, i64 %iv
  %lv = load float, ptr %arrayidx, align 4
  %cmp = icmp eq ptr %arrayidx, %z
  %sel = select i1 %cmp, float 10.0, float 20.0
  %add = fadd float %lv, %sel
  %arrayidx2 = getelementptr inbounds float, ptr %x, i64 %iv
  store float %add, ptr %arrayidx2, align 4
  %iv.next = add i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  ret void
}

define float @print_reduction(i64 %n, ptr noalias %y) {
; CHECK-LABEL: Checking a loop in 'print_reduction'
; CHECK:      VPlan 'Initial VPlan for VF={4},UF>=1' {
; CHECK-NEXT: Live-in vp<[[VFxUF:%.]]> = VF * UF
; CHECK-NEXT: Live-in vp<[[VTC:%.+]]> = vector-trip-count
; CHECK-NEXT: Live-in ir<%n> = original trip-count
; CHECK-EMPTY:
; CHECK-NEXT: vector.ph:
; CHECK-NEXT: Successor(s): vector loop
; CHECK-EMPTY:
; CHECK-NEXT: <x1> vector loop: {
; CHECK-NEXT: vector.body:
; CHECK-NEXT:   EMIT vp<[[CAN_IV:%.+]]> = CANONICAL-INDUCTION ir<0>, vp<[[CAN_IV_NEXT:%.+]]>
; CHECK-NEXT:   WIDEN-REDUCTION-PHI ir<%red> = phi ir<0.000000e+00>, ir<%red.next>
; CHECK-NEXT:   vp<[[STEPS:%.+]]> = SCALAR-STEPS vp<[[CAN_IV]]>, ir<1>
; CHECK-NEXT:   CLONE ir<%arrayidx> = getelementptr inbounds ir<%y>, vp<[[STEPS]]>
; CHECK-NEXT:   vp<[[VEC_PTR:%.+]]> = vector-pointer ir<%arrayidx>
; CHECK-NEXT:   WIDEN ir<%lv> = load vp<[[VEC_PTR]]>
; CHECK-NEXT:   REDUCE ir<%red.next> = ir<%red> + fast reduce.fadd (ir<%lv>)
; CHECK-NEXT:   EMIT vp<[[CAN_IV_NEXT]]> = add nuw vp<[[CAN_IV]]>, vp<[[VFxUF]]>
; CHECK-NEXT:   EMIT branch-on-count vp<[[CAN_IV_NEXT]]>, vp<[[VTC]]>
; CHECK-NEXT: No successors
; CHECK-NEXT: }
; CHECK-NEXT: Successor(s): middle.block
; CHECK-EMPTY:
; CHECK-NEXT: middle.block:
; CHECK-NEXT:   EMIT vp<[[RED_RES:%.+]]> = compute-reduction-result ir<%red>, ir<%red.next>
; CHECK-NEXT:   EMIT vp<[[RED_EX:%.+]]> = extract-from-end vp<[[RED_RES]]>, ir<1>
; CHECK-NEXT:   EMIT vp<[[CMP:%.+]]> = icmp eq ir<%n>, vp<[[VTC]]>
; CHECK-NEXT:   EMIT branch-on-cond vp<[[CMP]]>
; CHECK-NEXT: Successor(s): ir-bb<for.end>, scalar.ph
; CHECK-EMPTY:
; CHECK-NEXT: ir-bb<for.end>
; CHECK-NEXT: No successors
; CHECK-EMPTY:
; CHECK-NEXT: scalar.ph
; CHECK-NEXT: No successors
; CHECK-EMPTY:
; CHECK-NEXT: Live-out float %red.next.lcssa = vp<[[RED_EX]]>
; CHECK-NEXT: }
;
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %iv = phi i64 [ %iv.next, %for.body ], [ 0, %entry ]
  %red = phi float [ %red.next, %for.body ], [ 0.0, %entry ]
  %arrayidx = getelementptr inbounds float, ptr %y, i64 %iv
  %lv = load float, ptr %arrayidx, align 4
  %red.next = fadd fast float %lv, %red
  %iv.next = add i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  ret float %red.next
}

define void @print_reduction_with_invariant_store(i64 %n, ptr noalias %y, ptr noalias %dst) {
; CHECK-LABEL: Checking a loop in 'print_reduction_with_invariant_store'
; CHECK:      VPlan 'Initial VPlan for VF={4},UF>=1' {
; CHECK-NEXT: Live-in vp<[[VFxUF:%.]]> = VF * UF
; CHECK-NEXT: Live-in vp<[[VTC:%.+]]> = vector-trip-count
; CHECK-NEXT: Live-in ir<%n> = original trip-count
; CHECK-EMPTY:
; CHECK-NEXT: vector.ph:
; CHECK-NEXT: Successor(s): vector loop
; CHECK-EMPTY:
; CHECK-NEXT: <x1> vector loop: {
; CHECK-NEXT: vector.body:
; CHECK-NEXT:   EMIT vp<[[CAN_IV:%.+]]> = CANONICAL-INDUCTION ir<0>, vp<[[CAN_IV_NEXT:%.+]]>
; CHECK-NEXT:   WIDEN-REDUCTION-PHI ir<%red> = phi ir<0.000000e+00>, ir<%red.next>
; CHECK-NEXT:   vp<[[IV:%.+]]> = SCALAR-STEPS vp<[[CAN_IV]]>, ir<1>
; CHECK-NEXT:   CLONE ir<%arrayidx> = getelementptr inbounds ir<%y>, vp<[[IV]]>
; CHECK-NEXT:   vp<[[VEC_PTR:%.+]]> = vector-pointer ir<%arrayidx>
; CHECK-NEXT:   WIDEN ir<%lv> = load vp<[[VEC_PTR]]>
; CHECK-NEXT:   REDUCE ir<%red.next> = ir<%red> + fast reduce.fadd (ir<%lv>) (with final reduction value stored in invariant address sank outside of loop)
; CHECK-NEXT:   EMIT vp<[[CAN_IV_NEXT]]> = add nuw vp<[[CAN_IV]]>, vp<[[VFxUF]]>
; CHECK-NEXT:   EMIT branch-on-count vp<[[CAN_IV_NEXT]]>, vp<[[VTC]]>
; CHECK-NEXT: No successors
; CHECK-NEXT: }
; CHECK-NEXT: Successor(s): middle.block
; CHECK-EMPTY:
; CHECK-NEXT: middle.block:
; CHECK-NEXT:   EMIT vp<[[RED_RES:.+]]> = compute-reduction-result ir<%red>, ir<%red.next>
; CHECK-NEXT:   EMIT vp<[[CMP:%.+]]> = icmp eq ir<%n>, vp<[[VTC]]>
; CHECK-NEXT:   EMIT branch-on-cond vp<[[CMP]]>
; CHECK-NEXT: Successor(s): ir-bb<for.end>, scalar.ph
; CHECK-EMPTY:
; CHECK-NEXT: ir-bb<for.end>
; CHECK-NEXT: No successors
; CHECK-EMPTY:
; CHECK-NEXT: scalar.ph
; CHECK-NEXT: No successors
; CHECK-NEXT: }
;
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %iv = phi i64 [ %iv.next, %for.body ], [ 0, %entry ]
  %red = phi float [ %red.next, %for.body ], [ 0.0, %entry ]
  %arrayidx = getelementptr inbounds float, ptr %y, i64 %iv
  %lv = load float, ptr %arrayidx, align 4
  %red.next = fadd fast float %lv, %red
  store float %red.next, ptr %dst, align 4
  %iv.next = add i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  ret void
}

define void @print_replicate_predicated_phi(i64 %n, ptr %x) {
; CHECK-LABEL: Checking a loop in 'print_replicate_predicated_phi'
; CHECK:      VPlan 'Initial VPlan for VF={4},UF>=1' {
; CHECK-NEXT: Live-in vp<[[VFxUF:%.]]> = VF * UF
; CHECK-NEXT: Live-in vp<[[VTC:%.+]]> = vector-trip-count
; CHECK-NEXT: vp<[[TC:%.+]]> = original trip-count
; CHECK-EMPTY:
; CHECK-NEXT: ir-bb<entry>:
; CHECK-NEXT:  EMIT vp<[[TC]]> = EXPAND SCEV (1 smax %n)
; CHECK-NEXT: No successors
; CHECK-EMPTY:
; CHECK-NEXT: vector.ph:
; CHECK-NEXT: Successor(s): vector loop
; CHECK-EMPTY:
; CHECK-NEXT: <x1> vector loop: {
; CHECK-NEXT: vector.body:
; CHECK-NEXT:   EMIT vp<[[CAN_IV:%.+]]> = CANONICAL-INDUCTION ir<0>, vp<[[CAN_IV_NEXT:%.+]]>
; CHECK-NEXT:   WIDEN-INDUCTION %i = phi 0, %i.next, ir<1>
; CHECK-NEXT:   vp<[[STEPS:%.+]]> = SCALAR-STEPS vp<[[CAN_IV]]>, ir<1>
; CHECK-NEXT:   WIDEN ir<%cmp> = icmp ult ir<%i>, ir<5>
; CHECK-NEXT: Successor(s): pred.udiv
; CHECK-EMPTY:
; CHECK-NEXT: <xVFxUF> pred.udiv: {
; CHECK-NEXT:   pred.udiv.entry:
; CHECK-NEXT:     BRANCH-ON-MASK ir<%cmp>
; CHECK-NEXT:   Successor(s): pred.udiv.if, pred.udiv.continue
; CHECK-EMPTY:
; CHECK-NEXT:   pred.udiv.if:
; CHECK-NEXT:     REPLICATE ir<%tmp4> = udiv ir<%n>, vp<[[STEPS]]> (S->V)
; CHECK-NEXT:   Successor(s): pred.udiv.continue
; CHECK-EMPTY:
; CHECK-NEXT:   pred.udiv.continue:
; CHECK-NEXT:     PHI-PREDICATED-INSTRUCTION vp<[[PRED:%.+]]> = ir<%tmp4>
; CHECK-NEXT:   No successors
; CHECK-NEXT: }
; CHECK-NEXT: Successor(s): if.then.0
; CHECK-EMPTY:
; CHECK-NEXT: if.then.0:
; CHECK-NEXT:   BLEND ir<%d> = ir<0> vp<[[PRED]]>/ir<%cmp>
; CHECK-NEXT:   CLONE ir<%idx> = getelementptr ir<%x>, vp<[[STEPS]]>
; CHECK-NEXT:   vp<[[VEC_PTR:%.+]]> = vector-pointer ir<%idx>
; CHECK-NEXT:   WIDEN store vp<[[VEC_PTR]]>, ir<%d>
; CHECK-NEXT:   EMIT vp<[[CAN_IV_NEXT]]> = add nuw vp<[[CAN_IV]]>, vp<[[VFxUF]]>
; CHECK-NEXT:   EMIT branch-on-count vp<[[CAN_IV_NEXT]]>, vp<[[VTC]]>
; CHECK-NEXT: No successors
; CHECK-NEXT: }
; CHECK-NEXT: Successor(s): middle.block
; CHECK-EMPTY:
; CHECK-NEXT: middle.block:
; CHECK-NEXT:   EMIT vp<[[CMP:%.+]]> = icmp eq vp<[[TC]]>, vp<[[VTC]]>
; CHECK-NEXT:   EMIT branch-on-cond vp<[[CMP]]>
; CHECK-NEXT: Successor(s): ir-bb<for.end>, scalar.ph
; CHECK-EMPTY:
; CHECK-NEXT: ir-bb<for.end>
; CHECK-NEXT: No successors
; CHECK-EMPTY:
; CHECK-NEXT: scalar.ph
; CHECK-NEXT: No successors
; CHECK-NEXT: }
;
entry:
  br label %for.body

for.body:                                         ; preds = %for.inc, %entry
  %i = phi i64 [ 0, %entry ], [ %i.next, %for.inc ]
  %cmp = icmp ult i64 %i, 5
  br i1 %cmp, label %if.then, label %for.inc

if.then:                                          ; preds = %for.body
  %tmp4 = udiv i64 %n, %i
  br label %for.inc

for.inc:                                          ; preds = %if.then, %for.body
  %d = phi i64 [ 0, %for.body ], [ %tmp4, %if.then ]
  %idx = getelementptr i64, ptr %x, i64 %i
  store i64 %d, ptr %idx
  %i.next = add nuw nsw i64 %i, 1
  %cond = icmp slt i64 %i.next, %n
  br i1 %cond, label %for.body, label %for.end

for.end:                                          ; preds = %for.inc
  ret void
}

@AB = common global [1024 x i32] zeroinitializer, align 4
@CD = common global [1024 x i32] zeroinitializer, align 4

define void @print_interleave_groups(i32 %C, i32 %D) {
; CHECK-LABEL: Checking a loop in 'print_interleave_groups'
; CHECK:       VPlan 'Initial VPlan for VF={4},UF>=1' {
; CHECK-NEXT: Live-in vp<[[VFxUF:%.]]> = VF * UF
; CHECK-NEXT: Live-in vp<[[VTC:%.+]]> = vector-trip-count
; CHECK-NEXT: Live-in ir<256> = original trip-count
; CHECK-EMPTY:
; CHECK-NEXT: vector.ph:
; CHECK-NEXT: Successor(s): vector loop
; CHECK-EMPTY:
; CHECK-NEXT: <x1> vector loop: {
; CHECK-NEXT:  vector.body:
; CHECK-NEXT:   EMIT vp<[[CAN_IV:%.+]]> = CANONICAL-INDUCTION ir<0>, vp<[[CAN_IV_NEXT:%.+]]>
; CHECK-NEXT:   vp<[[DERIVED_IV:%.+]]> = DERIVED-IV ir<0> + vp<[[CAN_IV]]> * ir<4>
; CHECK-NEXT:   vp<[[STEPS:%.+]]> = SCALAR-STEPS vp<[[DERIVED_IV]]>, ir<4>
; CHECK-NEXT:   CLONE ir<%gep.AB.0> = getelementptr inbounds ir<@AB>, ir<0>, vp<[[STEPS]]>
; CHECK-NEXT:   INTERLEAVE-GROUP with factor 4 at %AB.0, ir<%gep.AB.0>
; CHECK-NEXT:     ir<%AB.0> = load from index 0
; CHECK-NEXT:     ir<%AB.1> = load from index 1
; CHECK-NEXT:     ir<%AB.3> = load from index 3
; CHECK-NEXT:   CLONE ir<%iv.plus.3> = add vp<[[STEPS]]>, ir<3>
; CHECK-NEXT:   WIDEN ir<%add> = add nsw ir<%AB.0>, ir<%AB.1>
; CHECK-NEXT:   CLONE ir<%gep.CD.3> = getelementptr inbounds ir<@CD>, ir<0>, ir<%iv.plus.3>
; CHECK-NEXT:   INTERLEAVE-GROUP with factor 4 at <badref>, ir<%gep.CD.3>
; CHECK-NEXT:     store ir<%add> to index 0
; CHECK-NEXT:     store ir<1> to index 1
; CHECK-NEXT:     store ir<2> to index 2
; CHECK-NEXT:     store ir<%AB.3> to index 3
; CHECK-NEXT:   EMIT vp<[[CAN_IV_NEXT]]> = add nuw vp<[[CAN_IV]]>, vp<[[VFxUF]]>
; CHECK-NEXT:   EMIT branch-on-count vp<[[CAN_IV_NEXT]]>, vp<[[VTC]]>
; CHECK-NEXT: No successors
; CHECK-NEXT: }
; CHECK-NEXT: Successor(s): middle.block
; CHECK-EMPTY:
; CHECK-NEXT: middle.block:
; CHECK-NEXT:   EMIT vp<[[CMP:%.+]]> = icmp eq ir<256>, vp<[[VTC]]>
; CHECK-NEXT:   EMIT branch-on-cond vp<[[CMP]]>
; CHECK-NEXT: Successor(s): ir-bb<for.end>, scalar.ph
; CHECK-EMPTY:
; CHECK-NEXT: ir-bb<for.end>
; CHECK-NEXT: No successors
; CHECK-EMPTY:
; CHECK-NEXT: scalar.ph
; CHECK-NEXT: No successors
; CHECK-NEXT: }
;
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %gep.AB.0= getelementptr inbounds [1024 x i32], ptr @AB, i64 0, i64 %iv
  %AB.0 = load i32, ptr %gep.AB.0, align 4
  %iv.plus.1 = add i64 %iv, 1
  %gep.AB.1 = getelementptr inbounds [1024 x i32], ptr @AB, i64 0, i64 %iv.plus.1
  %AB.1 = load i32, ptr %gep.AB.1, align 4
  %iv.plus.2 = add i64 %iv, 2
  %iv.plus.3 = add i64 %iv, 3
  %gep.AB.3 = getelementptr inbounds [1024 x i32], ptr @AB, i64 0, i64 %iv.plus.3
  %AB.3 = load i32, ptr %gep.AB.3, align 4
  %add = add nsw i32 %AB.0, %AB.1
  %gep.CD.0 = getelementptr inbounds [1024 x i32], ptr @CD, i64 0, i64 %iv
  store i32 %add, ptr %gep.CD.0, align 4
  %gep.CD.1 = getelementptr inbounds [1024 x i32], ptr @CD, i64 0, i64 %iv.plus.1
  store i32 1, ptr %gep.CD.1, align 4
  %gep.CD.2 = getelementptr inbounds [1024 x i32], ptr @CD, i64 0, i64 %iv.plus.2
  store i32 2, ptr %gep.CD.2, align 4
  %gep.CD.3 = getelementptr inbounds [1024 x i32], ptr @CD, i64 0, i64 %iv.plus.3
  store i32 %AB.3, ptr %gep.CD.3, align 4
  %iv.next = add nuw nsw i64 %iv, 4
  %cmp = icmp slt i64 %iv.next, 1024
  br i1 %cmp, label %for.body, label %for.end

for.end:
  ret void
}

define float @print_fmuladd_strict(ptr %a, ptr %b, i64 %n) {
; CHECK-LABEL: Checking a loop in 'print_fmuladd_strict'
; CHECK:      VPlan 'Initial VPlan for VF={4},UF>=1' {
; CHECK-NEXT: Live-in vp<[[VFxUF:%.]]> = VF * UF
; CHECK-NEXT: Live-in vp<[[VTC:%.+]]> = vector-trip-count
; CHECK-NEXT: Live-in ir<%n> = original trip-count
; CHECK-EMPTY:
; CHECK-NEXT: vector.ph:
; CHECK-NEXT: Successor(s): vector loop
; CHECK-EMPTY:
; CHECK-NEXT: <x1> vector loop: {
; CHECK-NEXT: vector.body:
; CHECK-NEXT:   EMIT vp<[[CAN_IV:%.+]]> = CANONICAL-INDUCTION ir<0>, vp<[[CAN_IV_NEXT:%.+]]>
; CHECK-NEXT:   WIDEN-REDUCTION-PHI ir<%sum.07> = phi ir<0.000000e+00>, ir<%muladd>
; CHECK-NEXT:   vp<[[STEPS:%.+]]> = SCALAR-STEPS vp<[[CAN_IV]]>, ir<1>
; CHECK-NEXT:   CLONE ir<%arrayidx> = getelementptr inbounds ir<%a>, vp<[[STEPS]]>
; CHECK-NEXT:   vp<[[VEC_PTR:%.+]]> = vector-pointer ir<%arrayidx>
; CHECK-NEXT:   WIDEN ir<%l.a> = load vp<[[VEC_PTR]]>
; CHECK-NEXT:   CLONE ir<%arrayidx2> = getelementptr inbounds ir<%b>, vp<[[STEPS]]>
; CHECK-NEXT:   vp<[[VEC_PTR2:%.+]]> = vector-pointer ir<%arrayidx2>
; CHECK-NEXT:   WIDEN ir<%l.b> = load vp<[[VEC_PTR2]]>
; CHECK-NEXT:   EMIT vp<[[FMUL:%.+]]> = fmul nnan ninf nsz ir<%l.a>, ir<%l.b>
; CHECK-NEXT:   REDUCE ir<[[MULADD:%.+]]> = ir<%sum.07> + nnan ninf nsz reduce.fadd (vp<[[FMUL]]>)
; CHECK-NEXT:   EMIT vp<[[CAN_IV_NEXT]]> = add nuw vp<[[CAN_IV]]>, vp<[[VFxUF]]>
; CHECK-NEXT:   EMIT branch-on-count vp<[[CAN_IV_NEXT]]>, vp<[[VTC]]>
; CHECK-NEXT:   No successors
; CHECK-NEXT: }
; CHECK-NEXT: Successor(s): middle.block
; CHECK-EMPTY:
; CHECK-NEXT: middle.block:
; CHECK-NEXT:   EMIT vp<[[RED_RES:%.+]]> = compute-reduction-result ir<%sum.07>, ir<[[MULADD]]>
; CHECK-NEXT:   EMIT vp<[[RED_EX:%.+]]> = extract-from-end vp<[[RED_RES]]>, ir<1>
; CHECK-NEXT:   EMIT vp<[[CMP:%.+]]> = icmp eq ir<%n>, vp<[[VTC]]>
; CHECK-NEXT:   EMIT branch-on-cond vp<[[CMP]]>
; CHECK-NEXT: Successor(s): ir-bb<for.end>, scalar.ph
; CHECK-EMPTY:
; CHECK-NEXT: ir-bb<for.end>
; CHECK-NEXT: No successors
; CHECK-EMPTY:
; CHECK-NEXT: scalar.ph
; CHECK-NEXT: No successors
; CHECK-EMPTY:
; CHECK-NEXT: Live-out float %muladd.lcssa = vp<[[RED_EX]]>
; CHECK-NEXT:}

entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %sum.07 = phi float [ 0.000000e+00, %entry ], [ %muladd, %for.body ]
  %arrayidx = getelementptr inbounds float, ptr %a, i64 %iv
  %l.a = load float, ptr %arrayidx, align 4
  %arrayidx2 = getelementptr inbounds float, ptr %b, i64 %iv
  %l.b = load float, ptr %arrayidx2, align 4
  %muladd = tail call nnan ninf nsz float @llvm.fmuladd.f32(float %l.a, float %l.b, float %sum.07)
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond.not = icmp eq i64 %iv.next, %n
  br i1 %exitcond.not, label %for.end, label %for.body

for.end:
  ret float %muladd
}

define void @debug_loc_vpinstruction(ptr nocapture %asd, ptr nocapture %bsd) !dbg !5 {
; CHECK-LABEL: Checking a loop in 'debug_loc_vpinstruction'
; CHECK:    VPlan 'Initial VPlan for VF={4},UF>=1' {
; CHECK-NEXT: Live-in vp<[[VFxUF:%.]]> = VF * UF
; CHECK-NEXT: Live-in vp<[[VTC:%.+]]> = vector-trip-count
; CHECK-NEXT: Live-in ir<128> = original trip-count
; CHECK-EMPTY:
; CHECK-NEXT: vector.ph:
; CHECK-NEXT: Successor(s): vector loop
; CHECK-EMPTY:
; CHECK-NEXT: <x1> vector loop: {
; CHECK-NEXT:  vector.body:
; CHECK-NEXT:    EMIT vp<[[CAN_IV:%.+]]> = CANONICAL-INDUCTION ir<0>, vp<[[CAN_IV_NEXT:%.+]]>
; CHECK-NEXT:    vp<[[STEPS:%.+]]> = SCALAR-STEPS vp<[[CAN_IV]]>, ir<1>
; CHECK-NEXT:    CLONE ir<%isd> = getelementptr inbounds ir<%asd>, vp<[[STEPS]]>
; CHECK-NEXT:    vp<[[VEC_PTR:%.+]]> = vector-pointer ir<%isd>
; CHECK-NEXT:    WIDEN ir<%lsd> = load vp<[[VEC_PTR]]>
; CHECK-NEXT:    WIDEN ir<%psd> = add nuw nsw ir<%lsd>, ir<23>
; CHECK-NEXT:    WIDEN ir<%cmp1> = icmp slt ir<%lsd>, ir<100>
; CHECK-NEXT:    EMIT vp<[[NOT1:%.+]]> = not ir<%cmp1>, !dbg /tmp/s.c:5:3
; CHECK-NEXT:    WIDEN ir<%cmp2> = icmp sge ir<%lsd>, ir<200>
; CHECK-NEXT:    EMIT vp<[[SEL1:%.+]]> = logical-and vp<[[NOT1]]>, ir<%cmp2>, !dbg /tmp/s.c:5:21
; CHECK-NEXT:    EMIT vp<[[OR1:%.+]]> = or vp<[[SEL1]]>, ir<%cmp1>
; CHECK-NEXT:  Successor(s): pred.sdiv
; CHECK-EMPTY:
; CHECK-NEXT:  <xVFxUF> pred.sdiv: {
; CHECK-NEXT:    pred.sdiv.entry:
; CHECK-NEXT:      BRANCH-ON-MASK vp<[[OR1]]>
; CHECK-NEXT:    Successor(s): pred.sdiv.if, pred.sdiv.continue
; CHECK-EMPTY:
; CHECK-NEXT:    pred.sdiv.if:
; CHECK-NEXT:      REPLICATE ir<%sd1> = sdiv ir<%psd>, ir<%lsd> (S->V)
; CHECK-NEXT:    Successor(s): pred.sdiv.continue
; CHECK-EMPTY:
; CHECK-NEXT:    pred.sdiv.continue:
; CHECK-NEXT:      PHI-PREDICATED-INSTRUCTION vp<[[PHI:%.+]]> = ir<%sd1>
; CHECK-NEXT:    No successors
; CHECK-NEXT:  }
; CHECK-NEXT:  Successor(s): if.then.0
; CHECK-EMPTY:
; CHECK-NEXT:  if.then.0:
; CHECK-NEXT:    EMIT vp<[[NOT2:%.+]]> = not ir<%cmp2>
; CHECK-NEXT:    EMIT vp<[[SEL2:%.+]]> = logical-and vp<[[NOT1]]>, vp<[[NOT2]]>
; CHECK-NEXT:    BLEND ir<%ysd.0> = vp<[[PHI]]> ir<%psd>/vp<[[SEL2]]>
; CHECK-NEXT:    vp<[[VEC_PTR2:%.+]]> = vector-pointer ir<%isd>
; CHECK-NEXT:    WIDEN store vp<[[VEC_PTR2]]>, ir<%ysd.0>
; CHECK-NEXT:    EMIT vp<[[CAN_IV_NEXT]]> = add nuw vp<[[CAN_IV]]>, vp<[[VFxUF]]>
; CHECK-NEXT:    EMIT branch-on-count vp<[[CAN_IV_NEXT]]>, vp<[[VTC]]>
; CHECK-NEXT:  No successors
; CHECK-NEXT:}
; CHECK-NEXT: Successor(s): middle.block
; CHECK-EMPTY:
; CHECK-NEXT: middle.block:
; CHECK-NEXT:   EMIT vp<[[CMP:%.+]]> = icmp eq ir<128>, vp<[[VTC]]>
; CHECK-NEXT:   EMIT branch-on-cond vp<[[CMP]]>
; CHECK-NEXT: Successor(s): ir-bb<exit>, scalar.ph
; CHECK-EMPTY:
; CHECK-NEXT: ir-bb<exit>
; CHECK-NEXT: No successors
; CHECK-EMPTY:
; CHECK-NEXT: scalar.ph
; CHECK-NEXT: No successors
; CHECK-NEXT:}
;
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %if.end ]
  %isd = getelementptr inbounds i32, ptr %asd, i64 %iv
  %lsd = load i32, ptr %isd, align 4
  %psd = add nuw nsw i32 %lsd, 23
  %cmp1 = icmp slt i32 %lsd, 100
  br i1 %cmp1, label %if.then, label %check, !dbg !7

check:
  %cmp2 = icmp sge i32 %lsd, 200
  br i1 %cmp2, label %if.then, label %if.end, !dbg !8

if.then:
  %sd1 = sdiv i32 %psd, %lsd
  br label %if.end

if.end:
  %ysd.0 = phi i32 [ %sd1, %if.then ], [ %psd, %check ]
  store i32 %ysd.0, ptr %isd, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 128
  br i1 %exitcond, label %exit, label %loop

exit:
  ret void
}

declare float @llvm.sqrt.f32(float) nounwind readnone
declare float @llvm.fmuladd.f32(float, float, float)

define void @print_expand_scev(i64 %y, ptr %ptr) {
; CHECK-LABEL: Checking a loop in 'print_expand_scev'
; CHECK: VPlan 'Initial VPlan for VF={4},UF>=1' {
; CHECK-NEXT: Live-in vp<[[VFxUF:%.]]> = VF * UF
; CHECK-NEXT: Live-in vp<[[VTC:%.+]]> = vector-trip-count
; CHECK-NEXT: vp<[[TC:%.+]]> = original trip-count
; CHECK-EMPTY:
; CHECK-NEXT: ir-bb<entry>:
; CHECK-NEXT:   EMIT vp<[[TC]]> = EXPAND SCEV (1 + ((15 + (%y /u 492802768830814060))<nuw><nsw> /u (1 + (%y /u 492802768830814060))<nuw><nsw>))<nuw><nsw>
; CHECK-NEXT:   EMIT vp<[[EXP_SCEV:%.+]]> = EXPAND SCEV (1 + (%y /u 492802768830814060))<nuw><nsw>
; CHECK-NEXT: No successors
; CHECK-EMPTY:
; CHECK-NEXT: vector.ph:
; CHECK-NEXT: Successor(s): vector loop
; CHECK-EMPTY:
; CHECK-NEXT: <x1> vector loop: {
; CHECK-NEXT:   vector.body:
; CHECK-NEXT:    EMIT vp<[[CAN_IV:%.+]]> = CANONICAL-INDUCTION ir<0>, vp<[[CAN_IV_NEXT:%.+]]>
; CHECK-NEXT:     WIDEN-INDUCTION\l" +
; CHECK-NEXT:     "  %iv = phi %iv.next, 0\l" +
; CHECK-NEXT:     "  ir<%v2>, vp<[[EXP_SCEV]]>
; CHECK-NEXT:     vp<[[DERIVED_IV:%.+]]> = DERIVED-IV ir<0> + vp<[[CAN_IV]]> * vp<[[EXP_SCEV]]>
; CHECK-NEXT:     vp<[[STEPS:%.+]]> = SCALAR-STEPS vp<[[DERIVED_IV]]>, vp<[[EXP_SCEV]]>
; CHECK-NEXT:     WIDEN ir<%v3> = add nuw ir<%v2>, ir<1>
; CHECK-NEXT:     REPLICATE ir<%gep> = getelementptr inbounds ir<%ptr>, vp<[[STEPS]]>
; CHECK-NEXT:     REPLICATE store ir<%v3>, ir<%gep>
; CHECK-NEXT:     EMIT vp<[[CAN_IV_NEXT]]> = add nuw vp<[[CAN_IV]]>, vp<[[VFxUF]]>
; CHECK-NEXT:     EMIT branch-on-count  vp<[[CAN_IV_NEXT]]>, vp<[[VTC]]>
; CHECK-NEXT:   No successors
; CHECK-NEXT: }
; CHECK-NEXT: Successor(s): middle.block
; CHECK-EMPTY:
; CHECK-NEXT: middle.block:
; CHECK-NEXT:   EMIT vp<[[CMP:%.+]]> = icmp eq vp<[[TC]]>, vp<[[VTC]]>
; CHECK-NEXT:   EMIT branch-on-cond vp<[[CMP]]>
; CHECK-NEXT: Successor(s): ir-bb<loop.exit>, scalar.ph
; CHECK-EMPTY:
; CHECK-NEXT: ir-bb<loop.exit>
; CHECK-NEXT: No successors
; CHECK-EMPTY:
; CHECK-NEXT: scalar.ph
; CHECK-NEXT: No successors
; CHECK-NEXT: }
;
entry:
  %div = udiv i64 %y, 492802768830814060
  %inc = add i64 %div, 1
  br label %loop

loop:                                             ; preds = %loop, %entry
  %iv = phi i64 [ %iv.next, %loop ], [ 0, %entry ]
  %v2 = trunc i64 %iv to i8
  %v3 = add nuw i8 %v2, 1
  %gep = getelementptr inbounds i8, ptr %ptr, i64 %iv
  store i8 %v3, ptr %gep

  %cmp15 = icmp slt i8 %v3, 10000
  %iv.next = add i64 %iv, %inc
  br i1 %cmp15, label %loop, label %loop.exit

loop.exit:
  ret void
}

define i32 @print_exit_value(ptr %ptr, i32 %off) {
; CHECK-LABEL: Checking a loop in 'print_exit_value'
; CHECK: VPlan 'Initial VPlan for VF={4},UF>=1' {
; CHECK-NEXT: Live-in vp<[[VFxUF:%.]]> = VF * UF
; CHECK-NEXT: Live-in vp<[[VTC:%.+]]> = vector-trip-count
; CHECK-NEXT: Live-in ir<1000> = original trip-count
; CHECK-EMPTY:
; CHECK-NEXT: vector.ph:
; CHECK-NEXT: Successor(s): vector loop
; CHECK-EMPTY:
; CHECK-NEXT: <x1> vector loop: {
; CHECK-NEXT:   vector.body:
; CHECK-NEXT:    EMIT vp<[[CAN_IV:%.+]]> = CANONICAL-INDUCTION ir<0>, vp<[[CAN_IV_NEXT:%.+]]>
; CHECK-NEXT:     WIDEN-INDUCTION %iv = phi 0, %iv.next, ir<1>
; CHECK-NEXT:     vp<[[STEPS:%.+]]>    = SCALAR-STEPS vp<[[CAN_IV]]>, ir<1>
; CHECK-NEXT:     CLONE ir<%gep> = getelementptr inbounds ir<%ptr>, vp<[[STEPS]]>
; CHECK-NEXT:     WIDEN ir<%add> = add ir<%iv>, ir<%off>
; CHECK-NEXT:     vp<[[VEC_PTR:%.+]]> = vector-pointer ir<%gep>
; CHECK-NEXT:     WIDEN store vp<[[VEC_PTR]]>, ir<0>
; CHECK-NEXT:     EMIT vp<[[CAN_IV_NEXT]]> = add nuw vp<[[CAN_IV]]>, vp<[[VFxUF]]>
; CHECK-NEXT:     EMIT branch-on-count vp<[[CAN_IV_NEXT]]>, vp<[[VTC]]>
; CHECK-NEXT:   No successors
; CHECK-NEXT: }
; CHECK-NEXT: Successor(s): middle.block
; CHECK-EMPTY:
; CHECK-NEXT: middle.block:
; CHECK-NEXT:   EMIT vp<[[EXIT:%.+]]> = extract-from-end ir<%add>, ir<1>
; CHECK-NEXT:   EMIT vp<[[CMP:%.+]]> = icmp eq ir<1000>, vp<[[VTC]]>
; CHECK-NEXT:   EMIT branch-on-cond vp<[[CMP]]>
; CHECK-NEXT: Successor(s): ir-bb<exit>, scalar.ph
; CHECK-EMPTY:
; CHECK-NEXT: ir-bb<exit>
; CHECK-NEXT: No successors
; CHECK-EMPTY:
; CHECK-NEXT: scalar.ph
; CHECK-NEXT: No successors
; CHECK-EMPTY:
; CHECK-NEXT: Live-out i32 %lcssa = vp<[[EXIT]]>
; CHECK-NEXT: }
;
entry:
  br label %loop

loop:
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %loop ]
  %gep = getelementptr inbounds i8, ptr %ptr, i32 %iv
  %add = add i32 %iv, %off
  store i8 0, ptr %gep
  %iv.next = add nsw i32 %iv, 1
  %ec = icmp eq i32 %iv.next, 1000
  br i1 %ec, label %exit, label %loop

exit:
  %lcssa = phi i32 [ %add, %loop ]
  ret i32 %lcssa
}

define void @print_fast_math_flags(i64 %n, ptr noalias %y, ptr noalias %x, ptr %z) {
; CHECK-LABEL: Checking a loop in 'print_fast_math_flags'
; CHECK:      VPlan 'Initial VPlan for VF={4},UF>=1' {
; CHECK-NEXT: Live-in vp<[[VFxUF:%.]]> = VF * UF
; CHECK-NEXT: Live-in vp<[[VTC:%.+]]> = vector-trip-count
; CHECK-NEXT: Live-in ir<%n> = original trip-count
; CHECK-EMPTY:
; CHECK-NEXT: vector.ph:
; CHECK-NEXT: Successor(s): vector loop
; CHECK-EMPTY:
; CHECK-NEXT: <x1> vector loop: {
; CHECK-NEXT: vector.body:
; CHECK-NEXT:   EMIT vp<[[CAN_IV:%.+]]> = CANONICAL-INDUCTION ir<0>, vp<[[CAN_IV_NEXT:%.+]]>
; CHECK-NEXT:   vp<[[STEPS:%.+]]> = SCALAR-STEPS vp<[[CAN_IV]]>, ir<1>
; CHECK-NEXT:   CLONE ir<%gep.y> = getelementptr inbounds ir<%y>, vp<[[STEPS]]>
; CHECK-NEXT:   vp<[[VEC_PTR:%.+]]> = vector-pointer ir<%gep.y>
; CHECK-NEXT:   WIDEN ir<%lv> = load vp<[[VEC_PTR]]>
; CHECK-NEXT:   WIDEN ir<%add> = fadd nnan ir<%lv>, ir<1.000000e+00>
; CHECK-NEXT:   WIDEN ir<%mul> = fmul reassoc nnan ninf nsz arcp contract afn ir<%add>, ir<2.000000e+00>
; CHECK-NEXT:   WIDEN ir<%div> = fdiv reassoc nsz contract ir<%mul>, ir<2.000000e+00>
; CHECK-NEXT:   CLONE ir<%gep.x> = getelementptr inbounds ir<%x>, vp<[[STEPS]]>
; CHECK-NEXT:   vp<[[VEC_PTR:%.+]]> = vector-pointer ir<%gep.x>
; CHECK-NEXT:   WIDEN store vp<[[VEC_PTR]]>, ir<%div>
; CHECK-NEXT:   EMIT vp<[[CAN_IV_NEXT]]> = add nuw vp<[[CAN_IV]]>, vp<[[VFxUF]]>
; CHECK-NEXT:   EMIT branch-on-count vp<[[CAN_IV_NEXT]]>, vp<[[VTC]]>
; CHECK-NEXT: No successors
; CHECK-NEXT: }
; CHECK-NEXT: Successor(s): middle.block
; CHECK-EMPTY:
; CHECK-NEXT: middle.block:
; CHECK-NEXT:   EMIT vp<[[CMP:%.+]]> = icmp eq ir<%n>, vp<[[VTC]]>
; CHECK-NEXT:   EMIT branch-on-cond vp<[[CMP]]>
; CHECK-NEXT: Successor(s): ir-bb<exit>, scalar.ph
; CHECK-EMPTY:
; CHECK-NEXT: ir-bb<exit>
; CHECK-NEXT: No successors
; CHECK-EMPTY:
; CHECK-NEXT: scalar.ph
; CHECK-NEXT: No successors
; CHECK-NEXT: }
;
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %gep.y = getelementptr inbounds float, ptr %y, i64 %iv
  %lv = load float, ptr %gep.y, align 4
  %add = fadd nnan float %lv, 1.0
  %mul = fmul fast float %add, 2.0
  %div = fdiv nsz reassoc contract float %mul, 2.0
  %gep.x = getelementptr inbounds float, ptr %x, i64 %iv
  store float %div, ptr %gep.x, align 4
  %iv.next = add i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, %n
  br i1 %exitcond, label %exit, label %loop

exit:
  ret void
}

define void @print_exact_flags(i64 %n, ptr noalias %x) {
; CHECK-LABEL: Checking a loop in 'print_exact_flags'
; CHECK:      VPlan 'Initial VPlan for VF={4},UF>=1' {
; CHECK-NEXT: Live-in vp<[[VFxUF:%.]]> = VF * UF
; CHECK-NEXT: Live-in vp<[[VTC:%.+]]> = vector-trip-count
; CHECK-NEXT: Live-in ir<%n> = original trip-count
; CHECK-EMPTY:
; CHECK-NEXT: vector.ph:
; CHECK-NEXT: Successor(s): vector loop
; CHECK-EMPTY:
; CHECK-NEXT: <x1> vector loop: {
; CHECK-NEXT: vector.body:
; CHECK-NEXT:   EMIT vp<[[CAN_IV:%.+]]> = CANONICAL-INDUCTION ir<0>, vp<[[CAN_IV_NEXT:%.+]]>
; CHECK-NEXT:   vp<[[STEPS:%.+]]> = SCALAR-STEPS vp<[[CAN_IV]]>, ir<1>
; CHECK-NEXT:   CLONE ir<%gep.x> = getelementptr inbounds ir<%x>, vp<[[STEPS]]>
; CHECK-NEXT:   vp<[[VEC_PTR:%.+]]> = vector-pointer ir<%gep.x>
; CHECK-NEXT:   WIDEN ir<%lv> = load vp<[[VEC_PTR]]>
; CHECK-NEXT:   WIDEN ir<%div.1> = udiv exact ir<%lv>, ir<20>
; CHECK-NEXT:   WIDEN ir<%div.2> = udiv ir<%lv>, ir<60>
; CHECK-NEXT:   WIDEN ir<%add> = add nuw nsw ir<%div.1>, ir<%div.2>
; CHECK-NEXT:   vp<[[VEC_PTR2:%.+]]> = vector-pointer ir<%gep.x>
; CHECK-NEXT:   WIDEN store vp<[[VEC_PTR2]]>, ir<%add>
; CHECK-NEXT:   EMIT vp<[[CAN_IV_NEXT]]> = add nuw vp<[[CAN_IV]]>, vp<[[VFxUF]]>
; CHECK-NEXT:   EMIT branch-on-count vp<[[CAN_IV_NEXT]]>, vp<[[VTC]]>
; CHECK-NEXT: No successors
; CHECK-NEXT: }
; CHECK-NEXT: Successor(s): middle.block
; CHECK-EMPTY:
; CHECK-NEXT: middle.block:
; CHECK-NEXT:   EMIT vp<[[CMP:%.+]]> = icmp eq ir<%n>, vp<[[VTC]]>
; CHECK-NEXT:   EMIT branch-on-cond vp<[[CMP]]>
; CHECK-NEXT: Successor(s): ir-bb<exit>, scalar.ph
; CHECK-EMPTY:
; CHECK-NEXT: ir-bb<exit>
; CHECK-NEXT: No successors
; CHECK-EMPTY:
; CHECK-NEXT: scalar.ph
; CHECK-NEXT: No successors
; CHECK-NEXT: }
;
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %gep.x = getelementptr inbounds i32, ptr %x, i64 %iv
  %lv = load i32, ptr %gep.x, align 4
  %div.1 = udiv exact i32 %lv, 20
  %div.2 = udiv i32 %lv, 60
  %add = add nsw nuw i32 %div.1, %div.2
  store i32 %add, ptr %gep.x, align 4
  %iv.next = add i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, %n
  br i1 %exitcond, label %exit, label %loop

exit:
  ret void
}

define void @print_call_flags(ptr readonly %src, ptr noalias %dest, i64 %n) {
; CHECK-LABEL: Checking a loop in 'print_call_flags'
; CHECK:      VPlan 'Initial VPlan for VF={4},UF>=1' {
; CHECK-NEXT: Live-in vp<[[VFxUF:%.]]> = VF * UF
; CHECK-NEXT: Live-in vp<[[VTC:%.+]]> = vector-trip-count
; CHECK-NEXT: Live-in ir<%n> = original trip-count
; CHECK-EMPTY:
; CHECK-NEXT: vector.ph:
; CHECK-NEXT: Successor(s): vector loop
; CHECK-EMPTY:
; CHECK-NEXT: <x1> vector loop: {
; CHECK-NEXT: vector.body:
; CHECK-NEXT:   EMIT vp<[[CAN_IV:%.+]]> = CANONICAL-INDUCTION ir<0>, vp<[[CAN_IV_NEXT:%.+]]>
; CHECK-NEXT:   vp<[[STEPS:%.+]]> = SCALAR-STEPS vp<[[CAN_IV]]>, ir<1>
; CHECK-NEXT:   CLONE ir<%ld.addr> = getelementptr inbounds ir<%src>, vp<[[STEPS]]>
; CHECK-NEXT:   vp<[[VEC_PTR:%.+]]> = vector-pointer ir<%ld.addr>
; CHECK-NEXT:   WIDEN ir<%ld.value> = load vp<[[VEC_PTR]]>
; CHECK-NEXT:   WIDEN ir<%ifcond> = fcmp oeq ir<%ld.value>, ir<5.000000e+00>
; CHECK-NEXT:  Successor(s): pred.call
; CHECK-EMPTY:
; CHECK-NEXT:  <xVFxUF> pred.call: {
; CHECK-NEXT:    pred.call.entry:
; CHECK-NEXT:      BRANCH-ON-MASK ir<%ifcond>
; CHECK-NEXT:    Successor(s): pred.call.if, pred.call.continue
; CHECK-EMPTY:
; CHECK-NEXT:    pred.call.if:
; CHECK-NEXT:      REPLICATE ir<%foo.ret.1> = call nnan ninf nsz @foo(ir<%ld.value>) (S->V)
; CHECK-NEXT:      REPLICATE ir<%foo.ret.2> = call @foo(ir<%ld.value>) (S->V)
; CHECK-NEXT:    Successor(s): pred.call.continue
; CHECK-EMPTY:
; CHECK-NEXT:    pred.call.continue:
; CHECK-NEXT:      PHI-PREDICATED-INSTRUCTION vp<[[PHI1:%.+]]> = ir<%foo.ret.1>
; CHECK-NEXT:      PHI-PREDICATED-INSTRUCTION vp<[[PHI2:%.+]]> = ir<%foo.ret.2>
; CHECK-NEXT:    No successors
; CHECK-NEXT:  }
; CHECK-NEXT:  Successor(s): if.then.1
; CHECK-EMPTY:
; CHECK-NEXT:  if.then.1:
; CHECK-NEXT:    WIDEN ir<%fadd> = fadd vp<[[PHI1]]>, vp<[[PHI2]]>
; CHECK-NEXT:    BLEND ir<%st.value> = ir<%ld.value> ir<%fadd>/ir<%ifcond>
; CHECK-NEXT:    CLONE ir<%st.addr> = getelementptr inbounds ir<%dest>, vp<[[STEPS]]>
; CHECK-NEXT:    vp<[[VEC_PTR2:%.+]]> = vector-pointer ir<%st.addr>
; CHECK-NEXT:    WIDEN store vp<[[VEC_PTR2]]>, ir<%st.value>
; CHECK-NEXT:   EMIT vp<[[CAN_IV_NEXT]]> = add nuw vp<[[CAN_IV]]>, vp<[[VFxUF]]>
; CHECK-NEXT:   EMIT branch-on-count vp<[[CAN_IV_NEXT]]>, vp<[[VTC]]>
; CHECK-NEXT: No successors
; CHECK-NEXT: }
; CHECK-NEXT: Successor(s): middle.block
; CHECK-EMPTY:
; CHECK-NEXT: middle.block:
; CHECK-NEXT:   EMIT vp<[[CMP:%.+]]> = icmp eq ir<%n>, vp<[[VTC]]>
; CHECK-NEXT:   EMIT branch-on-cond vp<[[CMP]]>
; CHECK-NEXT: Successor(s): ir-bb<end>, scalar.ph
; CHECK-EMPTY:
; CHECK-NEXT: ir-bb<end>
; CHECK-NEXT: No successors
; CHECK-EMPTY:
; CHECK-NEXT: scalar.ph
; CHECK-NEXT: No successors
; CHECK-NEXT: }
;
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.loop ]
  %ld.addr = getelementptr inbounds float, ptr %src, i64 %iv
  %ld.value = load float , ptr %ld.addr, align 8
  %ifcond = fcmp oeq float %ld.value, 5.0
  br i1 %ifcond, label %if.then, label %for.loop

if.then:
  %foo.ret.1 = call nnan nsz ninf float @foo(float %ld.value) #0
  %foo.ret.2 = call float @foo(float %ld.value) #0
  %fadd = fadd float %foo.ret.1, %foo.ret.2
  br label %for.loop

for.loop:
  %st.value = phi float [ %ld.value, %for.body ], [ %fadd, %if.then ]
  %st.addr = getelementptr inbounds float, ptr %dest, i64 %iv
  store float %st.value, ptr %st.addr, align 8
  %iv.next = add nsw nuw i64 %iv, 1
  %loopcond = icmp eq i64 %iv.next, %n
  br i1 %loopcond, label %end, label %for.body

end:
  ret void
}

; FIXME: Preserve disjoint flag on OR recipe.
define void @print_disjoint_flags(i64 %n, ptr noalias %x) {
; CHECK-LABEL: Checking a loop in 'print_disjoint_flags'
; CHECK:      VPlan 'Initial VPlan for VF={4},UF>=1' {
; CHECK-NEXT: Live-in vp<[[VFxUF:%.]]> = VF * UF
; CHECK-NEXT: Live-in vp<[[VTC:%.+]]> = vector-trip-count
; CHECK-NEXT: Live-in ir<%n> = original trip-count
; CHECK-EMPTY:
; CHECK-NEXT: vector.ph:
; CHECK-NEXT: Successor(s): vector loop
; CHECK-EMPTY:
; CHECK-NEXT: <x1> vector loop: {
; CHECK-NEXT: vector.body:
; CHECK-NEXT:   EMIT vp<[[CAN_IV:%.+]]> = CANONICAL-INDUCTION ir<0>, vp<[[CAN_IV_NEXT:%.+]]>
; CHECK-NEXT:   vp<[[STEPS:%.+]]> = SCALAR-STEPS vp<[[CAN_IV]]>, ir<1>
; CHECK-NEXT:   CLONE ir<%gep.x> = getelementptr inbounds ir<%x>, vp<[[STEPS]]>
; CHECK-NEXT:   vp<[[VEC_PTR:%.+]]> = vector-pointer ir<%gep.x>
; CHECK-NEXT:   WIDEN ir<%lv> = load vp<[[VEC_PTR]]>
; CHECK-NEXT:   WIDEN ir<%or.1> = or disjoint ir<%lv>, ir<1>
; CHECK-NEXT:   WIDEN ir<%or.2> = or ir<%lv>, ir<3>
; CHECK-NEXT:   WIDEN ir<%add> = add nuw nsw ir<%or.1>, ir<%or.2>
; CHECK-NEXT:   vp<[[VEC_PTR:%.+]]> = vector-pointer ir<%gep.x>
; CHECK-NEXT:   WIDEN store vp<[[VEC_PTR]]>, ir<%add>
; CHECK-NEXT:   EMIT vp<[[CAN_IV_NEXT]]> = add nuw vp<[[CAN_IV]]>, vp<[[VFxUF]]>
; CHECK-NEXT:   EMIT branch-on-count vp<[[CAN_IV_NEXT]]>, vp<[[VTC]]>
; CHECK-NEXT: No successors
; CHECK-NEXT: }
; CHECK-NEXT: Successor(s): middle.block
; CHECK-EMPTY:
; CHECK-NEXT: middle.block:
; CHECK-NEXT:   EMIT vp<[[CMP:%.+]]> = icmp eq ir<%n>, vp<[[VTC]]>
; CHECK-NEXT:   EMIT branch-on-cond vp<[[CMP]]>
; CHECK-NEXT: Successor(s): ir-bb<exit>, scalar.ph
; CHECK-EMPTY:
; CHECK-NEXT: ir-bb<exit>
; CHECK-NEXT: No successors
; CHECK-EMPTY:
; CHECK-NEXT: scalar.ph
; CHECK-NEXT: No successors
; CHECK-NEXT: }
;
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %gep.x = getelementptr inbounds i32, ptr %x, i64 %iv
  %lv = load i32, ptr %gep.x, align 4
  %or.1 = or disjoint i32 %lv, 1
  %or.2 = or i32 %lv, 3
  %add = add nsw nuw i32 %or.1, %or.2
  store i32 %add, ptr %gep.x, align 4
  %iv.next = add i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, %n
  br i1 %exitcond, label %exit, label %loop

exit:
  ret void
}

define void @zext_nneg(ptr noalias %p, ptr noalias %p1) {
; CHECK-LABEL: LV: Checking a loop in 'zext_nneg'
; CHECK:       VPlan 'Initial VPlan for VF={4},UF>=1' {
; CHECK-NEXT:  Live-in vp<[[VFxUF:%.]]> = VF * UF
; CHECK-NEXT:  Live-in vp<[[VTC:%.+]]> = vector-trip-count
; CHECK-NEXT:  Live-in ir<1000> = original trip-count
; CHECK-EMPTY:
; CHECK-NEXT:  vector.ph:
; CHECK-NEXT:  Successor(s): vector loop
; CHECK-EMPTY:
; CHECK-NEXT:  <x1> vector loop: {
; CHECK-NEXT:  vector.body:
; CHECK-NEXT:    EMIT vp<[[CAN_IV:%.+]]> = CANONICAL-INDUCTION ir<0>, vp<[[CAN_IV_NEXT:%.+]]>
; CHECK-NEXT:    vp<[[STEPS:%.+]]> = SCALAR-STEPS vp<[[CAN_IV]]>, ir<1>
; CHECK-NEXT:    CLONE ir<%idx> = getelementptr ir<%p>, vp<[[STEPS]]>
; CHECK-NEXT:    vp<[[VEC_PTR:%.+]]> = vector-pointer ir<%idx>
; CHECK-NEXT:    WIDEN ir<%l> = load vp<[[VEC_PTR]]>
; CHECK-NEXT:    WIDEN-CAST ir<%zext> = zext nneg ir<%l>
; CHECK-NEXT:    REPLICATE store ir<%zext>, ir<%p1>
; CHECK-NEXT:    EMIT vp<[[CAN_IV_NEXT]]> = add nuw vp<[[CAN_IV]]>
; CHECK-NEXT:    EMIT branch-on-count vp<[[CAN_IV_NEXT]]>, vp<[[VTC]]>
; CHECK-NEXT:  No successors
; CHECK-NEXT: }
;
entry:
  br label %body

body:
  %iv = phi i64 [ %next, %body ], [ 0, %entry ]
  %idx = getelementptr i32, ptr %p, i64 %iv
  %l = load i32, ptr %idx, align 8
  %zext = zext nneg i32 %l to i64
  store i64 %zext, ptr %p1, align 8
  %next = add i64 %iv, 1
  %cmp = icmp eq i64 %next, 1000
  br i1 %cmp, label %exit, label %body

exit:
  ret void
}

define i16 @print_first_order_recurrence_and_result(ptr %ptr) {
; CHECK-LABEL: 'print_first_order_recurrence_and_result'
; CHECK:      VPlan 'Initial VPlan for VF={4},UF>=1' {
; CHECK-NEXT: Live-in vp<[[VFxUF:%.+]]> = VF * UF
; CHECK-NEXT: Live-in vp<[[VTC:%.+]]> = vector-trip-count
; CHECK-NEXT: Live-in ir<1000> = original trip-count
; CHECK-EMPTY:
; CHECK-NEXT: vector.ph:
; CHECK-NEXT: Successor(s): vector loop
; CHECK-EMPTY:
; CHECK-NEXT: <x1> vector loop: {
; CHECK-NEXT:   vector.body:
; CHECK-NEXT:     EMIT vp<[[CAN_IV:%.+]]> = CANONICAL-INDUCTION
; CHECK-NEXT:     FIRST-ORDER-RECURRENCE-PHI ir<%for.1> = phi ir<22>, ir<%for.1.next>
; CHECK-NEXT:     vp<[[STEPS:%.+]]>    = SCALAR-STEPS vp<[[CAN_IV]]>, ir<1>
; CHECK-NEXT:     CLONE ir<%gep.ptr> = getelementptr inbounds ir<%ptr>, vp<[[STEPS]]>
; CHECK-NEXT:     vp<[[VEC_PTR:%.+]]> = vector-pointer ir<%gep.ptr>
; CHECK-NEXT:     WIDEN ir<%for.1.next> = load vp<[[VEC_PTR]]>
; CHECK-NEXT:     EMIT vp<[[FOR1_SPLICE:%.+]]> = first-order splice ir<%for.1>, ir<%for.1.next>
; CHECK-NEXT:     WIDEN ir<%add> = add vp<[[FOR1_SPLICE]]>, ir<1>
; CHECK-NEXT:     vp<[[VEC_PTR2:%.+]]> = vector-pointer ir<%gep.ptr>
; CHECK-NEXT:     WIDEN store vp<[[VEC_PTR2]]>, ir<%add>
; CHECK-NEXT:     EMIT vp<[[CAN_IV_NEXT:%.+]]> = add nuw vp<[[CAN_IV]]>, vp<[[VFxUF]]>
; CHECK-NEXT:     EMIT branch-on-count vp<[[CAN_IV_NEXT]]>, vp<[[VTC]]>
; CHECK-NEXT:   No successors
; CHECK-NEXT: }
; CHECK-NEXT: Successor(s): middle.block
; CHECK-EMPTY:
; CHECK-NEXT: middle.block:
; CHECK-NEXT:   EMIT vp<[[RESUME_1:%.+]]> = extract-from-end ir<%for.1.next>, ir<1>
; CHECK-NEXT:   EMIT vp<[[FOR_RESULT:%.+]]> = extract-from-end ir<%for.1.next>, ir<2>
; CHECK-NEXT:   EMIT vp<[[CMP:%.+]]> = icmp eq ir<1000>, vp<[[VTC]]>
; CHECK-NEXT:   EMIT branch-on-cond vp<[[CMP]]>
; CHECK-NEXT: Successor(s): ir-bb<exit>, scalar.ph
; CHECK-EMPTY:
; CHECK-NEXT: ir-bb<exit>
; CHECK-NEXT: No successors
; CHECK-EMPTY:
; CHECK-NEXT: scalar.ph
; CHECK-NEXT:   EMIT vp<[[RESUME_P:%.*]]> = resume-phi vp<[[RESUME_1]]>, ir<22>
; CHECK-NEXT: No successors
; CHECK-EMPTY:
; CHECK-NEXT:   Live-out i16 %for.1 = vp<[[RESUME_P]]>
; CHECK-NEXT:   Live-out i16 %for.1.lcssa = vp<[[FOR_RESULT]]>
; CHECK-NEXT: }
;
entry:
  br label %loop

loop:
  %for.1 = phi i16 [ 22, %entry ], [ %for.1.next, %loop ]
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %iv.next = add nuw nsw i64 %iv, 1
  %gep.ptr = getelementptr inbounds i16, ptr %ptr, i64 %iv
  %for.1.next = load i16, ptr %gep.ptr, align 2
  %add = add i16 %for.1, 1
  store i16 %add, ptr %gep.ptr
  %exitcond.not = icmp eq i64 %iv.next, 1000
  br i1 %exitcond.not, label %exit, label %loop

exit:
  ret i16 %for.1
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}

declare float @foo(float) #0
declare <2 x float> @vector_foo(<2 x float>, <2 x i1>)

; We need a vector variant in order to allow for vectorization at present, but
; we want to test scalarization of conditional calls. If we provide a variant
; with a different number of lanes than the VF we force via
; "-force-vector-width=4", then it should pass the legality checks but
; scalarize. TODO: Remove the requirement to have a variant.
attributes #0 = { readonly nounwind "vector-function-abi-variant"="_ZGV_LLVM_M2v_foo(vector_foo)" }

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: NoDebug, enums: !2)
!1 = !DIFile(filename: "/tmp/s.c", directory: "/tmp")
!2 = !{}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 7, !"PIC Level", i32 2}
!5 = distinct !DISubprogram(name: "f", scope: !1, file: !1, line: 4, type: !6, scopeLine: 4, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!6 = !DISubroutineType(types: !2)
!7 = !DILocation(line: 5, column: 3, scope: !5)
!8 = !DILocation(line: 5, column: 21, scope: !5)
