; REQUIRES: asserts

; RUN: opt -passes=loop-vectorize -debug-only=loop-vectorize -force-vector-interleave=1 -force-vector-width=4 -prefer-inloop-reductions -disable-output %s 2>&1 | FileCheck %s

; Tests for printing VPlans with reductions.

define float @print_reduction(i64 %n, ptr noalias %y) {
; CHECK-LABEL: Checking a loop in 'print_reduction'
; CHECK:      VPlan 'Initial VPlan for VF={4},UF>=1' {
; CHECK-NEXT: Live-in vp<[[VF:%.]]> = VF
; CHECK-NEXT: Live-in vp<[[VFxUF:%.]]> = VF * UF
; CHECK-NEXT: Live-in vp<[[VTC:%.+]]> = vector-trip-count
; CHECK-NEXT: Live-in ir<%n> = original trip-count
; CHECK-EMPTY:
; CHECK-NEXT: ir-bb<entry>:
; CHECK-NEXT: Successor(s): vector.ph
; CHECK-EMPTY:
; CHECK-NEXT: vector.ph:
; CHECK-NEXT: Successor(s): vector loop
; CHECK-EMPTY:
; CHECK-NEXT: <x1> vector loop: {
; CHECK-NEXT: vector.body:
; CHECK-NEXT:   EMIT vp<[[CAN_IV:%.+]]> = CANONICAL-INDUCTION ir<0>, vp<[[CAN_IV_NEXT:%.+]]>
; CHECK-NEXT:   WIDEN-REDUCTION-PHI ir<%red> = phi ir<0.000000e+00>, ir<%red.next>
; CHECK-NEXT:   vp<[[STEPS:%.+]]> = SCALAR-STEPS vp<[[CAN_IV]]>, ir<1>, vp<[[VF]]>
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
; CHECK-NEXT:   EMIT vp<[[RED_EX:%.+]]> = extract-last-element vp<[[RED_RES]]>
; CHECK-NEXT:   EMIT vp<[[CMP:%.+]]> = icmp eq ir<%n>, vp<[[VTC]]>
; CHECK-NEXT:   EMIT branch-on-cond vp<[[CMP]]>
; CHECK-NEXT: Successor(s): ir-bb<exit>, scalar.ph
; CHECK-EMPTY:
; CHECK-NEXT: scalar.ph
; CHECK-NEXT:   EMIT vp<[[RESUME_IV:%.+]]> = resume-phi vp<[[VTC]]>, ir<0>
; CHECK-NEXT:   EMIT vp<[[RED_RESUME:%.+]]> = resume-phi vp<[[RED_RES]]>, ir<0.000000e+00>
; CHECK-NEXT:  Successor(s): ir-bb<loop>
; CHECK-EMPTY:
; CHECK-NEXT:  ir-bb<loop>:
; CHECK-NEXT:    IR   %iv = phi i64 [ %iv.next, %loop ], [ 0, %entry ] (extra operand: vp<[[RESUME_IV]]> from scalar.ph)
; CHECK:         IR   %exitcond = icmp eq i64 %iv.next, %n
; CHECK-NEXT: No successors
; CHECK-EMPTY:
; CHECK-NEXT: ir-bb<exit>
; CHECK-NEXT:  IR %red.next.lcssa = phi float [ %red.next, %loop ] (extra operand: vp<[[RED_EX]]> from middle.block)
; CHECK-NEXT: No successors
; CHECK-NEXT: }
;
entry:
  br label %loop

loop:                                         ; preds = %entry, %loop
  %iv = phi i64 [ %iv.next, %loop ], [ 0, %entry ]
  %red = phi float [ %red.next, %loop ], [ 0.0, %entry ]
  %arrayidx = getelementptr inbounds float, ptr %y, i64 %iv
  %lv = load float, ptr %arrayidx, align 4
  %red.next = fadd fast float %lv, %red
  %iv.next = add i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, %n
  br i1 %exitcond, label %exit, label %loop

exit:                                          ; preds = %loop, %entry
  ret float %red.next
}

define void @print_reduction_with_invariant_store(i64 %n, ptr noalias %y, ptr noalias %dst) {
; CHECK-LABEL: Checking a loop in 'print_reduction_with_invariant_store'
; CHECK:      VPlan 'Initial VPlan for VF={4},UF>=1' {
; CHECK-NEXT: Live-in vp<[[VF:%.]]> = VF
; CHECK-NEXT: Live-in vp<[[VFxUF:%.]]> = VF * UF
; CHECK-NEXT: Live-in vp<[[VTC:%.+]]> = vector-trip-count
; CHECK-NEXT: Live-in ir<%n> = original trip-count
; CHECK-EMPTY:
; CHECK-NEXT: ir-bb<entry>:
; CHECK-NEXT: Successor(s): vector.ph
; CHECK-EMPTY:
; CHECK-NEXT: vector.ph:
; CHECK-NEXT: Successor(s): vector loop
; CHECK-EMPTY:
; CHECK-NEXT: <x1> vector loop: {
; CHECK-NEXT: vector.body:
; CHECK-NEXT:   EMIT vp<[[CAN_IV:%.+]]> = CANONICAL-INDUCTION ir<0>, vp<[[CAN_IV_NEXT:%.+]]>
; CHECK-NEXT:   WIDEN-REDUCTION-PHI ir<%red> = phi ir<0.000000e+00>, ir<%red.next>
; CHECK-NEXT:   vp<[[IV:%.+]]> = SCALAR-STEPS vp<[[CAN_IV]]>, ir<1>, vp<[[VF]]>
; CHECK-NEXT:   CLONE ir<%arrayidx> = getelementptr inbounds ir<%y>, vp<[[IV]]>
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
; CHECK-NEXT:   EMIT vp<[[RED_RES:.+]]> = compute-reduction-result ir<%red>, ir<%red.next>
; CHECK-NEXT:   CLONE store vp<[[RED_RES]]>, ir<%dst>
; CHECK-NEXT:   EMIT vp<[[CMP:%.+]]> = icmp eq ir<%n>, vp<[[VTC]]>
; CHECK-NEXT:   EMIT branch-on-cond vp<[[CMP]]>
; CHECK-NEXT: Successor(s): ir-bb<exit>, scalar.ph
; CHECK-EMPTY:
; CHECK-NEXT: scalar.ph
; CHECK-NEXT:   EMIT vp<[[RESUME_IV:%.+]]> = resume-phi vp<[[VTC]]>, ir<0>
; CHECK-NEXT:   EMIT vp<[[RED_RESUME:%.+]]> = resume-phi vp<[[RED_RES]]>, ir<0.000000e+00>
; CHECK-NEXT:  Successor(s): ir-bb<loop>
; CHECK-EMPTY:
; CHECK-NEXT:  ir-bb<loop>:
; CHECK-NEXT:    IR   %iv = phi i64 [ %iv.next, %loop ], [ 0, %entry ] (extra operand: vp<[[RESUME_IV]]> from scalar.ph)
; CHECK-NEXT:    IR   %red = phi float [ %red.next, %loop ], [ 0.000000e+00, %entry ]
; CHECK:         IR   %exitcond = icmp eq i64 %iv.next, %n
; CHECK-NEXT: No successors
; CHECK-EMPTY:
; CHECK-NEXT: ir-bb<exit>
; CHECK-NEXT: No successors
; CHECK-NEXT: }
;
entry:
  br label %loop

loop:                                         ; preds = %entry, %loop
  %iv = phi i64 [ %iv.next, %loop ], [ 0, %entry ]
  %red = phi float [ %red.next, %loop ], [ 0.0, %entry ]
  %arrayidx = getelementptr inbounds float, ptr %y, i64 %iv
  %lv = load float, ptr %arrayidx, align 4
  %red.next = fadd fast float %lv, %red
  store float %red.next, ptr %dst, align 4
  %iv.next = add i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, %n
  br i1 %exitcond, label %exit, label %loop

exit:                                          ; preds = %loop, %entry
  ret void
}

define float @print_fmuladd_strict(ptr %a, ptr %b, i64 %n) {
; CHECK-LABEL: Checking a loop in 'print_fmuladd_strict'
; CHECK:      VPlan 'Initial VPlan for VF={4},UF>=1' {
; CHECK-NEXT: Live-in vp<[[VF:%.]]> = VF
; CHECK-NEXT: Live-in vp<[[VFxUF:%.]]> = VF * UF
; CHECK-NEXT: Live-in vp<[[VTC:%.+]]> = vector-trip-count
; CHECK-NEXT: Live-in ir<%n> = original trip-count
; CHECK-EMPTY:
; CHECK-NEXT: ir-bb<entry>:
; CHECK-NEXT: Successor(s): vector.ph
; CHECK-EMPTY:
; CHECK-NEXT: vector.ph:
; CHECK-NEXT: Successor(s): vector loop
; CHECK-EMPTY:
; CHECK-NEXT: <x1> vector loop: {
; CHECK-NEXT: vector.body:
; CHECK-NEXT:   EMIT vp<[[CAN_IV:%.+]]> = CANONICAL-INDUCTION ir<0>, vp<[[CAN_IV_NEXT:%.+]]>
; CHECK-NEXT:   WIDEN-REDUCTION-PHI ir<%sum.07> = phi ir<0.000000e+00>, ir<%muladd>
; CHECK-NEXT:   vp<[[STEPS:%.+]]> = SCALAR-STEPS vp<[[CAN_IV]]>, ir<1>, vp<[[VF]]>
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
; CHECK-NEXT:   EMIT vp<[[RED_EX:%.+]]> = extract-last-element vp<[[RED_RES]]>
; CHECK-NEXT:   EMIT vp<[[CMP:%.+]]> = icmp eq ir<%n>, vp<[[VTC]]>
; CHECK-NEXT:   EMIT branch-on-cond vp<[[CMP]]>
; CHECK-NEXT: Successor(s): ir-bb<exit>, scalar.ph
; CHECK-EMPTY:
; CHECK-NEXT: scalar.ph
; CHECK-NEXT:   EMIT vp<[[RESUME_IV:%.+]]> = resume-phi vp<[[VTC]]>, ir<0>
; CHECK-NEXT:   EMIT vp<[[RED_RESUME:%.+]]> = resume-phi vp<[[RED_RES]]>, ir<0.000000e+00>
; CHECK-NEXT:  Successor(s): ir-bb<loop>
; CHECK-EMPTY:
; CHECK-NEXT:  ir-bb<loop>:
; CHECK-NEXT:    IR   %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ] (extra operand: vp<[[RESUME_IV]]> from scalar.ph)
; CHECK-NEXT:    IR   %sum.07 = phi float [ 0.000000e+00, %entry ], [ %muladd, %loop ] (extra operand: vp<[[RED_RESUME]]> from scalar.ph)
; CHECK:         IR   %exitcond.not = icmp eq i64 %iv.next, %n
; CHECK-NEXT: No successors
; CHECK-EMPTY:
; CHECK-NEXT: ir-bb<exit>
; CHECK-NEXT:   IR %muladd.lcssa = phi float [ %muladd, %loop ] (extra operand: vp<[[RED_EX]]> from middle.block)
; CHECK-NEXT: No successors
; CHECK-NEXT:}

entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %sum.07 = phi float [ 0.000000e+00, %entry ], [ %muladd, %loop ]
  %arrayidx = getelementptr inbounds float, ptr %a, i64 %iv
  %l.a = load float, ptr %arrayidx, align 4
  %arrayidx2 = getelementptr inbounds float, ptr %b, i64 %iv
  %l.b = load float, ptr %arrayidx2, align 4
  %muladd = tail call nnan ninf nsz float @llvm.fmuladd.f32(float %l.a, float %l.b, float %sum.07)
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond.not = icmp eq i64 %iv.next, %n
  br i1 %exitcond.not, label %exit, label %loop

exit:
  ret float %muladd
}

define i64 @find_last_iv(ptr %a, i64 %n, i64 %start) {
; CHECK-LABEL: Checking a loop in 'find_last_iv'
; CHECK:       VPlan 'Initial VPlan for VF={4},UF>=1' {
; CHECK:        <x1> vector loop: {
; CHECK-NEXT:     vector.body:
; CHECK-NEXT:     EMIT vp<[[CAN_IV:%.+]]> = CANONICAL-INDUCTION ir<0>, vp<%index.next>
; CHECK-NEXT:     ir<%iv> = WIDEN-INDUCTION ir<0>, ir<1>, vp<{{.+}}>
; CHECK-NEXT:     WIDEN-REDUCTION-PHI ir<%rdx> = phi ir<-9223372036854775808>, ir<%cond>
; CHECK-NEXT:     vp<[[SCALAR_STEPS:%.+]]> = SCALAR-STEPS vp<[[CAN_IV]]>, ir<1>
; CHECK-NEXT:     CLONE ir<%gep.a> = getelementptr inbounds ir<%a>, vp<[[SCALAR_STEPS]]>
; CHECK-NEXT:     vp<[[VEC_PTR:%.+]]> = vector-pointer ir<%gep.a>
; CHECK-NEXT:     WIDEN ir<%l.a> = load vp<[[VEC_PTR]]>
; CHECK-NEXT:     WIDEN ir<%cmp2> = icmp eq ir<%l.a>, ir<%start>
; CHECK-NEXT:     WIDEN-SELECT ir<%cond> = select  ir<%cmp2>, ir<%iv>, ir<%rdx>
; CHECK-NEXT:     EMIT vp<%index.next> = add nuw vp<[[CAN_IV]]>, vp<{{.+}}>
; CHECK-NEXT:     EMIT branch-on-count vp<%index.next>, vp<{{.+}}>
; CHECK-NEXT:   No successors
; CHECK-NEXT: }
; CHECK-NEXT: Successor(s): middle.block
; CHECK-EMPTY:
; CHECK-NEXT: middle.block:
; CHECK-NEXT:   EMIT vp<[[RDX_RES:%.+]]> = compute-find-last-iv-result ir<%rdx>, ir<%start>, ir<%cond>
; CHECK-NEXT:   EMIT vp<[[EXT:%.+]]> = extract-last-element vp<[[RDX_RES]]>
; CHECK-NEXT:   EMIT vp<%cmp.n> = icmp eq ir<%n>, vp<{{.+}}>
; CHECK-NEXT:   EMIT branch-on-cond vp<%cmp.n>
; CHECK-NEXT: Successor(s): ir-bb<exit>, scalar.ph
; CHECK-EMPTY:
; CHECK-NEXT: scalar.ph:
; CHECK-NEXT:   EMIT vp<%bc.resume.val> = resume-phi vp<{{.+}}>, ir<0>
; CHECK-NEXT:   EMIT vp<%bc.merge.rdx> = resume-phi vp<[[RDX_RES]]>, ir<%start>
;
; CHECK:      ir-bb<exit>:
; CHECK-NEXT:   IR   %cond.lcssa = phi i64 [ %cond, %loop ] (extra operand: vp<[[EXT]]> from middle.block)
; CHECK-NEXT: No successors
;
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %inc, %loop ]
  %rdx = phi i64 [ %start, %entry ], [ %cond, %loop ]
  %gep.a = getelementptr inbounds i64, ptr %a, i64 %iv
  %l.a = load i64, ptr %gep.a, align 8
  %cmp2 = icmp eq i64 %l.a, %start
  %cond = select i1 %cmp2, i64 %iv, i64 %rdx
  %inc = add nuw nsw i64 %iv, 1
  %exitcond.not = icmp eq i64 %inc, %n
  br i1 %exitcond.not, label %exit, label %loop

exit:
  ret i64 %cond
}
