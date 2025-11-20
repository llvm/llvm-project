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
; CHECK-NEXT: Successor(s): scalar.ph, vector.ph
; CHECK-EMPTY:
; CHECK-NEXT: vector.ph:
; CHECK-NEXT:   EMIT vp<[[RDX_START:%.+]]> = reduction-start-vector fast ir<0.000000e+00>, ir<0.000000e+00>, ir<1>
; CHECK-NEXT: Successor(s): vector loop
; CHECK-EMPTY:
; CHECK-NEXT: <x1> vector loop: {
; CHECK-NEXT: vector.body:
; CHECK-NEXT:   EMIT vp<[[CAN_IV:%.+]]> = CANONICAL-INDUCTION ir<0>, vp<[[CAN_IV_NEXT:%.+]]>
; CHECK-NEXT:   WIDEN-REDUCTION-PHI ir<%red> = phi vp<[[RDX_START]]>, ir<%red.next>
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
; CHECK-NEXT:   EMIT vp<[[RED_RES:%.+]]> = compute-reduction-result fast ir<%red>, ir<%red.next>
; CHECK-NEXT:   EMIT vp<[[CMP:%.+]]> = icmp eq ir<%n>, vp<[[VTC]]>
; CHECK-NEXT:   EMIT branch-on-cond vp<[[CMP]]>
; CHECK-NEXT: Successor(s): ir-bb<exit>, scalar.ph
; CHECK-EMPTY:
; CHECK-NEXT: ir-bb<exit>
; CHECK-NEXT:  IR %red.next.lcssa = phi float [ %red.next, %loop ] (extra operand: vp<[[RED_RES]]> from middle.block)
; CHECK-NEXT: No successors
; CHECK-EMPTY:
; CHECK-NEXT: scalar.ph
; CHECK-NEXT:   EMIT-SCALAR vp<[[RESUME_IV:%.+]]> = phi [ vp<[[VTC]]>, middle.block ], [ ir<0>, ir-bb<entry> ]
; CHECK-NEXT:   EMIT-SCALAR vp<[[RED_RESUME:%.+]]> = phi [ vp<[[RED_RES]]>, middle.block ], [ ir<0.000000e+00>, ir-bb<entry> ]
; CHECK-NEXT:  Successor(s): ir-bb<loop>
; CHECK-EMPTY:
; CHECK-NEXT:  ir-bb<loop>:
; CHECK-NEXT:    IR   %iv = phi i64 [ %iv.next, %loop ], [ 0, %entry ] (extra operand: vp<[[RESUME_IV]]> from scalar.ph)
; CHECK:         IR   %exitcond = icmp eq i64 %iv.next, %n
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
; CHECK-NEXT: Successor(s): scalar.ph, vector.ph
; CHECK-EMPTY:
; CHECK-NEXT: vector.ph:
; CHECK-NEXT:   EMIT vp<[[RDX_START:%.+]]> = reduction-start-vector fast ir<0.000000e+00>, ir<0.000000e+00>, ir<1>
; CHECK-NEXT: Successor(s): vector loop
; CHECK-EMPTY:
; CHECK-NEXT: <x1> vector loop: {
; CHECK-NEXT: vector.body:
; CHECK-NEXT:   EMIT vp<[[CAN_IV:%.+]]> = CANONICAL-INDUCTION ir<0>, vp<[[CAN_IV_NEXT:%.+]]>
; CHECK-NEXT:   WIDEN-REDUCTION-PHI ir<%red> = phi vp<[[RDX_START]]>, ir<%red.next>
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
; CHECK-NEXT:   EMIT vp<[[RED_RES:.+]]> = compute-reduction-result fast ir<%red>, ir<%red.next>
; CHECK-NEXT:   CLONE store vp<[[RED_RES]]>, ir<%dst>
; CHECK-NEXT:   EMIT vp<[[CMP:%.+]]> = icmp eq ir<%n>, vp<[[VTC]]>
; CHECK-NEXT:   EMIT branch-on-cond vp<[[CMP]]>
; CHECK-NEXT: Successor(s): ir-bb<exit>, scalar.ph
; CHECK-EMPTY:
; CHECK-NEXT: ir-bb<exit>
; CHECK-NEXT: No successors
; CHECK-EMPTY:
; CHECK-NEXT: scalar.ph
; CHECK-NEXT:   EMIT-SCALAR vp<[[RESUME_IV:%.+]]> = phi [ vp<[[VTC]]>, middle.block ], [ ir<0>, ir-bb<entry> ]
; CHECK-NEXT:   EMIT-SCALAR vp<[[RED_RESUME:%.+]]> = phi [ vp<[[RED_RES]]>, middle.block ], [ ir<0.000000e+00>, ir-bb<entry> ]
; CHECK-NEXT:  Successor(s): ir-bb<loop>
; CHECK-EMPTY:
; CHECK-NEXT:  ir-bb<loop>:
; CHECK-NEXT:    IR   %iv = phi i64 [ %iv.next, %loop ], [ 0, %entry ] (extra operand: vp<[[RESUME_IV]]> from scalar.ph)
; CHECK-NEXT:    IR   %red = phi float [ %red.next, %loop ], [ 0.000000e+00, %entry ]
; CHECK:         IR   %exitcond = icmp eq i64 %iv.next, %n
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
; CHECK-NEXT: Successor(s): scalar.ph, vector.ph
; CHECK-EMPTY:
; CHECK-NEXT: vector.ph:
; CHECK-NEXT:   EMIT vp<[[RDX_START:%.+]]> = reduction-start-vector nnan ninf nsz ir<0.000000e+00>, ir<0.000000e+00>, ir<1>
; CHECK-NEXT: Successor(s): vector loop
; CHECK-EMPTY:
; CHECK-NEXT: <x1> vector loop: {
; CHECK-NEXT: vector.body:
; CHECK-NEXT:   EMIT vp<[[CAN_IV:%.+]]> = CANONICAL-INDUCTION ir<0>, vp<[[CAN_IV_NEXT:%.+]]>
; CHECK-NEXT:   WIDEN-REDUCTION-PHI ir<%sum.07> = phi vp<[[RDX_START]]>, ir<%muladd>
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
; CHECK-NEXT:   EMIT vp<[[RED_RES:%.+]]> = compute-reduction-result nnan ninf nsz ir<%sum.07>, ir<[[MULADD]]>
; CHECK-NEXT:   EMIT vp<[[CMP:%.+]]> = icmp eq ir<%n>, vp<[[VTC]]>
; CHECK-NEXT:   EMIT branch-on-cond vp<[[CMP]]>
; CHECK-NEXT: Successor(s): ir-bb<exit>, scalar.ph
; CHECK-EMPTY:
; CHECK-NEXT: ir-bb<exit>
; CHECK-NEXT:   IR %muladd.lcssa = phi float [ %muladd, %loop ] (extra operand: vp<[[RED_RES]]> from middle.block)
; CHECK-NEXT: No successors
; CHECK-EMPTY:
; CHECK-NEXT: scalar.ph
; CHECK-NEXT:   EMIT-SCALAR vp<[[RESUME_IV:%.+]]> = phi [ vp<[[VTC]]>, middle.block ], [ ir<0>, ir-bb<entry> ]
; CHECK-NEXT:   EMIT-SCALAR vp<[[RED_RESUME:%.+]]> = phi [ vp<[[RED_RES]]>, middle.block ], [ ir<0.000000e+00>, ir-bb<entry> ]
; CHECK-NEXT:  Successor(s): ir-bb<loop>
; CHECK-EMPTY:
; CHECK-NEXT:  ir-bb<loop>:
; CHECK-NEXT:    IR   %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ] (extra operand: vp<[[RESUME_IV]]> from scalar.ph)
; CHECK-NEXT:    IR   %sum.07 = phi float [ 0.000000e+00, %entry ], [ %muladd, %loop ] (extra operand: vp<[[RED_RESUME]]> from scalar.ph)
; CHECK:         IR   %exitcond.not = icmp eq i64 %iv.next, %n
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
; CHECK-NEXT:     ir<%iv> = WIDEN-INDUCTION nuw nsw ir<0>, ir<1>, vp<{{.+}}>
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
; CHECK-NEXT:   EMIT vp<[[RDX_RES:%.+]]> = compute-find-iv-result ir<%rdx>, ir<%start>, ir<-9223372036854775808>, ir<%cond>
; CHECK-NEXT:   EMIT vp<%cmp.n> = icmp eq ir<%n>, vp<{{.+}}>
; CHECK-NEXT:   EMIT branch-on-cond vp<%cmp.n>
; CHECK-NEXT: Successor(s): ir-bb<exit>, scalar.ph
; CHECK-EMPTY:
; CHECK:      ir-bb<exit>:
; CHECK-NEXT:   IR   %cond.lcssa = phi i64 [ %cond, %loop ] (extra operand: vp<[[RDX_RES]]> from middle.block)
; CHECK-NEXT: No successors
; CHECK-EMPTY:
; CHECK-NEXT: scalar.ph:
; CHECK-NEXT:   EMIT-SCALAR vp<%bc.resume.val> = phi [ vp<{{.+}}>, middle.block ], [ ir<0>, ir-bb<entry> ]
; CHECK-NEXT:   EMIT-SCALAR vp<%bc.merge.rdx> = phi [ vp<[[RDX_RES]]>, middle.block ], [ ir<%start>, ir-bb<entry> ]
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

define i64 @print_extended_reduction(ptr nocapture readonly %x, ptr nocapture readonly %y, i32 %n) {
; CHECK-LABEL: 'print_extended_reduction'
; CHECK:      VPlan 'Initial VPlan for VF={4},UF>=1' {
; CHECK-NEXT: Live-in vp<[[VF:%.+]]> = VF
; CHECK-NEXT: Live-in vp<[[VFxUF:%.+]]> = VF * UF
; CHECK-NEXT: Live-in vp<[[VTC:%.+]]> = vector-trip-count
; CHECK-NEXT: Live-in ir<%n> = original trip-count
; CHECK-EMPTY:
; CHECK:      vector.ph:
; CHECK-NEXT:   EMIT vp<[[RDX_START:%.+]]> = reduction-start-vector ir<0>, ir<0>, ir<1>
; CHECK-NEXT: Successor(s): vector loop
; CHECK-EMPTY:
; CHECK-NEXT: <x1> vector loop: {
; CHECK-NEXT:   vector.body:
; CHECK-NEXT:     EMIT vp<[[IV:%.+]]> = CANONICAL-INDUCTION ir<0>, vp<[[IV_NEXT:%.+]]>
; CHECK-NEXT:     WIDEN-REDUCTION-PHI ir<[[RDX:%.+]]> = phi vp<[[RDX_START]]>, vp<[[RDX_NEXT:%.+]]>
; CHECK-NEXT:     vp<[[STEPS:%.+]]> = SCALAR-STEPS vp<[[IV]]>, ir<1>
; CHECK-NEXT:     CLONE ir<%arrayidx> = getelementptr inbounds ir<%x>, vp<[[STEPS]]>
; CHECK-NEXT:     vp<[[ADDR:%.+]]> = vector-pointer ir<%arrayidx>
; CHECK-NEXT:     WIDEN ir<[[LOAD:%.+]]> = load vp<[[ADDR]]>
; CHECK-NEXT:     EXPRESSION vp<[[RDX_NEXT]]> = ir<[[RDX]]> + reduce.add (ir<[[LOAD]]> zext to i64)
; CHECK-NEXT:     EMIT vp<[[IV_NEXT]]> = add nuw vp<[[IV]]>, vp<[[VFxUF]]>
; CHECK-NEXT:     EMIT branch-on-count vp<[[IV_NEXT]]>, vp<[[VTC]]>
; CHECK-NEXT:   No successors
; CHECK-NEXT: }
;
entry:
  br label %loop

loop:
  %iv = phi i32 [ %iv.next, %loop ], [ 0, %entry ]
  %rdx = phi i64 [ %rdx.next, %loop ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds i32, ptr %x, i32 %iv
  %load0 = load i32, ptr %arrayidx, align 4
  %conv0 = zext i32 %load0 to i64
  %rdx.next = add nsw i64 %rdx, %conv0
  %iv.next = add nuw nsw i32 %iv, 1
  %exitcond = icmp eq i32 %iv.next, %n
  br i1 %exitcond, label %exit, label %loop

exit:
  %r.0.lcssa = phi i64 [ %rdx.next, %loop ]
  ret i64 %r.0.lcssa
}

define i64 @print_mulacc(ptr nocapture readonly %x, ptr nocapture readonly %y, i32 %n) {
; CHECK-LABEL: 'print_mulacc'
; CHECK:      VPlan 'Initial VPlan for VF={4},UF>=1' {
; CHECK-NEXT: Live-in vp<[[VF:%.+]]> = VF
; CHECK-NEXT: Live-in vp<[[VFxUF:%.+]]> = VF * UF
; CHECK-NEXT: Live-in vp<[[VTC:%.+]]> = vector-trip-count
; CHECK-NEXT: Live-in ir<%n> = original trip-count
; CHECK-EMPTY:
; CHECK:      vector.ph:
; CHECK-NEXT:   EMIT vp<[[RDX_START:%.+]]> = reduction-start-vector ir<0>, ir<0>, ir<1>
; CHECK-NEXT: Successor(s): vector loop
; CHECK-EMPTY:
; CHECK-NEXT: <x1> vector loop: {
; CHECK-NEXT:   vector.body:
; CHECK-NEXT:     EMIT vp<[[IV:%.+]]> = CANONICAL-INDUCTION ir<0>, vp<[[IV_NEXT:%.+]]>
; CHECK-NEXT:     WIDEN-REDUCTION-PHI ir<[[RDX:%.+]]> = phi vp<[[RDX_START]]>, vp<[[RDX_NEXT:%.+]]>
; CHECK-NEXT:     vp<[[STEPS:%.+]]> = SCALAR-STEPS vp<[[IV]]>, ir<1>
; CHECK-NEXT:     CLONE ir<[[ARRAYIDX0:%.+]]> = getelementptr inbounds ir<%x>, vp<[[STEPS]]>
; CHECK-NEXT:     vp<[[ADDR0:%.+]]> = vector-pointer ir<[[ARRAYIDX0]]>
; CHECK-NEXT:     WIDEN ir<[[LOAD0:%.+]]> = load vp<[[ADDR0]]>
; CHECK-NEXT:     CLONE ir<[[ARRAYIDX1:%.+]]> = getelementptr inbounds ir<%y>, vp<[[STEPS]]>
; CHECK-NEXT:     vp<[[ADDR1:%.+]]> = vector-pointer ir<[[ARRAYIDX1]]>
; CHECK-NEXT:     WIDEN ir<[[LOAD1:%.+]]> = load vp<[[ADDR1]]>
; CHECK-NEXT:     EXPRESSION vp<[[RDX_NEXT]]> = ir<[[RDX]]> + reduce.add (mul nsw ir<[[LOAD0]]>, ir<[[LOAD1]]>)
; CHECK-NEXT:     EMIT vp<[[IV_NEXT]]> = add nuw vp<[[IV]]>, vp<[[VFxUF]]>
; CHECK-NEXT:     EMIT branch-on-count vp<[[IV_NEXT]]>, vp<[[VTC]]>
; CHECK-NEXT:   No successors
; CHECK-NEXT: }
;
entry:
  br label %loop

loop:
  %iv = phi i32 [ %iv.next, %loop ], [ 0, %entry ]
  %rdx = phi i64 [ %rdx.next, %loop ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds i64, ptr %x, i32 %iv
  %load0 = load i64, ptr %arrayidx, align 4
  %arrayidx1 = getelementptr inbounds i64, ptr %y, i32 %iv
  %load1 = load i64, ptr %arrayidx1, align 4
  %mul = mul nsw i64 %load0, %load1
  %rdx.next = add nsw i64 %rdx, %mul
  %iv.next = add nuw nsw i32 %iv, 1
  %exitcond = icmp eq i32 %iv.next, %n
  br i1 %exitcond, label %exit, label %loop

exit:
  %r.0.lcssa = phi i64 [ %rdx.next, %loop ]
  ret i64 %r.0.lcssa
}

define i64 @print_mulacc_extended(ptr nocapture readonly %x, ptr nocapture readonly %y, i32 %n) {
; CHECK-LABEL: 'print_mulacc_extended'
; CHECK:      VPlan 'Initial VPlan for VF={4},UF>=1' {
; CHECK-NEXT: Live-in vp<[[VF:%.+]]> = VF
; CHECK-NEXT: Live-in vp<[[VFxUF:%.+]]> = VF * UF
; CHECK-NEXT: Live-in vp<[[VTC:%.+]]> = vector-trip-count
; CHECK-NEXT: Live-in ir<%n> = original trip-count
; CHECK-EMPTY:
; CHECK:      vector.ph:
; CHECK-NEXT:   EMIT vp<[[RDX_START:%.+]]> = reduction-start-vector ir<0>, ir<0>, ir<1>
; CHECK-NEXT: Successor(s): vector loop
; CHECK-EMPTY:
; CHECK-NEXT: <x1> vector loop: {
; CHECK-NEXT:   vector.body:
; CHECK-NEXT:     EMIT vp<[[IV:%.+]]> = CANONICAL-INDUCTION ir<0>, vp<[[IV_NEXT:%.+]]>
; CHECK-NEXT:     WIDEN-REDUCTION-PHI ir<[[RDX:%.+]]> = phi vp<[[RDX_START]]>, vp<[[RDX_NEXT:%.+]]>
; CHECK-NEXT:     vp<[[STEPS:%.+]]> = SCALAR-STEPS vp<[[IV]]>, ir<1>
; CHECK-NEXT:     CLONE ir<[[ARRAYIDX0:%.+]]> = getelementptr inbounds ir<%x>, vp<[[STEPS]]>
; CHECK-NEXT:     vp<[[ADDR0:%.+]]> = vector-pointer ir<[[ARRAYIDX0]]>
; CHECK-NEXT:     WIDEN ir<[[LOAD0:%.+]]> = load vp<[[ADDR0]]>
; CHECK-NEXT:     CLONE ir<[[ARRAYIDX1:%.+]]> = getelementptr inbounds ir<%y>, vp<[[STEPS]]>
; CHECK-NEXT:     vp<[[ADDR1:%.+]]> = vector-pointer ir<[[ARRAYIDX1]]>
; CHECK-NEXT:     WIDEN ir<[[LOAD1:%.+]]> = load vp<[[ADDR1]]>
; CHECK-NEXT:     EXPRESSION vp<[[RDX_NEXT:%.+]]> = ir<[[RDX]]> + reduce.add (mul nsw (ir<[[LOAD0]]> sext to i64), (ir<[[LOAD1]]> sext to i64))
; CHECK-NEXT:     EMIT vp<[[IV_NEXT]]> = add nuw vp<[[IV]]>, vp<[[VFxUF]]>
; CHECK-NEXT:     EMIT branch-on-count vp<[[IV_NEXT]]>, vp<[[VTC]]>
; CHECK-NEXT:   No successors
; CHECK-NEXT: }
;
entry:
  br label %loop

loop:
  %iv = phi i32 [ %iv.next, %loop ], [ 0, %entry ]
  %rdx = phi i64 [ %rdx.next, %loop ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds i16, ptr %x, i32 %iv
  %load0 = load i16, ptr %arrayidx, align 4
  %arrayidx1 = getelementptr inbounds i16, ptr %y, i32 %iv
  %load1 = load i16, ptr %arrayidx1, align 4
  %conv0 = sext i16 %load0 to i32
  %conv1 = sext i16 %load1 to i32
  %mul = mul nsw i32 %conv0, %conv1
  %conv = sext i32 %mul to i64
  %rdx.next = add nsw i64 %rdx, %conv
  %iv.next = add nuw nsw i32 %iv, 1
  %exitcond = icmp eq i32 %iv.next, %n
  br i1 %exitcond, label %exit, label %loop

exit:
  %r.0.lcssa = phi i64 [ %rdx.next, %loop ]
  ret i64 %r.0.lcssa
}

define i64 @print_extended_sub_reduction(ptr nocapture readonly %x, ptr nocapture readonly %y, i32 %n) {
; CHECK-LABEL: 'print_extended_sub_reduction'
; CHECK:      VPlan 'Initial VPlan for VF={4},UF>=1' {
; CHECK-NEXT: Live-in vp<[[VF:%.+]]> = VF
; CHECK-NEXT: Live-in vp<[[VFxUF:%.+]]> = VF * UF
; CHECK-NEXT: Live-in vp<[[VTC:%.+]]> = vector-trip-count
; CHECK-NEXT: Live-in ir<%n> = original trip-count
; CHECK-EMPTY:
; CHECK:      vector.ph:
; CHECK-NEXT:   EMIT vp<[[RDX_START:%.+]]> = reduction-start-vector ir<0>, ir<0>, ir<1>
; CHECK-NEXT: Successor(s): vector loop
; CHECK-EMPTY:
; CHECK-NEXT: <x1> vector loop: {
; CHECK-NEXT:   vector.body:
; CHECK-NEXT:     EMIT vp<[[IV:%.+]]> = CANONICAL-INDUCTION ir<0>, vp<[[IV_NEXT:%.+]]>
; CHECK-NEXT:     WIDEN-REDUCTION-PHI ir<[[RDX:%.+]]> = phi vp<[[RDX_START]]>, vp<[[RDX_NEXT:%.+]]>
; CHECK-NEXT:     vp<[[STEPS:%.+]]> = SCALAR-STEPS vp<[[IV]]>, ir<1>
; CHECK-NEXT:     CLONE ir<%arrayidx> = getelementptr inbounds ir<%x>, vp<[[STEPS]]>
; CHECK-NEXT:     vp<[[ADDR:%.+]]> = vector-pointer ir<%arrayidx>
; CHECK-NEXT:     WIDEN ir<[[LOAD:%.+]]> = load vp<[[ADDR]]>
; CHECK-NEXT:     EXPRESSION vp<[[RDX_NEXT]]> = ir<[[RDX]]> + reduce.sub (ir<[[LOAD]]> zext to i64)
; CHECK-NEXT:     EMIT vp<[[IV_NEXT]]> = add nuw vp<[[IV]]>, vp<[[VFxUF]]>
; CHECK-NEXT:     EMIT branch-on-count vp<[[IV_NEXT]]>, vp<[[VTC]]>
; CHECK-NEXT:   No successors
; CHECK-NEXT: }
;
entry:
  br label %loop

loop:
  %iv = phi i32 [ %iv.next, %loop ], [ 0, %entry ]
  %rdx = phi i64 [ %rdx.next, %loop ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds i32, ptr %x, i32 %iv
  %load0 = load i32, ptr %arrayidx, align 4
  %conv0 = zext i32 %load0 to i64
  %rdx.next = sub nsw i64 %rdx, %conv0
  %iv.next = add nuw nsw i32 %iv, 1
  %exitcond = icmp eq i32 %iv.next, %n
  br i1 %exitcond, label %exit, label %loop

exit:
  %r.0.lcssa = phi i64 [ %rdx.next, %loop ]
  ret i64 %r.0.lcssa
}

define i32 @print_mulacc_sub(ptr %a, ptr %b) {
; CHECK-LABEL: 'print_mulacc_sub'
; CHECK:      VPlan 'Initial VPlan for VF={4},UF>=1' {
; CHECK-NEXT: Live-in vp<%0> = VF
; CHECK-NEXT: Live-in vp<%1> = VF * UF
; CHECK-NEXT: Live-in vp<%2> = vector-trip-count
; CHECK-NEXT: Live-in ir<1024> = original trip-count
; CHECK-EMPTY:
; CHECK-NEXT: ir-bb<entry>:
; CHECK-NEXT: Successor(s): scalar.ph, vector.ph
; CHECK-EMPTY:
; CHECK-NEXT: vector.ph:
; CHECK-NEXT:   EMIT vp<%3> = reduction-start-vector ir<0>, ir<0>, ir<1>
; CHECK-NEXT: Successor(s): vector loop
; CHECK-EMPTY:
; CHECK-NEXT: <x1> vector loop: {
; CHECK-NEXT:   vector.body:
; CHECK-NEXT:     EMIT vp<%4> = CANONICAL-INDUCTION ir<0>, vp<%index.next>
; CHECK-NEXT:     WIDEN-REDUCTION-PHI ir<%accum> = phi vp<%3>, vp<%8>
; CHECK-NEXT:     vp<%5> = SCALAR-STEPS vp<%4>, ir<1>, vp<%0>
; CHECK-NEXT:     CLONE ir<%gep.a> = getelementptr ir<%a>, vp<%5>
; CHECK-NEXT:     vp<%6> = vector-pointer ir<%gep.a>
; CHECK-NEXT:     WIDEN ir<%load.a> = load vp<%6>
; CHECK-NEXT:     CLONE ir<%gep.b> = getelementptr ir<%b>, vp<%5>
; CHECK-NEXT:     vp<%7> = vector-pointer ir<%gep.b>
; CHECK-NEXT:     WIDEN ir<%load.b> = load vp<%7>
; CHECK-NEXT:     EXPRESSION vp<%8> = ir<%accum> + reduce.sub (mul (ir<%load.b> zext to i32), (ir<%load.a> zext to i32))
; CHECK-NEXT:     EMIT vp<%index.next> = add nuw vp<%4>, vp<%1>
; CHECK-NEXT:     EMIT branch-on-count vp<%index.next>, vp<%2>
; CHECK-NEXT:   No successors
; CHECK-NEXT: }
; CHECK-NEXT: Successor(s): middle.block
; CHECK-EMPTY:
; CHECK-NEXT: middle.block:
; CHECK-NEXT:   EMIT vp<%10> = compute-reduction-result ir<%accum>, vp<%8>
; CHECK-NEXT:   EMIT vp<%cmp.n> = icmp eq ir<1024>, vp<%2>
; CHECK-NEXT:   EMIT branch-on-cond vp<%cmp.n>
; CHECK-NEXT: Successor(s): ir-bb<exit>, scalar.ph
; CHECK-EMPTY:
; CHECK-NEXT: ir-bb<exit>:
; CHECK-NEXT:   IR   %add.lcssa = phi i32 [ %add, %loop ] (extra operand: vp<%10> from middle.block)
; CHECK-NEXT: No successors
; CHECK-EMPTY:
; CHECK-NEXT: scalar.ph:
; CHECK-NEXT:   EMIT-SCALAR vp<%bc.resume.val> = phi [ vp<%2>, middle.block ], [ ir<0>, ir-bb<entry> ]
; CHECK-NEXT:   EMIT-SCALAR vp<%bc.merge.rdx> = phi [ vp<%10>, middle.block ], [ ir<0>, ir-bb<entry> ]
; CHECK-NEXT: Successor(s): ir-bb<loop>
; CHECK-EMPTY:
; CHECK-NEXT: ir-bb<loop>:
; CHECK-NEXT:   IR   %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ] (extra operand: vp<%bc.resume.val> from scalar.ph)
; CHECK-NEXT:   IR   %accum = phi i32 [ 0, %entry ], [ %add, %loop ] (extra operand: vp<%bc.merge.rdx> from scalar.ph)
; CHECK-NEXT:   IR   %gep.a = getelementptr i8, ptr %a, i64 %iv
; CHECK-NEXT:   IR   %load.a = load i8, ptr %gep.a, align 1
; CHECK-NEXT:   IR   %ext.a = zext i8 %load.a to i32
; CHECK-NEXT:   IR   %gep.b = getelementptr i8, ptr %b, i64 %iv
; CHECK-NEXT:   IR   %load.b = load i8, ptr %gep.b, align 1
; CHECK-NEXT:   IR   %ext.b = zext i8 %load.b to i32
; CHECK-NEXT:   IR   %mul = mul i32 %ext.b, %ext.a
; CHECK-NEXT:   IR   %add = sub i32 %accum, %mul
; CHECK-NEXT:   IR   %iv.next = add i64 %iv, 1
; CHECK-NEXT:   IR   %exitcond.not = icmp eq i64 %iv.next, 1024
; CHECK-NEXT: No successors
; CHECK-NEXT: }
; CHECK:      VPlan 'Final VPlan for VF={4},UF={1}' {
; CHECK-NEXT: Live-in ir<1024> = vector-trip-count
; CHECK-NEXT: Live-in ir<1024> = original trip-count
; CHECK-EMPTY:
; CHECK-NEXT: ir-bb<entry>:
; CHECK-NEXT: Successor(s): vector.ph
; CHECK-EMPTY:
; CHECK-NEXT: vector.ph:
; CHECK-NEXT: Successor(s): vector.body
; CHECK-EMPTY:
; CHECK-NEXT: vector.body:
; CHECK-NEXT:   EMIT-SCALAR vp<%index> = phi [ ir<0>, vector.ph ], [ vp<%index.next>, vector.body ]
; CHECK-NEXT:   WIDEN-REDUCTION-PHI ir<%accum> = phi ir<0>, ir<%add>
; CHECK-NEXT:   CLONE ir<%gep.a> = getelementptr ir<%a>, vp<%index>
; CHECK-NEXT:   WIDEN ir<%load.a> = load ir<%gep.a>
; CHECK-NEXT:   CLONE ir<%gep.b> = getelementptr ir<%b>, vp<%index>
; CHECK-NEXT:   WIDEN ir<%load.b> = load ir<%gep.b>
; CHECK-NEXT:   WIDEN-CAST ir<%ext.b> = zext ir<%load.b> to i32
; CHECK-NEXT:   WIDEN-CAST ir<%ext.a> = zext ir<%load.a> to i32
; CHECK-NEXT:   WIDEN ir<%mul> = mul ir<%ext.b>, ir<%ext.a>
; CHECK-NEXT:   REDUCE ir<%add> = ir<%accum> + reduce.sub (ir<%mul>)
; CHECK-NEXT:   EMIT vp<%index.next> = add nuw vp<%index>, ir<4>
; CHECK-NEXT:   EMIT branch-on-count vp<%index.next>, ir<1024>
; CHECK-NEXT: Successor(s): middle.block, vector.body
; CHECK-EMPTY:
; CHECK-NEXT: middle.block:
; CHECK-NEXT:   EMIT vp<[[RED_RESULT:%.+]]> = compute-reduction-result ir<%accum>, ir<%add>
; CHECK-NEXT: Successor(s): ir-bb<exit>
; CHECK-EMPTY:
; CHECK-NEXT: ir-bb<exit>:
; CHECK-NEXT:   IR   %add.lcssa = phi i32 [ %add, %loop ] (extra operand: vp<[[RED_RESULT]]> from middle.block)
; CHECK-NEXT: No successors
; CHECK-NEXT: }
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %accum = phi i32 [ 0, %entry ], [ %add, %loop ]
  %gep.a = getelementptr i8, ptr %a, i64 %iv
  %load.a = load i8, ptr %gep.a, align 1
  %ext.a = zext i8 %load.a to i32
  %gep.b = getelementptr i8, ptr %b, i64 %iv
  %load.b = load i8, ptr %gep.b, align 1
  %ext.b = zext i8 %load.b to i32
  %mul = mul i32 %ext.b, %ext.a
  %add = sub i32 %accum, %mul
  %iv.next = add i64 %iv, 1
  %exitcond.not = icmp eq i64 %iv.next, 1024
  br i1 %exitcond.not, label %exit, label %loop

exit:
  ret i32 %add
}

define i32 @print_mulacc_negated(ptr %a, ptr %b) {
; CHECK-LABEL: 'print_mulacc_negated'
; CHECK:      VPlan 'Initial VPlan for VF={4},UF>=1' {
; CHECK-NEXT: Live-in vp<%0> = VF
; CHECK-NEXT: Live-in vp<%1> = VF * UF
; CHECK-NEXT: Live-in vp<%2> = vector-trip-count
; CHECK-NEXT: Live-in ir<1024> = original trip-count
; CHECK-EMPTY:
; CHECK-NEXT: ir-bb<entry>:
; CHECK-NEXT: Successor(s): scalar.ph, vector.ph
; CHECK-EMPTY:
; CHECK-NEXT: vector.ph:
; CHECK-NEXT:   EMIT vp<%3> = reduction-start-vector ir<0>, ir<0>, ir<1>
; CHECK-NEXT: Successor(s): vector loop
; CHECK-EMPTY:
; CHECK-NEXT: <x1> vector loop: {
; CHECK-NEXT:   vector.body:
; CHECK-NEXT:     EMIT vp<%4> = CANONICAL-INDUCTION ir<0>, vp<%index.next>
; CHECK-NEXT:     WIDEN-REDUCTION-PHI ir<%accum> = phi vp<%3>, vp<%8>
; CHECK-NEXT:     vp<%5> = SCALAR-STEPS vp<%4>, ir<1>, vp<%0>
; CHECK-NEXT:     CLONE ir<%gep.a> = getelementptr ir<%a>, vp<%5>
; CHECK-NEXT:     vp<%6> = vector-pointer ir<%gep.a>
; CHECK-NEXT:     WIDEN ir<%load.a> = load vp<%6>
; CHECK-NEXT:     CLONE ir<%gep.b> = getelementptr ir<%b>, vp<%5>
; CHECK-NEXT:     vp<%7> = vector-pointer ir<%gep.b>
; CHECK-NEXT:     WIDEN ir<%load.b> = load vp<%7>
; CHECK-NEXT:     EXPRESSION vp<%8> = ir<%accum> + reduce.add (sub (0, mul (ir<%load.b> zext to i32), (ir<%load.a> zext to i32)))
; CHECK-NEXT:     EMIT vp<%index.next> = add nuw vp<%4>, vp<%1>
; CHECK-NEXT:     EMIT branch-on-count vp<%index.next>, vp<%2>
; CHECK-NEXT:   No successors
; CHECK-NEXT: }
; CHECK-NEXT: Successor(s): middle.block
; CHECK-EMPTY:
; CHECK-NEXT: middle.block:
; CHECK-NEXT:   EMIT vp<%10> = compute-reduction-result ir<%accum>, vp<%8>
; CHECK-NEXT:   EMIT vp<%cmp.n> = icmp eq ir<1024>, vp<%2>
; CHECK-NEXT:   EMIT branch-on-cond vp<%cmp.n>
; CHECK-NEXT: Successor(s): ir-bb<exit>, scalar.ph
; CHECK-EMPTY:
; CHECK-NEXT: ir-bb<exit>:
; CHECK-NEXT:   IR   %add.lcssa = phi i32 [ %add, %loop ] (extra operand: vp<%10> from middle.block)
; CHECK-NEXT: No successors
; CHECK-EMPTY:
; CHECK-NEXT: scalar.ph:
; CHECK-NEXT:   EMIT-SCALAR vp<%bc.resume.val> = phi [ vp<%2>, middle.block ], [ ir<0>, ir-bb<entry> ]
; CHECK-NEXT:   EMIT-SCALAR vp<%bc.merge.rdx> = phi [ vp<%10>, middle.block ], [ ir<0>, ir-bb<entry> ]
; CHECK-NEXT: Successor(s): ir-bb<loop>
; CHECK-EMPTY:
; CHECK-NEXT: ir-bb<loop>:
; CHECK-NEXT:   IR   %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ] (extra operand: vp<%bc.resume.val> from scalar.ph)
; CHECK-NEXT:   IR   %accum = phi i32 [ 0, %entry ], [ %add, %loop ] (extra operand: vp<%bc.merge.rdx> from scalar.ph)
; CHECK-NEXT:   IR   %gep.a = getelementptr i8, ptr %a, i64 %iv
; CHECK-NEXT:   IR   %load.a = load i8, ptr %gep.a, align 1
; CHECK-NEXT:   IR   %ext.a = zext i8 %load.a to i32
; CHECK-NEXT:   IR   %gep.b = getelementptr i8, ptr %b, i64 %iv
; CHECK-NEXT:   IR   %load.b = load i8, ptr %gep.b, align 1
; CHECK-NEXT:   IR   %ext.b = zext i8 %load.b to i32
; CHECK-NEXT:   IR   %mul = mul i32 %ext.b, %ext.a
; CHECK-NEXT:   IR   %sub = sub i32 0, %mul
; CHECK-NEXT:   IR   %add = add i32 %accum, %sub
; CHECK-NEXT:   IR   %iv.next = add i64 %iv, 1
; CHECK-NEXT:   IR   %exitcond.not = icmp eq i64 %iv.next, 1024
; CHECK-NEXT: No successors
; CHECK-NEXT: }
; CHECK:      VPlan 'Final VPlan for VF={4},UF={1}' {
; CHECK-NEXT: Live-in ir<1024> = vector-trip-count
; CHECK-NEXT: Live-in ir<1024> = original trip-count
; CHECK-EMPTY:
; CHECK-NEXT: ir-bb<entry>:
; CHECK-NEXT: Successor(s): vector.ph
; CHECK-EMPTY:
; CHECK-NEXT: vector.ph:
; CHECK-NEXT: Successor(s): vector.body
; CHECK-EMPTY:
; CHECK-NEXT: vector.body:
; CHECK-NEXT:   EMIT-SCALAR vp<%index> = phi [ ir<0>, vector.ph ], [ vp<%index.next>, vector.body ]
; CHECK-NEXT:   WIDEN-REDUCTION-PHI ir<%accum> = phi ir<0>, ir<%add>
; CHECK-NEXT:   CLONE ir<%gep.a> = getelementptr ir<%a>, vp<%index>
; CHECK-NEXT:   WIDEN ir<%load.a> = load ir<%gep.a>
; CHECK-NEXT:   CLONE ir<%gep.b> = getelementptr ir<%b>, vp<%index>
; CHECK-NEXT:   WIDEN ir<%load.b> = load ir<%gep.b>
; CHECK-NEXT:   WIDEN-CAST ir<%ext.b> = zext ir<%load.b> to i32
; CHECK-NEXT:   WIDEN-CAST ir<%ext.a> = zext ir<%load.a> to i32
; CHECK-NEXT:   WIDEN ir<%mul> = mul ir<%ext.b>, ir<%ext.a>
; CHECK-NEXT:   WIDEN ir<%sub> = sub ir<0>, ir<%mul>
; CHECK-NEXT:   REDUCE ir<%add> = ir<%accum> + reduce.add (ir<%sub>)
; CHECK-NEXT:   EMIT vp<%index.next> = add nuw vp<%index>, ir<4>
; CHECK-NEXT:   EMIT branch-on-count vp<%index.next>, ir<1024>
; CHECK-NEXT: Successor(s): middle.block, vector.body
; CHECK-EMPTY:
; CHECK-NEXT: middle.block:
; CHECK-NEXT:   EMIT vp<[[RED_RESULT:%.+]]> = compute-reduction-result ir<%accum>, ir<%add>
; CHECK-NEXT: Successor(s): ir-bb<exit>
; CHECK-EMPTY:
; CHECK-NEXT: ir-bb<exit>:
; CHECK-NEXT:   IR   %add.lcssa = phi i32 [ %add, %loop ] (extra operand: vp<[[RED_RESULT]]> from middle.block)
; CHECK-NEXT: No successors
; CHECK-NEXT: }
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %accum = phi i32 [ 0, %entry ], [ %add, %loop ]
  %gep.a = getelementptr i8, ptr %a, i64 %iv
  %load.a = load i8, ptr %gep.a, align 1
  %ext.a = zext i8 %load.a to i32
  %gep.b = getelementptr i8, ptr %b, i64 %iv
  %load.b = load i8, ptr %gep.b, align 1
  %ext.b = zext i8 %load.b to i32
  %mul = mul i32 %ext.b, %ext.a
  %sub = sub i32 0, %mul
  %add = add i32 %accum, %sub
  %iv.next = add i64 %iv, 1
  %exitcond.not = icmp eq i64 %iv.next, 1024
  br i1 %exitcond.not, label %exit, label %loop

exit:
  ret i32 %add
}

define i64 @print_mulacc_sub_extended(ptr nocapture readonly %x, ptr nocapture readonly %y, i32 %n) {
; CHECK-LABEL: 'print_mulacc_sub_extended'
; CHECK:      VPlan 'Initial VPlan for VF={4},UF>=1' {
; CHECK-NEXT: Live-in vp<[[VF:%.+]]> = VF
; CHECK-NEXT: Live-in vp<[[VFxUF:%.+]]> = VF * UF
; CHECK-NEXT: Live-in vp<[[VTC:%.+]]> = vector-trip-count
; CHECK-NEXT: Live-in ir<%n> = original trip-count
; CHECK-EMPTY:
; CHECK:      vector.ph:
; CHECK-NEXT:   EMIT vp<[[RDX_START:%.+]]> = reduction-start-vector ir<0>, ir<0>, ir<1>
; CHECK-NEXT: Successor(s): vector loop
; CHECK-EMPTY:
; CHECK-NEXT: <x1> vector loop: {
; CHECK-NEXT:   vector.body:
; CHECK-NEXT:     EMIT vp<[[IV:%.+]]> = CANONICAL-INDUCTION ir<0>, vp<[[IV_NEXT:%.+]]>
; CHECK-NEXT:     WIDEN-REDUCTION-PHI ir<[[RDX:%.+]]> = phi vp<[[RDX_START]]>, vp<[[RDX_NEXT:%.+]]>
; CHECK-NEXT:     vp<[[STEPS:%.+]]> = SCALAR-STEPS vp<[[IV]]>, ir<1>
; CHECK-NEXT:     CLONE ir<[[ARRAYIDX0:%.+]]> = getelementptr inbounds ir<%x>, vp<[[STEPS]]>
; CHECK-NEXT:     vp<[[ADDR0:%.+]]> = vector-pointer ir<[[ARRAYIDX0]]>
; CHECK-NEXT:     WIDEN ir<[[LOAD0:%.+]]> = load vp<[[ADDR0]]>
; CHECK-NEXT:     CLONE ir<[[ARRAYIDX1:%.+]]> = getelementptr inbounds ir<%y>, vp<[[STEPS]]>
; CHECK-NEXT:     vp<[[ADDR1:%.+]]> = vector-pointer ir<[[ARRAYIDX1]]>
; CHECK-NEXT:     WIDEN ir<[[LOAD1:%.+]]> = load vp<[[ADDR1]]>
; CHECK-NEXT:     EXPRESSION vp<[[RDX_NEXT:%.+]]> = ir<[[RDX]]> + reduce.sub (mul nsw (ir<[[LOAD0]]> sext to i64), (ir<[[LOAD1]]> sext to i64))
; CHECK-NEXT:     EMIT vp<[[IV_NEXT]]> = add nuw vp<[[IV]]>, vp<[[VFxUF]]>
; CHECK-NEXT:     EMIT branch-on-count vp<[[IV_NEXT]]>, vp<[[VTC]]>
; CHECK-NEXT:   No successors
; CHECK-NEXT: }
;
entry:
  br label %loop

loop:
  %iv = phi i32 [ %iv.next, %loop ], [ 0, %entry ]
  %rdx = phi i64 [ %rdx.next, %loop ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds i16, ptr %x, i32 %iv
  %load0 = load i16, ptr %arrayidx, align 4
  %arrayidx1 = getelementptr inbounds i16, ptr %y, i32 %iv
  %load1 = load i16, ptr %arrayidx1, align 4
  %conv0 = sext i16 %load0 to i32
  %conv1 = sext i16 %load1 to i32
  %mul = mul nsw i32 %conv0, %conv1
  %conv = sext i32 %mul to i64
  %rdx.next = sub nsw i64 %rdx, %conv
  %iv.next = add nuw nsw i32 %iv, 1
  %exitcond = icmp eq i32 %iv.next, %n
  br i1 %exitcond, label %exit, label %loop

exit:
  %r.0.lcssa = phi i64 [ %rdx.next, %loop ]
  ret i64 %r.0.lcssa
}

define i64 @print_mulacc_duplicate_extends(ptr nocapture readonly %x, ptr nocapture readonly %y, i32 %n) {
; CHECK-LABEL: 'print_mulacc_duplicate_extends'
; CHECK:      VPlan 'Initial VPlan for VF={4},UF>=1' {
; CHECK-NEXT: Live-in vp<[[VF:%.+]]> = VF
; CHECK-NEXT: Live-in vp<[[VFxUF:%.+]]> = VF * UF
; CHECK-NEXT: Live-in vp<[[VTC:%.+]]> = vector-trip-count
; CHECK-NEXT: Live-in ir<%n> = original trip-count
; CHECK-EMPTY:
; CHECK:      vector.ph:
; CHECK-NEXT:   EMIT vp<[[RDX_START:%.+]]> = reduction-start-vector ir<0>, ir<0>, ir<1>
; CHECK-NEXT: Successor(s): vector loop
; CHECK-EMPTY:
; CHECK-NEXT: <x1> vector loop: {
; CHECK-NEXT:   vector.body:
; CHECK-NEXT:     EMIT vp<[[IV:%.+]]> = CANONICAL-INDUCTION ir<0>, vp<[[IV_NEXT:%.+]]>
; CHECK-NEXT:     WIDEN-REDUCTION-PHI ir<[[RDX:%.+]]> = phi vp<[[RDX_START]]>, vp<[[RDX_NEXT:%.+]]>
; CHECK-NEXT:     vp<[[STEPS:%.+]]> = SCALAR-STEPS vp<[[IV]]>, ir<1>
; CHECK-NEXT:     CLONE ir<[[ARRAYIDX0:%.+]]> = getelementptr inbounds ir<%x>, vp<[[STEPS]]>
; CHECK-NEXT:     vp<[[ADDR0:%.+]]> = vector-pointer ir<[[ARRAYIDX0]]>
; CHECK-NEXT:     WIDEN ir<[[LOAD0:%.+]]> = load vp<[[ADDR0]]>
; CHECK-NEXT:     EXPRESSION vp<[[RDX_NEXT:%.+]]> = ir<[[RDX]]> + reduce.sub (mul nsw (ir<[[LOAD0]]> sext to i64), (ir<[[LOAD0]]> sext to i64))
; CHECK-NEXT:     EMIT vp<[[IV_NEXT]]> = add nuw vp<[[IV]]>, vp<[[VFxUF]]>
; CHECK-NEXT:     EMIT branch-on-count vp<[[IV_NEXT]]>, vp<[[VTC]]>
; CHECK-NEXT:   No successors
; CHECK-NEXT: }
;
entry:
  br label %loop

loop:
  %iv = phi i32 [ %iv.next, %loop ], [ 0, %entry ]
  %rdx = phi i64 [ %rdx.next, %loop ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds i16, ptr %x, i32 %iv
  %load0 = load i16, ptr %arrayidx, align 4
  %conv0 = sext i16 %load0 to i32
  %mul = mul nsw i32 %conv0, %conv0
  %conv = sext i32 %mul to i64
  %rdx.next = sub nsw i64 %rdx, %conv
  %iv.next = add nuw nsw i32 %iv, 1
  %exitcond = icmp eq i32 %iv.next, %n
  br i1 %exitcond, label %exit, label %loop

exit:
  %r.0.lcssa = phi i64 [ %rdx.next, %loop ]
  ret i64 %r.0.lcssa
}

define i32 @print_mulacc_extended_const(ptr %start, ptr %end) {
; CHECK-LABEL: 'print_mulacc_extended_const'
; CHECK:       VPlan 'Initial VPlan for VF={4},UF>=1' {
; CHECK-NEXT:  Live-in vp<[[VF:%.+]]> = VF
; CHECK-NEXT:  Live-in vp<[[VFxUF:%.+]]> = VF * UF
; CHECK-NEXT:  Live-in vp<[[VTC:%.+]]> = vector-trip-count
; CHECK-NEXT:  vp<%3> = original trip-count
; CHECK-EMPTY:
; CHECK-NEXT:  ir-bb<entry>:
; CHECK-NEXT:    EMIT vp<%3> = EXPAND SCEV (1 + (-1 * (ptrtoint ptr %start to i64)) + (ptrtoint ptr %end to i64))
; CHECK-NEXT:  Successor(s): scalar.ph, vector.ph
; CHECK-EMPTY:
; CHECK-NEXT:  vector.ph:
; CHECK-NEXT:    vp<[[DER_IV:%.+]]> = DERIVED-IV ir<%start> + vp<[[VTC]]> * ir<1>
; CHECK-NEXT:    EMIT vp<[[RDX_START:%.+]]> = reduction-start-vector ir<0>, ir<0>, ir<1>
; CHECK-NEXT:  Successor(s): vector loop
; CHECK-EMPTY:
; CHECK-NEXT:  <x1> vector loop: {
; CHECK-NEXT:    vector.body:
; CHECK-NEXT:      EMIT vp<[[CAN_IV:%.+]]> = CANONICAL-INDUCTION ir<0>, vp<[[IV_NEXT:%.+]]>
; CHECK-NEXT:      WIDEN-REDUCTION-PHI ir<[[RDX:%.+]]> = phi vp<[[RDX_START]]>, vp<[[RDX_NEXT:%.+]]>
; CHECK-NEXT:      vp<[[STEPS:%.+]]> = SCALAR-STEPS vp<[[CAN_IV]]>, ir<1>, vp<[[VF]]>
; CHECK-NEXT:      EMIT vp<%next.gep> = ptradd ir<%start>, vp<[[STEPS]]>
; CHECK-NEXT:      vp<[[VEC_PTR:%.+]]> = vector-pointer vp<%next.gep>
; CHECK-NEXT:      WIDEN ir<%l> = load vp<[[VEC_PTR]]>
; CHECK-NEXT:      EXPRESSION vp<[[RDX_NEXT]]> = ir<[[RDX]]> + reduce.add (mul (ir<%l> zext to i32), (ir<63> zext to i32))
; CHECK-NEXT:      EMIT vp<[[IV_NEXT]]> = add nuw vp<[[CAN_IV]]>, vp<[[VFxUF]]>
; CHECK-NEXT:      EMIT branch-on-count vp<[[IV_NEXT]]>, vp<[[VTC]]>
; CHECK-NEXT:    No successors
; CHECK-NEXT:  }
; CHECK-NEXT:  Successor(s): middle.block
; CHECK-EMPTY:
; CHECK-NEXT:  middle.block:
; CHECK-NEXT:    EMIT vp<%11> = compute-reduction-result ir<[[RDX]]>, vp<[[RDX_NEXT]]>
; CHECK-NEXT:    EMIT vp<%cmp.n> = icmp eq vp<%3>, vp<[[VTC]]>
; CHECK-NEXT:    EMIT branch-on-cond vp<%cmp.n>
entry:
  br label %loop

loop:
  %ptr.iv = phi ptr [ %start, %entry ], [ %gep.iv.next, %loop ]
  %red = phi i32 [ 0, %entry ], [ %red.next, %loop ]
  %l = load i8, ptr %ptr.iv, align 1
  %l.ext = zext i8 %l to i32
  %mul = mul i32 %l.ext, 63
  %red.next = add i32 %red, %mul
  %gep.iv.next = getelementptr i8, ptr %ptr.iv, i64 1
  %ec = icmp eq ptr %ptr.iv, %end
  br i1 %ec, label %exit, label %loop

exit:
  ret i32 %red.next
}

define i32 @print_mulacc_extended_const_lhs(ptr %start, ptr %end) {
; CHECK-LABEL: 'print_mulacc_extended_const_lhs'
; CHECK:       VPlan 'Initial VPlan for VF={4},UF>=1' {
; CHECK-NEXT:  Live-in vp<[[VF:%.+]]> = VF
; CHECK-NEXT:  Live-in vp<[[VFxUF:%.+]]> = VF * UF
; CHECK-NEXT:  Live-in vp<[[VTC:%.+]]> = vector-trip-count
; CHECK-NEXT:  vp<%3> = original trip-count
; CHECK-EMPTY:
; CHECK-NEXT:  ir-bb<entry>:
; CHECK-NEXT:    EMIT vp<%3> = EXPAND SCEV (1 + (-1 * (ptrtoint ptr %start to i64)) + (ptrtoint ptr %end to i64))
; CHECK-NEXT:  Successor(s): scalar.ph, vector.ph
; CHECK-EMPTY:
; CHECK-NEXT:  vector.ph:
; CHECK-NEXT:    vp<[[DER_IV:%.+]]> = DERIVED-IV ir<%start> + vp<[[VTC]]> * ir<1>
; CHECK-NEXT:    EMIT vp<[[RDX_START:%.+]]> = reduction-start-vector ir<0>, ir<0>, ir<1>
; CHECK-NEXT:  Successor(s): vector loop
; CHECK-EMPTY:
; CHECK-NEXT:  <x1> vector loop: {
; CHECK-NEXT:    vector.body:
; CHECK-NEXT:      EMIT vp<[[CAN_IV:%.+]]> = CANONICAL-INDUCTION ir<0>, vp<[[IV_NEXT:%.+]]>
; CHECK-NEXT:      WIDEN-REDUCTION-PHI ir<[[RDX:%.+]]> = phi vp<[[RDX_START]]>, vp<[[RDX_NEXT:%.+]]>
; CHECK-NEXT:      vp<[[STEPS:%.+]]> = SCALAR-STEPS vp<[[CAN_IV]]>, ir<1>, vp<[[VF]]>
; CHECK-NEXT:      EMIT vp<%next.gep> = ptradd ir<%start>, vp<[[STEPS]]>
; CHECK-NEXT:      vp<[[VEC_PTR:%.+]]> = vector-pointer vp<%next.gep>
; CHECK-NEXT:      WIDEN ir<%l> = load vp<[[VEC_PTR]]>
; CHECK-NEXT:      WIDEN-CAST ir<%l.ext> = zext ir<%l> to i32
; CHECK-NEXT:      EXPRESSION vp<[[RDX_NEXT]]> = ir<[[RDX]]> + reduce.add (mul ir<63>, ir<%l.ext>)
; CHECK-NEXT:      EMIT vp<[[IV_NEXT]]> = add nuw vp<[[CAN_IV]]>, vp<[[VFxUF]]>
; CHECK-NEXT:      EMIT branch-on-count vp<[[IV_NEXT]]>, vp<[[VTC]]>
; CHECK-NEXT:    No successors
; CHECK-NEXT:  }
; CHECK-NEXT:  Successor(s): middle.block
; CHECK-EMPTY:
; CHECK-NEXT:  middle.block:
; CHECK-NEXT:    EMIT vp<%11> = compute-reduction-result ir<[[RDX]]>, vp<[[RDX_NEXT]]>
; CHECK-NEXT:    EMIT vp<%cmp.n> = icmp eq vp<%3>, vp<[[VTC]]>
; CHECK-NEXT:    EMIT branch-on-cond vp<%cmp.n>
entry:
  br label %loop

loop:
  %ptr.iv = phi ptr [ %start, %entry ], [ %gep.iv.next, %loop ]
  %red = phi i32 [ 0, %entry ], [ %red.next, %loop ]
  %l = load i8, ptr %ptr.iv, align 1
  %l.ext = zext i8 %l to i32
  %mul = mul i32 63, %l.ext
  %red.next = add i32 %red, %mul
  %gep.iv.next = getelementptr i8, ptr %ptr.iv, i64 1
  %ec = icmp eq ptr %ptr.iv, %end
  br i1 %ec, label %exit, label %loop

exit:
  ret i32 %red.next
}

; Constants >= 128 cannot be treated as sign-extended, so the expression shouldn't extend 128
define i32 @print_mulacc_not_extended_const(ptr %start, ptr %end) {
; CHECK-LABEL: 'print_mulacc_not_extended_const'
; CHECK:       VPlan 'Initial VPlan for VF={4},UF>=1' {
; CHECK-NEXT:  Live-in vp<[[VF:%.+]]> = VF
; CHECK-NEXT:  Live-in vp<[[VFxUF:%.+]]> = VF * UF
; CHECK-NEXT:  Live-in vp<[[VTC:%.+]]> = vector-trip-count
; CHECK-NEXT:  vp<%3> = original trip-count
; CHECK-EMPTY:
; CHECK-NEXT:  ir-bb<entry>:
; CHECK-NEXT:    EMIT vp<%3> = EXPAND SCEV (1 + (-1 * (ptrtoint ptr %start to i64)) + (ptrtoint ptr %end to i64))
; CHECK-NEXT:  Successor(s): scalar.ph, vector.ph
; CHECK-EMPTY:
; CHECK-NEXT:  vector.ph:
; CHECK-NEXT:    vp<[[DER_IV:%.+]]> = DERIVED-IV ir<%start> + vp<[[VTC]]> * ir<1>
; CHECK-NEXT:    EMIT vp<[[RDX_START:%.+]]> = reduction-start-vector ir<0>, ir<0>, ir<1>
; CHECK-NEXT:  Successor(s): vector loop
; CHECK-EMPTY:
; CHECK-NEXT:  <x1> vector loop: {
; CHECK-NEXT:    vector.body:
; CHECK-NEXT:      EMIT vp<[[CAN_IV:%.+]]> = CANONICAL-INDUCTION ir<0>, vp<[[IV_NEXT:%.+]]>
; CHECK-NEXT:      WIDEN-REDUCTION-PHI ir<[[RDX:%.+]]> = phi vp<[[RDX_START]]>, vp<[[RDX_NEXT:%.+]]>
; CHECK-NEXT:      vp<[[STEPS:%.+]]> = SCALAR-STEPS vp<[[CAN_IV]]>, ir<1>, vp<[[VF]]>
; CHECK-NEXT:      EMIT vp<%next.gep> = ptradd ir<%start>, vp<[[STEPS]]>
; CHECK-NEXT:      vp<[[VEC_PTR:%.+]]> = vector-pointer vp<%next.gep>
; CHECK-NEXT:      WIDEN ir<%l> = load vp<[[VEC_PTR]]>
; CHECK-NEXT:      WIDEN-CAST ir<%l.ext> = sext ir<%l> to i32
; CHECK-NEXT:      EXPRESSION vp<[[RDX_NEXT]]> = ir<[[RDX]]> + reduce.add (mul ir<%l.ext>, ir<128>)
; CHECK-NEXT:      EMIT vp<[[IV_NEXT]]> = add nuw vp<[[CAN_IV]]>, vp<[[VFxUF]]>
; CHECK-NEXT:      EMIT branch-on-count vp<[[IV_NEXT]]>, vp<[[VTC]]>
; CHECK-NEXT:    No successors
; CHECK-NEXT:  }
; CHECK-NEXT:  Successor(s): middle.block
; CHECK-EMPTY:
; CHECK-NEXT:  middle.block:
; CHECK-NEXT:    EMIT vp<%11> = compute-reduction-result ir<[[RDX:%.+]]>, vp<[[RDX_NEXT]]>
; CHECK-NEXT:    EMIT vp<%cmp.n> = icmp eq vp<%3>, vp<[[VTC]]>
; CHECK-NEXT:    EMIT branch-on-cond vp<%cmp.n>
entry:
  br label %loop

loop:
  %ptr.iv = phi ptr [ %start, %entry ], [ %gep.iv.next, %loop ]
  %red = phi i32 [ 0, %entry ], [ %red.next, %loop ]
  %l = load i8, ptr %ptr.iv, align 1
  %l.ext = sext i8 %l to i32
  %mul = mul i32 %l.ext, 128
  %red.next = add i32 %red, %mul
  %gep.iv.next = getelementptr i8, ptr %ptr.iv, i64 1
  %ec = icmp eq ptr %ptr.iv, %end
  br i1 %ec, label %exit, label %loop

exit:
  %red.next.lcssa = phi i32 [ %red.next, %loop ]
  ret i32 %red.next.lcssa
}

define i64 @print_ext_mulacc_extended_const(ptr %start, ptr %end) {
; CHECK-LABEL: 'print_ext_mulacc_extended_const'
; CHECK:       VPlan 'Initial VPlan for VF={4},UF>=1' {
; CHECK-NEXT:  Live-in vp<[[VF:%.+]]> = VF
; CHECK-NEXT:  Live-in vp<[[VFxUF:%.+]]> = VF * UF
; CHECK-NEXT:  Live-in vp<[[VTC:%.+]]> = vector-trip-count
; CHECK-NEXT:  vp<%3> = original trip-count
; CHECK-EMPTY:
; CHECK-NEXT:  ir-bb<entry>:
; CHECK-NEXT:    EMIT vp<%3> = EXPAND SCEV (1 + (-1 * (ptrtoint ptr %start to i64)) + (ptrtoint ptr %end to i64))
; CHECK-NEXT:  Successor(s): scalar.ph, vector.ph
; CHECK-EMPTY:
; CHECK-NEXT:  vector.ph:
; CHECK-NEXT:    vp<[[DER_IV:%.+]]> = DERIVED-IV ir<%start> + vp<[[VTC]]> * ir<1>
; CHECK-NEXT:    EMIT vp<[[RDX_START:%.+]]> = reduction-start-vector ir<0>, ir<0>, ir<1>
; CHECK-NEXT:  Successor(s): vector loop
; CHECK-EMPTY:
; CHECK-NEXT:  <x1> vector loop: {
; CHECK-NEXT:    vector.body:
; CHECK-NEXT:      EMIT vp<[[CAN_IV:%.+]]> = CANONICAL-INDUCTION ir<0>, vp<[[IV_NEXT:%.+]]>
; CHECK-NEXT:      WIDEN-REDUCTION-PHI ir<[[RDX:%.+]]> = phi vp<[[RDX_START]]>, vp<[[RDX_NEXT:%.+]]>
; CHECK-NEXT:      vp<[[STEPS:%.+]]> = SCALAR-STEPS vp<[[CAN_IV]]>, ir<1>, vp<[[VF]]>
; CHECK-NEXT:      EMIT vp<%next.gep> = ptradd ir<%start>, vp<[[STEPS]]>
; CHECK-NEXT:      vp<[[VEC_PTR:%.+]]> = vector-pointer vp<%next.gep>
; CHECK-NEXT:      WIDEN ir<%l> = load vp<[[VEC_PTR]]>
; CHECK-NEXT:      EXPRESSION vp<[[RDX_NEXT]]> = ir<[[RDX]]> + reduce.add (mul (ir<%l> zext to i64), (ir<63> zext to i64))
; CHECK-NEXT:      EMIT vp<[[IV_NEXT]]> = add nuw vp<[[CAN_IV]]>, vp<[[VFxUF]]>
; CHECK-NEXT:      EMIT branch-on-count vp<[[IV_NEXT]]>, vp<[[VTC]]>
; CHECK-NEXT:    No successors
; CHECK-NEXT:  }
; CHECK-NEXT:  Successor(s): middle.block
; CHECK-EMPTY:
; CHECK-NEXT:  middle.block:
; CHECK-NEXT:    EMIT vp<%11> = compute-reduction-result ir<[[RDX]]>, vp<[[RDX_NEXT]]>
; CHECK-NEXT:    EMIT vp<%cmp.n> = icmp eq vp<%3>, vp<[[VTC]]>
; CHECK-NEXT:    EMIT branch-on-cond vp<%cmp.n>
entry:
  br label %loop

loop:
  %ptr.iv = phi ptr [ %start, %entry ], [ %gep.iv.next, %loop ]
  %red = phi i64 [ 0, %entry ], [ %red.next, %loop ]
  %l = load i8, ptr %ptr.iv, align 1
  %l.ext = zext i8 %l to i32
  %mul = mul i32 %l.ext, 63
  %mul.ext = zext i32 %mul to i64
  %red.next = add i64 %red, %mul.ext
  %gep.iv.next = getelementptr i8, ptr %ptr.iv, i64 1
  %ec = icmp eq ptr %ptr.iv, %end
  br i1 %ec, label %exit, label %loop

exit:
  ret i64 %red.next
}

; Constants >= 128 cannot be treated as sign-extended, so the expression shouldn't extend 128
define i64 @print_ext_mulacc_not_extended_const(ptr %start, ptr %end) {
; CHECK-LABEL: 'print_ext_mulacc_not_extended_const'
; CHECK:       VPlan 'Initial VPlan for VF={4},UF>=1' {
; CHECK-NEXT:  Live-in vp<[[VF:%.+]]> = VF
; CHECK-NEXT:  Live-in vp<[[VFxUF:%.+]]> = VF * UF
; CHECK-NEXT:  Live-in vp<[[VTC:%.+]]> = vector-trip-count
; CHECK-NEXT:  vp<%3> = original trip-count
; CHECK-EMPTY:
; CHECK-NEXT:  ir-bb<entry>:
; CHECK-NEXT:    EMIT vp<%3> = EXPAND SCEV (1 + (-1 * (ptrtoint ptr %start to i64)) + (ptrtoint ptr %end to i64))
; CHECK-NEXT:  Successor(s): scalar.ph, vector.ph
; CHECK-EMPTY:
; CHECK-NEXT:  vector.ph:
; CHECK-NEXT:    vp<[[DER_IV:%.+]]> = DERIVED-IV ir<%start> + vp<[[VTC]]> * ir<1>
; CHECK-NEXT:    EMIT vp<[[RDX_START:%.+]]> = reduction-start-vector ir<0>, ir<0>, ir<1>
; CHECK-NEXT:  Successor(s): vector loop
; CHECK-EMPTY:
; CHECK-NEXT:  <x1> vector loop: {
; CHECK-NEXT:    vector.body:
; CHECK-NEXT:      EMIT vp<[[CAN_IV:%.+]]> = CANONICAL-INDUCTION ir<0>, vp<[[IV_NEXT:%.+]]>
; CHECK-NEXT:      WIDEN-REDUCTION-PHI ir<[[RDX:%.+]]> = phi vp<[[RDX_START]]>, vp<[[RDX_NEXT:%.+]]>
; CHECK-NEXT:      vp<[[STEPS:%.+]]> = SCALAR-STEPS vp<[[CAN_IV]]>, ir<1>, vp<[[VF]]>
; CHECK-NEXT:      EMIT vp<%next.gep> = ptradd ir<%start>, vp<[[STEPS]]>
; CHECK-NEXT:      vp<[[VEC_PTR:%.+]]> = vector-pointer vp<%next.gep>
; CHECK-NEXT:      WIDEN ir<%l> = load vp<[[VEC_PTR]]>
; CHECK-NEXT:      WIDEN-CAST ir<%l.ext> = sext ir<%l> to i32
; CHECK-NEXT:      WIDEN ir<%mul> = mul ir<%l.ext>, ir<128>
; CHECK-NEXT:      EXPRESSION vp<[[RDX_NEXT]]> = ir<[[RDX]]> + reduce.add (ir<%mul> sext to i64)
; CHECK-NEXT:      EMIT vp<[[IV_NEXT]]> = add nuw vp<[[CAN_IV]]>, vp<[[VFxUF]]>
; CHECK-NEXT:      EMIT branch-on-count vp<[[IV_NEXT]]>, vp<[[VTC]]>
; CHECK-NEXT:    No successors
; CHECK-NEXT:  }
; CHECK-NEXT:  Successor(s): middle.block
; CHECK-EMPTY:
; CHECK-NEXT:  middle.block:
; CHECK-NEXT:    EMIT vp<%11> = compute-reduction-result ir<[[RDX]]>, vp<[[RDX_NEXT]]>
; CHECK-NEXT:    EMIT vp<%cmp.n> = icmp eq vp<%3>, vp<[[VTC]]>
; CHECK-NEXT:    EMIT branch-on-cond vp<%cmp.n>
entry:
  br label %loop

loop:
  %ptr.iv = phi ptr [ %start, %entry ], [ %gep.iv.next, %loop ]
  %red = phi i64 [ 0, %entry ], [ %red.next, %loop ]
  %l = load i8, ptr %ptr.iv, align 1
  %l.ext = sext i8 %l to i32
  %mul = mul i32 %l.ext, 128
  %mul.ext = sext i32 %mul to i64
  %red.next = add i64 %red, %mul.ext
  %gep.iv.next = getelementptr i8, ptr %ptr.iv, i64 1
  %ec = icmp eq ptr %ptr.iv, %end
  br i1 %ec, label %exit, label %loop

exit:
  %red.next.lcssa = phi i64 [ %red.next, %loop ]
  ret i64 %red.next.lcssa
}

; This reduce.add(ext(mul(ext(A), ext(B)))) can't be turned into an
; ExtMulAccReduction VPExpressionRecipe since the mul has two users.
; It can however be turned into an ExtendedReduction since that one doesn't
; modify the mul's operands.
define i64 @print_ext_mul_two_uses(i64 %n, ptr %a, i16 %b, i32 %c) {
; CHECK:       VPlan 'Initial VPlan for VF={4},UF>=1' {
; CHECK-NEXT:  Live-in vp<%0> = VF * UF
; CHECK-NEXT:  Live-in vp<%1> = vector-trip-count
; CHECK-NEXT:  vp<%2> = original trip-count
; CHECK-EMPTY:
; CHECK-NEXT:  ir-bb<entry>:
; CHECK-NEXT:    EMIT vp<%2> = EXPAND SCEV (1 + %n)
; CHECK-NEXT:  Successor(s): scalar.ph, vector.ph
; CHECK-EMPTY:
; CHECK-NEXT:  vector.ph:
; CHECK-NEXT:    EMIT vp<%3> = reduction-start-vector ir<0>, ir<0>, ir<1>
; CHECK-NEXT:    WIDEN-CAST ir<%conv> = sext ir<%b> to i32
; CHECK-NEXT:    WIDEN ir<%mul> = mul ir<%conv>, ir<%conv>
; CHECK-NEXT:  Successor(s): vector loop
; CHECK-EMPTY:
; CHECK-NEXT:  <x1> vector loop: {
; CHECK-NEXT:    vector.body:
; CHECK-NEXT:      EMIT vp<%4> = CANONICAL-INDUCTION ir<0>, vp<%index.next>
; CHECK-NEXT:      WIDEN-REDUCTION-PHI ir<%res2> = phi vp<%3>, vp<%5>
; CHECK-NEXT:      CLONE ir<%load> = load ir<%a>
; CHECK-NEXT:      WIDEN-CAST ir<%load.ext> = sext ir<%load> to i32
; CHECK-NEXT:      WIDEN-CAST ir<%load.ext.ext> = sext ir<%load.ext> to i64
; CHECK-NEXT:      EXPRESSION vp<%5> = ir<%res2> + reduce.add (ir<%mul> zext to i64)
; CHECK-NEXT:      EMIT vp<%index.next> = add nuw vp<%4>, vp<%0>
; CHECK-NEXT:      EMIT branch-on-count vp<%index.next>, vp<%1>
; CHECK-NEXT:    No successors
; CHECK-NEXT:  }
; CHECK-NEXT:  Successor(s): middle.block
; CHECK-EMPTY:
; CHECK-NEXT:  middle.block:
; CHECK-NEXT:    EMIT vp<%7> = compute-reduction-result ir<%res2>, vp<%5>
; CHECK-NEXT:    EMIT vp<%vector.recur.extract> = extract-last-element ir<%load.ext.ext>
; CHECK-NEXT:    EMIT vp<%cmp.n> = icmp eq vp<%2>, vp<%1>
; CHECK-NEXT:    EMIT branch-on-cond vp<%cmp.n>
; CHECK-NEXT:  Successor(s): ir-bb<exit>, scalar.ph
; CHECK-EMPTY:
; CHECK-NEXT:  ir-bb<exit>:
; CHECK-NEXT:    IR   %add.lcssa = phi i64 [ %add, %loop ] (extra operand: vp<%7> from middle.block)
; CHECK-NEXT:  No successors
; CHECK-EMPTY:
; CHECK-NEXT:  scalar.ph:
; CHECK-NEXT:    EMIT-SCALAR vp<%bc.resume.val> = phi [ vp<%1>, middle.block ], [ ir<0>, ir-bb<entry> ]
; CHECK-NEXT:    EMIT-SCALAR vp<%scalar.recur.init> = phi [ vp<%vector.recur.extract>, middle.block ], [ ir<0>, ir-bb<entry> ]
; CHECK-NEXT:    EMIT-SCALAR vp<%bc.merge.rdx> = phi [ vp<%7>, middle.block ], [ ir<0>, ir-bb<entry> ]
; CHECK-NEXT:  Successor(s): ir-bb<loop>
; CHECK-EMPTY:
; CHECK-NEXT:  ir-bb<loop>:
; CHECK-NEXT:    IR   %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ] (extra operand: vp<%bc.resume.val> from scalar.ph)
; CHECK-NEXT:    IR   %res1 = phi i64 [ 0, %entry ], [ %load.ext.ext, %loop ] (extra operand: vp<%scalar.recur.init> from scalar.ph)
; CHECK-NEXT:    IR   %res2 = phi i64 [ 0, %entry ], [ %add, %loop ] (extra operand: vp<%bc.merge.rdx> from scalar.ph)
; CHECK-NEXT:    IR   %load = load i16, ptr %a, align 2
; CHECK-NEXT:    IR   %iv.next = add i64 %iv, 1
; CHECK-NEXT:    IR   %conv = sext i16 %b to i32
; CHECK-NEXT:    IR   %mul = mul i32 %conv, %conv
; CHECK-NEXT:    IR   %mul.ext = zext i32 %mul to i64
; CHECK-NEXT:    IR   %add = add i64 %res2, %mul.ext
; CHECK-NEXT:    IR   %second_use = or i32 %mul, %c
; CHECK-NEXT:    IR   %load.ext = sext i16 %load to i32
; CHECK-NEXT:    IR   %load.ext.ext = sext i32 %load.ext to i64
; CHECK-NEXT:    IR   %exitcond740.not = icmp eq i64 %iv, %n
; CHECK-NEXT:  No successors
; CHECK-NEXT:  }
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %res1 = phi i64 [ 0, %entry ], [ %load.ext.ext, %loop ]
  %res2 = phi i64 [ 0, %entry ], [ %add, %loop ]
  %load = load i16, ptr %a, align 2
  %iv.next = add i64 %iv, 1
  %conv = sext i16 %b to i32
  %mul = mul i32 %conv, %conv
  %mul.ext = zext i32 %mul to i64
  %add = add i64 %res2, %mul.ext
  %second_use = or i32 %mul, %c ; this value is otherwise unused, but that's sufficient for the test
  %load.ext = sext i16 %load to i32
  %load.ext.ext = sext i32 %load.ext to i64
  %exitcond740.not = icmp eq i64 %iv, %n
  br i1 %exitcond740.not, label %exit, label %loop

exit:
  ret i64 %add
}
