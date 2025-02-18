; REQUIRES: asserts
; RUN: opt -passes=loop-vectorize -force-vector-interleave=1 -force-vector-width=8 -S -debug %s 2>&1 | FileCheck %s

define void @iv_no_binary_op_in_descriptor(i1 %c, ptr %dst) {
; CHECK-LABEL: LV: Checking a loop in 'iv_no_binary_op_in_descriptor'
; CHECK:      VPlan 'Initial VPlan for VF={8},UF>=1' {
; CHECK-NEXT: Live-in vp<[[VF:%.+]]> = VF
; CHECK-NEXT: Live-in vp<[[VFxUF:%.+]]> = VF * UF
; CHECK-NEXT: Live-in vp<[[VEC_TC:%.+]]> = vector-trip-count
; CHECK-NEXT: Live-in ir<1000> = original trip-count
; CHECK-EMPTY:
; CHECK-NEXT: ir-bb<entry>:
; CHECK-NEXT: Successor(s): vector.ph
; CHECK-EMPTY:
; CHECK-NEXT: vector.ph:
; CHECK-NEXT: Successor(s): vector loop
; CHECK-EMPTY:
; CHECK-NEXT: <x1> vector loop: {
; CHECK-NEXT:   vector.body:
; CHECK-NEXT:     EMIT vp<[[CAN_IV:%.+]]> = CANONICAL-INDUCTION
; CHECK-NEXT:     ir<%iv> = WIDEN-INDUCTION ir<0>, ir<1>, vp<[[VF]]>
; CHECK-NEXT:     vp<[[STEPS:%.+]]>    = SCALAR-STEPS vp<[[CAN_IV]]>, ir<1>
; CHECK-NEXT:     CLONE ir<%gep> = getelementptr inbounds ir<%dst>, vp<[[STEPS:%.+]]>
; CHECK-NEXT:     vp<[[VEC_PTR:%.+]]> = vector-pointer ir<%gep>
; CHECK-NEXT:     WIDEN store vp<[[VEC_PTR]]>, ir<%iv>
; CHECK-NEXT:     EMIT vp<[[CAN_INC:%.+]]> = add nuw vp<[[CAN_IV]]>, vp<[[VFxUF]]>
; CHECK-NEXT:     EMIT branch-on-count vp<[[CAN_INC]]>, vp<[[VEC_TC]]>
; CHECK-NEXT:   No successors
; CHECK-NEXT: }
; CHECK-NEXT: Successor(s): middle.block
; CHECK-EMPTY:
 ; CHECK-NEXT: middle.block:
; CHECK-NEXT:    EMIT vp<[[CMP:%.+]]> = icmp eq ir<1000>, vp<[[VEC_TC]]>
; CHECK-NEXT:    EMIT branch-on-cond vp<[[CMP]]>
; CHECK-NEXT:  Successor(s): ir-bb<exit>, scalar.ph
; CHECK-EMPTY:
; CHECK-NEXT:  scalar.ph:
; CHECK-NEXT:    EMIT vp<[[RESUME:%.+]]> = resume-phi vp<[[VEC_TC]]>, ir<0>
; CHECK-NEXT:  Successor(s): ir-bb<loop.header>
; CHECK-EMPTY:
; CHECK-NEXT:  ir-bb<loop.header>:
; CHECK-NEXT:    IR   %iv = phi i64 [ 0, %entry ], [ %iv.next.p, %loop.latch ] (extra operand: vp<[[RESUME]]> from scalar.ph)
; CHECK:         IR   %iv.next = add i64 %iv, 1
; CHECK-NEXT:  No successors
; CHECK-EMPTY:
; CHECK-NEXT:  ir-bb<exit>:
; CHECK-NEXT:  No successors
; CHECK-NEXT: }
;
entry:
  br label %loop.header

loop.header:
  %iv = phi i64 [ 0, %entry ], [ %iv.next.p, %loop.latch ]
  %gep = getelementptr inbounds i64, ptr %dst, i64 %iv
  store i64 %iv, ptr %gep, align 8
  %iv.next = add i64 %iv, 1
  br label %loop.latch

loop.latch:
  %iv.next.p = phi i64 [ %iv.next, %loop.header ]
  %exitcond.not = icmp eq i64 %iv.next.p, 1000
  br i1 %exitcond.not, label %exit, label %loop.header

exit:
  ret void
}

; Check that VPWidenIntOrFPInductionRecipe is expanded into smaller recipes in
; the final VPlan.
define void @iv_expand(ptr %p, i64 %n) {
; CHECK-LABEL: LV: Checking a loop in 'iv_expand'
; CHECK:      VPlan 'Initial VPlan for VF={8},UF>=1' {
; CHECK:      <x1> vector loop: {
; CHECK-NEXT:   vector.body:
; CHECK-NEXT:     EMIT vp<%3> = CANONICAL-INDUCTION ir<0>, vp<%index.next>
; CHECK-NEXT:     ir<%i> = WIDEN-INDUCTION  ir<0>, ir<1>, vp<%0>
; CHECK-NEXT:     vp<%4> = SCALAR-STEPS vp<%3>, ir<1>
; CHECK-NEXT:     CLONE ir<%q> = getelementptr ir<%p>, vp<%4>
; CHECK-NEXT:     vp<%5> = vector-pointer ir<%q>
; CHECK-NEXT:     WIDEN ir<%x> = load vp<%5>
; CHECK-NEXT:     WIDEN ir<%y> = add ir<%x>, ir<%i>
; CHECK-NEXT:     vp<%6> = vector-pointer ir<%q>
; CHECK-NEXT:     WIDEN store vp<%6>, ir<%y>
; CHECK-NEXT:     EMIT vp<%index.next> = add nuw vp<%3>, vp<%1>
; CHECK-NEXT:     EMIT branch-on-count vp<%index.next>, vp<%2>
; CHECK-NEXT:   No successors
; CHECK-NEXT: }
; CHECK-NEXT: Successor(s): middle.block
; CHECK:      VPlan 'Final VPlan for VF={8},UF={1}'
; CHECK:      <x1> vector loop: {
; CHECK-NEXT:   vector.body:
; CHECK-NEXT:     SCALAR-PHI vp<%3> = phi ir<0>, vp<%index.next>
; CHECK-NEXT:     WIDEN-PHI ir<%i> = phi vp<%induction>, vp<%vec.ind.next>
; CHECK-NEXT:     vp<%4> = SCALAR-STEPS vp<%3>, ir<1>
; CHECK-NEXT:     CLONE ir<%q> = getelementptr ir<%p>, vp<%4>
; CHECK-NEXT:     vp<%5> = vector-pointer ir<%q>
; CHECK-NEXT:     WIDEN ir<%x> = load vp<%5>
; CHECK-NEXT:     WIDEN ir<%y> = add ir<%x>, ir<%i>
; CHECK-NEXT:     vp<%6> = vector-pointer ir<%q>
; CHECK-NEXT:     WIDEN store vp<%6>, ir<%y>
; CHECK-NEXT:     EMIT vp<%index.next> = add nuw vp<%3>, ir<8>
; CHECK-NEXT:     EMIT vp<%vec.ind.next> = add ir<%i>, vp<%2>
; CHECK-NEXT:     EMIT branch-on-count vp<%index.next>, ir<%n.vec>
; CHECK-NEXT:   No successors
; CHECK-NEXT: }
; CHECK-NEXT: Successor(s): middle.block
entry:
  br label %loop
loop:
  %i = phi i64 [0, %entry], [%i.next, %loop]
  %q = getelementptr i64, ptr %p, i64 %i
  %x = load i64, ptr %q
  %y = add i64 %x, %i
  store i64 %y, ptr %q
  %i.next = add i64 %i, 1
  %done = icmp eq i64 %i.next, %n
  br i1 %done, label %exit, label %loop
exit:
  ret void
}
