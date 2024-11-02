; REQUIRES: asserts
; RUN: opt -passes=loop-vectorize -force-vector-interleave=1 -force-vector-width=8 -S -debug %s 2>&1 | FileCheck %s

define void @iv_no_binary_op_in_descriptor(i1 %c, ptr %dst) {
; CHECK-LABEL: LV: Checking a loop in 'iv_no_binary_op_in_descriptor'
; CHECK:      VPlan 'Initial VPlan for VF={8},UF>=1' {
; CHECK-NEXT: Live-in vp<[[VFxUF:%.+]]> = VF * UF
; CHECK-NEXT: Live-in vp<[[VEC_TC:%.+]]> = vector-trip-count
; CHECK-NEXT: Live-in ir<1000> = original trip-count
; CHECK-EMPTY:
; CHECK-NEXT: vector.ph:
; CHECK-NEXT: Successor(s): vector loop
; CHECK-EMPTY:
; CHECK-NEXT: <x1> vector loop: {
; CHECK-NEXT:   vector.body:
; CHECK-NEXT:     EMIT vp<[[CAN_IV:%.+]]> = CANONICAL-INDUCTION
; CHECK-NEXT:     WIDEN-INDUCTION %iv = phi 0, %iv.next.p, ir<1>
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
; CHECK-NEXT:  ir-bb<exit>:
; CHECK-NEXT:  No successors
; CHECK-EMPTY:
; CHECK-NEXT:  scalar.ph:
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
