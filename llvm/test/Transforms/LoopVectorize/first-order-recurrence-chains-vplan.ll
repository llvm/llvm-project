; REQUIRES: asserts

; RUN: opt -passes=loop-vectorize -force-vector-width=4 -force-vector-interleave=1 -debug-only=loop-vectorize -disable-output -S %s 2>&1 | FileCheck %s

define void @test_chained_first_order_recurrences_1(ptr %ptr) {
; CHECK-LABEL: 'test_chained_first_order_recurrences_1'
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
; CHECK-NEXT:     FIRST-ORDER-RECURRENCE-PHI ir<%for.2> = phi ir<33>, vp<[[FOR1_SPLICE:%.+]]>
; CHECK-NEXT:     vp<[[STEPS:%.+]]>    = SCALAR-STEPS vp<[[CAN_IV]]>, ir<1>
; CHECK-NEXT:     CLONE ir<%gep.ptr> = getelementptr inbounds ir<%ptr>, vp<[[STEPS]]>
; CHECK-NEXT:     vp<[[VEC_PTR:%.+]]> = vector-pointer ir<%gep.ptr>
; CHECK-NEXT:     WIDEN ir<%for.1.next> = load vp<[[VEC_PTR]]>
; CHECK-NEXT:     EMIT vp<[[FOR1_SPLICE]]> = first-order splice ir<%for.1>, ir<%for.1.next>
; CHECK-NEXT:     EMIT vp<[[FOR2_SPLICE:%.+]]> = first-order splice ir<%for.2>, vp<[[FOR1_SPLICE]]>
; CHECK-NEXT:     WIDEN ir<%add> = add vp<[[FOR1_SPLICE]]>, vp<[[FOR2_SPLICE]]>
; CHECK-NEXT:     vp<[[VEC_PTR2:%.+]]> = vector-pointer ir<%gep.ptr>
; CHECK-NEXT:     WIDEN store vp<[[VEC_PTR2]]>, ir<%add>
; CHECK-NEXT:     EMIT vp<[[CAN_IV_NEXT:%.+]]> = add nuw vp<[[CAN_IV]]>, vp<[[VFxUF]]>
; CHECK-NEXT:     EMIT branch-on-count vp<[[CAN_IV_NEXT]]>, vp<[[VTC]]>
; CHECK-NEXT:   No successors
; CHECK-NEXT: }
; CHECK-NEXT: Successor(s): middle.block
; CHECK-EMPTY:
; CHECK-NEXT: middle.block:
; CHECK-NEXT:    EMIT vp<[[RESUME_1:%.+]]> = extract-from-end ir<%for.1.next>, ir<1>
; CHECK-NEXT:    EMIT vp<[[RESUME_2:%.+]]> = extract-from-end vp<[[FOR1_SPLICE]]>, ir<1>
; CHECK-NEXT:    EMIT vp<[[CMP:%.+]]> = icmp eq ir<1000>, vp<[[VTC]]>
; CHECK-NEXT:    EMIT branch-on-cond vp<[[CMP]]>
; CHECK-NEXT:  Successor(s): ir-bb<exit>, scalar.ph
; CHECK-EMPTY:
; CHECK-NEXT:  ir-bb<exit>
; CHECK-NEXT:  No successors
; CHECK-EMPTY:
; CHECK-NEXT:  scalar.ph
; CHECK-NEXT:    EMIT vp<[[RESUME_1_P:%.*]]> = resume-phi vp<[[RESUME_1]]>, ir<22>
; CHECK-NEXT:    EMIT vp<[[RESUME_2_P:%.*]]> = resume-phi vp<[[RESUME_2]]>, ir<33>
; CHECK-NEXT:  No successors
; CHECK-EMPTY:
; CHECK-NEXT: Live-out i16 %for.1 = vp<[[RESUME_1_P]]>
; CHECK-NEXT: Live-out i16 %for.2 = vp<[[RESUME_2_P]]>
; CHECK-NEXT: }
;
entry:
  br label %loop

loop:
  %for.1 = phi i16 [ 22, %entry ], [ %for.1.next, %loop ]
  %for.2 = phi i16 [ 33, %entry ], [ %for.1, %loop ]
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %iv.next = add nuw nsw i64 %iv, 1
  %gep.ptr = getelementptr inbounds i16, ptr %ptr, i64 %iv
  %for.1.next = load i16, ptr %gep.ptr, align 2
  %add = add i16 %for.1, %for.2
  store i16 %add, ptr %gep.ptr
  %exitcond.not = icmp eq i64 %iv.next, 1000
  br i1 %exitcond.not, label %exit, label %loop

exit:
  ret void
}

define void @test_chained_first_order_recurrences_3(ptr %ptr) {
; CHECK-LABEL: 'test_chained_first_order_recurrences_3'
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
; CHECK-NEXT:     FIRST-ORDER-RECURRENCE-PHI ir<%for.2> = phi ir<33>, vp<[[FOR1_SPLICE:%.+]]>
; CHECK-NEXT:     FIRST-ORDER-RECURRENCE-PHI ir<%for.3> = phi ir<33>, vp<[[FOR2_SPLICE:%.+]]>
; CHECK-NEXT:     vp<[[STEPS:%.+]]>    = SCALAR-STEPS vp<[[CAN_IV]]>, ir<1>
; CHECK-NEXT:     CLONE ir<%gep.ptr> = getelementptr inbounds ir<%ptr>, vp<[[STEPS]]>
; CHECK-NEXT:     vp<[[VEC_PTR:%.+]]> = vector-pointer ir<%gep.ptr>
; CHECK-NEXT:     WIDEN ir<%for.1.next> = load vp<[[VEC_PTR]]>
; CHECK-NEXT:     EMIT vp<[[FOR1_SPLICE]]> = first-order splice ir<%for.1>, ir<%for.1.next>
; CHECK-NEXT:     EMIT vp<[[FOR2_SPLICE]]> = first-order splice ir<%for.2>, vp<[[FOR1_SPLICE]]>
; CHECK-NEXT:     EMIT vp<[[FOR3_SPLICE:%.+]]> = first-order splice ir<%for.3>, vp<[[FOR2_SPLICE]]>
; CHECK-NEXT:     WIDEN ir<%add.1> = add vp<[[FOR1_SPLICE]]>, vp<[[FOR2_SPLICE]]>
; CHECK-NEXT:     WIDEN ir<%add.2> = add ir<%add.1>, vp<[[FOR3_SPLICE]]>
; CHECK-NEXT:     vp<[[VEC_PTR2:%.+]]> = vector-pointer ir<%gep.ptr>
; CHECK-NEXT:     WIDEN store vp<[[VEC_PTR2]]>, ir<%add.2>
; CHECK-NEXT:     EMIT vp<[[CAN_IV_NEXT:%.+]]> = add nuw vp<[[CAN_IV]]>, vp<[[VFxUF]]>
; CHECK-NEXT:     EMIT branch-on-count vp<[[CAN_IV_NEXT]]>, vp<[[VTC]]>
; CHECK-NEXT:   No successors
; CHECK-NEXT: }
; CHECK-NEXT: Successor(s): middle.block
; CHECK-EMPTY:
; CHECK-NEXT: middle.block:
; CHECK-NEXT:    EMIT vp<[[RESUME_1:%.+]]> = extract-from-end ir<%for.1.next>, ir<1>
; CHECK-NEXT:    EMIT vp<[[RESUME_2:%.+]]> = extract-from-end vp<[[FOR1_SPLICE]]>, ir<1>
; CHECK-NEXT:    EMIT vp<[[RESUME_3:%.+]]> = extract-from-end vp<[[FOR2_SPLICE]]>, ir<1>
; CHECK-NEXT:    EMIT vp<[[CMP:%.+]]> = icmp eq ir<1000>, vp<[[VTC]]>
; CHECK-NEXT:    EMIT branch-on-cond vp<[[CMP]]>
; CHECK-NEXT:  Successor(s): ir-bb<exit>, scalar.ph
; CHECK-EMPTY:
; CHECK-NEXT:  ir-bb<exit>
; CHECK-NEXT:  No successors
; CHECK-EMPTY:
; CHECK-NEXT:  scalar.ph
; CHECK-NEXT:    EMIT vp<[[RESUME_1_P:%.*]]> = resume-phi vp<[[RESUME_1]]>, ir<22>
; CHECK-NEXT:    EMIT vp<[[RESUME_2_P:%.*]]> = resume-phi vp<[[RESUME_2]]>, ir<33>
; CHECK-NEXT:    EMIT vp<[[RESUME_3_P:%.*]]> = resume-phi vp<[[RESUME_3]]>, ir<33>
; CHECK-NEXT: No successors
; CHECK-EMPTY:
; CHECK-NEXT: Live-out i16 %for.1 = vp<[[RESUME_1_P]]>
; CHECK-NEXT: Live-out i16 %for.2 = vp<[[RESUME_2_P]]>
; CHECK-NEXT: Live-out i16 %for.3 = vp<[[RESUME_3_P]]>
; CHECK-NEXT: }
;
entry:
  br label %loop

loop:
  %for.1 = phi i16 [ 22, %entry ], [ %for.1.next, %loop ]
  %for.2 = phi i16 [ 33, %entry ], [ %for.1, %loop ]
  %for.3 = phi i16 [ 33, %entry ], [ %for.2, %loop ]
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %iv.next = add nuw nsw i64 %iv, 1
  %gep.ptr = getelementptr inbounds i16, ptr %ptr, i64 %iv
  %for.1.next = load i16, ptr %gep.ptr, align 2
  %add.1 = add i16 %for.1, %for.2
  %add.2 = add i16 %add.1, %for.3
  store i16 %add.2, ptr %gep.ptr
  %exitcond.not = icmp eq i64 %iv.next, 1000
  br i1 %exitcond.not, label %exit, label %loop

exit:
  ret void
}

; This test has two FORs (for.x and for.y) where incoming value from the previous
; iteration (for.x.prev) of one FOR (for.y) depends on another FOR (for.x). Due to
; this dependency all uses of the former FOR (for.y) should be sunk after
; incoming value from the previous iteration (for.x.prev) of te latter FOR (for.y).
; That means side-effecting user (store i64 %for.y.i64, ptr %gep) of the latter
; FOR (for.y) should be moved which is not currently supported.
define i32 @test_chained_first_order_recurrences_4(ptr %base) {
; CHECK-LABEL: 'test_chained_first_order_recurrences_4'
; CHECK: No VPlans built.

entry:
  br label %loop

ret:
  ret i32 0

loop:
  %iv = phi i64 [ %iv.next, %loop ], [ 0, %entry ]
  %for.x = phi i64 [ %for.x.next, %loop ], [ 0, %entry ]
  %for.y = phi i32 [ %for.x.prev, %loop ], [ 0, %entry ]
  %iv.next = add i64 %iv, 1
  %gep = getelementptr i64, ptr %base, i64 %iv
  %for.x.prev = trunc i64 %for.x to i32
  %for.y.i64 = sext i32 %for.y to i64
  store i64 %for.y.i64, ptr %gep
  %for.x.next = mul i64 0, 0
  %icmp = icmp ugt i64 %iv, 4096
  br i1 %icmp, label %ret, label %loop
}
