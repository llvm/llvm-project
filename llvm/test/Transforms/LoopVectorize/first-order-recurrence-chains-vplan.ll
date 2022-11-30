; REQUIRES: asserts

; RUN: opt -passes=loop-vectorize -force-vector-width=4 -force-vector-interleave=1 -debug-only=loop-vectorize -disable-output -S %s 2>&1 | FileCheck %s

define void @test_chained_first_order_recurrences_1(i16* %ptr) {
; CHECK-LABEL: 'test_chained_first_order_recurrences_1'
; CHECK:      VPlan 'Initial VPlan for VF={4},UF>=1' {
; CHECK-NEXT: Live-in vp<%1> = vector-trip-count
; CHECK-EMPTY:
; CHECK-NEXT: vector.ph:
; CHECK-NEXT: Successor(s): vector loop
; CHECK-EMPTY:
; CHECK-NEXT: <x1> vector loop: {
; CHECK-NEXT:   vector.body:
; CHECK-NEXT:     EMIT vp<%2> = CANONICAL-INDUCTION
; CHECK-NEXT:     FIRST-ORDER-RECURRENCE-PHI ir<%for.1> = phi ir<22>, ir<%for.1.next>
; CHECK-NEXT:     FIRST-ORDER-RECURRENCE-PHI ir<%for.2> = phi ir<33>, vp<%8>
; CHECK-NEXT:     vp<%5>    = SCALAR-STEPS vp<%2>, ir<1>
; CHECK-NEXT:     CLONE ir<%gep.ptr> = getelementptr ir<%ptr>, vp<%5>
; CHECK-NEXT:     WIDEN ir<%for.1.next> = load ir<%gep.ptr>
; CHECK-NEXT:     EMIT vp<%8> = first-order splice ir<%for.1> ir<%for.1.next>
; CHECK-NEXT:     EMIT vp<%9> = first-order splice ir<%for.2> vp<%8>
; CHECK-NEXT:     WIDEN ir<%add> = add vp<%8>, vp<%9>
; CHECK-NEXT:     WIDEN store ir<%gep.ptr>, ir<%add>
; CHECK-NEXT:     EMIT vp<%11> = VF * UF +(nuw)  vp<%2>
; CHECK-NEXT:     EMIT branch-on-count  vp<%11> vp<%1>
; CHECK-NEXT:   No successors
; CHECK-NEXT: }
; CHECK-NEXT: Successor(s): middle.block
; CHECK-EMPTY:
; CHECK-NEXT: middle.block:
; CHECK-NEXT: No successors
; CHECK-NEXT: }
;
entry:
  br label %loop

loop:
  %for.1 = phi i16 [ 22, %entry ], [ %for.1.next, %loop ]
  %for.2 = phi i16 [ 33, %entry ], [ %for.1, %loop ]
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %iv.next = add nuw nsw i64 %iv, 1
  %gep.ptr = getelementptr inbounds i16, i16* %ptr, i64 %iv
  %for.1.next = load i16, i16* %gep.ptr, align 2
  %add = add i16 %for.1, %for.2
  store i16 %add, i16* %gep.ptr
  %exitcond.not = icmp eq i64 %iv.next, 1000
  br i1 %exitcond.not, label %exit, label %loop

exit:
  ret void
}

define void @test_chained_first_order_recurrences_3(i16* %ptr) {
; CHECK-LABEL: 'test_chained_first_order_recurrences_3'
; CHECK:      VPlan 'Initial VPlan for VF={4},UF>=1' {
; CHECK-NEXT: Live-in vp<%1> = vector-trip-count
; CHECK-EMPTY:
; CHECK-NEXT: vector.ph:
; CHECK-NEXT: Successor(s): vector loop
; CHECK-EMPTY:
; CHECK-NEXT: <x1> vector loop: {
; CHECK-NEXT:   vector.body:
; CHECK-NEXT:     EMIT vp<%2> = CANONICAL-INDUCTION
; CHECK-NEXT:     FIRST-ORDER-RECURRENCE-PHI ir<%for.1> = phi ir<22>, ir<%for.1.next>
; CHECK-NEXT:     FIRST-ORDER-RECURRENCE-PHI ir<%for.2> = phi ir<33>, vp<%9>
; CHECK-NEXT:     FIRST-ORDER-RECURRENCE-PHI ir<%for.3> = phi ir<33>, vp<%10>
; CHECK-NEXT:     vp<%6>    = SCALAR-STEPS vp<%2>, ir<1>
; CHECK-NEXT:     CLONE ir<%gep.ptr> = getelementptr ir<%ptr>, vp<%6>
; CHECK-NEXT:     WIDEN ir<%for.1.next> = load ir<%gep.ptr>
; CHECK-NEXT:     EMIT vp<%9> = first-order splice ir<%for.1> ir<%for.1.next>
; CHECK-NEXT:     EMIT vp<%10> = first-order splice ir<%for.2> vp<%9>
; CHECK-NEXT:     EMIT vp<%11> = first-order splice ir<%for.3> vp<%10>
; CHECK-NEXT:     WIDEN ir<%add.1> = add vp<%9>, vp<%10>
; CHECK-NEXT:     WIDEN ir<%add.2> = add ir<%add.1>, vp<%11>
; CHECK-NEXT:     WIDEN store ir<%gep.ptr>, ir<%add.2>
; CHECK-NEXT:     EMIT vp<%14> = VF * UF +(nuw)  vp<%2>
; CHECK-NEXT:     EMIT branch-on-count  vp<%14> vp<%1>
; CHECK-NEXT:   No successors
; CHECK-NEXT: }
; CHECK-NEXT: Successor(s): middle.block
; CHECK-EMPTY:
; CHECK-NEXT: middle.block:
; CHECK-NEXT: No successors
; CHECK-NEXT: }

; CHECK-NOT: vector.body:
;
entry:
  br label %loop

loop:
  %for.1 = phi i16 [ 22, %entry ], [ %for.1.next, %loop ]
  %for.2 = phi i16 [ 33, %entry ], [ %for.1, %loop ]
  %for.3 = phi i16 [ 33, %entry ], [ %for.2, %loop ]
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %iv.next = add nuw nsw i64 %iv, 1
  %gep.ptr = getelementptr inbounds i16, i16* %ptr, i64 %iv
  %for.1.next = load i16, i16* %gep.ptr, align 2
  %add.1 = add i16 %for.1, %for.2
  %add.2 = add i16 %add.1, %for.3
  store i16 %add.2, i16* %gep.ptr
  %exitcond.not = icmp eq i64 %iv.next, 1000
  br i1 %exitcond.not, label %exit, label %loop

exit:
  ret void
}
