; RUN: opt -passes=loop-vectorize -force-vector-width=8 -force-vector-interleave=2 -disable-output -debug -S %s 2>&1 | FileCheck --check-prefixes=CHECK %s

target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"

; REQUIRES: asserts

; Check if the vector loop condition can be simplified to true for a given
; VF/IC combination.
define void @test_tc_less_than_16(ptr %A, i64 %N) {
; CHECK:      LV: Scalarizing:  %cmp =
; CHECK:      VPlan 'Initial VPlan for VF={8},UF>=1' {
; CHECK-NEXT: Live-in vp<[[VFxUF:%.+]]> = VF * UF
; CHECK-NEXT: Live-in vp<[[VTC:%.+]]> = vector-trip-count
; CHECK-NEXT: vp<[[TC:%.+]]> = original trip-count
; CHECK-EMPTY:
; CHECK-NEXT: ir-bb<entry>:
; CHECK-NEXT:   IR %and = and i64 %N, 15
; CHECK-NEXT:   EMIT vp<[[TC]]> = EXPAND SCEV (zext i4 (trunc i64 %N to i4) to i64)
; CHECK-NEXT: No successors
; CHECK-EMPTY:
; CHECK-NEXT: vector.ph:
; CHECK-NEXT: Successor(s): vector loop
; CHECK-EMPTY:
; CHECK-NEXT: <x1> vector loop: {
; CHECK-NEXT:   vector.body:
; CHECK-NEXT:     EMIT vp<[[CAN_IV:%.+]]> = CANONICAL-INDUCTION ir<0>, vp<[[CAN_IV_NEXT:%.+]]>
; CHECK-NEXT:     vp<[[STEPS:%.+]]> = SCALAR-STEPS vp<[[CAN_IV]]>, ir<1>
; CHECK-NEXT:     EMIT vp<[[PADD:%.+]]> = ptradd ir<%A>, vp<[[STEPS]]>
; CHECK-NEXT:     vp<[[VPTR:%.]]> = vector-pointer vp<[[PADD]]>
; CHECK-NEXT:     WIDEN ir<%l> = load vp<[[VPTR]]>
; CHECK-NEXT:     WIDEN ir<%add> = add nsw ir<%l>, ir<10>
; CHECK-NEXT:     vp<[[VPTR2:%.+]]> = vector-pointer vp<[[PADD]]>
; CHECK-NEXT:     WIDEN store vp<[[VPTR2]]>, ir<%add>
; CHECK-NEXT:     EMIT vp<[[CAN_IV_NEXT]]> = add nuw vp<[[CAN_IV:%.+]]>, vp<[[VFxUF]]>
; CHECK-NEXT:     EMIT branch-on-count vp<[[CAN_IV_NEXT]]>, vp<[[VTC]]>
; CHECK-NEXT:   No successors
; CHECK-NEXT: }
; CHECK-NEXT: Successor(s): middle.block
; CHECK-EMPTY:
; CHECK-NEXT: middle.block:
; CHECK-NEXT:   EMIT vp<[[C:%.+]]> = icmp eq vp<[[TC]]>, vp<[[VTC]]>
; CHECK-NEXT:   EMIT branch-on-cond vp<[[C]]>
; CHECK-NEXT: Successor(s): ir-bb<exit>, scalar.ph
; CHECK-EMPTY:
; CHECK-NEXT: ir-bb<exit>:
; CHECK-NEXT: No successors
; CHECK-EMPTY:
; CHECK-NEXT: scalar.ph:
; CHECK-NEXT: No successors
; CHECK-NEXT: }
;
; CHECK: Executing best plan with VF=8, UF=2
; CHECK-NEXT: VPlan 'Final VPlan for VF={8},UF={2}' {
; CHECK-NEXT: Live-in vp<[[VFxUF:%.+]]> = VF * UF
; CHECK-NEXT: Live-in vp<[[VTC:%.+]]> = vector-trip-count
; CHECK-NEXT: vp<[[TC:%.+]]> = original trip-count
; CHECK-EMPTY:
; CHECK-NEXT: ir-bb<entry>:
; CHECK-NEXT:   IR %and = and i64 %N, 15
; CHECK-NEXT:   EMIT vp<[[TC]]> = EXPAND SCEV (zext i4 (trunc i64 %N to i4) to i64)
; CHECK-NEXT: No successors
; CHECK-EMPTY:
; CHECK-NEXT: vector.ph:
; CHECK-NEXT: Successor(s): vector loop
; CHECK-EMPTY:
; CHECK-NEXT: <x1> vector loop: {
; CHECK-NEXT:   vector.body:
; CHECK-NEXT:     EMIT vp<[[CAN_IV:%.+]]> = CANONICAL-INDUCTION ir<0>, vp<[[CAN_IV_NEXT:%.+]]>
; CHECK-NEXT:     vp<[[STEPS:%.+]]> = SCALAR-STEPS vp<[[CAN_IV]]>, ir<1>
; CHECK-NEXT:     EMIT vp<[[PADD:%.+]]> = ptradd ir<%A>, vp<[[STEPS]]>
; CHECK-NEXT:     vp<[[VPTR:%.]]> = vector-pointer vp<[[PADD]]>
; CHECK-NEXT:     WIDEN ir<%l> = load vp<[[VPTR]]>
; CHECK-NEXT:     WIDEN ir<%add> = add nsw ir<%l>, ir<10>
; CHECK-NEXT:     vp<[[VPTR2:%.+]]> = vector-pointer vp<[[PADD]]>
; CHECK-NEXT:     WIDEN store vp<[[VPTR2]]>, ir<%add>
; CHECK-NEXT:     EMIT vp<[[CAN_IV_NEXT]]> = add nuw vp<[[CAN_IV:%.+]]>, vp<[[VFxUF]]>
; CHECK-NEXT:     EMIT branch-on-cond ir<true>
; CHECK-NEXT:   No successors
; CHECK-NEXT: }
; CHECK-NEXT: Successor(s): middle.block
; CHECK-EMPTY:
; CHECK-NEXT: middle.block:
; CHECK-NEXT:   EMIT vp<[[C:%.+]]> = icmp eq vp<[[TC]]>, vp<[[VTC]]>
; CHECK-NEXT:   EMIT branch-on-cond vp<[[C]]>
; CHECK-NEXT: Successor(s): ir-bb<exit>, scalar.ph
; CHECK-EMPTY:
; CHECK-NEXT: ir-bb<exit>:
; CHECK-NEXT: No successors
; CHECK-EMPTY:
; CHECK-NEXT: scalar.ph:
; CHECK-NEXT: No successors
; CHECK-NEXT: }
;
entry:
  %and = and i64 %N, 15
  br label %loop

loop:
  %iv = phi i64 [ %and, %entry ], [ %iv.next, %loop ]
  %p.src = phi ptr [ %A, %entry ], [ %p.src.next, %loop ]
  %p.src.next = getelementptr inbounds i8, ptr %p.src, i64 1
  %l = load i8, ptr %p.src, align 1
  %add = add nsw i8 %l, 10
  store i8 %add, ptr %p.src
  %iv.next = add nsw i64 %iv, -1
  %cmp = icmp eq i64 %iv.next, 0
  br i1 %cmp, label %exit, label %loop

exit:
  ret void
}
