; REQUIRES: asserts
; RUN: opt -passes=loop-vectorize -debug-only=loop-vectorize -force-vector-interleave=1 -disable-output %s 2>&1 | FileCheck %s

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64-none-unknown-elf"

; Tests for printing VPlans that are enabled under AArch64

define void @print_partial_reduction(ptr %a, ptr %b) {
; CHECK-LABEL: Checking a loop in 'print_partial_reduction'
; CHECK:      VPlan 'Initial VPlan for VF={2,4},UF>=1' {
; CHECK-NEXT: Live-in vp<[[VFxUF:%.]]> = VF * UF
; CHECK-NEXT: Live-in vp<[[VEC_TC:%.+]]> = vector-trip-count
; CHECK-NEXT: Live-in ir<0> = original trip-count
; CHECK-EMPTY:
; CHECK-NEXT: vector.ph:
; CHECK-NEXT: Successor(s): vector loop
; CHECK-EMPTY:
; CHECK-NEXT: <x1> vector loop: {
; CHECK-NEXT: vector.body:
; CHECK-NEXT:   EMIT vp<[[CAN_IV:%.+]]> = CANONICAL-INDUCTION ir<0>, vp<[[CAN_IV_NEXT:%.+]]>
; CHECK-NEXT:   WIDEN-REDUCTION-PHI ir<[[ACC:%.+]]> = phi ir<0>, ir<%add>
; CHECK-NEXT:   vp<[[STEPS:%.+]]> = SCALAR-STEPS vp<[[CAN_IV]]>, ir<1>
; CHECK-NEXT:   CLONE ir<%arrayidx> = getelementptr ir<%a>, vp<[[STEPS]]>
; CHECK-NEXT:   vp<%4> = vector-pointer ir<%arrayidx>
; CHECK-NEXT:   WIDEN ir<%1> = load vp<%4>
; CHECK-NEXT:   WIDEN-CAST ir<%conv> = zext ir<%1> to i32
; CHECK-NEXT:   CLONE ir<%arrayidx2> = getelementptr ir<%b>, vp<[[STEPS]]>
; CHECK-NEXT:   vp<%5> = vector-pointer ir<%arrayidx2>
; CHECK-NEXT:   WIDEN ir<%2> = load vp<%5>
; CHECK-NEXT:   WIDEN-CAST ir<%conv3> = zext ir<%2> to i32
; CHECK-NEXT:   WIDEN ir<%mul> = mul ir<%conv3>, ir<%conv>
; CHECK-NEXT:   WIDEN ir<%add> = add ir<%mul>, ir<[[ACC]]>
; CHECK-NEXT:   EMIT vp<[[CAN_IV_NEXT]]> = add nuw vp<[[CAN_IV]]>, vp<[[VFxUF]]>
; CHECK-NEXT:   EMIT branch-on-count vp<[[CAN_IV_NEXT]]>, vp<[[VEC_TC]]>
; CHECK-NEXT: No successors
; CHECK-NEXT: }
; CHECK-NEXT: Successor(s): middle.block
; CHECK-EMPTY:
; CHECK-NEXT: middle.block:
; CHECK-NEXT:   EMIT vp<%8> = compute-reduction-result ir<[[ACC]]>, ir<%add>
; CHECK-NEXT:   EMIT vp<%9> = extract-from-end vp<%8>, ir<1>
; CHECK-NEXT:   EMIT vp<%10> = icmp eq ir<0>, vp<%1>
; CHECK-NEXT:   EMIT branch-on-cond vp<%10>
; CHECK-NEXT: Successor(s): ir-bb<for.cond.cleanup.loopexit>, scalar.ph
; CHECK-EMPTY:
; CHECK-NEXT: ir-bb<for.cond.cleanup.loopexit>:
; CHECK-NEXT:   IR   %add.lcssa = phi i32 [ %add, %for.body ] (extra operand: vp<%9>)
; CHECK-NEXT:   IR   %0 = lshr i32 %add.lcssa, 0
; CHECK-NEXT: No successors
; CHECK-EMPTY:
; CHECK-NEXT: scalar.ph:
; CHECK-NEXT: No successors
; CHECK-NEXT: }
; CHECK:      VPlan 'Initial VPlan for VF={8,16},UF>=1' {
; CHECK-NEXT: Live-in vp<[[VFxUF:%.]]> = VF * UF
; CHECK-NEXT: Live-in vp<[[VEC_TC:%.+]]> = vector-trip-count
; CHECK-NEXT: Live-in ir<0> = original trip-count
; CHECK-EMPTY:
; CHECK-NEXT: vector.ph:
; CHECK-NEXT: Successor(s): vector loop
; CHECK-EMPTY:
; CHECK-NEXT: <x1> vector loop: {
; CHECK-NEXT: vector.body:
; CHECK-NEXT:   EMIT vp<[[CAN_IV:%.+]]> = CANONICAL-INDUCTION ir<0>, vp<[[CAN_IV_NEXT:%.+]]>
; CHECK-NEXT:   WIDEN-REDUCTION-PHI ir<[[ACC:%.+]]> = phi ir<0>, ir<%add> (VF scaled by 1/4)
; CHECK-NEXT:   vp<[[STEPS:%.+]]> = SCALAR-STEPS vp<[[CAN_IV]]>, ir<1>
; CHECK-NEXT:   CLONE ir<%arrayidx> = getelementptr ir<%a>, vp<[[STEPS]]>
; CHECK-NEXT:   vp<%4> = vector-pointer ir<%arrayidx>
; CHECK-NEXT:   WIDEN ir<%1> = load vp<%4>
; CHECK-NEXT:   WIDEN-CAST ir<%conv> = zext ir<%1> to i32
; CHECK-NEXT:   CLONE ir<%arrayidx2> = getelementptr ir<%b>, vp<[[STEPS]]>
; CHECK-NEXT:   vp<%5> = vector-pointer ir<%arrayidx2>
; CHECK-NEXT:   WIDEN ir<%2> = load vp<%5>
; CHECK-NEXT:   WIDEN-CAST ir<%conv3> = zext ir<%2> to i32
; CHECK-NEXT:   WIDEN ir<%mul> = mul ir<%conv3>, ir<%conv>
; CHECK-NEXT:   PARTIAL-REDUCE ir<%add> = add ir<%mul>, ir<[[ACC]]>
; CHECK-NEXT:   EMIT vp<[[CAN_IV_NEXT]]> = add nuw vp<[[CAN_IV]]>, vp<[[VFxUF]]>
; CHECK-NEXT:   EMIT branch-on-count vp<[[CAN_IV_NEXT]]>, vp<[[VEC_TC]]>
; CHECK-NEXT: No successors
; CHECK-NEXT: }
; CHECK-NEXT: Successor(s): middle.block
; CHECK-EMPTY:
; CHECK-NEXT: middle.block:
; CHECK-NEXT:   EMIT vp<%8> = compute-reduction-result ir<[[ACC]]>, ir<%add>
; CHECK-NEXT:   EMIT vp<%9> = extract-from-end vp<%8>, ir<1>
; CHECK-NEXT:   EMIT vp<%10> = icmp eq ir<0>, vp<%1>
; CHECK-NEXT:   EMIT branch-on-cond vp<%10>
; CHECK-NEXT: Successor(s): ir-bb<for.cond.cleanup.loopexit>, scalar.ph
; CHECK-EMPTY:
; CHECK-NEXT: ir-bb<for.cond.cleanup.loopexit>:
; CHECK-NEXT:   IR   %add.lcssa = phi i32 [ %add, %for.body ] (extra operand: vp<%9>)
; CHECK-NEXT:   IR   %0 = lshr i32 %add.lcssa, 0
; CHECK-NEXT: No successors
; CHECK-EMPTY:
; CHECK-NEXT: scalar.ph:
; CHECK-NEXT: No successors
; CHECK-NEXT: }
;
entry:
  br label %for.body

for.cond.cleanup.loopexit:                        ; preds = %for.body
  %0 = lshr i32 %add, 0
  ret void

for.body:                                         ; preds = %for.body, %entry
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %acc.010 = phi i32 [ 0, %entry ], [ %add, %for.body ]
  %arrayidx = getelementptr i8, ptr %a, i64 %indvars.iv
  %1 = load i8, ptr %arrayidx, align 1
  %conv = zext i8 %1 to i32
  %arrayidx2 = getelementptr i8, ptr %b, i64 %indvars.iv
  %2 = load i8, ptr %arrayidx2, align 1
  %conv3 = zext i8 %2 to i32
  %mul = mul i32 %conv3, %conv
  %add = add i32 %mul, %acc.010
  %indvars.iv.next = add i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, 0
  br i1 %exitcond.not, label %for.cond.cleanup.loopexit, label %for.body
}
