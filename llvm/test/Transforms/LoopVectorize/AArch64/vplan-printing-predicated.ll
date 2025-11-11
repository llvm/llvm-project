; REQUIRES: asserts
; RUN: opt -passes=loop-vectorize -debug-only=loop-vectorize -mattr=+sve2 -force-vector-width=4 -prefer-predicate-over-epilogue=predicate-else-scalar-epilogue -force-tail-folding-style=data-and-control -disable-output %s 2>&1 | FileCheck %s

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64-none-unknown-elf"

; Tests for printing predicated VPlans.

define dso_local void @alias_mask(ptr noalias %a, ptr %b, ptr %c, i32 %n) {
; CHECK-LABEL: 'alias_mask'
; CHECK: VPlan 'Final VPlan for VF={4},UF={1}' {
; CHECK-NEXT: Live-in ir<%wide.trip.count> = original trip-count
; CHECK-EMPTY:
; CHECK-NEXT: ir-bb<for.body.preheader>:
; CHECK-NEXT:   IR   %wide.trip.count = zext nneg i32 %n to i64
; CHECK-NEXT: Successor(s): ir-bb<vector.memcheck>
; CHECK-EMPTY:
; CHECK-NEXT: ir-bb<vector.memcheck>:
; CHECK-NEXT:   IR   %0 = sub i64 %c1, %b2
; CHECK-NEXT:   IR   %diff.check = icmp ult i64 %0, 4
; CHECK-NEXT:   EMIT vp<[[ALIAS_MASK:%.+]]> = ALIAS-LANE-MASK ir<%b2>, ir<%c3> (write-after-read)
; CHECK-NEXT:   EMIT vp<[[POPCOUNT:%.+]]> = popcount vp<[[ALIAS_MASK]]>
; CHECK-NEXT:   EMIT vp<%4> = icmp eq vp<[[POPCOUNT]]>, ir<0>
; CHECK-NEXT:   EMIT branch-on-cond vp<%4>
; CHECK-NEXT: Successor(s): ir-bb<scalar.ph>, vector.ph
; CHECK-EMPTY:
; CHECK-NEXT: vector.ph:
; CHECK-NEXT:   EMIT vp<%active.lane.mask.entry> = active lane mask ir<0>, ir<%wide.trip.count>, ir<1>
; CHECK-NEXT: Successor(s): vector.body
; CHECK-EMPTY:
; CHECK-NEXT: vector.body:
; CHECK-NEXT:   EMIT-SCALAR vp<%index> = phi [ ir<0>, vector.ph ], [ vp<%index.next>, vector.body ]
; CHECK-NEXT:   ACTIVE-LANE-MASK-PHI vp<[[ACTIVE_LANE_MASK:%.+]]> = phi vp<%active.lane.mask.entry>, vp<%active.lane.mask.next>
; CHECK-NEXT:   CLONE ir<%arrayidx> = getelementptr inbounds ir<%a>, vp<%index>
; CHECK-NEXT:   EMIT vp<[[MASK:%.+]]> = and vp<[[ACTIVE_LANE_MASK]]>, vp<[[ALIAS_MASK]]>
; CHECK-NEXT:   WIDEN ir<%1> = load ir<%arrayidx>, vp<[[MASK]]>
; CHECK-NEXT:   CLONE ir<%arrayidx2> = getelementptr inbounds ir<%b>, vp<%index>
; CHECK-NEXT:   WIDEN ir<[[ALIAS_MASK]]> = load ir<%arrayidx2>, vp<[[MASK]]>
; CHECK-NEXT:   WIDEN ir<%add> = add ir<[[ALIAS_MASK]]>, ir<%1>
; CHECK-NEXT:   CLONE ir<%arrayidx6> = getelementptr inbounds ir<%c>, vp<%index>
; CHECK-NEXT:   WIDEN store ir<%arrayidx6>, ir<%add>, vp<[[MASK]]>
; CHECK-NEXT:   EMIT vp<%index.next> = add vp<%index>, vp<[[POPCOUNT]]>
; CHECK-NEXT:   EMIT vp<%active.lane.mask.next> = active lane mask vp<%index.next>, ir<%wide.trip.count>, ir<1>
; CHECK-NEXT:   EMIT vp<%8> = not vp<%active.lane.mask.next>
; CHECK-NEXT:   EMIT branch-on-cond vp<%8>
; CHECK-NEXT: Successor(s): middle.block, vector.body
; CHECK-EMPTY:
; CHECK-NEXT: middle.block:
; CHECK-NEXT: Successor(s): ir-bb<for.cond.cleanup.loopexit>
; CHECK-EMPTY:
; CHECK-NEXT: ir-bb<for.cond.cleanup.loopexit>:
; CHECK-NEXT: No successors
; CHECK-EMPTY:
; CHECK-NEXT: ir-bb<scalar.ph>:
; CHECK-NEXT: Successor(s): ir-bb<for.body>
; CHECK-EMPTY:
; CHECK-NEXT: ir-bb<for.body>:
; CHECK-NEXT:   IR   %indvars.iv = phi i64 [ 0, %scalar.ph ], [ %indvars.iv.next, %for.body ] (extra operand: ir<0> from ir-bb<scalar.ph>)
; CHECK-NEXT:   IR   %arrayidx = getelementptr inbounds i8, ptr %a, i64 %indvars.iv
; CHECK-NEXT:   IR   %1 = load i8, ptr %arrayidx, align 1
; CHECK-NEXT:   IR   %arrayidx2 = getelementptr inbounds i8, ptr %b, i64 %indvars.iv
; CHECK-NEXT:   IR   %2 = load i8, ptr %arrayidx2, align 1
; CHECK-NEXT:   IR   %add = add i8 %2, %1
; CHECK-NEXT:   IR   %arrayidx6 = getelementptr inbounds i8, ptr %c, i64 %indvars.iv
; CHECK-NEXT:   IR   store i8 %add, ptr %arrayidx6, align 1
; CHECK-NEXT:   IR   %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
; CHECK-NEXT:   IR   %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
; CHECK-NEXT: No successors
; CHECK-NEXT: }
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
