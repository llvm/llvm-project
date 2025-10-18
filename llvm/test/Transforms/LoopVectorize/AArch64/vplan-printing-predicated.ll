; REQUIRES: asserts
; RUN: opt -passes=loop-vectorize -debug-only=loop-vectorize -mattr=+sve2 -force-vector-width=4 -prefer-predicate-over-epilogue=predicate-else-scalar-epilogue -force-tail-folding-style=data-and-control -disable-output %s 2>&1 | FileCheck %s

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64-none-unknown-elf"

; Tests for printing predicated VPlans.

define dso_local void @alias_mask(ptr noalias %a, ptr %b, ptr %c, i32 %n) {
; CHECK-LABEL: 'alias_mask'
; CHECK:      VPlan 'Initial VPlan for VF={4},UF>=1' {
; CHECK-NEXT: Live-in vp<%0> = VF
; CHECK-NEXT: vp<%3> = original trip-count
; CHECK-EMPTY:
; CHECK-NEXT: ir-bb<for.body.preheader>:
; CHECK-NEXT:   IR   %wide.trip.count = zext nneg i32 %n to i64
; CHECK-NEXT:   EMIT vp<%3> = EXPAND SCEV (zext i32 %n to i64)
; CHECK-NEXT:   EMIT vp<%4> = EXPAND SCEV (ptrtoint ptr %c to i64)
; CHECK-NEXT:   EMIT vp<%5> = EXPAND SCEV (ptrtoint ptr %b to i64)
; CHECK-NEXT: Successor(s): scalar.ph, vector.ph
; CHECK-EMPTY:
; CHECK-NEXT: vector.ph:
; CHECK-NEXT:   EMIT vp<%6> = ALIAS-LANE-MASK vp<%5>, vp<%4> (write-after-read)
; CHECK-NEXT:   EMIT vp<%popcount> = popcount vp<%6>
; CHECK-NEXT:   EMIT vp<%index.part.next> = VF * Part + ir<0>
; CHECK-NEXT:   EMIT vp<%active.lane.mask.entry> = active lane mask vp<%index.part.next>, vp<%3>
; CHECK-NEXT: Successor(s): vector loop
; CHECK-EMPTY:
; CHECK-NEXT: <x1> vector loop: {
; CHECK-NEXT:   vector.body:
; CHECK-NEXT:     EMIT vp<%7> = CANONICAL-INDUCTION ir<0>, vp<%index.next>
; CHECK-NEXT:     ACTIVE-LANE-MASK-PHI vp<%8> = phi vp<%active.lane.mask.entry>, vp<%active.lane.mask.next>
; CHECK-NEXT:     vp<%9> = SCALAR-STEPS vp<%7>, ir<1>, vp<%0>
; CHECK-NEXT:     EMIT vp<%10> = and vp<%8>, vp<%6>
; CHECK-NEXT:     CLONE ir<%arrayidx> = getelementptr inbounds ir<%a>, vp<%9>
; CHECK-NEXT:     vp<%11> = vector-pointer ir<%arrayidx>
; CHECK-NEXT:     WIDEN ir<%0> = load vp<%11>, vp<%10>
; CHECK-NEXT:     CLONE ir<%arrayidx2> = getelementptr inbounds ir<%b>, vp<%9>
; CHECK-NEXT:     vp<%12> = vector-pointer ir<%arrayidx2>
; CHECK-NEXT:     WIDEN ir<%1> = load vp<%12>, vp<%10>
; CHECK-NEXT:     WIDEN ir<%add> = add ir<%1>, ir<%0>
; CHECK-NEXT:     CLONE ir<%arrayidx6> = getelementptr inbounds ir<%c>, vp<%9>
; CHECK-NEXT:     vp<%13> = vector-pointer ir<%arrayidx6>
; CHECK-NEXT:     WIDEN store vp<%13>, ir<%add>, vp<%10>
; CHECK-NEXT:     EMIT vp<%index.next> = add vp<%7>, vp<%popcount>
; CHECK-NEXT:     EMIT vp<%14> = VF * Part + vp<%index.next>
; CHECK-NEXT:     EMIT vp<%active.lane.mask.next> = active lane mask vp<%14>, vp<%3>
; CHECK-NEXT:     EMIT vp<%15> = not vp<%active.lane.mask.next>
; CHECK-NEXT:     EMIT branch-on-cond vp<%15>
; CHECK-NEXT:   No successors
; CHECK-NEXT: }
; CHECK-NEXT: Successor(s): middle.block
; CHECK-EMPTY:
; CHECK-NEXT: middle.block:
; CHECK-NEXT: Successor(s): ir-bb<for.cond.cleanup.loopexit>
; CHECK-EMPTY:
; CHECK-NEXT: ir-bb<for.cond.cleanup.loopexit>:
; CHECK-NEXT: No successors
; CHECK-EMPTY:
; CHECK-NEXT: scalar.ph:
; CHECK-NEXT:   EMIT-SCALAR vp<%bc.resume.val> = phi [ ir<0>, ir-bb<for.body.preheader> ]
; CHECK-NEXT: Successor(s): ir-bb<for.body>
; CHECK-EMPTY:
; CHECK-NEXT: ir-bb<for.body>:
; CHECK-NEXT:   IR   %indvars.iv = phi i64 [ 0, %for.body.preheader ], [ %indvars.iv.next, %for.body ] (extra operand: vp<%bc.resume.val> from scalar.ph)
; CHECK-NEXT:   IR   %arrayidx = getelementptr inbounds i8, ptr %a, i64 %indvars.iv
; CHECK-NEXT:   IR   %0 = load i8, ptr %arrayidx, align 1
; CHECK-NEXT:   IR   %arrayidx2 = getelementptr inbounds i8, ptr %b, i64 %indvars.iv
; CHECK-NEXT:   IR   %1 = load i8, ptr %arrayidx2, align 1
; CHECK-NEXT:   IR   %add = add i8 %1, %0
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
