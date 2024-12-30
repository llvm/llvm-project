; REQUIRES: asserts

; RUN: opt -passes=loop-vectorize -force-vector-interleave=2 -force-vector-width=1 -debug -disable-output %s 2>&1 | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"

; Make sure recipes with side-effects are not sunk.
define void @sink_with_sideeffects(i1 %c, ptr %ptr) {
; CHECK-LABEL: sink_with_sideeffects
; CHECK:      VPlan 'Initial VPlan for VF={1},UF>=1' {
; CHECK-NEXT: Live-in vp<[[VFxUF:%.+]]> = VF * UF
; CHECK-NEXT: Live-in vp<[[VEC_TC:%.+]]> = vector-trip-count
; CHECK-NEXT: ir<0> = original trip-count
; CHECK-EMPTY:
; CHECK-NEXT: ir-bb<entry>:
; CHECK-NEXT: Successor(s): vector.ph
; CHECK-EMPTY:
; CHECK-NEXT: vector.ph:
; CHECK-NEXT:   vp<[[END:%.+]]> = DERIVED-IV ir<0> + vp<[[VEC_TC]]> * ir<-1>
; CHECK-NEXT: Successor(s): vector loop
; CHECK-EMPTY:
; CHECK-NEXT: <x1> vector loop: {
; CHECK-NEXT: vector.body:
; CHECK-NEXT:   EMIT vp<[[CAN_IV:%.+]]> = CANONICAL-INDUCTION
; CHECK-NEXT:   vp<[[STEPS:%.+]]> = SCALAR-STEPS vp<[[CAN_IV]]>, ir<1>
; CHECK-NEXT:   CLONE ir<%tmp2> = getelementptr ir<%ptr>, vp<[[STEPS]]>
; CHECK-NEXT:   CLONE ir<%tmp3> = load ir<%tmp2>
; CHECK-NEXT:   CLONE store ir<0>, ir<%tmp2>
; CHECK-NEXT: Successor(s): pred.store

; CHECK:      <xVFxUF> pred.store: {
; CHECK-NEXT:   pred.store.entry:
; CHECK-NEXT:     BRANCH-ON-MASK ir<%c>
; CHECK-NEXT:   Successor(s): pred.store.if, pred.store.continue

; CHECK:      pred.store.if:
; CHECK-NEXT:   CLONE store ir<%tmp3>, ir<%tmp2>
; CHECK-NEXT:   Successor(s): pred.store.continue

; CHECK:      pred.store.continue:
; CHECK-NEXT:   No successors
; CHECK-NEXT: }

; CHECK:      if.then.0:
; CHECK-NEXT:  EMIT vp<[[CAN_IV_NEXT:%.+]]> = add nuw vp<[[CAN_IV]]>, vp<[[VFxUF]]>
; CHECK-NEXT:  EMIT branch-on-count vp<[[CAN_IV_NEXT]]>, vp<[[VEC_TC]]>
; CHECK-NEXT: No successors
; CHECK-NEXT: }
; CHECK-NEXT: Successor(s): middle.block
; CHECK-EMPTY:
; CHECK-NEXT: middle.block:
; CHECK-NEXT:    EMIT vp<[[CMP:%.+]]> = icmp eq ir<0>, vp<[[VEC_TC]]>
; CHECK-NEXT:    EMIT branch-on-cond vp<[[CMP]]>
; CHECK-NEXT:  Successor(s): ir-bb<for.end>, scalar.ph
; CHECK-EMPTY:
; CHECK-NEXT:  scalar.ph:
; CHECK-NEXT:    EMIT vp<[[RESUME1:%.+]]> = resume-phi vp<[[VEC_TC]]>, ir<0>
; CHECK-NEXT:    EMIT vp<[[RESUME2:%.+]]>.1 = resume-phi vp<[[END]]>, ir<0>
; CHECK-NEXT:  Successor(s): ir-bb<for.body>
; CHECK-EMPTY:
; CHECK-NEXT:  ir-bb<for.body>:
; CHECK-NEXT:    IR   %tmp0 = phi i64 [ %tmp6, %for.inc ], [ 0, %entry ] (extra operand: vp<[[RESUME1]]> from scalar.ph)
; CHECK-NEXT:    IR   %tmp1 = phi i64 [ %tmp7, %for.inc ], [ 0, %entry ] (extra operand: vp<[[RESUME2]]>.1 from scalar.ph)
; CHECK:         IR   %tmp5 = trunc i32 %tmp4 to i8
; CHECK-NEXT:  No successors
; CHECK-EMPTY:
; CHECK-NEXT:  ir-bb<for.end>:
; CHECK-NEXT:  No successors
; CHECK-NEXT: }
;
entry:
  br label %for.body

for.body:
  %tmp0 = phi i64 [ %tmp6, %for.inc ], [ 0, %entry ]
  %tmp1 = phi i64 [ %tmp7, %for.inc ], [ 0, %entry ]
  %tmp2 = getelementptr i8, ptr %ptr, i64 %tmp0
  %tmp3 = load i8, ptr %tmp2, align 1
  store i8 0, ptr %tmp2
  %tmp4 = zext i8 %tmp3 to i32
  %tmp5 = trunc i32 %tmp4 to i8
  br i1 %c, label %if.then, label %for.inc

if.then:
  store i8 %tmp5, ptr %tmp2, align 1
  br label %for.inc

for.inc:
  %tmp6 = add nuw nsw i64 %tmp0, 1
  %tmp7 = add i64 %tmp1, -1
  %tmp8 = icmp eq i64 %tmp7, 0
  br i1 %tmp8, label %for.end, label %for.body

for.end:
  ret void
}
