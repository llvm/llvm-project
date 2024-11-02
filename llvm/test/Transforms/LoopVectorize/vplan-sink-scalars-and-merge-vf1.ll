; REQUIRES: asserts

; RUN: opt -passes=loop-vectorize -force-vector-interleave=2 -force-vector-width=1 -debug -disable-output %s 2>&1 | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"

; Make sure recipes with side-effects are not sunk.
define void @sink_with_sideeffects(i1 %c, i8* %ptr) {
; CHECK-LABEL: sink_with_sideeffects
; CHECK:      VPlan 'Initial VPlan for VF={1},UF>=1' {
; CHECK-NEXT: Live-in vp<[[VEC_TC:%.+]]> = vector-trip-count
; CHECK-EMPTY:
; CHECK-NEXT: vector.ph:
; CHECK-NEXT: Successor(s): vector loop
; CHECK-EMPTY:
; CHECK-NEXT: <x1> vector loop: {
; CHECK-NEXT: vector.body:
; CHECK-NEXT:   EMIT vp<[[CAN_IV:%.+]]> = CANONICAL-INDUCTION
; CHECK-NEXT:   vp<[[STEPS:%.+]]> = SCALAR-STEPS vp<[[CAN_IV]]>, ir<1>
; CHECK-NEXT:   CLONE ir<%tmp2> = getelementptr ir<%ptr>, vp<[[STEPS]]>
; CHECK-NEXT:   CLONE ir<%tmp3> = load ir<%tmp2>
; CHECK-NEXT:   CLONE store ir<0>, ir<%tmp2>
; CHECK-NEXT:   CLONE ir<%tmp4> = zext ir<%tmp3>
; CHECK-NEXT:   CLONE ir<%tmp5> = trunc ir<%tmp4>
; CHECK-NEXT: Successor(s): if.then

; CHECK:      if.then:
; CHECK-NEXT: Successor(s): pred.store

; CHECK:      <xVFxUF> pred.store: {
; CHECK-NEXT:   pred.store.entry:
; CHECK-NEXT:     BRANCH-ON-MASK ir<%c>
; CHECK-NEXT:   Successor(s): pred.store.if, pred.store.continue

; CHECK:      pred.store.if:
; CHECK-NEXT:   CLONE store ir<%tmp5>, ir<%tmp2>
; CHECK-NEXT:   Successor(s): pred.store.continue

; CHECK:      pred.store.continue:
; CHECK-NEXT:   No successors
; CHECK-NEXT: }

; CHECK:      if.then.0:
; CHECK-NEXT: Successor(s): for.inc

; CHECK:      for.inc:
; CHECK-NEXT:  EMIT vp<[[CAN_IV_NEXT:%.+]]> = VF * UF +(nuw) vp<[[CAN_IV]]>
; CHECK-NEXT:  EMIT branch-on-count vp<[[CAN_IV_NEXT]]> vp<[[VEC_TC]]>
; CHECK-NEXT: No successors
; CHECK-NEXT: }
; CHECK-NEXT: Successor(s): middle.block
; CHECK-EMPTY:
; CHECK-NEXT: middle.block:
; CHECK-NEXT: No successors
; CHECK-NEXT: }
;
entry:
  br label %for.body

for.body:
  %tmp0 = phi i64 [ %tmp6, %for.inc ], [ 0, %entry ]
  %tmp1 = phi i64 [ %tmp7, %for.inc ], [ 0, %entry ]
  %tmp2 = getelementptr i8, i8* %ptr, i64 %tmp0
  %tmp3 = load i8, i8* %tmp2, align 1
  store i8 0, i8* %tmp2
  %tmp4 = zext i8 %tmp3 to i32
  %tmp5 = trunc i32 %tmp4 to i8
  br i1 %c, label %if.then, label %for.inc

if.then:
  store i8 %tmp5, i8* %tmp2, align 1
  br label %for.inc

for.inc:
  %tmp6 = add nuw nsw i64 %tmp0, 1
  %tmp7 = add i64 %tmp1, -1
  %tmp8 = icmp eq i64 %tmp7, 0
  br i1 %tmp8, label %for.end, label %for.body

for.end:
  ret void
}
