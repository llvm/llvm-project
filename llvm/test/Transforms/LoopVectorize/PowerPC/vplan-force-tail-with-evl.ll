; REQUIRES: asserts

; RUN: opt -passes=loop-vectorize -debug-only=loop-vectorize \
; RUN: -force-tail-folding-style=data-with-evl \
; RUN: -prefer-predicate-over-epilogue=predicate-dont-vectorize \
; RUN: -mtriple=powerpc64le-unknown-linux-gnu \
; RUN: -mcpu=pwr10 -disable-output < %s 2>&1 | FileCheck %s

define void @foo(ptr noalias %a, ptr noalias %b, ptr noalias %c, i64 %N) {
; CHECK-LABEL: VPlan 'Initial VPlan for VF={2,4},UF>=1' {
; CHECK-NEXT: Live-in vp<[[VF:%.+]]> = VF
; CHECK-NEXT: Live-in vp<[[VFxUF:%.+]]> = VF * UF
; CHECK-NEXT: Live-in vp<[[VTC:%.+]]> = vector-trip-count
; CHECK-NEXT: Live-in vp<[[BTC:%.+]]> = backedge-taken count
; CHECK-NEXT: Live-in ir<%N> = original trip-count
; CHECK-EMPTY:
; CHECK-NEXT: ir-bb<entry>:
; CHECK-NEXT: Successor(s): vector.ph
; CHECK-EMPTY:
; CHECK-NEXT: vector.ph:
; CHECK-NEXT: Successor(s): vector loop
; CHECK-EMPTY:
; CHECK-NEXT: <x1> vector loop: {
; CHECK-NEXT:   vector.body:
; CHECK-NEXT:     EMIT vp<[[CAN_IV:%.+]]> = CANONICAL-INDUCTION ir<0>, vp<[[CAN_INC:%.*]]>
; CHECK-NEXT:     ir<%iv> = WIDEN-INDUCTION ir<0>, ir<1>, vp<[[VF]]>
; CHECK-NEXT:     EMIT vp<[[CMP:%.+]]> = icmp ule ir<%iv>, vp<[[BTC]]>
; CHECK-NEXT:   Successor(s): pred.store
; CHECK-EMPTY:
; CHECK-NEXT:  <xVFxUF> pred.store: {
; CHECK-NEXT:    pred.store.entry:
; CHECK-NEXT:      BRANCH-ON-MASK vp<[[CMP]]>
; CHECK-NEXT:    Successor(s): pred.store.if, pred.store.continue
; CHECK-EMPTY:
; CHECK-NEXT:    pred.store.if:
; CHECK-NEXT:      vp<[[STEPS:%.+]]> = SCALAR-STEPS vp<[[CAN_IV]]>, ir<1>
; CHECK-NEXT:      REPLICATE ir<%arrayidx> = getelementptr inbounds ir<%b>, vp<[[STEPS]]>
; CHECK-NEXT:      REPLICATE ir<%0> = load ir<%arrayidx>
; CHECK-NEXT:      REPLICATE ir<%arrayidx2> = getelementptr inbounds ir<%c>, vp<[[STEPS]]>
; CHECK-NEXT:      REPLICATE ir<%1> = load ir<%arrayidx2>
; CHECK-NEXT:      REPLICATE ir<%arrayidx4> = getelementptr inbounds ir<%a>, vp<[[STEPS]]>
; CHECK-NEXT:      REPLICATE ir<%add> = add nsw ir<%1>, ir<%0>
; CHECK-NEXT:      REPLICATE store ir<%add>, ir<%arrayidx4>
; CHECK-NEXT:    Successor(s): pred.store.continue
; CHECK-EMPTY:
; CHECK-NEXT:    pred.store.continue:
; CHECK-NEXT:    No successors
; CHECK-NEXT:  }
; CHECK-NEXT:  Successor(s): for.body.2
; CHECK-EMPTY:
; CHECK-NEXT:  for.body.2:
; CHECK-NEXT:     EMIT vp<[[CAN_INC:%.+]]> = add vp<[[CAN_IV]]>, vp<[[VFxUF]]>
; CHECK-NEXT:     EMIT branch-on-count vp<[[CAN_INC]]>, vp<[[VTC]]>
; CHECK-NEXT:   No successors
; CHECK-NEXT: }
;
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %arrayidx = getelementptr inbounds i32, ptr %b, i64 %iv
  %0 = load i32, ptr %arrayidx, align 4
  %arrayidx2 = getelementptr inbounds i32, ptr %c, i64 %iv
  %1 = load i32, ptr %arrayidx2, align 4
  %add = add nsw i32 %1, %0
  %arrayidx4 = getelementptr inbounds i32, ptr %a, i64 %iv
  store i32 %add, ptr %arrayidx4, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond.not = icmp eq i64 %iv.next, %N
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body

for.cond.cleanup:
  ret void
}

define void @safe_dep(ptr %p) {
; CHECK-LABEL: VPlan 'Initial VPlan for VF={2},UF>=1' {
; CHECK-NEXT: Live-in vp<[[VFxUF:%.+]]> = VF * UF
; CHECK-NEXT: Live-in vp<[[VTC:%.+]]> = vector-trip-count
; CHECK-NEXT: Live-in ir<512> = original trip-count
; CHECK-EMPTY:
; CHECK-NEXT: ir-bb<entry>:
; CHECK-NEXT: Successor(s): vector.ph
; CHECK-EMPTY:
; CHECK-NEXT: vector.ph:
; CHECK-NEXT: Successor(s): vector loop
; CHECK-EMPTY:
; CHECK-NEXT: <x1> vector loop: {
; CHECK-NEXT:   vector.body:
; CHECK-NEXT:     EMIT vp<[[CAN_IV:%.+]]> = CANONICAL-INDUCTION ir<0>, vp<[[CAN_INC:%.+]]>
; CHECK-NEXT:     vp<[[STEPS:%.+]]> = SCALAR-STEPS vp<[[CAN_IV]]>, ir<1>
; CHECK-NEXT:     CLONE ir<%a1> = getelementptr ir<%p>, vp<[[STEPS]]>
; CHECK-NEXT:     vp<[[VPTR1:%.+]]> = vector-pointer ir<%a1>
; CHECK-NEXT:     WIDEN ir<%v> = load vp<[[VPTR1]]>
; CHECK-NEXT:     CLONE ir<%offset> = add vp<[[STEPS]]>, ir<100>
; CHECK-NEXT:     CLONE ir<%a2> = getelementptr ir<%p>, ir<%offset>
; CHECK-NEXT:     vp<[[VPTR2:%.+]]> = vector-pointer ir<%a2>
; CHECK-NEXT:     WIDEN store vp<[[VPTR2]]>, ir<%v>
; CHECK-NEXT:     EMIT vp<[[CAN_INC]]> = add nuw vp<[[CAN_IV]]>, vp<[[VFxUF]]>
; CHECK-NEXT:     EMIT branch-on-count vp<[[CAN_INC]]>, vp<[[VTC]]>
; CHECK-NEXT:   No successors
; CHECK-NEXT: }
;
entry:
  br label %loop

loop:
  %iv = phi i64 [0, %entry], [%iv.next, %loop]
  %a1 = getelementptr i64, ptr %p, i64 %iv
  %v = load i64, ptr %a1, align 32
  %offset = add i64 %iv, 100
  %a2 = getelementptr i64, ptr %p, i64 %offset
  store i64 %v, ptr %a2, align 32
  %iv.next = add i64 %iv, 1
  %cmp = icmp ne i64 %iv, 511
  br i1 %cmp, label %loop, label %exit

exit:
  ret void
}

