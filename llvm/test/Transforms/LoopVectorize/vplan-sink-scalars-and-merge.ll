; REQUIRES: asserts

; RUN: opt -passes=loop-vectorize -force-vector-interleave=1 -force-vector-width=2 -debug -disable-output %s 2>&1 | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"

@a = common global [2048 x i32] zeroinitializer, align 16
@b = common global [2048 x i32] zeroinitializer, align 16
@c = common global [2048 x i32] zeroinitializer, align 16


; CHECK-LABEL: LV: Checking a loop in 'sink1'
; CHECK:      VPlan 'Initial VPlan for VF={2},UF>=1' {
; CHECK-NEXT: Live-in vp<[[VEC_TC:%.+]]> = vector-trip-count
; CHECK-EMPTY:
; CHECK-NEXT: Live-in vp<[[BTC:%.+]]> = backedge-taken count
; CHECK-EMPTY:
; CHECK-NEXT: vector.ph:
; CHECK-NEXT: Successor(s): vector loop
; CHECK-EMPTY:
; CHECK-NEXT: <x1> vector loop: {
; CHECK-NEXT: vector.body:
; CHECK-NEXT:   EMIT vp<[[CAN_IV:%.+]]> = CANONICAL-INDUCTION
; CHECK-NEXT:   WIDEN-INDUCTION %iv = phi 0, %iv.next, ir<1>
; CHECK-NEXT:   EMIT vp<[[MASK:%.+]]> = icmp ule ir<%iv> vp<[[BTC]]>
; CHECK-NEXT: Successor(s): pred.store

; CHECK:      <xVFxUF> pred.store: {
; CHECK-NEXT:   pred.store.entry:
; CHECK-NEXT:     BRANCH-ON-MASK vp<[[MASK]]>
; CHECK-NEXT:   Successor(s): pred.store.if, pred.store.continue

; CHECK:      pred.store.if:
; CHECK-NEXT:     vp<[[STEPS:%.+]]> = SCALAR-STEPS vp<[[CAN_IV]]>, ir<1>
; CHECK-NEXT:     REPLICATE ir<%gep.b> = getelementptr ir<@b>, ir<0>, vp<[[STEPS]]>
; CHECK-NEXT:     REPLICATE ir<%lv.b> = load ir<%gep.b>
; CHECK-NEXT:     REPLICATE ir<%add> = add ir<%lv.b>, ir<10>
; CHECK-NEXT:     REPLICATE ir<%gep.a> = getelementptr ir<@a>, ir<0>, vp<[[STEPS]]
; CHECK-NEXT:     REPLICATE ir<%mul> = mul ir<2>, ir<%add>
; CHECK-NEXT:     REPLICATE store ir<%mul>, ir<%gep.a>
; CHECK-NEXT:   Successor(s): pred.store.continue

; CHECK:      pred.store.continue:
; CHECK-NEXT:   PHI-PREDICATED-INSTRUCTION vp<[[PRED:%.+]]> = ir<%lv.b>
; CHECK-NEXT:   No successors
; CHECK-NEXT: }

; CHECK:      loop.1:
; CHECK-NEXT:   EMIT vp<[[CAN_IV_NEXT:%.+]]> = VF * UF + vp<[[CAN_IV]]>
; CHECK-NEXT:   EMIT branch-on-count vp<[[CAN_IV_NEXT]]> vp<[[VEC_TC]]>
; CHECK-NEXT: No successors
; CHECK-NEXT: }
;
define void @sink1(i32 %k) {
entry:
  br label %loop

loop:
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %loop ]
  %gep.b = getelementptr inbounds [2048 x i32], ptr @b, i32 0, i32 %iv
  %lv.b  = load i32, ptr %gep.b, align 4
  %add = add i32 %lv.b, 10
  %mul = mul i32 2, %add
  %gep.a = getelementptr inbounds [2048 x i32], ptr @a, i32 0, i32 %iv
  store i32 %mul, ptr %gep.a, align 4
  %iv.next = add i32 %iv, 1
  %large = icmp sge i32 %iv, 8
  %exitcond = icmp eq i32 %iv, %k
  %realexit = or i1 %large, %exitcond
  br i1 %realexit, label %exit, label %loop

exit:
  ret void
}

; CHECK-LABEL: LV: Checking a loop in 'sink2'
; CHECK:      VPlan 'Initial VPlan for VF={2},UF>=1' {
; CHECK-NEXT: Live-in vp<[[VEC_TC:%.+]]> = vector-trip-count
; CHECK-EMPTY:
; CHECK-NEXT: Live-in vp<[[BTC:%.+]]> = backedge-taken count
; CHECK-EMPTY:
; CHECK-NEXT: vector.ph:
; CHECK-NEXT: Successor(s): vector loop
; CHECK-EMPTY:
; CHECK-NEXT: <x1> vector loop: {
; CHECK-NEXT: vector.body:
; CHECK-NEXT:   EMIT vp<[[CAN_IV:%.+]]> = CANONICAL-INDUCTION
; CHECK-NEXT:   WIDEN-INDUCTION %iv = phi 0, %iv.next, ir<1>
; CHECK-NEXT:   EMIT vp<[[MASK:%.+]]> = icmp ule ir<%iv> vp<[[BTC]]>
; CHECK-NEXT: Successor(s): pred.load

; CHECK:      <xVFxUF> pred.load: {
; CHECK-NEXT:   pred.load.entry:
; CHECK-NEXT:     BRANCH-ON-MASK vp<[[MASK]]>
; CHECK-NEXT:   Successor(s): pred.load.if, pred.load.continue

; CHECK:      pred.load.if:
; CHECK-NEXT:     vp<[[STEPS:%.+]]> = SCALAR-STEPS vp<[[CAN_IV]]>, ir<1>
; CHECK-NEXT:     REPLICATE ir<%gep.b> = getelementptr ir<@b>, ir<0>, vp<[[STEPS]]>
; CHECK-NEXT:     REPLICATE ir<%lv.b> = load ir<%gep.b>
; CHECK-NEXT:   Successor(s): pred.load.continue

; CHECK:      pred.load.continue:
; CHECK-NEXT:     PHI-PREDICATED-INSTRUCTION vp<[[PRED:%.+]]> = ir<%lv.b>
; CHECK-NEXT:   No successors
; CHECK-NEXT: }

; CHECK:      loop.0:
; CHECK-NEXT:   WIDEN ir<%mul> = mul ir<%iv>, ir<2>
; CHECK-NEXT: Successor(s): pred.store

; CHECK:      <xVFxUF> pred.store: {
; CHECK-NEXT:   pred.store.entry:
; CHECK-NEXT:     BRANCH-ON-MASK vp<[[MASK]]>
; CHECK-NEXT:   Successor(s): pred.store.if, pred.store.continue

; CHECK:       pred.store.if:
; CHECK-NEXT:     REPLICATE ir<%gep.a> = getelementptr ir<@a>, ir<0>, ir<%mul>
; CHECK-NEXT:     REPLICATE ir<%add> = add vp<[[PRED]]>, ir<10>
; CHECK-NEXT:     REPLICATE store ir<%add>, ir<%gep.a>
; CHECK-NEXT:   Successor(s): pred.store.continue

; CHECK:      pred.store.continue:
; CHECK-NEXT:   No successors
; CHECK-NEXT: }

; CHECK:       loop.1:
; CHECK-NEXT:   EMIT vp<[[CAN_IV_NEXT:%.+]]> = VF * UF + vp<[[CAN_IV]]>
; CHECK-NEXT:   EMIT branch-on-count vp<[[CAN_IV_NEXT]]> vp<[[VEC_TC]]>
; CHECK-NEXT: No successors
; CHECK-NEXT: }
;
define void @sink2(i32 %k) {
entry:
  br label %loop

loop:
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %loop ]
  %gep.b = getelementptr inbounds [2048 x i32], ptr @b, i32 0, i32 %iv
  %lv.b  = load i32, ptr %gep.b, align 4
  %add = add i32 %lv.b, 10
  %mul = mul i32 %iv, 2
  %gep.a = getelementptr inbounds [2048 x i32], ptr @a, i32 0, i32 %mul
  store i32 %add, ptr %gep.a, align 4
  %iv.next = add i32 %iv, 1
  %large = icmp sge i32 %iv, 8
  %exitcond = icmp eq i32 %iv, %k
  %realexit = or i1 %large, %exitcond
  br i1 %realexit, label %exit, label %loop

exit:
  ret void
}

; CHECK-LABEL: LV: Checking a loop in 'sink3'
; CHECK:      VPlan 'Initial VPlan for VF={2},UF>=1' {
; CHECK-NEXT: Live-in vp<[[VEC_TC:%.+]]> = vector-trip-count
; CHECK-EMPTY:
; CHECK-NEXT: Live-in vp<[[BTC:%.+]]> = backedge-taken count
; CHECK-EMPTY:
; CHECK-NEXT: vector.ph:
; CHECK-NEXT: Successor(s): vector loop
; CHECK-EMPTY:
; CHECK-NEXT: <x1> vector loop: {
; CHECK-NEXT: vector.body:
; CHECK-NEXT:   EMIT vp<[[CAN_IV:%.+]]> = CANONICAL-INDUCTION
; CHECK-NEXT:   WIDEN-INDUCTION %iv = phi 0, %iv.next, ir<1>
; CHECK-NEXT:   EMIT vp<[[MASK:%.+]]> = icmp ule ir<%iv> vp<[[BTC]]>
; CHECK-NEXT: Successor(s): pred.load

; CHECK:      <xVFxUF> pred.load: {
; CHECK-NEXT:   pred.load.entry:
; CHECK-NEXT:     BRANCH-ON-MASK vp<[[MASK]]>
; CHECK-NEXT:   Successor(s): pred.load.if, pred.load.continue

; CHECK:       pred.load.if:
; CHECK-NEXT:     vp<[[STEPS:%.+]]> = SCALAR-STEPS vp<[[CAN_IV]]>, ir<1>
; CHECK-NEXT:     REPLICATE ir<%gep.b> = getelementptr ir<@b>, ir<0>, vp<[[STEPS]]>
; CHECK-NEXT:     REPLICATE ir<%lv.b> = load ir<%gep.b> (S->V)
; CHECK-NEXT:   Successor(s): pred.load.continue

; CHECK:       pred.load.continue:
; CHECK-NEXT:     PHI-PREDICATED-INSTRUCTION vp<[[PRED:%.+]]> = ir<%lv.b>
; CHECK-NEXT:   No successors
; CHECK-NEXT: }

; CHECK:      loop.0:
; CHECK-NEXT:   WIDEN ir<%add> = add vp<[[PRED]]>, ir<10>
; CHECK-NEXT:   WIDEN ir<%mul> = mul ir<%iv>, ir<%add>
; CHECK-NEXT: Successor(s): pred.store

; CHECK:      <xVFxUF> pred.store: {
; CHECK-NEXT:   pred.store.entry:
; CHECK-NEXT:     BRANCH-ON-MASK vp<[[MASK]]>
; CHECK-NEXT:   Successor(s): pred.store.if, pred.store.continue

; CHECK:      pred.store.if:
; CHECK-NEXT:     REPLICATE ir<%gep.a> = getelementptr ir<@a>, ir<0>, ir<%mul>
; CHECK-NEXT:     REPLICATE store ir<%add>, ir<%gep.a>
; CHECK-NEXT:   Successor(s): pred.store.continue

; CHECK:      pred.store.continue:
; CHECK-NEXT:   No successors
; CHECK-NEXT: }

; CHECK:      loop.1:
; CHECK-NEXT:   EMIT vp<[[CAN_IV_NEXT:%.+]]> = VF * UF + vp<[[CAN_IV]]>
; CHECK-NEXT:   EMIT branch-on-count vp<[[CAN_IV_NEXT]]> vp<[[VEC_TC]]>
; CHECK-NEXT: No successors
; CHECK-NEXT: }
;
define void @sink3(i32 %k) {
entry:
  br label %loop

loop:
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %loop ]
  %gep.b = getelementptr inbounds [2048 x i32], ptr @b, i32 0, i32 %iv
  %lv.b  = load i32, ptr %gep.b, align 4
  %add = add i32 %lv.b, 10
  %mul = mul i32 %iv, %add
  %gep.a = getelementptr inbounds [2048 x i32], ptr @a, i32 0, i32 %mul
  store i32 %add, ptr %gep.a, align 4
  %iv.next = add i32 %iv, 1
  %large = icmp sge i32 %iv, 8
  %exitcond = icmp eq i32 %iv, %k
  %realexit = or i1 %large, %exitcond
  br i1 %realexit, label %exit, label %loop

exit:
  ret void
}

; Make sure we do not sink uniform instructions.
define void @uniform_gep(i64 %k, ptr noalias %A, ptr noalias %B) {
; CHECK-LABEL: LV: Checking a loop in 'uniform_gep'
; CHECK:      VPlan 'Initial VPlan for VF={2},UF>=1' {
; CHECK-NEXT: Live-in vp<[[VEC_TC:%.+]]> = vector-trip-count
; CHECK-EMPTY:
; CHECK-NEXT: Live-in vp<[[BTC:%.+]]> = backedge-taken count
; CHECK-EMPTY:
; CHECK-NEXT: vector.ph:
; CHECK-NEXT: Successor(s): vector loop
; CHECK-EMPTY:
; CHECK-NEXT: <x1> vector loop: {
; CHECK-NEXT: vector.body:
; CHECK-NEXT:   EMIT vp<[[CAN_IV:%.+]]> = CANONICAL-INDUCTION
; CHECK-NEXT:   WIDEN-INDUCTION %iv = phi 21, %iv.next, ir<1>
; CHECK-NEXT:   vp<[[DERIVED_IV:%.+]]> = DERIVED-IV ir<21> + vp<[[CAN_IV]]> * ir<1>
; CHECK-NEXT:   EMIT vp<[[WIDE_CAN_IV:%.+]]> = WIDEN-CANONICAL-INDUCTION vp<[[CAN_IV]]>
; CHECK-NEXT:   EMIT vp<[[MASK:%.+]]> = icmp ule vp<[[WIDE_CAN_IV]]> vp<[[BTC]]>
; CHECK-NEXT:   CLONE ir<%gep.A.uniform> = getelementptr ir<%A>, ir<0>
; CHECK-NEXT:   CLONE ir<%lv> = load ir<%gep.A.uniform>
; CHECK-NEXT:   WIDEN ir<%cmp> = icmp ult ir<%iv>, ir<%k>
; CHECK-NEXT:   EMIT vp<[[NOT2:%.+]]> = not ir<%cmp>
; CHECK-NEXT:   EMIT vp<[[MASK2:%.+]]> = select vp<[[MASK]]> vp<[[NOT2]]> ir<false>
; CHECK-NEXT: Successor(s): pred.store
; CHECK-EMPTY:
; CHECK-NEXT: <xVFxUF> pred.store: {
; CHECK-NEXT:   pred.store.entry:
; CHECK-NEXT:     BRANCH-ON-MASK vp<[[MASK2]]>
; CHECK-NEXT:   Successor(s): pred.store.if, pred.store.continue
; CHECK-EMPTY:
; CHECK-NEXT:   pred.store.if:
; CHECK-NEXT:     vp<[[STEPS:%.+]]> = SCALAR-STEPS vp<[[DERIVED_IV]]>, ir<1>
; CHECK-NEXT:     REPLICATE ir<%gep.B> = getelementptr ir<%B>, vp<[[STEPS]]>
; CHECK-NEXT:     REPLICATE store ir<%lv>, ir<%gep.B>
; CHECK-NEXT:   Successor(s): pred.store.continue
; CHECK-EMPTY:
; CHECK-NEXT:   pred.store.continue:
; CHECK-NEXT:   No successors
; CHECK-NEXT: }
; CHECK-NEXT: Successor(s): loop.then.0
; CHECK-EMPTY:
; CHECK-NEXT: loop.then.0:
; CHECK-NEXT:   EMIT vp<[[CAN_IV_NEXT:%.+]]> = VF * UF + vp<[[CAN_IV]]>
; CHECK-NEXT:   EMIT branch-on-count vp<[[CAN_IV_NEXT]]> vp<[[VEC_TC]]>
; CHECK-NEXT: No successors
; CHECK-NEXT: }
;
entry:
  br label %loop

loop:
  %iv = phi i64 [ 21, %entry ], [ %iv.next, %loop.latch ]
  %gep.A.uniform = getelementptr inbounds i16, ptr %A, i64 0
  %gep.B = getelementptr inbounds i16, ptr %B, i64 %iv
  %lv = load i16, ptr %gep.A.uniform, align 1
  %cmp = icmp ult i64 %iv, %k
  br i1 %cmp, label %loop.latch, label %loop.then

loop.then:
  store i16 %lv, ptr %gep.B, align 1
  br label %loop.latch

loop.latch:
  %iv.next = add nsw i64 %iv, 1
  %cmp179 = icmp slt i64 %iv.next, 32
  br i1 %cmp179, label %loop, label %exit
exit:
  ret void
}

; Loop with predicated load.
define void @pred_cfg1(i32 %k, i32 %j) {
; CHECK-LABEL: LV: Checking a loop in 'pred_cfg1'
; CHECK:      VPlan 'Initial VPlan for VF={2},UF>=1' {
; CHECK-NEXT: Live-in vp<[[VEC_TC:%.+]]> = vector-trip-count
; CHECK-EMPTY:
; CHECK-NEXT: Live-in vp<[[BTC:%.+]]> = backedge-taken count
; CHECK-EMPTY:
; CHECK-NEXT: vector.ph:
; CHECK-NEXT: Successor(s): vector loop
; CHECK-EMPTY:
; CHECK-NEXT: <x1> vector loop: {
; CHECK-NEXT: vector.body:
; CHECK-NEXT:   EMIT vp<[[CAN_IV:%.+]]> = CANONICAL-INDUCTION
; CHECK-NEXT:   WIDEN-INDUCTION %iv = phi 0, %iv.next, ir<1>
; CHECK-NEXT:   EMIT vp<[[MASK1:%.+]]> = icmp ule ir<%iv> vp<[[BTC]]>
; CHECK-NEXT:   WIDEN ir<%c.1> = icmp ult ir<%iv>, ir<%j>
; CHECK-NEXT:   WIDEN ir<%mul> = mul ir<%iv>, ir<10>
; CHECK-NEXT:   EMIT vp<[[MASK2:%.+]]> = select vp<[[MASK1]]> ir<%c.1> ir<false>
; CHECK-NEXT: Successor(s): pred.load
; CHECK-EMPTY:
; CHECK-NEXT: <xVFxUF> pred.load: {
; CHECK-NEXT:   pred.load.entry:
; CHECK-NEXT:     BRANCH-ON-MASK vp<[[MASK2]]>
; CHECK-NEXT:   Successor(s): pred.load.if, pred.load.continue
; CHECK-EMPTY:
; CHECK-NEXT:   pred.load.if:
; CHECK-NEXT:     vp<[[STEPS:%.+]]> = SCALAR-STEPS vp<[[CAN_IV]]>, ir<1>
; CHECK-NEXT:     REPLICATE ir<%gep.b> = getelementptr ir<@b>, ir<0>, vp<[[STEPS]]>
; CHECK-NEXT:     REPLICATE ir<%lv.b> = load ir<%gep.b> (S->V)
; CHECK-NEXT:   Successor(s): pred.load.continue
; CHECK-EMPTY:
; CHECK-NEXT:   pred.load.continue:
; CHECK-NEXT:     PHI-PREDICATED-INSTRUCTION vp<[[PRED:%.+]]> = ir<%lv.b>
; CHECK-NEXT:   No successors
; CHECK-NEXT: }
; CHECK-NEXT: Successor(s): then.0.0
; CHECK-EMPTY:
; CHECK-NEXT: then.0.0:
; CHECK-NEXT:   EMIT vp<[[NOT:%.+]]> = not ir<%c.1>
; CHECK-NEXT:   EMIT vp<[[MASK3:%.+]]> = select vp<[[MASK1]]> vp<[[NOT]]> ir<false>
; CHECK-NEXT:   BLEND %p = ir<0>/vp<[[MASK3]]> vp<[[PRED]]>/vp<[[MASK2]]>
; CHECK-NEXT:   EMIT vp<[[OR:%.+]]> = or vp<[[MASK2]]> vp<[[MASK3]]>
; CHECK-NEXT: Successor(s): pred.store
; CHECK-EMPTY:
; CHECK-NEXT: <xVFxUF> pred.store: {
; CHECK-NEXT:   pred.store.entry:
; CHECK-NEXT:     BRANCH-ON-MASK vp<[[OR]]>
; CHECK-NEXT:   Successor(s): pred.store.if, pred.store.continue
; CHECK-EMPTY:
; CHECK-NEXT:   pred.store.if:
; CHECK-NEXT:     REPLICATE ir<%gep.a> = getelementptr ir<@a>, ir<0>, ir<%mul>
; CHECK-NEXT:     REPLICATE store ir<%p>, ir<%gep.a>
; CHECK-NEXT:   Successor(s): pred.store.continue
; CHECK-EMPTY:
; CHECK-NEXT:   pred.store.continue:
; CHECK-NEXT:   No successors
; CHECK-NEXT: }
; CHECK-NEXT: Successor(s): next.0.0
; CHECK-EMPTY:
; CHECK-NEXT: next.0.0:
; CHECK-NEXT:   EMIT vp<[[CAN_IV_NEXT:%.+]]> = VF * UF + vp<[[CAN_IV]]>
; CHECK-NEXT:   EMIT branch-on-count vp<[[CAN_IV_NEXT]]> vp<[[VEC_TC]]>
; CHECK-NEXT: No successors
; CHECK-NEXT: }
;
entry:
  br label %loop

loop:
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %next.0 ]
  %gep.b = getelementptr inbounds [2048 x i32], ptr @b, i32 0, i32 %iv
  %c.1 = icmp ult i32 %iv, %j
  %mul = mul i32 %iv, 10
  %gep.a = getelementptr inbounds [2048 x i32], ptr @a, i32 0, i32 %mul
  br i1 %c.1, label %then.0, label %next.0

then.0:
  %lv.b  = load i32, ptr %gep.b, align 4
  br label %next.0

next.0:
  %p = phi i32 [ 0, %loop ], [ %lv.b, %then.0 ]
  store i32 %p, ptr %gep.a, align 4
  %iv.next = add i32 %iv, 1
  %large = icmp sge i32 %iv, 8
  %exitcond = icmp eq i32 %iv, %k
  %realexit = or i1 %large, %exitcond
  br i1 %realexit, label %exit, label %loop

exit:
  ret void
}

; Loop with predicated load and store in separate blocks, store depends on
; loaded value.
define void @pred_cfg2(i32 %k, i32 %j) {
; CHECK-LABEL: LV: Checking a loop in 'pred_cfg2'
; CHECK:      VPlan 'Initial VPlan for VF={2},UF>=1' {
; CHECK-NEXT: Live-in vp<[[VEC_TC:%.+]]> = vector-trip-count
; CHECK-EMPTY:
; CHECK-NEXT: Live-in vp<[[BTC:%.+]]> = backedge-taken count
; CHECK-EMPTY:
; CHECK-NEXT: vector.ph:
; CHECK-NEXT: Successor(s): vector loop
; CHECK-EMPTY:
; CHECK-NEXT: <x1> vector loop: {
; CHECK-NEXT: vector.body:
; CHECK-NEXT:   EMIT vp<[[CAN_IV:%.+]]> = CANONICAL-INDUCTION
; CHECK-NEXT:   WIDEN-INDUCTION %iv = phi 0, %iv.next, ir<1>
; CHECK-NEXT:   EMIT vp<[[MASK1:%.+]]> = icmp ule ir<%iv> vp<[[BTC]]>
; CHECK-NEXT:   WIDEN ir<%mul> = mul ir<%iv>, ir<10>
; CHECK-NEXT:   WIDEN ir<%c.0> = icmp ult ir<%iv>, ir<%j>
; CHECK-NEXT:   WIDEN ir<%c.1> = icmp ugt ir<%iv>, ir<%j>
; CHECK-NEXT:   EMIT vp<[[MASK2:%.+]]> = select vp<[[MASK1]]> ir<%c.0> ir<false>
; CHECK-NEXT: Successor(s): pred.load
; CHECK-EMPTY:
; CHECK-NEXT: <xVFxUF> pred.load: {
; CHECK-NEXT:   pred.load.entry:
; CHECK-NEXT:     BRANCH-ON-MASK vp<[[MASK2]]>
; CHECK-NEXT:   Successor(s): pred.load.if, pred.load.continue
; CHECK-EMPTY:
; CHECK-NEXT:   pred.load.if:
; CHECK-NEXT:     vp<[[STEPS:%.+]]> = SCALAR-STEPS vp<[[CAN_IV]]>, ir<1>
; CHECK-NEXT:     REPLICATE ir<%gep.b> = getelementptr ir<@b>, ir<0>, vp<[[STEPS]]>
; CHECK-NEXT:     REPLICATE ir<%lv.b> = load ir<%gep.b> (S->V)
; CHECK-NEXT:   Successor(s): pred.load.continue
; CHECK-EMPTY:
; CHECK-NEXT:   pred.load.continue:
; CHECK-NEXT:     PHI-PREDICATED-INSTRUCTION vp<[[PRED:%.+]]> = ir<%lv.b>
; CHECK-NEXT:   No successors
; CHECK-NEXT: }
; CHECK-NEXT: Successor(s): then.0.0
; CHECK-EMPTY:
; CHECK-NEXT: then.0.0:
; CHECK-NEXT:   EMIT vp<[[NOT:%.+]]> = not ir<%c.0>
; CHECK-NEXT:   EMIT vp<[[MASK3:%.+]]> = select vp<[[MASK1]]> vp<[[NOT]]> ir<false>
; CHECK-NEXT:   BLEND %p = ir<0>/vp<[[MASK3]]> vp<[[PRED]]>/vp<[[MASK2]]>
; CHECK-NEXT:   EMIT vp<[[OR:%.+]]> = or vp<[[MASK2]]> vp<[[MASK3]]>
; CHECK-NEXT:   EMIT vp<[[MASK4:%.+]]> = select vp<[[OR]]> ir<%c.1> ir<false>
; CHECK-NEXT: Successor(s): pred.store
; CHECK-EMPTY:
; CHECK-NEXT: <xVFxUF> pred.store: {
; CHECK-NEXT:   pred.store.entry:
; CHECK-NEXT:     BRANCH-ON-MASK vp<[[MASK4]]>
; CHECK-NEXT:   Successor(s): pred.store.if, pred.store.continue
; CHECK-EMPTY:
; CHECK-NEXT:   pred.store.if:
; CHECK-NEXT:     REPLICATE ir<%gep.a> = getelementptr ir<@a>, ir<0>, ir<%mul>
; CHECK-NEXT:     REPLICATE store ir<%p>, ir<%gep.a>
; CHECK-NEXT:   Successor(s): pred.store.continue
; CHECK-EMPTY:
; CHECK-NEXT:   pred.store.continue:
; CHECK-NEXT:   No successors
; CHECK-NEXT: }
; CHECK-NEXT: Successor(s): then.1.0
; CHECK-EMPTY:
; CHECK-NEXT: then.1.0:
; CHECK-NEXT:   EMIT vp<[[CAN_IV_NEXT:%.+]]> = VF * UF + vp<[[CAN_IV]]>
; CHECK-NEXT:   EMIT branch-on-count vp<[[CAN_IV_NEXT]]> vp<[[VEC_TC]]>
; CHECK-NEXT: No successors
; CHECK-NEXT: }
;
entry:
  br label %loop

loop:
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %next.1 ]
  %gep.b = getelementptr inbounds [2048 x i32], ptr @b, i32 0, i32 %iv
  %mul = mul i32 %iv, 10
  %gep.a = getelementptr inbounds [2048 x i32], ptr @a, i32 0, i32 %mul
  %c.0 = icmp ult i32 %iv, %j
  %c.1 = icmp ugt i32 %iv, %j
  br i1 %c.0, label %then.0, label %next.0

then.0:
  %lv.b  = load i32, ptr %gep.b, align 4
  br label %next.0

next.0:
  %p = phi i32 [ 0, %loop ], [ %lv.b, %then.0 ]
  br i1 %c.1, label %then.1, label %next.1

then.1:
  store i32 %p, ptr %gep.a, align 4
  br label %next.1

next.1:
  %iv.next = add i32 %iv, 1
  %large = icmp sge i32 %iv, 8
  %exitcond = icmp eq i32 %iv, %k
  %realexit = or i1 %large, %exitcond
  br i1 %realexit, label %exit, label %loop

exit:
  ret void
}

; Loop with predicated load and store in separate blocks, store does not depend
; on loaded value.
define void @pred_cfg3(i32 %k, i32 %j) {
; CHECK-LABEL: LV: Checking a loop in 'pred_cfg3'
; CHECK:      VPlan 'Initial VPlan for VF={2},UF>=1' {
; CHECK-NEXT: Live-in vp<[[VEC_TC:%.+]]> = vector-trip-count
; CHECK-EMPTY:
; CHECK-NEXT: Live-in vp<[[BTC:%.+]]> = backedge-taken count
; CHECK-EMPTY:
; CHECK-NEXT: vector.ph:
; CHECK-NEXT: Successor(s): vector loop
; CHECK-EMPTY:
; CHECK-NEXT: <x1> vector loop: {
; CHECK-NEXT: vector.body:
; CHECK-NEXT:   EMIT vp<[[CAN_IV:%.+]]> = CANONICAL-INDUCTION
; CHECK-NEXT:   WIDEN-INDUCTION %iv = phi 0, %iv.next, ir<1>
; CHECK-NEXT:   EMIT vp<[[MASK1:%.+]]> = icmp ule ir<%iv> vp<[[BTC]]>
; CHECK-NEXT:   WIDEN ir<%mul> = mul ir<%iv>, ir<10>
; CHECK-NEXT:   WIDEN ir<%c.0> = icmp ult ir<%iv>, ir<%j>
; CHECK-NEXT:   EMIT vp<[[MASK2:%.+]]> = select vp<[[MASK1:%.+]]> ir<%c.0> ir<false>
; CHECK-NEXT: Successor(s): pred.load
; CHECK-EMPTY:
; CHECK-NEXT: <xVFxUF> pred.load: {
; CHECK-NEXT:   pred.load.entry:
; CHECK-NEXT:     BRANCH-ON-MASK vp<[[MASK2]]>
; CHECK-NEXT:   Successor(s): pred.load.if, pred.load.continue
; CHECK-EMPTY:
; CHECK-NEXT:   pred.load.if:
; CHECK-NEXT:     vp<[[STEPS:%.+]]> = SCALAR-STEPS vp<[[CAN_IV]]>, ir<1>
; CHECK-NEXT:     REPLICATE ir<%gep.b> = getelementptr ir<@b>, ir<0>, vp<[[STEPS]]>
; CHECK-NEXT:     REPLICATE ir<%lv.b> = load ir<%gep.b>
; CHECK-NEXT:   Successor(s): pred.load.continue
; CHECK-EMPTY:
; CHECK-NEXT:   pred.load.continue:
; CHECK-NEXT:     PHI-PREDICATED-INSTRUCTION vp<[[PRED:%.+]]> = ir<%lv.b>
; CHECK-NEXT:   No successors
; CHECK-NEXT: }
; CHECK-NEXT: Successor(s): then.0.0
; CHECK-EMPTY:
; CHECK-NEXT: then.0.0:
; CHECK-NEXT:   EMIT vp<[[NOT:%.+]]> = not ir<%c.0>
; CHECK-NEXT:   EMIT vp<[[MASK3:%.+]]> = select vp<[[MASK1]]> vp<[[NOT]]> ir<false>
; CHECK-NEXT:   BLEND %p = ir<0>/vp<[[MASK3]]> vp<[[PRED]]>/vp<[[MASK2]]>
; CHECK-NEXT:   EMIT vp<[[MASK4:%.+]]> = or vp<[[MASK2]]> vp<[[MASK3]]>
; CHECK-NEXT:   EMIT vp<[[MASK5:%.+]]> = select vp<[[MASK4]]> ir<%c.0> ir<false>
; CHECK-NEXT: Successor(s): pred.store
; CHECK-EMPTY:
; CHECK-NEXT: <xVFxUF> pred.store: {
; CHECK-NEXT:   pred.store.entry:
; CHECK-NEXT:     BRANCH-ON-MASK vp<[[MASK5]]>
; CHECK-NEXT:   Successor(s): pred.store.if, pred.store.continue
; CHECK-EMPTY:
; CHECK-NEXT:   pred.store.if:
; CHECK-NEXT:     REPLICATE ir<%gep.a> = getelementptr ir<@a>, ir<0>, ir<%mul>
; CHECK-NEXT:     REPLICATE store ir<0>, ir<%gep.a>
; CHECK-NEXT:     REPLICATE ir<%gep.c> = getelementptr ir<@c>, ir<0>, ir<%mul>
; CHECK-NEXT:     REPLICATE store ir<%p>, ir<%gep.c>
; CHECK-NEXT:   Successor(s): pred.store.continue
; CHECK-EMPTY:
; CHECK-NEXT:   pred.store.continue:
; CHECK-NEXT:   No successors
; CHECK-NEXT: }
; CHECK-NEXT: Successor(s): then.1.1
; CHECK-EMPTY:
; CHECK-NEXT: then.1.1:
; CHECK-NEXT:   EMIT vp<[[CAN_IV_NEXT:%.+]]> = VF * UF + vp<[[CAN_IV]]>
; CHECK-NEXT:   EMIT branch-on-count vp<[[CAN_IV_NEXT]]> vp<[[VEC_TC]]>
; CHECK-NEXT: No successors
; CHECK-NEXT: }
;
entry:
  br label %loop

loop:
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %next.1 ]
  %gep.b = getelementptr inbounds [2048 x i32], ptr @b, i32 0, i32 %iv
  %mul = mul i32 %iv, 10
  %gep.a = getelementptr inbounds [2048 x i32], ptr @a, i32 0, i32 %mul
  %gep.c = getelementptr inbounds [2048 x i32], ptr @c, i32 0, i32 %mul
  %c.0 = icmp ult i32 %iv, %j
  br i1 %c.0, label %then.0, label %next.0

then.0:
  %lv.b  = load i32, ptr %gep.b, align 4
  br label %next.0

next.0:
  %p = phi i32 [ 0, %loop ], [ %lv.b, %then.0 ]
  br i1 %c.0, label %then.1, label %next.1

then.1:
  store i32 0, ptr %gep.a, align 4
  store i32 %p, ptr %gep.c, align 4
  br label %next.1

next.1:
  %iv.next = add i32 %iv, 1
  %large = icmp sge i32 %iv, 8
  %exitcond = icmp eq i32 %iv, %k
  %realexit = or i1 %large, %exitcond
  br i1 %realexit, label %exit, label %loop

exit:
  ret void
}

define void @merge_3_replicate_region(i32 %k, i32 %j) {
; CHECK-LABEL: LV: Checking a loop in 'merge_3_replicate_region'
; CHECK:      VPlan 'Initial VPlan for VF={2},UF>=1' {
; CHECK-NEXT: Live-in vp<[[VEC_TC:%.+]]> = vector-trip-count
; CHECK-EMPTY:
; CHECK-NEXT: Live-in vp<[[BTC:%.+]]> = backedge-taken count
; CHECK-EMPTY:
; CHECK-NEXT: vector.ph:
; CHECK-NEXT: Successor(s): vector loop
; CHECK-EMPTY:
; CHECK-NEXT: <x1> vector loop: {
; CHECK-NEXT: vector.body:
; CHECK-NEXT:   EMIT vp<[[CAN_IV:%.+]]> = CANONICAL-INDUCTION
; CHECK-NEXT:   WIDEN-INDUCTION %iv = phi 0, %iv.next, ir<1>
; CHECK-NEXT:   vp<[[STEPS:%.+]]> = SCALAR-STEPS vp<[[CAN_IV]]>, ir<1>
; CHECK-NEXT:   EMIT vp<[[MASK:%.+]]> = icmp ule ir<%iv> vp<[[BTC]]>
; CHECK-NEXT: Successor(s): pred.store
; CHECK-EMPTY:
; CHECK-NEXT: <xVFxUF> pred.store: {
; CHECK-NEXT:   pred.store.entry:
; CHECK-NEXT:     BRANCH-ON-MASK vp<[[MASK]]>
; CHECK-NEXT:   Successor(s): pred.store.if, pred.store.continue
; CHECK-EMPTY:
; CHECK-NEXT:   pred.store.if:
; CHECK-NEXT:     REPLICATE ir<%gep.a> = getelementptr ir<@a>, ir<0>, vp<[[STEPS]]>
; CHECK-NEXT:     REPLICATE ir<%lv.a> = load ir<%gep.a>
; CHECK-NEXT:     REPLICATE ir<%gep.b> = getelementptr ir<@b>, ir<0>, vp<[[STEPS]]>
; CHECK-NEXT:     REPLICATE ir<%lv.b> = load ir<%gep.b>
; CHECK-NEXT:     REPLICATE ir<%gep.c> = getelementptr ir<@c>, ir<0>, vp<[[STEPS]]>
; CHECK-NEXT:     REPLICATE store ir<%lv.a>, ir<%gep.c>
; CHECK-NEXT:     REPLICATE store ir<%lv.b>, ir<%gep.a>
; CHECK-NEXT:   Successor(s): pred.store.continue
; CHECK-EMPTY:
; CHECK-NEXT:   pred.store.continue:
; CHECK-NEXT:     PHI-PREDICATED-INSTRUCTION vp<[[PRED1:%.+]]> = ir<%lv.a>
; CHECK-NEXT:     PHI-PREDICATED-INSTRUCTION vp<[[PRED2:%.+]]> = ir<%lv.b>
; CHECK-NEXT:   No successors
; CHECK-NEXT: }
; CHECK-NEXT: Successor(s): loop.3
; CHECK-EMPTY:
; CHECK-NEXT: loop.3:
; CHECK-NEXT:   WIDEN ir<%c.0> = icmp ult ir<%iv>, ir<%j>
; CHECK-NEXT:   WIDEN ir<%mul> = mul vp<[[PRED1]]>, vp<[[PRED2]]>
; CHECK-NEXT:   EMIT vp<[[MASK2:%.+]]> = select vp<[[MASK]]> ir<%c.0> ir<false>
; CHECK-NEXT: Successor(s): pred.store
; CHECK-EMPTY:
; CHECK-NEXT: <xVFxUF> pred.store: {
; CHECK-NEXT:   pred.store.entry:
; CHECK-NEXT:     BRANCH-ON-MASK vp<[[MASK2]]>
; CHECK-NEXT:   Successor(s): pred.store.if, pred.store.continue
; CHECK-EMPTY:
; CHECK-NEXT:   pred.store.if:
; CHECK-NEXT:     REPLICATE ir<%gep.c.1> = getelementptr ir<@c>, ir<0>, vp<[[STEPS]]>
; CHECK-NEXT:     REPLICATE store ir<%mul>, ir<%gep.c.1>
; CHECK-NEXT:   Successor(s): pred.store.continue
; CHECK-EMPTY:
; CHECK-NEXT:   pred.store.continue:
; CHECK-NEXT:   No successors
; CHECK-NEXT: }
; CHECK-NEXT: Successor(s): then.0.0
; CHECK-EMPTY:
; CHECK-NEXT: then.0.0:
; CHECK-NEXT:   EMIT vp<[[CAN_IV_NEXT:%.+]]> = VF * UF + vp<[[CAN_IV]]>
; CHECK-NEXT:   EMIT branch-on-count vp<[[CAN_IV_NEXT]]> vp<[[VEC_TC]]>
; CHECK-NEXT: No successors
; CHECK-NEXT: }
;
entry:
  br label %loop

loop:
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %latch ]
  %gep.a = getelementptr inbounds [2048 x i32], ptr @a, i32 0, i32 %iv
  %lv.a  = load i32, ptr %gep.a, align 4
  %gep.b = getelementptr inbounds [2048 x i32], ptr @b, i32 0, i32 %iv
  %lv.b  = load i32, ptr %gep.b, align 4
  %gep.c = getelementptr inbounds [2048 x i32], ptr @c, i32 0, i32 %iv
  store i32 %lv.a, ptr %gep.c, align 4
  store i32 %lv.b, ptr %gep.a, align 4
  %c.0 = icmp ult i32 %iv, %j
  br i1 %c.0, label %then.0, label %latch

then.0:
  %mul = mul i32 %lv.a, %lv.b
  %gep.c.1 = getelementptr inbounds [2048 x i32], ptr @c, i32 0, i32 %iv
  store i32 %mul, ptr %gep.c.1, align 4
  br label %latch

latch:
  %iv.next = add i32 %iv, 1
  %large = icmp sge i32 %iv, 8
  %exitcond = icmp eq i32 %iv, %k
  %realexit = or i1 %large, %exitcond
  br i1 %realexit, label %exit, label %loop

exit:
  ret void
}


define void @update_2_uses_in_same_recipe_in_merged_block(i32 %k) {
; CHECK-LABEL: LV: Checking a loop in 'update_2_uses_in_same_recipe_in_merged_block'
; CHECK:      VPlan 'Initial VPlan for VF={2},UF>=1' {
; CHECK-NEXT: Live-in vp<[[VEC_TC:%.+]]> = vector-trip-count
; CHECK-EMPTY:
; CHECK-NEXT: Live-in vp<[[BTC:%.+]]> = backedge-taken count
; CHECK-EMPTY:
; CHECK-NEXT: vector.ph:
; CHECK-NEXT: Successor(s): vector loop
; CHECK-EMPTY:
; CHECK-NEXT: <x1> vector loop: {
; CHECK-NEXT: vector.body:
; CHECK-NEXT:   EMIT vp<[[CAN_IV:%.+]]> = CANONICAL-INDUCTION
; CHECK-NEXT:   WIDEN-INDUCTION %iv = phi 0, %iv.next, ir<1>
; CHECK-NEXT:   EMIT vp<[[MASK:%.+]]> = icmp ule ir<%iv> vp<[[BTC]]>
; CHECK-NEXT: Successor(s): pred.store
; CHECK-EMPTY:
; CHECK-NEXT: <xVFxUF> pred.store: {
; CHECK-NEXT:   pred.store.entry:
; CHECK-NEXT:     BRANCH-ON-MASK vp<[[MASK]]>
; CHECK-NEXT:   Successor(s): pred.store.if, pred.store.continue
; CHECK-EMPTY:
; CHECK-NEXT:   pred.store.if:
; CHECK-NEXT:     vp<[[STEPS:%.+]]> = SCALAR-STEPS vp<[[CAN_IV]]>, ir<1>
; CHECK-NEXT:     REPLICATE ir<%gep.a> = getelementptr ir<@a>, ir<0>, vp<[[STEPS]]>
; CHECK-NEXT:     REPLICATE ir<%lv.a> = load ir<%gep.a>
; CHECK-NEXT:     REPLICATE ir<%div> = sdiv ir<%lv.a>, ir<%lv.a>
; CHECK-NEXT:     REPLICATE store ir<%div>, ir<%gep.a>
; CHECK-NEXT:   Successor(s): pred.store.continue
; CHECK-EMPTY:
; CHECK-NEXT:   pred.store.continue:
; CHECK-NEXT:     PHI-PREDICATED-INSTRUCTION vp<[[PRED1:%.+]]> = ir<%lv.a>
; CHECK-NEXT:     PHI-PREDICATED-INSTRUCTION vp<[[PRED2:%.+]]> = ir<%div>
; CHECK-NEXT:   No successors
; CHECK-NEXT: }
; CHECK-NEXT: Successor(s): loop.2
; CHECK-EMPTY:
; CHECK-NEXT: loop.2:
; CHECK-NEXT:   EMIT vp<[[CAN_IV_NEXT:%.+]]> = VF * UF + vp<[[CAN_IV]]>
; CHECK-NEXT:   EMIT branch-on-count vp<[[CAN_IV_NEXT]]> vp<[[VEC_TC]]>
; CHECK-NEXT: No successors
; CHECK-NEXT: }
;
entry:
  br label %loop

loop:
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %loop ]
  %gep.a = getelementptr inbounds [2048 x i32], ptr @a, i32 0, i32 %iv
  %lv.a  = load i32, ptr %gep.a, align 4
  %div = sdiv i32 %lv.a, %lv.a
  store i32 %div, ptr %gep.a, align 4
  %iv.next = add i32 %iv, 1
  %large = icmp sge i32 %iv, 8
  %exitcond = icmp eq i32 %iv, %k
  %realexit = or i1 %large, %exitcond
  br i1 %realexit, label %exit, label %loop

exit:
  ret void
}

define void @recipe_in_merge_candidate_used_by_first_order_recurrence(i32 %k) {
; CHECK-LABEL: LV: Checking a loop in 'recipe_in_merge_candidate_used_by_first_order_recurrence'
; CHECK:      VPlan 'Initial VPlan for VF={2},UF>=1' {
; CHECK-NEXT: Live-in vp<[[VEC_TC:%.+]]> = vector-trip-count
; CHECK-EMPTY:
; CHECK-NEXT: Live-in vp<[[BTC:%.+]]> = backedge-taken count
; CHECK-EMPTY:
; CHECK-NEXT: vector.ph:
; CHECK-NEXT: Successor(s): vector loop
; CHECK-EMPTY:
; CHECK-NEXT: <x1> vector loop: {
; CHECK-NEXT: vector.body:
; CHECK-NEXT:   EMIT vp<[[CAN_IV:%.+]]> = CANONICAL-INDUCTION
; CHECK-NEXT:   WIDEN-INDUCTION %iv = phi 0, %iv.next, ir<1>
; CHECK-NEXT:   FIRST-ORDER-RECURRENCE-PHI ir<%for> = phi ir<0>, vp<[[PRED:%.+]]>
; CHECK-NEXT:   vp<[[STEPS:%.+]]> = SCALAR-STEPS vp<[[CAN_IV]]>, ir<1>
; CHECK-NEXT:   EMIT vp<[[MASK:%.+]]> = icmp ule ir<%iv> vp<[[BTC]]>
; CHECK-NEXT:   REPLICATE ir<%gep.a> = getelementptr ir<@a>, ir<0>, vp<[[STEPS]]>
; CHECK-NEXT: Successor(s): pred.load
; CHECK-EMPTY:
; CHECK-NEXT: <xVFxUF> pred.load: {
; CHECK-NEXT:   pred.load.entry:
; CHECK-NEXT:     BRANCH-ON-MASK vp<[[MASK]]>
; CHECK-NEXT:   Successor(s): pred.load.if, pred.load.continue
; CHECK-EMPTY:
; CHECK-NEXT:   pred.load.if:
; CHECK-NEXT:     REPLICATE ir<%lv.a> = load ir<%gep.a>
; CHECK-NEXT:   Successor(s): pred.load.continue
; CHECK-EMPTY:
; CHECK-NEXT:   pred.load.continue:
; CHECK-NEXT:     PHI-PREDICATED-INSTRUCTION vp<[[PRED]]> = ir<%lv.a>
; CHECK-NEXT:   No successors
; CHECK-NEXT: }
; CHECK-NEXT: Successor(s): loop.0
; CHECK-EMPTY:
; CHECK-NEXT: loop.0:
; CHECK-NEXT:   EMIT vp<[[SPLICE:%.+]]> = first-order splice ir<%for> vp<[[PRED]]>
; CHECK-NEXT: Successor(s): pred.store
; CHECK-EMPTY:
; CHECK-NEXT: <xVFxUF> pred.store: {
; CHECK-NEXT:   pred.store.entry:
; CHECK-NEXT:     BRANCH-ON-MASK vp<[[MASK]]>
; CHECK-NEXT:   Successor(s): pred.store.if, pred.store.continue
; CHECK-EMPTY:
; CHECK-NEXT:   pred.store.if:
; CHECK-NEXT:     REPLICATE ir<%div> = sdiv vp<[[SPLICE]]>, vp<[[PRED]]>
; CHECK-NEXT:     REPLICATE store ir<%div>, ir<%gep.a>
; CHECK-NEXT:   Successor(s): pred.store.continue
; CHECK-EMPTY:
; CHECK-NEXT:   pred.store.continue:
; CHECK-NEXT:     PHI-PREDICATED-INSTRUCTION vp<[[PRED2:%.+]]> = ir<%div>
; CHECK-NEXT:   No successors
; CHECK-NEXT: }
; CHECK-NEXT: Successor(s): loop.2
; CHECK-EMPTY:
; CHECK-NEXT: loop.2:
; CHECK-NEXT:   EMIT vp<[[CAN_IV_NEXT:%.+]]> = VF * UF + vp<[[CAN_IV]]>
; CHECK-NEXT:   EMIT branch-on-count vp<[[CAN_IV_NEXT]]> vp<[[VEC_TC]]>
; CHECK-NEXT: No successors
; CHECK-NEXT: }
;
entry:
  br label %loop

loop:
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %loop ]
  %for = phi i32 [ 0, %entry ], [ %lv.a, %loop ]
  %gep.a = getelementptr inbounds [2048 x i32], ptr @a, i32 0, i32 %iv
  %lv.a  = load i32, ptr %gep.a, align 4
  %div = sdiv i32 %for, %lv.a
  store i32 %div, ptr %gep.a, align 4
  %iv.next = add i32 %iv, 1
  %large = icmp sge i32 %iv, 8
  %exitcond = icmp eq i32 %iv, %k
  %realexit = or i1 %large, %exitcond
  br i1 %realexit, label %exit, label %loop

exit:
  ret void
}

define void @update_multiple_users(ptr noalias %src, ptr noalias %dst, i1 %c) {
; CHECK-LABEL: LV: Checking a loop in 'update_multiple_users'
; CHECK:      VPlan 'Initial VPlan for VF={2},UF>=1' {
; CHECK-NEXT: Live-in vp<[[VEC_TC:%.+]]> = vector-trip-count
; CHECK-EMPTY:
; CHECK-NEXT: vector.ph:
; CHECK-NEXT: Successor(s): vector loop
; CHECK-EMPTY:
; CHECK-NEXT: <x1> vector loop: {
; CHECK-NEXT: vector.body:
; CHECK-NEXT:   EMIT vp<[[CAN_IV:%.+]]> = CANONICAL-INDUCTION
; CHECK-NEXT: Successor(s): pred.store
; CHECK-EMPTY:
; CHECK-NEXT: <xVFxUF> pred.store: {
; CHECK-NEXT:   pred.store.entry:
; CHECK-NEXT:     BRANCH-ON-MASK ir<%c>
; CHECK-NEXT:   Successor(s): pred.store.if, pred.store.continue
; CHECK-EMPTY:
; CHECK-NEXT:   pred.store.if:
; CHECK-NEXT:     REPLICATE ir<%l1> = load ir<%src>
; CHECK-NEXT:     REPLICATE ir<%l2> = trunc ir<%l1>
; CHECK-NEXT:     REPLICATE ir<%cmp> = icmp ir<%l1>, ir<0>
; CHECK-NEXT:     REPLICATE ir<%sel> = select ir<%cmp>, ir<5>, ir<%l2>
; CHECK-NEXT:     REPLICATE store ir<%sel>, ir<%dst>
; CHECK-NEXT:   Successor(s): pred.store.continue
; CHECK-EMPTY:
; CHECK-NEXT:   pred.store.continue:
; CHECK-NEXT:     PHI-PREDICATED-INSTRUCTION vp<[[PRED:%.+]]> = ir<%l1>
; CHECK-NEXT:   No successors
; CHECK-NEXT: }
; CHECK-NEXT: Successor(s): loop.then.1
; CHECK-EMPTY:
; CHECK-NEXT: loop.then.1:
; CHECK-NEXT:   EMIT vp<[[CAN_IV_NEXT:%.+]]> = VF * UF +(nuw) vp<[[CAN_IV]]>
; CHECK-NEXT:   EMIT branch-on-count vp<[[CAN_IV_NEXT]]> vp<[[VEC_TC]]>
; CHECK-NEXT: No successors
; CHECK-NEXT: }
;
entry:
  br label %loop.header

loop.header:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop.latch ]
  br i1 %c, label %loop.then, label %loop.latch

loop.then:
  %l1 = load i16, ptr %src, align 2
  %l2 = trunc i16 %l1 to i8
  %cmp = icmp eq i16 %l1, 0
  %sel = select i1 %cmp, i8 5, i8 %l2
  store i8 %sel, ptr %dst, align 1
  %sext.l1 = sext i16 %l1 to i32
  br label %loop.latch

loop.latch:
  %iv.next = add nsw i64 %iv, 1
  %ec = icmp eq i64 %iv.next, 999
  br i1 %ec, label %exit, label %loop.header

exit:
  ret void
}

define void @sinking_requires_duplication(ptr %addr) {
; CHECK-LABEL: LV: Checking a loop in 'sinking_requires_duplication'
; CHECK:      VPlan 'Initial VPlan for VF={2},UF>=1' {
; CHECK-NEXT: Live-in vp<[[VEC_TC:%.+]]> = vector-trip-count
; CHECK-EMPTY:
; CHECK-NEXT: vector.ph:
; CHECK-NEXT: Successor(s): vector loop
; CHECK-EMPTY:
; CHECK-NEXT: <x1> vector loop: {
; CHECK-NEXT: vector.body:
; CHECK-NEXT:   EMIT vp<[[CAN_IV:%.+]]> = CANONICAL-INDUCTION
; CHECK-NEXT:   vp<[[STEPS:%.+]]> = SCALAR-STEPS vp<[[CAN_IV]]>, ir<1>
; CHECK-NEXT:   CLONE ir<%gep> = getelementptr ir<%addr>, vp<[[STEPS]]>
; CHECK-NEXT:   WIDEN ir<%0> = load ir<%gep>
; CHECK-NEXT:   WIDEN ir<%pred> = fcmp oeq ir<%0>, ir<0.000000e+00>
; CHECK-NEXT:   EMIT vp<[[MASK:%.+]]> = not ir<%pred>
; CHECK-NEXT: Successor(s): pred.store
; CHECK-EMPTY:
; CHECK-NEXT: <xVFxUF> pred.store: {
; CHECK-NEXT:   pred.store.entry:
; CHECK-NEXT:     BRANCH-ON-MASK vp<[[MASK]]>
; CHECK-NEXT:   Successor(s): pred.store.if, pred.store.continue
; CHECK-EMPTY:
; CHECK-NEXT:   pred.store.if:
; CHECK-NEXT:     REPLICATE ir<%gep> = getelementptr ir<%addr>, vp<[[STEPS]]>
; CHECK-NEXT:     REPLICATE store ir<1.000000e+01>, ir<%gep>
; CHECK-NEXT:   Successor(s): pred.store.continue
; CHECK-EMPTY:
; CHECK-NEXT:   pred.store.continue:
; CHECK-NEXT:   No successors
; CHECK-NEXT: }
; CHECK-NEXT: Successor(s): then.0
; CHECK-EMPTY:
; CHECK-NEXT: then.0:
; CHECK-NEXT:   EMIT vp<[[CAN_IV_NEXT:%.+]]> = VF * UF +(nuw) vp<[[CAN_IV]]>
; CHECK-NEXT:   EMIT branch-on-count vp<[[CAN_IV_NEXT]]> vp<[[VEC_TC]]>
; CHECK-NEXT: No successors
; CHECK-NEXT: }
;
entry:
  br label %loop.header

loop.header:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop.latch ]
  %gep = getelementptr float, ptr %addr, i64 %iv
  %exitcond.not = icmp eq i64 %iv, 200
  br i1 %exitcond.not, label %exit, label %loop.body

loop.body:
  %0 = load float, ptr %gep, align 4
  %pred = fcmp oeq float %0, 0.0
  br i1 %pred, label %loop.latch, label %then

then:
  store float 10.0, ptr %gep, align 4
  br label %loop.latch

loop.latch:
  %iv.next = add nuw nsw i64 %iv, 1
  br label %loop.header

exit:
  ret void
}

; Test case with a dead GEP between the load and store regions. Dead recipes
; need to be removed before merging.
define void @merge_with_dead_gep_between_regions(i32 %n, ptr noalias %src, ptr noalias %dst) optsize {
; CHECK-LABEL: LV: Checking a loop in 'merge_with_dead_gep_between_regions'
; CHECK:      VPlan 'Initial VPlan for VF={2},UF>=1' {
; CHECK-NEXT: Live-in vp<[[VEC_TC:%.+]]> = vector-trip-count
; CHECK-EMPTY:
; CHECK-NEXT: Live-in vp<[[BTC:%.+]]> = backedge-taken count
; CHECK-EMPTY:
; CHECK-NEXT: vector.ph:
; CHECK-NEXT: Successor(s): vector loop
; CHECK-EMPTY:
; CHECK-NEXT: <x1> vector loop: {
; CHECK-NEXT:   vector.body:
; CHECK-NEXT:     EMIT vp<[[CAN_IV:%.+]]> = CANONICAL-INDUCTION
; CHECK-NEXT:     vp<[[DERIVED_IV:%.+]]> = DERIVED-IV ir<%n> + vp<[[CAN_IV]]> * ir<-1>
; CHECK-NEXT:     EMIT vp<[[WIDE_IV:%.+]]> = WIDEN-CANONICAL-INDUCTION vp<[[CAN_IV]]>
; CHECK-NEXT:     EMIT vp<[[MASK:%.+]]> = icmp ule vp<[[WIDE_IV]]> vp<[[BTC]]>
; CHECK-NEXT:   Successor(s): pred.store
; CHECK-EMPTY:
; CHECK-NEXT:   <xVFxUF> pred.store: {
; CHECK-NEXT:     pred.store.entry:
; CHECK-NEXT:       BRANCH-ON-MASK vp<[[MASK]]>
; CHECK-NEXT:     Successor(s): pred.store.if, pred.store.continue
; CHECK-EMPTY:
; CHECK-NEXT:     pred.store.if:
; CHECK-NEXT:       vp<[[SCALAR_STEPS:%.+]]>    = SCALAR-STEPS vp<[[DERIVED_IV]]>, ir<-1>
; CHECK-NEXT:       REPLICATE ir<%gep.src> = getelementptr ir<%src>, vp<[[SCALAR_STEPS]]>
; CHECK-NEXT:       REPLICATE ir<%l> = load ir<%gep.src>
; CHECK-NEXT:       REPLICATE ir<%gep.dst> = getelementptr ir<%dst>, vp<[[SCALAR_STEPS]]>
; CHECK-NEXT:       REPLICATE store ir<%l>, ir<%gep.dst>
; CHECK-NEXT:     Successor(s): pred.store.continue
; CHECK-EMPTY:
; CHECK-NEXT:     pred.store.continue:
; CHECK-NEXT:       PHI-PREDICATED-INSTRUCTION vp<[[P_LOAD:%.+]]> = ir<%l>
; CHECK-NEXT:     No successors
; CHECK-NEXT:   }
; CHECK-NEXT:   Successor(s): loop.1
; CHECK-EMPTY:
; CHECK-NEXT:   loop.1:
; CHECK-NEXT:     EMIT vp<[[CAN_IV_NEXT:%.+]]> = VF * UF +  vp<[[CAN_IV]]>
; CHECK-NEXT:     EMIT branch-on-count  vp<[[CAN_IV_NEXT]]> vp<[[VEC_TC]]>
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
  %iv = phi i32[ %n, %entry ], [ %iv.next, %loop ]
  %iv.next = add nsw i32 %iv, -1
  %gep.src = getelementptr inbounds i32, ptr %src, i32 %iv
  %l = load i32, ptr %gep.src, align 16
  %dead_gep = getelementptr inbounds i32, ptr %dst, i64 1
  %gep.dst = getelementptr inbounds i32, ptr %dst, i32 %iv
  store i32 %l, ptr %gep.dst, align 16
  %ec = icmp eq i32 %iv.next, 0
  br i1 %ec, label %exit, label %loop

exit:
  ret void
}

define void @ptr_induction_remove_dead_recipe(ptr %start, ptr %end) {
; CHECK-LABEL: LV: Checking a loop in 'ptr_induction_remove_dead_recipe'
; CHECK:       VPlan 'Initial VPlan for VF={2},UF>=1' {
; CHECK-NEXT: Live-in vp<[[VEC_TC:%.+]]> = vector-trip-count
; CHECK-EMPTY:
; CHECK-NEXT: vector.ph:
; CHECK-NEXT: Successor(s): vector loop
; CHECK-EMPTY:
; CHECK-NEXT: <x1> vector loop: {
; CHECK-NEXT:   vector.body:
; CHECK-NEXT:     EMIT vp<[[CAN_IV:%.+]]> = CANONICAL-INDUCTION
; CHECK-NEXT:     EMIT ir<%ptr.iv> = WIDEN-POINTER-INDUCTION ir<%start>, -1
; CHECK-NEXT:     CLONE ir<%ptr.iv.next> = getelementptr ir<%ptr.iv>, ir<-1>
; CHECK-NEXT:     WIDEN ir<%l> = load ir<%ptr.iv.next>
; CHECK-NEXT:     WIDEN ir<%c.1> = icmp eq ir<%l>, ir<0>
; CHECK-NEXT:     EMIT vp<[[NEG:%.+]]> = not ir<%c.1>
; CHECK-NEXT:   Successor(s): pred.store
; CHECK-EMPTY:
; CHECK-NEXT:   <xVFxUF> pred.store: {
; CHECK-NEXT:     pred.store.entry:
; CHECK-NEXT:       BRANCH-ON-MASK vp<[[NEG]]>
; CHECK-NEXT:     Successor(s): pred.store.if, pred.store.continue
; CHECK-EMPTY:
; CHECK-NEXT:     pred.store.if:
; CHECK-NEXT:       REPLICATE ir<%ptr.iv.next> = getelementptr ir<%ptr.iv>, ir<-1>
; CHECK-NEXT:       REPLICATE store ir<95>, ir<%ptr.iv.next>
; CHECK-NEXT:     Successor(s): pred.store.continue
; CHECK-EMPTY:
; CHECK-NEXT:     pred.store.continue:
; CHECK-NEXT:     No successors
; CHECK-NEXT:   }
; CHECK-NEXT:   Successor(s): if.then.0
; CHECK-EMPTY:
; CHECK-NEXT:   if.then.0:
; CHECK-NEXT:     EMIT vp<[[CAN_IV_NEXT:%.+]]> = VF * UF +(nuw)  vp<[[CAN_IV]]>
; CHECK-NEXT:     EMIT branch-on-count  vp<[[CAN_IV_NEXT]]> vp<[[VEC_TC]]>
; CHECK-NEXT:   No successors
; CHECK-NEXT: }
; CHECK-NEXT: Successor(s): middle.block
; CHECK-EMPTY:
; CHECK-NEXT: middle.block:
; CHECK-NEXT: No successors
; CHECK-NEXT: }
;
entry:
  br label %loop.header

loop.header:
  %ptr.iv = phi ptr [ %start, %entry ], [ %ptr.iv.next, %loop.latch ]
  %ptr.iv.next = getelementptr inbounds i8, ptr %ptr.iv, i64 -1
  %l = load i8, ptr %ptr.iv.next, align 1
  %c.1 = icmp eq i8 %l, 0
  br i1 %c.1, label %loop.latch, label %if.then

if.then:
  store i8 95, ptr %ptr.iv.next, align 1
  br label %loop.latch

loop.latch:
  %c.2 = icmp eq ptr %ptr.iv.next, %end
  br i1 %c.2, label %exit, label %loop.header

exit:
  ret void
}
