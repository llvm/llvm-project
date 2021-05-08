; REQUIRES: asserts

; RUN: opt -loop-vectorize -force-vector-interleave=1 -force-vector-width=2 -debug -disable-output %s 2>&1 | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"

@a = common global [2048 x i32] zeroinitializer, align 16
@b = common global [2048 x i32] zeroinitializer, align 16
@c = common global [2048 x i32] zeroinitializer, align 16


; CHECK-LABEL: LV: Checking a loop in "sink1"
; CHECK:      VPlan 'Initial VPlan for VF={2},UF>=1' {
; CHECK-NEXT: loop:
; CHECK-NEXT:   WIDEN-INDUCTION %indvars.iv = phi 0, %indvars.iv.next
; CHECK-NEXT:   EMIT vp<%2> = icmp ule ir<%indvars.iv> vp<%0>
; CHECK-NEXT:   REPLICATE ir<%gep.b> = getelementptr ir<@b>, ir<0>, ir<%indvars.iv>
; CHECK-NEXT: Successor(s): pred.load

; CHECK:       <xVFxUF> pred.load: {
; CHECK-NEXT:   pred.load.entry:
; CHECK-NEXT:     BRANCH-ON-MASK vp<%2>
; CHECK-NEXT:   Successor(s): pred.load.if, pred.load.continue
; CHECK-NEXT:   CondBit: vp<%2> (loop)

; CHECK:       pred.load.if:
; CHECK-NEXT:     REPLICATE ir<%lv.b> = load ir<%gep.b>
; CHECK-NEXT:   Successor(s): pred.load.continue

; CHECK:      pred.load.continue:
; CHECK-NEXT:     PHI-PREDICATED-INSTRUCTION vp<%5> = ir<%lv.b>
; CHECK-NEXT:   No successors
; CHECK-NEXT: }

; CHECK:      loop.0:
; CHECK-NEXT:   REPLICATE ir<%add> = add vp<%5>, ir<10>
; CHECK-NEXT:   REPLICATE ir<%mul> = mul ir<2>, ir<%add>
; CHECK-NEXT:   REPLICATE ir<%gep.a> = getelementptr ir<@a>, ir<0>, ir<%indvars.iv>
; CHECK-NEXT: Successor(s): pred.store

; CHECK:      <xVFxUF> pred.store: {
; CHECK-NEXT:   pred.store.entry:
; CHECK-NEXT:     BRANCH-ON-MASK vp<%2>
; CHECK-NEXT:   Successor(s): pred.store.if, pred.store.continue
; CHECK-NEXT:   CondBit: vp<%2> (loop)

; CHECK:      pred.store.if:
; CHECK-NEXT:     REPLICATE store ir<%mul>, ir<%gep.a>
; CHECK-NEXT:   Successor(s): pred.store.continue

; CHECK:      pred.store.continue:
; CHECK-NEXT:   No successors
; CHECK-NEXT: }

; CHECK:      loop.1:
; CHECK-NEXT:   CLONE ir<%large> = icmp ir<%indvars.iv>, ir<8>
; CHECK-NEXT:   CLONE ir<%exitcond> = icmp ir<%indvars.iv>, ir<%k>
; CHECK-NEXT: No successors
; CHECK-NEXT: }
;
define void @sink1(i32 %k) {
entry:
  br label %loop

loop:
  %indvars.iv = phi i32 [ 0, %entry ], [ %indvars.iv.next, %loop ]
  %gep.b = getelementptr inbounds [2048 x i32], [2048 x i32]* @b, i32 0, i32 %indvars.iv
  %lv.b  = load i32, i32* %gep.b, align 4
  %add = add i32 %lv.b, 10
  %mul = mul i32 2, %add
  %gep.a = getelementptr inbounds [2048 x i32], [2048 x i32]* @a, i32 0, i32 %indvars.iv
  store i32 %mul, i32* %gep.a, align 4
  %indvars.iv.next = add i32 %indvars.iv, 1
  %large = icmp sge i32 %indvars.iv, 8
  %exitcond = icmp eq i32 %indvars.iv, %k
  %realexit = or i1 %large, %exitcond
  br i1 %realexit, label %exit, label %loop

exit:
  ret void
}

; CHECK-LABEL: LV: Checking a loop in "sink2"
; CHECK:      VPlan 'Initial VPlan for VF={2},UF>=1' {
; CHECK-NEXT: loop:
; CHECK-NEXT:   WIDEN-INDUCTION %indvars.iv = phi 0, %indvars.iv.next
; CHECK-NEXT:   EMIT vp<%2> = icmp ule ir<%indvars.iv> vp<%0>
; CHECK-NEXT:   REPLICATE ir<%gep.b> = getelementptr ir<@b>, ir<0>, ir<%indvars.iv>
; CHECK-NEXT: Successor(s): pred.load

; CHECK:      <xVFxUF> pred.load: {
; CHECK-NEXT:   pred.load.entry:
; CHECK-NEXT:     BRANCH-ON-MASK vp<%2>
; CHECK-NEXT:   Successor(s): pred.load.if, pred.load.continue
; CHECK-NEXT:   CondBit: vp<%2> (loop)

; CHECK:      pred.load.if:
; CHECK-NEXT:     REPLICATE ir<%lv.b> = load ir<%gep.b>
; CHECK-NEXT:   Successor(s): pred.load.continue

; CHECK:      pred.load.continue:
; CHECK-NEXT:     PHI-PREDICATED-INSTRUCTION vp<%5> = ir<%lv.b>
; CHECK-NEXT:   No successors
; CHECK-NEXT: }

; CHECK:      loop.0:
; CHECK-NEXT:   REPLICATE ir<%add> = add vp<%5>, ir<10>
; CHECK-NEXT:   WIDEN ir<%mul> = mul ir<%indvars.iv>, ir<2>
; CHECK-NEXT:   REPLICATE ir<%gep.a> = getelementptr ir<@a>, ir<0>, ir<%mul>
; CHECK-NEXT: Successor(s): pred.store

; CHECK:      <xVFxUF> pred.store: {
; CHECK-NEXT:   pred.store.entry:
; CHECK-NEXT:     BRANCH-ON-MASK vp<%2>
; CHECK-NEXT:   Successor(s): pred.store.if, pred.store.continue
; CHECK-NEXT:   CondBit: vp<%2> (loop)

; CHECK:       pred.store.if:
; CHECK-NEXT:     REPLICATE store ir<%add>, ir<%gep.a>
; CHECK-NEXT:   Successor(s): pred.store.continue

; CHECK:      pred.store.continue:
; CHECK-NEXT:   No successors
; CHECK-NEXT: }

; CHECK:       loop.1:
; CHECK-NEXT:   CLONE ir<%large> = icmp ir<%indvars.iv>, ir<8>
; CHECK-NEXT:   CLONE ir<%exitcond> = icmp ir<%indvars.iv>, ir<%k>
; CHECK-NEXT: No successors
; CHECK-NEXT: }
;
define void @sink2(i32 %k) {
entry:
  br label %loop

loop:
  %indvars.iv = phi i32 [ 0, %entry ], [ %indvars.iv.next, %loop ]
  %gep.b = getelementptr inbounds [2048 x i32], [2048 x i32]* @b, i32 0, i32 %indvars.iv
  %lv.b  = load i32, i32* %gep.b, align 4
  %add = add i32 %lv.b, 10
  %mul = mul i32 %indvars.iv, 2
  %gep.a = getelementptr inbounds [2048 x i32], [2048 x i32]* @a, i32 0, i32 %mul
  store i32 %add, i32* %gep.a, align 4
  %indvars.iv.next = add i32 %indvars.iv, 1
  %large = icmp sge i32 %indvars.iv, 8
  %exitcond = icmp eq i32 %indvars.iv, %k
  %realexit = or i1 %large, %exitcond
  br i1 %realexit, label %exit, label %loop

exit:
  ret void
}

; CHECK-LABEL: LV: Checking a loop in "sink3"
; CHECK:      VPlan 'Initial VPlan for VF={2},UF>=1' {
; CHECK-NEXT: loop:
; CHECK-NEXT:   WIDEN-INDUCTION %indvars.iv = phi 0, %indvars.iv.next
; CHECK-NEXT:   EMIT vp<%2> = icmp ule ir<%indvars.iv> vp<%0>
; CHECK-NEXT:   REPLICATE ir<%gep.b> = getelementptr ir<@b>, ir<0>, ir<%indvars.iv>
; CHECK-NEXT: Successor(s): pred.load

; CHECK:      <xVFxUF> pred.load: {
; CHECK-NEXT:   pred.load.entry:
; CHECK-NEXT:     BRANCH-ON-MASK vp<%2>
; CHECK-NEXT:   Successor(s): pred.load.if, pred.load.continue
; CHECK-NEXT:   CondBit: vp<%2> (loop)

; CHECK:       pred.load.if:
; CHECK-NEXT:     REPLICATE ir<%lv.b> = load ir<%gep.b> (S->V)
; CHECK-NEXT:   Successor(s): pred.load.continue

; CHECK:       pred.load.continue:
; CHECK-NEXT:     PHI-PREDICATED-INSTRUCTION vp<%5> = ir<%lv.b>
; CHECK-NEXT:   No successors
; CHECK-NEXT: }

; CHECK:      loop.0:
; CHECK-NEXT:   WIDEN ir<%add> = add vp<%5>, ir<10>
; CHECK-NEXT:   WIDEN ir<%mul> = mul ir<%indvars.iv>, ir<%add>
; CHECK-NEXT:   REPLICATE ir<%gep.a> = getelementptr ir<@a>, ir<0>, ir<%mul>
; CHECK-NEXT: Successor(s): pred.store

; CHECK:      <xVFxUF> pred.store: {
; CHECK-NEXT:   pred.store.entry:
; CHECK-NEXT:     BRANCH-ON-MASK vp<%2>
; CHECK-NEXT:   Successor(s): pred.store.if, pred.store.continue
; CHECK-NEXT:   CondBit: vp<%2> (loop)

; CHECK:      pred.store.if:
; CHECK-NEXT:     REPLICATE store ir<%add>, ir<%gep.a>
; CHECK-NEXT:   Successor(s): pred.store.continue

; CHECK:      pred.store.continue:
; CHECK-NEXT:   No successors
; CHECK-NEXT: }

; CHECK:      loop.1:
; CHECK-NEXT:   CLONE ir<%large> = icmp ir<%indvars.iv>, ir<8>
; CHECK-NEXT:   CLONE ir<%exitcond> = icmp ir<%indvars.iv>, ir<%k>
; CHECK-NEXT: No successors
; CHECK-NEXT: }
;
define void @sink3(i32 %k) {
entry:
  br label %loop

loop:
  %indvars.iv = phi i32 [ 0, %entry ], [ %indvars.iv.next, %loop ]
  %gep.b = getelementptr inbounds [2048 x i32], [2048 x i32]* @b, i32 0, i32 %indvars.iv
  %lv.b  = load i32, i32* %gep.b, align 4
  %add = add i32 %lv.b, 10
  %mul = mul i32 %indvars.iv, %add
  %gep.a = getelementptr inbounds [2048 x i32], [2048 x i32]* @a, i32 0, i32 %mul
  store i32 %add, i32* %gep.a, align 4
  %indvars.iv.next = add i32 %indvars.iv, 1
  %large = icmp sge i32 %indvars.iv, 8
  %exitcond = icmp eq i32 %indvars.iv, %k
  %realexit = or i1 %large, %exitcond
  br i1 %realexit, label %exit, label %loop

exit:
  ret void
}
