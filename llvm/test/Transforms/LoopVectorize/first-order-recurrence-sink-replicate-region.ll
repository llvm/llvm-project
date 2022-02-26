; REQUIRES: asserts
; RUN: opt < %s -loop-vectorize -force-vector-width=2 -force-vector-interleave=1 -disable-output -debug-only=loop-vectorize 2>&1 | FileCheck %s

target datalayout = "e-m:e-i64:64-i128:128-n32:64-S128"

; Test cases for PR50009, which require sinking a replicate-region due to a
; first-order recurrence.

define void @sink_replicate_region_1(i32 %x, i8* %ptr) optsize {
; CHECK-LABEL: sink_replicate_region_1
; CHECK:      VPlan 'Initial VPlan for VF={2},UF>=1' {
; CHECK-NEXT: Live-in vp<[[VEC_TC:%.+]]> = vector-trip-count
; CHECK-EMPTY:
; CHECK-NEXT: Live-in vp<[[BTC:%.+]]> = backedge-taken count
; CHECK-EMPTY:
; CHECK-NEXT: <x1> vector loop: {
; CHECK-NEXT: loop:
; CHECK-NEXT:   EMIT vp<[[CAN_IV:%.+]]> = CANONICAL-INDUCTION
; CHECK-NEXT:   FIRST-ORDER-RECURRENCE-PHI ir<%0> = phi ir<0>, ir<%conv>
; CHECK-NEXT:   WIDEN-INDUCTION %iv = phi 0, %iv.next
; CHECK-NEXT:   EMIT vp<[[MASK:%.+]]> = icmp ule ir<%iv> vp<[[BTC]]>
; CHECK-NEXT: Successor(s): loop.0
; CHECK-EMPTY:
; CHECK-NEXT: loop.0:
; CHECK-NEXT: Successor(s): pred.load
; CHECK-EMPTY:
; CHECK-NEXT: <xVFxUF> pred.load: {
; CHECK-NEXT:   pred.load.entry:
; CHECK-NEXT:     BRANCH-ON-MASK vp<[[MASK]]>
; CHECK-NEXT:   Successor(s): pred.load.if, pred.load.continue
; CHECK-NEXT:   CondBit: vp<[[MASK]]> (loop)
; CHECK-EMPTY:
; CHECK-NEXT:   pred.load.if:
; CHECK-NEXT:     REPLICATE ir<%gep> = getelementptr ir<%ptr>, ir<%iv>
; CHECK-NEXT:     REPLICATE ir<%lv> = load ir<%gep> (S->V)
; CHECK-NEXT:   Successor(s): pred.load.continue
; CHECK-EMPTY:
; CHECK-NEXT:   pred.load.continue:
; CHECK-NEXT:     PHI-PREDICATED-INSTRUCTION vp<[[PRED1:%.+]]> = ir<%lv>
; CHECK-NEXT:   No successors
; CHECK-NEXT: }
; CHECK-NEXT: Successor(s): loop.1
; CHECK-EMPTY:
; CHECK-NEXT: loop.1:
; CHECK-NEXT:   WIDEN ir<%conv> = sext vp<[[PRED1]]>
; CHECK-NEXT:   EMIT vp<[[SPLICE:%.+]]> = first-order splice ir<%0> ir<%conv>
; CHECK-NEXT: Successor(s): pred.srem
; CHECK-EMPTY:
; CHECK-NEXT: <xVFxUF> pred.srem: {
; CHECK-NEXT:   pred.srem.entry:
; CHECK-NEXT:     BRANCH-ON-MASK vp<[[MASK]]>
; CHECK-NEXT:   Successor(s): pred.srem.if, pred.srem.continue
; CHECK-NEXT:   CondBit: vp<[[MASK]]> (loop)
; CHECK-EMPTY:
; CHECK-NEXT:   pred.srem.if:
; CHECK-NEXT:     REPLICATE ir<%rem> = srem vp<[[SPLICE]]>, ir<%x> (S->V)
; CHECK-NEXT:   Successor(s): pred.srem.continue
; CHECK-EMPTY:
; CHECK-NEXT:   pred.srem.continue:
; CHECK-NEXT:     PHI-PREDICATED-INSTRUCTION vp<[[PRED2:%.+]]> = ir<%rem>
; CHECK-NEXT:   No successors
; CHECK-NEXT: }
; CHECK-NEXT: Successor(s): loop.1.split
; CHECK-EMPTY:
; CHECK-NEXT: loop.1.split:
; CHECK-NEXT:   WIDEN ir<%add> = add ir<%conv>, vp<[[PRED2]]>
; CHECK-NEXT:   EMIT vp<[[CAN_IV_NEXT:%.+]]> = VF * UF + vp<[[CAN_IV]]>
; CHECK-NEXT:   EMIT branch-on-count vp<[[CAN_IV_NEXT]]> vp<[[VEC_TC]]>
; CHECK-NEXT: No successors
; CHECK-NEXT: }
; CHECK-NEXT: No successors
; CHECK-NEXT: }
;
entry:
  br label %loop

loop:
  %0 = phi i32 [ 0, %entry ], [ %conv, %loop ]
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %loop ]
  %rem = srem i32 %0, %x
  %gep = getelementptr i8, i8* %ptr, i32 %iv
  %lv = load i8, i8* %gep
  %conv = sext i8 %lv to i32
  %add = add i32 %conv, %rem
  %iv.next = add nsw i32 %iv, 1
  %ec = icmp eq i32 %iv.next, 20001
  br i1 %ec, label %exit, label %loop

exit:
  ret void
}

define void @sink_replicate_region_2(i32 %x, i8 %y, i32* %ptr) optsize {
; CHECK-LABEL: sink_replicate_region_2
; CHECK:      VPlan 'Initial VPlan for VF={2},UF>=1' {
; CHECK-NEXT: Live-in vp<[[VEC_TC:%.+]]> = vector-trip-count
; CHECK-EMPTY:
; CHECK-NEXT: Live-in vp<[[BTC:%.+]]> = backedge-taken count
; CHECK-EMPTY:
; CHECK-NEXT: <x1> vector loop: {
; CHECK-NEXT: loop:
; CHECK-NEXT:   EMIT vp<[[CAN_IV:%.+]]> = CANONICAL-INDUCTION
; CHECK-NEXT:   FIRST-ORDER-RECURRENCE-PHI ir<%recur> = phi ir<0>, ir<%recur.next>
; CHECK-NEXT:   WIDEN-INDUCTION %iv = phi 0, %iv.next
; CHECK-NEXT:   EMIT vp<[[MASK:%.+]]> = icmp ule ir<%iv> vp<[[BTC]]>
; CHECK-NEXT: Successor(s): loop.0
; CHECK-EMPTY:
; CHECK-NEXT: loop.0:
; CHECK-NEXT:   WIDEN ir<%recur.next> = sext ir<%y>
; CHECK-NEXT:   EMIT vp<[[SPLICE:%.+]]> = first-order splice ir<%recur> ir<%recur.next>
; CHECK-NEXT: Successor(s): loop.0.split
; CHECK-EMPTY:
; CHECK-NEXT: loop.0.split:
; CHECK-NEXT:   Successor(s): pred.store
; CHECK-EMPTY:
; CHECK-NEXT: <xVFxUF> pred.store: {
; CHECK-NEXT:  pred.store.entry:
; CHECK-NEXT:    BRANCH-ON-MASK vp<[[MASK]]>
; CHECK-NEXT:  Successor(s): pred.store.if, pred.store.continue
; CHECK-NEXT:  CondBit: vp<[[MASK]]> (loop)
; CHECK-EMPTY:
; CHECK-NEXT:  pred.store.if:
; CHECK-NEXT:     REPLICATE ir<%rem> = srem vp<[[SPLICE]]>, ir<%x>
; CHECK-NEXT:     REPLICATE ir<%add> = add ir<%rem>, ir<%recur.next>
; CHECK-NEXT:     REPLICATE ir<%gep> = getelementptr ir<%ptr>, ir<%iv>
; CHECK-NEXT:     REPLICATE store ir<%add>, ir<%gep>
; CHECK-NEXT:   Successor(s): pred.store.continue
; CHECK-EMPTY:
; CHECK-NEXT:   pred.store.continue:
; CHECK-NEXT:     PHI-PREDICATED-INSTRUCTION vp<[[PRED:%.+]]> = ir<%rem>
; CHECK-NEXT:   No successors
; CHECK-NEXT: }
; CHECK-NEXT: Successor(s): loop.1
; CHECK-EMPTY:
; CHECK-NEXT: loop.1:
; CHECK-NEXT:   EMIT vp<[[CAN_IV_NEXT:%.+]]> = VF * UF + vp<[[CAN_IV]]>
; CHECK-NEXT:   EMIT branch-on-count vp<[[CAN_IV_NEXT]]> vp<[[VEC_TC]]>
; CHECK-NEXT: No successors
; CHECK-NEXT: }
; CHECK-NEXT: No successors
; CHECK-NEXT: }
;
entry:
  br label %loop

loop:
  %recur = phi i32 [ 0, %entry ], [ %recur.next, %loop ]
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %loop ]
  %rem = srem i32 %recur, %x
  %recur.next = sext i8 %y to i32
  %add = add i32 %rem, %recur.next
  %gep = getelementptr i32, i32* %ptr, i32 %iv
  store i32 %add, i32* %gep
  %iv.next = add nsw i32 %iv, 1
  %ec = icmp eq i32 %iv.next, 20001
  br i1 %ec, label %exit, label %loop

exit:
  ret void
}

define i32 @sink_replicate_region_3_reduction(i32 %x, i8 %y, i32* %ptr) optsize {
; CHECK-LABEL: sink_replicate_region_3_reduction
; CHECK:      VPlan 'Initial VPlan for VF={2},UF>=1' {
; CHECK-NEXT: Live-in vp<[[VEC_TC:%.+]]> = vector-trip-count
; CHECK-EMPTY:
; CHECK-NEXT: Live-in vp<[[BTC:%.+]]> = backedge-taken count
; CHECK-EMPTY:
; CHECK-NEXT: <x1> vector loop: {
; CHECK-NEXT: loop:
; CHECK-NEXT:   EMIT vp<[[CAN_IV:%.+]]> = CANONICAL-INDUCTION
; CHECK-NEXT:   FIRST-ORDER-RECURRENCE-PHI ir<%recur> = phi ir<0>, ir<%recur.next>
; CHECK-NEXT:   WIDEN-REDUCTION-PHI ir<%and.red> = phi ir<1234>, ir<%and.red.next>
; CHECK-NEXT:   EMIT vp<[[WIDEN_CAN:%.+]]> = WIDEN-CANONICAL-INDUCTION vp<[[CAN_IV]]>
; CHECK-NEXT:   EMIT vp<[[MASK:%.+]]> = icmp ule vp<[[WIDEN_CAN]]> vp<[[BTC]]>
; CHECK-NEXT: Successor(s): loop.0
; CHECK-EMPTY:
; CHECK-NEXT: loop.0:
; CHECK-NEXT:   WIDEN ir<%recur.next> = sext ir<%y>
; CHECK-NEXT:   EMIT vp<[[SPLICE:%.+]]> = first-order splice ir<%recur> ir<%recur.next>
; CHECK-NEXT: Successor(s): pred.srem
; CHECK-EMPTY:
; CHECK-NEXT: <xVFxUF> pred.srem: {
; CHECK-NEXT:   pred.srem.entry:
; CHECK-NEXT:     BRANCH-ON-MASK vp<[[MASK]]>
; CHECK-NEXT:   Successor(s): pred.srem.if, pred.srem.continue
; CHECK-NEXT:   CondBit: vp<[[MASK]]> (loop)
; CHECK-EMPTY:
; CHECK-NEXT:   pred.srem.if:
; CHECK-NEXT:     REPLICATE ir<%rem> = srem vp<[[SPLICE]]>, ir<%x> (S->V)
; CHECK-NEXT:   Successor(s): pred.srem.continue
; CHECK-EMPTY:
; CHECK-NEXT:   pred.srem.continue:
; CHECK-NEXT:     PHI-PREDICATED-INSTRUCTION vp<[[PRED:%.+]]> = ir<%rem>
; CHECK-NEXT:   No successors
; CHECK-NEXT: }
; CHECK-NEXT: Successor(s): loop.0.split
; CHECK-EMPTY:
; CHECK-NEXT: loop.0.split:
; CHECK-NEXT:   WIDEN ir<%add> = add vp<[[PRED]]>, ir<%recur.next>
; CHECK-NEXT:   WIDEN ir<%and.red.next> = and ir<%and.red>, ir<%add>
; CHECK-NEXT:   EMIT vp<[[SEL:%.+]]> = select vp<[[MASK]]> ir<%and.red.next> ir<%and.red>
; CHECK-NEXT:   EMIT vp<[[CAN_IV_NEXT:%.+]]> = VF * UF + vp<[[CAN_IV]]>
; CHECK-NEXT:   EMIT branch-on-count vp<[[CAN_IV_NEXT]]> vp<[[VEC_TC]]>
; CHECK-NEXT: No successors
; CHECK-NEXT: }
; CHECK-NEXT: No successors
; CHECK-NEXT: }
;
entry:
  br label %loop

loop:
  %recur = phi i32 [ 0, %entry ], [ %recur.next, %loop ]
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %loop ]
  %and.red = phi i32 [ 1234, %entry ], [ %and.red.next, %loop ]
  %rem = srem i32 %recur, %x
  %recur.next = sext i8 %y to i32
  %add = add i32 %rem, %recur.next
  %and.red.next = and i32 %and.red, %add
  %iv.next = add nsw i32 %iv, 1
  %ec = icmp eq i32 %iv.next, 20001
  br i1 %ec, label %exit, label %loop

exit:
  %res = phi i32 [ %and.red.next, %loop ]
  ret i32 %res
}

; To sink the replicate region containing %rem, we need to split the block
; containing %conv at the end, because %conv is the last recipe in the block.
define void @sink_replicate_region_4_requires_split_at_end_of_block(i32 %x, i8* %ptr) optsize {
; CHECK-LABEL: sink_replicate_region_4_requires_split_at_end_of_block
; CHECK:      VPlan 'Initial VPlan for VF={2},UF>=1' {
; CHECK-NEXT: Live-in vp<[[VEC_TC:%.+]]> = vector-trip-count
; CHECK-EMPTY:
; CHECK-NEXT: Live-in vp<[[BTC:%.+]]> = backedge-taken count
; CHECK-EMPTY:
; CHECK-NEXT: <x1> vector loop: {
; CHECK-NEXT: loop:
; CHECK-NEXT:   EMIT vp<[[CAN_IV:%.+]]> = CANONICAL-INDUCTION
; CHECK-NEXT:   FIRST-ORDER-RECURRENCE-PHI ir<%0> = phi ir<0>, ir<%conv>
; CHECK-NEXT:   WIDEN-INDUCTION %iv = phi 0, %iv.next
; CHECK-NEXT:   EMIT vp<[[MASK:%.+]]> = icmp ule ir<%iv> vp<[[BTC]]>
; CHECK-NEXT:   REPLICATE ir<%gep> = getelementptr ir<%ptr>, ir<%iv>
; CHECK-NEXT: Successor(s): loop.0
; CHECK-EMPTY:
; CHECK-NEXT: loop.0:
; CHECK-NEXT: Successor(s): pred.load
; CHECK-EMPTY:
; CHECK-NEXT: <xVFxUF> pred.load: {
; CHECK-NEXT:   pred.load.entry:
; CHECK-NEXT:     BRANCH-ON-MASK vp<[[MASK]]>
; CHECK-NEXT:   Successor(s): pred.load.if, pred.load.continue
; CHECK-NEXT:   CondBit: vp<[[MASK]]> (loop)
; CHECK-EMPTY:
; CHECK-NEXT:   pred.load.if:
; CHECK-NEXT:     REPLICATE ir<%lv> = load ir<%gep> (S->V)
; CHECK-NEXT:   Successor(s): pred.load.continue
; CHECK-EMPTY:
; CHECK-NEXT:   pred.load.continue:
; CHECK-NEXT:     PHI-PREDICATED-INSTRUCTION vp<[[PRED:%.+]]> = ir<%lv>
; CHECK-NEXT:   No successors
; CHECK-NEXT: }
; CHECK-NEXT: Successor(s): loop.1
; CHECK-EMPTY:
; CHECK-NEXT: loop.1:
; CHECK-NEXT:   WIDEN ir<%conv> = sext vp<[[PRED]]>
; CHECK-NEXT:   EMIT vp<[[SPLICE:%.+]]> = first-order splice ir<%0> ir<%conv>
; CHECK-NEXT: Successor(s): loop.1.split

; CHECK:      loop.1.split:
; CHECK-NEXT: Successor(s): pred.load

; CHECK:      <xVFxUF> pred.load: {
; CHECK-NEXT:   pred.load.entry:
; CHECK-NEXT:     BRANCH-ON-MASK vp<[[MASK]]>
; CHECK-NEXT:   Successor(s): pred.load.if, pred.load.continue
; CHECK-NEXT:   CondBit: vp<[[MASK]]> (loop)

; CHECK:        pred.load.if:
; CHECK-NEXT:     REPLICATE ir<%rem> = srem vp<[[SPLICE]]>, ir<%x> (S->V)
; CHECK-NEXT:     REPLICATE ir<%lv.2> = load ir<%gep> (S->V)
; CHECK-NEXT:   Successor(s): pred.load.continue

; CHECK:        pred.load.continue:
; CHECK-NEXT:     PHI-PREDICATED-INSTRUCTION vp<[[PRED1:%.+]]> = ir<%rem>
; CHECK-NEXT:     PHI-PREDICATED-INSTRUCTION vp<[[PRED2:%.+]]> = ir<%lv.2>
; CHECK-NEXT:   No successors
; CHECK-NEXT: }

; CHECK:      loop.2:
; CHECK-NEXT:   WIDEN ir<%add.1> = add ir<%conv>, vp<[[PRED1]]>
; CHECK-NEXT:   WIDEN ir<%conv.lv.2> = sext vp<[[PRED2]]>
; CHECK-NEXT:   WIDEN ir<%add> = add ir<%add.1>, ir<%conv.lv.2>
; CHECK-NEXT:   EMIT vp<[[CAN_IV_NEXT:%.+]]> = VF * UF + vp<[[CAN_IV]]>
; CHECK-NEXT:   EMIT branch-on-count vp<[[CAN_IV_NEXT]]> vp<[[VEC_TC]]>
; CHECK-NEXT: No successors
; CHECK-NEXT: }
; CHECK-NEXT: No successors
; CHECK-NEXT: }
;
entry:
  br label %loop

loop:
  %0 = phi i32 [ 0, %entry ], [ %conv, %loop ]
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %loop ]
  %gep = getelementptr i8, i8* %ptr, i32 %iv
  %rem = srem i32 %0, %x
  %lv = load i8, i8* %gep
  %conv = sext i8 %lv to i32
  %lv.2 = load i8, i8* %gep
  %add.1 = add i32 %conv, %rem
  %conv.lv.2 = sext i8 %lv.2 to i32
  %add = add i32 %add.1, %conv.lv.2
  %iv.next = add nsw i32 %iv, 1
  %ec = icmp eq i32 %iv.next, 20001
  br i1 %ec, label %exit, label %loop

exit:
  ret void
}

; Test case that requires sinking a recipe in a replicate region after another replicate region.
define void @sink_replicate_region_after_replicate_region(i32* %ptr, i32 %x, i8 %y) optsize {
; CHECK-LABEL: sink_replicate_region_after_replicate_region
; CHECK:      VPlan 'Initial VPlan for VF={2},UF>=1' {
; CHECK-NEXT: Live-in vp<[[VEC_TC:%.+]]> = vector-trip-count
; CHECK-EMPTY:
; CHECK-NEXT: Live-in vp<[[BTC:%.+]]> = backedge-taken count
; CHECK-EMPTY:
; CHECK-NEXT: <x1> vector loop: {
; CHECK-NEXT: loop:
; CHECK-NEXT:   EMIT vp<[[CAN_IV:%.+]]> = CANONICAL-INDUCTION
; CHECK-NEXT:   FIRST-ORDER-RECURRENCE-PHI ir<%recur> = phi ir<0>, ir<%recur.next>
; CHECK-NEXT:   WIDEN-INDUCTION %iv = phi 0, %iv.next
; CHECK-NEXT:   EMIT vp<[[MASK:%.+]]> = icmp ule ir<%iv> vp<[[BTC]]>
; CHECK-NEXT: Successor(s): loop.0
; CHECK-EMPTY:
; CHECK-NEXT: loop.0:
; CHECK-NEXT: Successor(s): loop.1
; CHECK-EMPTY:
; CHECK-NEXT:  loop.1:
; CHECK-NEXT:   WIDEN ir<%recur.next> = sext ir<%y>
; CHECK-NEXT:   EMIT vp<[[SPLICE:%.+]]> = first-order splice ir<%recur> ir<%recur.next>
; CHECK-NEXT: Successor(s): pred.srem
; CHECK-EMPTY:
; CHECK-NEXT: <xVFxUF> pred.srem: {
; CHECK-NEXT:   pred.srem.entry:
; CHECK-NEXT:     BRANCH-ON-MASK vp<[[MASK]]>
; CHECK-NEXT:   Successor(s): pred.srem.if, pred.srem.continue
; CHECK-NEXT:   CondBit: vp<[[MASK]]> (loop)
; CHECK-EMPTY:
; CHECK-NEXT:   pred.srem.if:
; CHECK-NEXT:     REPLICATE ir<%rem> = srem vp<[[SPLICE]]>, ir<%x>
; CHECK-NEXT:   Successor(s): pred.srem.continue
; CHECK-EMPTY:
; CHECK-NEXT:   pred.srem.continue:
; CHECK-NEXT:     PHI-PREDICATED-INSTRUCTION vp<[[PRED:%.+]]> = ir<%rem>
; CHECK-NEXT:   No successors
; CHECK-NEXT: }
; CHECK-NEXT: Successor(s): loop.1.split
; CHECK-EMPTY:
; CHECK-NEXT: loop.1.split:
; CHECK-NEXT: Successor(s): pred.store
; CHECK-EMPTY:
; CHECK-NEXT: <xVFxUF> pred.store: {
; CHECK-NEXT:   pred.store.entry:
; CHECK-NEXT:     BRANCH-ON-MASK vp<[[MASK]]>
; CHECK-NEXT:   Successor(s): pred.store.if, pred.store.continue
; CHECK-NEXT:   CondBit: vp<[[MASK]]> (loop)
; CHECK-EMPTY:
; CHECK-NEXT:   pred.store.if:
; CHECK-NEXT:     REPLICATE ir<%rem.div> = sdiv ir<20>, vp<[[PRED]]>
; CHECK-NEXT:     REPLICATE ir<%gep> = getelementptr ir<%ptr>, ir<%iv>
; CHECK-NEXT:     REPLICATE store ir<%rem.div>, ir<%gep>
; CHECK-NEXT:   Successor(s): pred.store.continue
; CHECK-EMPTY:
; CHECK-NEXT:   pred.store.continue:
; CHECK-NEXT:     PHI-PREDICATED-INSTRUCTION vp<[[PRED2:%.+]]> = ir<%rem.div>
; CHECK-NEXT:   No successors
; CHECK-NEXT: }
; CHECK-NEXT: Successor(s): loop.2
; CHECK-EMPTY:
; CHECK-NEXT: loop.2:
; CHECK-NEXT:   EMIT vp<[[CAN_IV_NEXT:%.+]]> = VF * UF + vp<[[CAN_IV]]>
; CHECK-NEXT:   EMIT branch-on-count vp<[[CAN_IV_NEXT]]> vp<[[VEC_TC]]>
; CHECK-NEXT: No successors
; CHECK-NEXT: }
; CHECK-NEXT: No successors
; CHECK-NEXT: }
;
entry:
  br label %loop

loop:                                             ; preds = %loop, %entry
  %recur = phi i32 [ 0, %entry ], [ %recur.next, %loop ]
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %loop ]
  %rem = srem i32 %recur, %x
  %rem.div = sdiv i32 20, %rem
  %recur.next = sext i8 %y to i32
  %gep = getelementptr i32, i32* %ptr, i32 %iv
  store i32 %rem.div, i32* %gep
  %iv.next = add nsw i32 %iv, 1
  %C = icmp sgt i32 %iv.next, %recur.next
  br i1 %C, label %exit, label %loop

exit:                                             ; preds = %loop
  ret void
}
