; REQUIRES: asserts

; RUN: opt -passes=loop-vectorize -force-vector-width=1 -force-vector-interleave=2 -debug -disable-output %s 2>&1 | FileCheck --check-prefix=DBG %s
; RUN: opt -passes=loop-vectorize -force-vector-width=1 -force-vector-interleave=2 -S %s | FileCheck %s

; DBG-LABEL: 'test_scalarize_call'
; DBG:      VPlan 'Initial VPlan for VF={1},UF>=1' {
; DBG-NEXT: Live-in vp<[[VFxUF:%.+]]> = VF * UF
; DBG-NEXT: Live-in vp<[[VEC_TC:%.+]]> = vector-trip-count
; DBG-NEXT: vp<[[TC:%.+]]> = original trip-count
; DBG-EMPTY:
; DBG-NEXT: ph:
; DBG-NEXT:  EMIT vp<[[TC]]> = EXPAND SCEV (1000 + (-1 * %start))
; DBG-NEXT: No successors
; DBG-EMPTY:
; DBG-NEXT: vector.ph:
; DBG-NEXT: Successor(s): vector loop
; DBG-EMPTY:
; DBG-NEXT: <x1> vector loop: {
; DBG-NEXT:   vector.body:
; DBG-NEXT:     EMIT vp<[[CAN_IV:%.+]]> = CANONICAL-INDUCTION
; DBG-NEXT:     vp<[[DERIVED_IV:%.+]]> = DERIVED-IV ir<%start> + vp<[[CAN_IV]]> * ir<1>
; DBG-NEXT:     vp<[[IV_STEPS:%.]]>    = SCALAR-STEPS vp<[[DERIVED_IV]]>, ir<1>
; DBG-NEXT:     CLONE ir<%min> = call @llvm.smin.i32(vp<[[IV_STEPS]]>, ir<65535>)
; DBG-NEXT:     CLONE ir<%arrayidx> = getelementptr inbounds ir<%dst>, vp<[[IV_STEPS]]>
; DBG-NEXT:     CLONE store ir<%min>, ir<%arrayidx>
; DBG-NEXT:     EMIT vp<[[INC:%.+]]> = add nuw vp<[[CAN_IV]]>, vp<[[VFxUF]]>
; DBG-NEXT:     EMIT branch-on-count vp<[[INC]]>, vp<[[VEC_TC]]>
; DBG-NEXT:   No successors
; DBG-NEXT: }
;
define void @test_scalarize_call(i32 %start, ptr %dst) {
; CHECK-LABEL: @test_scalarize_call(
; CHECK:       vector.body:
; CHECK-NEXT:    [[INDEX:%.*]] = phi i32 [ 0, %vector.ph ], [ [[INDEX_NEXT:%.*]], %vector.body ]
; CHECK-NEXT:    [[OFFSET_IDX:%.*]] = add i32 %start, [[INDEX]]
; CHECK-NEXT:    [[INDUCTION:%.*]] = add i32 [[OFFSET_IDX]], 0
; CHECK-NEXT:    [[INDUCTION1:%.*]] = add i32 [[OFFSET_IDX]], 1
; CHECK-NEXT:    [[TMP1:%.*]] = tail call i32 @llvm.smin.i32(i32 [[INDUCTION]], i32 65535)
; CHECK-NEXT:    [[TMP2:%.*]] = tail call i32 @llvm.smin.i32(i32 [[INDUCTION1]], i32 65535)
; CHECK-NEXT:    [[TMP3:%.*]] = getelementptr inbounds i32, ptr [[DST:%.*]], i32 [[INDUCTION]]
; CHECK-NEXT:    [[TMP4:%.*]] = getelementptr inbounds i32, ptr [[DST]], i32 [[INDUCTION1]]
; CHECK-NEXT:    store i32 [[TMP1]], ptr [[TMP3]], align 8
; CHECK-NEXT:    store i32 [[TMP2]], ptr [[TMP4]], align 8
; CHECK-NEXT:    [[INDEX_NEXT]] = add nuw i32 [[INDEX]], 2
; CHECK-NEXT:    [[TMP5:%.*]] = icmp eq i32 [[INDEX_NEXT]], %n.vec
; CHECK-NEXT:    br i1 [[TMP5]], label %middle.block, label %vector.body
; CHECK:       middle.block:
;
entry:
  br label %loop

loop:
  %iv = phi i32 [ %start, %entry ], [ %iv.next, %loop ]
  %min = tail call i32 @llvm.smin.i32(i32 %iv, i32 65535)
  %arrayidx = getelementptr inbounds i32 , ptr %dst, i32 %iv
  store i32 %min, ptr %arrayidx, align 8
  %iv.next = add nsw i32 %iv, 1
  %tobool.not = icmp eq i32 %iv.next, 1000
  br i1 %tobool.not, label %exit, label %loop

exit:
  ret void
}

declare i32 @llvm.smin.i32(i32, i32)


; DBG-LABEL: 'test_scalarize_with_branch_cond'

; DBG:       Live-in vp<[[VFxUF:%.+]]> = VF * UF
; DBG-NEXT:  Live-in vp<[[VEC_TC:%.+]]> = vector-trip-count
; DBG-NEXT:  Live-in ir<1000> = original trip-count
; DBG-EMPTY:
; DBG-NEXT: vector.ph:
; DBG-NEXT: Successor(s): vector loop
; DBG-EMPTY:
; DBG-NEXT: <x1> vector loop: {
; DBG-NEXT:   vector.body:
; DBG-NEXT:     EMIT vp<[[CAN_IV:%.+]]> = CANONICAL-INDUCTION
; DBG-NEXT:     vp<[[DERIVED_IV:%.+]]> = DERIVED-IV ir<false> + vp<[[CAN_IV]]> * ir<true>
; DBG-NEXT:     vp<[[STEPS1:%.+]]>    = SCALAR-STEPS vp<[[DERIVED_IV]]>, ir<true>
; DBG-NEXT:   Successor(s): pred.store
; DBG-EMPTY:
; DBG-NEXT:   <xVFxUF> pred.store: {
; DBG-NEXT:     pred.store.entry:
; DBG-NEXT:       BRANCH-ON-MASK vp<[[STEPS1]]>
; DBG-NEXT:     Successor(s): pred.store.if, pred.store.continue
; DBG-EMPTY:
; DBG-NEXT:     pred.store.if:
; DBG-NEXT:       vp<[[STEPS2:%.+]]>    = SCALAR-STEPS vp<[[CAN_IV]]>, ir<1>
; DBG-NEXT:       CLONE ir<%gep.src> = getelementptr inbounds ir<%src>, vp<[[STEPS2]]>
; DBG-NEXT:       CLONE ir<%l> = load ir<%gep.src>
; DBG-NEXT:       CLONE ir<%gep.dst> = getelementptr inbounds ir<%dst>, vp<[[STEPS2]]>
; DBG-NEXT:       CLONE store ir<%l>, ir<%gep.dst>
; DBG-NEXT:     Successor(s): pred.store.continue
; DBG-EMPTY:
; DBG-NEXT:     pred.store.continue:
; DBG-NEXT:       PHI-PREDICATED-INSTRUCTION vp<{{.+}}> = ir<%l>
; DBG-NEXT:     No successors
; DBG-NEXT:   }
; DBG-NEXT:   Successor(s): cond.false.1
; DBG-EMPTY:
; DBG-NEXT:   cond.false.1:
; DBG-NEXT:     EMIT vp<[[CAN_IV_INC:%.+]]> = add nuw vp<[[CAN_IV]]>, vp<[[VFxUF]]>
; DBG-NEXT:     EMIT branch-on-count vp<[[CAN_IV_INC]]>, vp<[[VEC_TC]]>
; DBG-NEXT:   No successors
; DBG-NEXT: }
; DBG-NEXT: Successor(s): middle.block
; DBG-EMPTY:
; DBG-NEXT: middle.block:
; DBG-NEXT: No successors
; DBG-NEXT: }

define void @test_scalarize_with_branch_cond(ptr %src, ptr %dst) {
; CHECK-LABEL: @test_scalarize_with_branch_cond(
; CHECK:       vector.body:
; CHECK-NEXT:    [[INDEX:%.*]] = phi i64 [ 0, %vector.ph ], [ [[INDEX_NEXT:%.*]], %pred.store.continue5 ]
; CHECK-NEXT:    [[TMP0:%.*]] = trunc i64 [[INDEX]] to i1
; CHECK-NEXT:    [[OFFSET_IDX:%.*]] = sub i1 false, [[TMP0]]
; CHECK-NEXT:    [[INDUCTION:%.*]] = add i1 [[OFFSET_IDX]], false
; CHECK-NEXT:    [[INDUCTION3:%.*]] = add i1 [[OFFSET_IDX]], true
; CHECK-NEXT:    br i1 [[INDUCTION]], label %pred.store.if, label %pred.store.continue
; CHECK:       pred.store.if:
; CHECK-NEXT:    [[INDUCTION4:%.*]] = add i64 [[INDEX]], 0
; CHECK-NEXT:    [[TMP3:%.*]] = getelementptr inbounds i32, ptr %src, i64 [[INDUCTION4]]
; CHECK-NEXT:    [[TMP4:%.*]] = load i32, ptr [[TMP3]], align 4
; CHECK-NEXT:    [[TMP1:%.*]] = getelementptr inbounds i32, ptr %dst, i64 [[INDUCTION4]]
; CHECK-NEXT:    store i32 [[TMP4]], ptr [[TMP1]], align 4
; CHECK-NEXT:    br label %pred.store.continue
; CHECK:       pred.store.continue:
; CHECK-NEXT:    [[TMP5:%.*]] = phi i32 [ poison, %vector.body ], [ [[TMP4]], %pred.store.if ]
; CHECK-NEXT:    br i1 [[INDUCTION3]], label %pred.store.if4, label %pred.store.continue5
; CHECK:       pred.store.if4:
; CHECK-NEXT:    [[INDUCTION5:%.*]] = add i64 [[INDEX]], 1
; CHECK-NEXT:    [[TMP6:%.*]] = getelementptr inbounds i32, ptr %src, i64 [[INDUCTION5]]
; CHECK-NEXT:    [[TMP7:%.*]] = load i32, ptr [[TMP6]], align 4
; CHECK-NEXT:    [[TMP2:%.*]] = getelementptr inbounds i32, ptr %dst, i64 [[INDUCTION5]]
; CHECK-NEXT:    store i32 [[TMP7]], ptr [[TMP2]], align 4
; CHECK-NEXT:    br label %pred.store.continue5
; CHECK:       pred.store.continue5:
; CHECK-NEXT:    [[TMP8:%.*]] = phi i32 [ poison, %pred.store.continue ], [ [[TMP7]], %pred.store.if4 ]
; CHECK-NEXT:    [[INDEX_NEXT]] = add nuw i64 [[INDEX]], 2
; CHECK-NEXT:    [[TMP9:%.*]] = icmp eq i64 [[INDEX_NEXT]], 1000
; CHECK-NEXT:    br i1 [[TMP9]], label %middle.block, label %vector.body
; CHECK:       middle.block:
;
entry:
  br label %loop.header

loop.header:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop.latch ]
  %d = phi i1 [ false, %entry ], [ %d.next, %loop.latch ]
  %d.next = xor i1 %d, true
  br i1 %d, label %cond.false, label %loop.latch

cond.false:
  %gep.src = getelementptr inbounds i32, ptr %src, i64 %iv
  %gep.dst = getelementptr inbounds i32, ptr %dst, i64 %iv
  %l = load i32, ptr %gep.src, align 4
  store i32 %l, ptr %gep.dst
  br label %loop.latch

loop.latch:
  %iv.next = add nsw i64 %iv, 1
  %ec = icmp eq i64 %iv.next, 1000
  br i1 %ec, label %exit, label %loop.header

exit:
  ret void
}

; Make sure the widened induction gets replaced by scalar-steps for plans
; including the scalar VF, if it is used in first-order recurrences.

; DBG-LABEL: 'first_order_recurrence_using_induction'
; DBG:      VPlan 'Initial VPlan for VF={1},UF>=1' {
; DBG-NEXT: Live-in vp<[[VFxUF:%.+]]> = VF * UF
; DBG-NEXT: Live-in vp<[[VTC:%.+]]> = vector-trip-count
; DBG-NEXT: vp<[[TC:%.+]]> = original trip-count
; DBG-EMPTY:
; DBG-NEXT: ph:
; DBG-NEXT:  EMIT vp<[[TC]]> = EXPAND SCEV (zext i32 (1 smax %n) to i64)
; DBG-NEXT: No successors
; DBG-EMPTY:
; DBG-NEXT: vector.ph:
; DBG-NEXT: Successor(s): vector loop
; DBG-EMPTY:
; DBG-NEXT: <x1> vector loop: {
; DBG-NEXT:   vector.body:
; DBG-NEXT:     EMIT vp<[[CAN_IV:%.+]]> = CANONICAL-INDUCTION
; DBG-NEXT:     FIRST-ORDER-RECURRENCE-PHI ir<%for> = phi ir<0>, vp<[[SCALAR_STEPS:.+]]>
; DBG-NEXT:     vp<[[DERIVED_IV:%.+]]> = DERIVED-IV ir<0> + vp<[[CAN_IV]]> * ir<1> (truncated to i32)
; DBG-NEXT:     vp<[[SCALAR_STEPS]]> = SCALAR-STEPS vp<[[DERIVED_IV]]>, ir<1>
; DBG-NEXT:     EMIT vp<[[SPLICE:%.+]]> = first-order splice ir<%for>, vp<[[SCALAR_STEPS]]>
; DBG-NEXT:     CLONE store vp<[[SPLICE]]>, ir<%dst>
; DBG-NEXT:     EMIT vp<[[IV_INC:%.+]]> = add nuw vp<[[CAN_IV]]>, vp<[[VFxUF]]>
; DBG-NEXT:     EMIT branch-on-count vp<[[IV_INC]]>, vp<[[VTC]]>
; DBG-NEXT:   No successors
; DBG-NEXT: }
; DBG-NEXT: Successor(s): middle.block
; DBG-EMPTY:
; DBG-NEXT: middle.block:
; DBG-NEXT: No successors
; DBG-NEXT: }

define void @first_order_recurrence_using_induction(i32 %n, ptr %dst) {
; CHECK-LABEL: @first_order_recurrence_using_induction(
; CHECK:       vector.body:
; CHECK-NEXT:    [[INDEX:%.*]] = phi i64 [ 0, %vector.ph ], [ [[INDEX_NEXT:%.*]], %vector.body ]
; CHECK-NEXT:    [[VECTOR_RECUR:%.*]] = phi i32 [ 0, %vector.ph ], [ [[INDUCTION1:%.*]], %vector.body ]
; CHECK-NEXT:    [[TMP3:%.*]] = trunc i64 [[INDEX]] to i32
; CHECK-NEXT:    [[INDUCTION:%.*]] = add i32 [[TMP3]], 0
; CHECK-NEXT:    [[INDUCTION1]] = add i32 [[TMP3]], 1
; CHECK-NEXT:    store i32 [[VECTOR_RECUR]], ptr [[DST:%.*]], align 4
; CHECK-NEXT:    store i32 [[INDUCTION]], ptr [[DST]], align 4
; CHECK-NEXT:    [[INDEX_NEXT]] = add nuw i64 [[INDEX]], 2
; CHECK-NEXT:    [[TMP4:%.*]] = icmp eq i64 [[INDEX_NEXT]], %n.vec
; CHECK-NEXT:    br i1 [[TMP4]], label %middle.block, label %vector.body
; CHECK:       middle.block:
;
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ],[ %iv.next, %loop ]
  %for = phi i32 [ 0, %entry ], [ %iv.trunc, %loop ]
  %iv.trunc = trunc i64 %iv to i32
  store i32 %for, ptr %dst
  %iv.next = add nuw nsw i64 %iv, 1
  %iv.next.trunc = trunc i64 %iv.next to i32
  %ec = icmp slt i32 %iv.next.trunc, %n
  br i1 %ec, label %loop, label %exit

exit:
  ret void
}

define i16 @reduction_with_casts() {
; CHECK-LABEL: define i16 @reduction_with_casts() {
; CHECK:       vector.body:
; CHECK-NEXT:    [[INDEX:%.*]] = phi i32 [ 0, [[VECTOR_PH:%.+]] ], [ [[INDEX_NEXT:%.*]], [[VECTOR_BODY:%.+]] ]
; CHECK-NEXT:    [[VEC_PHI:%.*]] = phi i32 [ 0, [[VECTOR_PH]] ], [ [[TMP2:%.*]], [[VECTOR_BODY]] ]
; CHECK-NEXT:    [[VEC_PHI1:%.*]] = phi i32 [ 0, [[VECTOR_PH]] ], [ [[TMP3:%.*]], [[VECTOR_BODY]] ]
; CHECK-NEXT:    [[TMP0:%.*]] = and i32 [[VEC_PHI]], 65535
; CHECK-NEXT:    [[TMP1:%.*]] = and i32 [[VEC_PHI1]], 65535
; CHECK-NEXT:    [[TMP2]] = add i32 [[TMP0]], 1
; CHECK-NEXT:    [[TMP3]] = add i32 [[TMP1]], 1
; CHECK-NEXT:    [[INDEX_NEXT]] = add nuw i32 [[INDEX]], 2
; CHECK-NEXT:    [[TMP4:%.*]] = icmp eq i32 [[INDEX_NEXT]], 9998
; CHECK-NEXT:    br i1 [[TMP4]], label [[MIDDLE_BLOCK:%.*]], label [[VECTOR_BODY]]
; CHECK:       middle.block:
; CHECK-NEXT:    [[BIN_RDX:%.*]] = add i32 [[TMP3]], [[TMP2]]
; CHECK-NEXT:    br i1 false, label [[EXIT:%.*]], label %scalar.ph
;
entry:
  br label %loop

loop:
  %count.0.in1 = phi i32 [ 0, %entry ], [ %add, %loop ]
  %iv = phi i16 [ 1, %entry ], [ %iv.next, %loop ]
  %conv1 = and i32 %count.0.in1, 65535
  %add = add nuw nsw i32 %conv1, 1
  %iv.next = add i16 %iv, 1
  %cmp = icmp eq i16 %iv.next, 10000
  br i1 %cmp, label %exit, label %loop

exit:
  %add.lcssa = phi i32 [ %add, %loop ]
  %count.0 = trunc i32 %add.lcssa to i16
  ret i16 %count.0
}

define void @scalarize_ptrtoint(ptr %src, ptr %dst) {
; CHECK:       vector.body:
; CHECK-NEXT:    [[INDEX:%.*]] = phi i64 [ 0, %vector.ph ], [ [[INDEX_NEXT:%.*]], %vector.body ]
; CHECK-NEXT:    [[TMP0:%.*]] = add i64 [[INDEX]], 0
; CHECK-NEXT:    [[TMP1:%.*]] = add i64 [[INDEX]], 1
; CHECK-NEXT:    [[TMP2:%.*]] = getelementptr ptr, ptr %src, i64 [[TMP0]]
; CHECK-NEXT:    [[TMP3:%.*]] = getelementptr ptr, ptr %src, i64 [[TMP1]]
; CHECK-NEXT:    [[TMP4:%.*]] = load ptr, ptr [[TMP2]], align 8
; CHECK-NEXT:    [[TMP5:%.*]] = load ptr, ptr [[TMP3]], align 8
; CHECK-NEXT:    [[TMP6:%.*]] = ptrtoint ptr [[TMP4]] to i64
; CHECK-NEXT:    [[TMP7:%.*]] = ptrtoint ptr [[TMP5]] to i64
; CHECK-NEXT:    [[TMP8:%.*]] = add i64 [[TMP6]], 10
; CHECK-NEXT:    [[TMP9:%.*]] = add i64 [[TMP7]], 10
; CHECK-NEXT:    [[TMP10:%.*]] = inttoptr i64 [[TMP8]] to ptr
; CHECK-NEXT:    [[TMP11:%.*]] = inttoptr i64 [[TMP9]] to ptr
; CHECK-NEXT:    store ptr [[TMP10]], ptr %dst, align 8
; CHECK-NEXT:    store ptr [[TMP11]], ptr %dst, align 8
; CHECK-NEXT:    [[INDEX_NEXT]] = add nuw i64 [[INDEX]], 2
; CHECK-NEXT:    [[TMP12:%.*]] = icmp eq i64 [[INDEX_NEXT]], 0
; CHECK-NEXT:    br i1 [[TMP12]], label %middle.block, label %vector.body

entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %gep = getelementptr ptr, ptr %src, i64 %iv
  %l = load ptr, ptr %gep, align 8
  %cast = ptrtoint ptr %l to i64
  %add = add i64 %cast, 10
  %cast.2 = inttoptr i64 %add to ptr
  store ptr %cast.2, ptr %dst, align 8
  %iv.next = add i64 %iv, 1
  %ec = icmp eq i64 %iv.next, 0
  br i1 %ec, label %exit, label %loop

exit:
  ret void
}
