; REQUIRES: asserts

; RUN: opt -passes=loop-vectorize -force-vector-width=1 -force-vector-interleave=2 -debug -disable-output %s 2>&1 | FileCheck --check-prefix=DBG %s
; RUN: opt -passes=loop-vectorize -force-vector-width=1 -force-vector-interleave=2 -S %s | FileCheck %s

; DBG-LABEL: 'test_scalarize_call'
; DBG:      VPlan 'Initial VPlan for VF={1},UF>=1' {
; DBG-NEXT: Live-in vp<[[VEC_TC:%.+]]> = vector-trip-count
; DBG-EMPTY:
; DBG-NEXT: vector.ph:
; DBG-NEXT: Successor(s): vector loop
; DBG-EMPTY:
; DBG-NEXT: <x1> vector loop: {
; DBG-NEXT:   vector.body:
; DBG-NEXT:     EMIT vp<[[CAN_IV:%.+]]> = CANONICAL-INDUCTION
; DBG-NEXT:     vp<[[IV_STEPS:%.]]>    = SCALAR-STEPS vp<[[CAN_IV]]>, ir<%start>, ir<1>
; DBG-NEXT:     CLONE ir<%min> = call @llvm.smin.i32(vp<[[IV_STEPS]]>, ir<65535>)
; DBG-NEXT:     CLONE ir<%arrayidx> = getelementptr ir<%dst>, vp<[[IV_STEPS]]>
; DBG-NEXT:     CLONE store ir<%min>, ir<%arrayidx>
; DBG-NEXT:     EMIT vp<[[INC:%.+]]> = VF * UF +(nuw)  vp<[[CAN_IV]]>
; DBG-NEXT:     EMIT branch-on-count  vp<[[INC]]> vp<[[VEC_TC]]>
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

; DBG:       Live-in vp<[[VEC_TC:%.+]]> = vector-trip-count
; DBG-EMPTY:
; DBG-NEXT: vector.ph:
; DBG-NEXT: Successor(s): vector loop
; DBG-EMPTY:
; DBG-NEXT: <x1> vector loop: {
; DBG-NEXT:   vector.body:
; DBG-NEXT:     EMIT vp<[[CAN_IV:%.+]]> = CANONICAL-INDUCTION
; DBG-NEXT:     vp<[[STEPS1:%.+]]>    = SCALAR-STEPS vp<[[CAN_IV]]>, ir<false>, ir<true>
; DBG-NEXT:     vp<[[STEPS2:%.+]]>    = SCALAR-STEPS vp<[[CAN_IV]]>, ir<0>, ir<1>
; DBG-NEXT:   Successor(s): cond.false
; DBG-EMPTY:
; DBG-NEXT:   cond.false:
; DBG-NEXT:     CLONE ir<%gep.src> = getelementptr ir<%src>, vp<[[STEPS2]]>
; DBG-NEXT:     CLONE ir<%gep.dst> = getelementptr ir<%dst>, vp<[[STEPS2]]>
; DBG-NEXT:   Successor(s): cond.false.0
; DBG-EMPTY:
; DBG-NEXT:   cond.false.0:
; DBG-NEXT:   Successor(s): pred.store
; DBG-EMPTY:
; DBG-NEXT:   <xVFxUF> pred.store: {
; DBG-NEXT:     pred.store.entry:
; DBG-NEXT:       BRANCH-ON-MASK vp<[[STEPS1]]>
; DBG-NEXT:     Successor(s): pred.store.if, pred.store.continue
; DBG-EMPTY:
; DBG-NEXT:     pred.store.if:
; DBG-NEXT:       CLONE ir<%l> = load ir<%gep.src>
; DBG-NEXT:       CLONE store ir<%l>, ir<%dst>
; DBG-NEXT:     Successor(s): pred.store.continue
; DBG-EMPTY:
; DBG-NEXT:     pred.store.continue:
; DBG-NEXT:       PHI-PREDICATED-INSTRUCTION vp<{{.+}}> = ir<%l>
; DBG-NEXT:     No successors
; DBG-NEXT:   }
; DBG-NEXT:   Successor(s): cond.false.1
; DBG-EMPTY:
; DBG-NEXT:   cond.false.1:
; DBG-NEXT:   Successor(s): loop.latch
; DBG-EMPTY:
; DBG-NEXT:   loop.latch:
; DBG-NEXT:     EMIT vp<[[CAN_IV_INC:%.+]]> = VF * UF +(nuw)  vp<[[CAN_IV]]>
; DBG-NEXT:     EMIT branch-on-count  vp<[[CAN_IV_INC]]> vp<[[VEC_TC]]>
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
; CHECK-NEXT:    [[INDEX:%.*]] = phi i64 [ 0, %vector.ph ], [ [[INDEX_NEXT:%.*]], %pred.store.continue7 ]
; CHECK-NEXT:    [[TMP0:%.*]] = trunc i64 [[INDEX]] to i1
; CHECK-NEXT:    [[OFFSET_IDX:%.*]] = sub i1 false, [[TMP0]]
; CHECK-NEXT:    [[INDUCTION:%.*]] = add i1 [[OFFSET_IDX]], false
; CHECK-NEXT:    [[INDUCTION3:%.*]] = add i1 [[OFFSET_IDX]], true
; CHECK-NEXT:    [[INDUCTION4:%.*]] = add i64 [[INDEX]], 0
; CHECK-NEXT:    [[INDUCTION5:%.*]] = add i64 [[INDEX]], 1
; CHECK-NEXT:    [[TMP1:%.*]] = getelementptr inbounds i32, ptr %dst, i64 [[INDUCTION4]]
; CHECK-NEXT:    [[TMP2:%.*]] = getelementptr inbounds i32, ptr %dst, i64 [[INDUCTION5]]
; CHECK-NEXT:    br i1 [[INDUCTION]], label %pred.store.if, label %pred.store.continue
; CHECK:       pred.store.if:
; CHECK-NEXT:    [[TMP3:%.*]] = getelementptr inbounds i32, ptr %src, i64 [[INDUCTION4]]
; CHECK-NEXT:    [[TMP4:%.*]] = load i32, ptr [[TMP3]], align 4
; CHECK-NEXT:    store i32 [[TMP4]], ptr %dst, align 4
; CHECK-NEXT:    br label %pred.store.continue
; CHECK:       pred.store.continue:
; CHECK-NEXT:    [[TMP5:%.*]] = phi i32 [ poison, %vector.body ], [ [[TMP4]], %pred.store.if ]
; CHECK-NEXT:    br i1 [[INDUCTION3]], label %pred.store.if6, label %pred.store.continue7
; CHECK:       pred.store.if6:
; CHECK-NEXT:    [[TMP6:%.*]] = getelementptr inbounds i32, ptr %src, i64 [[INDUCTION5]]
; CHECK-NEXT:    [[TMP7:%.*]] = load i32, ptr [[TMP6]], align 4
; CHECK-NEXT:    store i32 [[TMP7]], ptr %dst, align 4
; CHECK-NEXT:    br label %pred.store.continue7
; CHECK:       pred.store.continue7:
; CHECK-NEXT:    [[TMP8:%.*]] = phi i32 [ poison, %pred.store.continue ], [ [[TMP7]], %pred.store.if6 ]
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
  store i32 %l, ptr %dst
  br label %loop.latch

loop.latch:
  %iv.next = add nsw i64 %iv, 1
  %ec = icmp eq i64 %iv.next, 1000
  br i1 %ec, label %exit, label %loop.header

exit:
  ret void
}
