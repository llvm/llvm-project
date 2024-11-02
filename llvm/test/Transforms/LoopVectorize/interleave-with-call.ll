; REQUIRES: asserts

; RUN: opt -passes=loop-vectorize -force-vector-width=1 -force-vector-interleave=2 -debug -S %s 2>&1 | FileCheck %s

; CHECK:      VPlan 'Initial VPlan for VF={1},UF>=1' {
; CHECK-NEXT: Live-in vp<[[VEC_TC:%.+]]> = vector-trip-count
; CHECK-EMPTY:
; CHECK-NEXT: vector.ph:
; CHECK-NEXT: Successor(s): vector loop
; CHECK-EMPTY:
; CHECK-NEXT: <x1> vector loop: {
; CHECK-NEXT:   vector.body:
; CHECK-NEXT:     EMIT vp<[[CAN_IV:%.+]]> = CANONICAL-INDUCTION
; CHECK-NEXT:     vp<[[IV_STEPS:%.]]>    = SCALAR-STEPS vp<[[CAN_IV]]>, ir<%start>, ir<1>
; CHECK-NEXT:     CLONE ir<%min> = call @llvm.smin.i32(vp<[[IV_STEPS]]>, ir<65535>)
; CHECK-NEXT:     CLONE ir<%arrayidx> = getelementptr ir<%dst>, vp<[[IV_STEPS]]>
; CHECK-NEXT:     CLONE store ir<%min>, ir<%arrayidx>
; CHECK-NEXT:     EMIT vp<[[INC:%.+]]> = VF * UF +(nuw)  vp<[[CAN_IV]]>
; CHECK-NEXT:     EMIT branch-on-count  vp<[[INC]]> vp<[[VEC_TC]]>
; CHECK-NEXT:   No successors
; CHECK-NEXT: }
;
define void @test(i32 %start, ptr %dst) {
; CHECK-LABEL: @test(
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
