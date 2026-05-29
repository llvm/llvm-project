; REQUIRES: asserts
; RUN: opt -passes=loop-vectorize -mtriple=aarch64-linux-gnu -mattr=+sve -S %s | FileCheck %s --check-prefix=IR
; RUN: opt -passes=loop-vectorize -mtriple=aarch64-linux-gnu -mattr=+sve -debug-only=loop-vectorize -disable-output %s 2>&1 | FileCheck %s --check-prefix=DBG

target triple = "aarch64-unknown-linux-gnu"

define void @tc3_udiv_i8_reject(ptr noalias %a, ptr noalias %b,
                                ptr noalias %c) #0 {
; IR-LABEL: define void @tc3_udiv_i8_reject(
; IR-NOT: vector.body
; IR-LABEL: define void @tc3_udiv_i8_user_vf2(
; IR-NOT: vector.body
; IR-LABEL: define void @tc3_smin_i8_reject(
; IR-NOT: vector.body
; IR-LABEL: define void @tc3_udiv_i8_forced(
; IR: vector.body:
;
; DBG-LABEL: LV: Checking a loop in 'tc3_udiv_i8_reject'
; DBG: LV: Picking VF=2 with 1 scalar iteration remaining.
; DBG: LV: Scalar loop costs: 9.
; DBG: Cost for VF 2: 19
; DBG: LV: Rejecting VF 2 for one-scalar-tail low trip count: vector cost 37 >= scalar cost 27.
; DBG: LV: Selecting VF: 1.
; DBG: LV: Vectorization is possible but not beneficial.
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %pa = getelementptr inbounds i8, ptr %a, i64 %iv
  %pb = getelementptr inbounds i8, ptr %b, i64 %iv
  %pc = getelementptr inbounds i8, ptr %c, i64 %iv
  %va = load i8, ptr %pa, align 1
  %vb = load i8, ptr %pb, align 1
  %div = udiv i8 %va, %vb
  store i8 %div, ptr %pc, align 1
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 3
  br i1 %exitcond, label %exit, label %loop

exit:
  ret void
}

define void @tc3_udiv_i8_user_vf2(ptr noalias %a, ptr noalias %b,
                                  ptr noalias %c) #0 {
; DBG-LABEL: LV: Checking a loop in 'tc3_udiv_i8_user_vf2'
; DBG: LV: Using user VF 2.
; DBG: LV: Scalar loop costs: 9.
; DBG: Cost for VF 2: 19
; DBG: LV: Rejecting VF 2 for one-scalar-tail low trip count: vector cost 37 >= scalar cost 27.
; DBG: LV: Vectorization is possible but not beneficial.
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %pa = getelementptr inbounds i8, ptr %a, i64 %iv
  %pb = getelementptr inbounds i8, ptr %b, i64 %iv
  %pc = getelementptr inbounds i8, ptr %c, i64 %iv
  %va = load i8, ptr %pa, align 1
  %vb = load i8, ptr %pb, align 1
  %div = udiv i8 %va, %vb
  store i8 %div, ptr %pc, align 1
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 3
  br i1 %exitcond, label %exit, label %loop, !llvm.loop !2

exit:
  ret void
}

define void @tc3_smin_i8_reject(ptr noalias %a, ptr noalias %b) #0 {
; DBG-LABEL: LV: Checking a loop in 'tc3_smin_i8_reject'
; DBG: LV: Picking VF=2 with 1 scalar iteration remaining.
; DBG: LV: Scalar loop costs: 10.
; DBG: Cost for VF 2: 15
; DBG: LV: Rejecting VF 2 for one-scalar-tail low trip count: vector cost 35 >= scalar cost 30.
; DBG: LV: Selecting VF: 1.
; DBG: LV: Vectorization is possible but not beneficial.
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %arrayidx = getelementptr inbounds i8, ptr %b, i64 %iv
  %0 = load i8, ptr %arrayidx, align 1
  %arrayidx2 = getelementptr inbounds i8, ptr %a, i64 %iv
  %1 = load i8, ptr %arrayidx2, align 1
  %min = tail call i8 @llvm.smin.i8(i8 %0, i8 %1)
  store i8 %min, ptr %arrayidx, align 1
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 3
  br i1 %exitcond, label %exit, label %loop

exit:
  ret void
}

define void @tc3_udiv_i8_forced(ptr noalias %a, ptr noalias %b,
                                ptr noalias %c) #0 {
; DBG-LABEL: LV: Checking a loop in 'tc3_udiv_i8_forced'
; DBG-NOT: Rejecting VF 2
; DBG: LV: Selecting VF: 2.
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %pa = getelementptr inbounds i8, ptr %a, i64 %iv
  %pb = getelementptr inbounds i8, ptr %b, i64 %iv
  %pc = getelementptr inbounds i8, ptr %c, i64 %iv
  %va = load i8, ptr %pa, align 1
  %vb = load i8, ptr %pb, align 1
  %div = udiv i8 %va, %vb
  store i8 %div, ptr %pc, align 1
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 3
  br i1 %exitcond, label %exit, label %loop, !llvm.loop !0

exit:
  ret void
}

declare i8 @llvm.smin.i8(i8, i8)

attributes #0 = { vscale_range(1,16) "target-features"="+sve" }

!0 = distinct !{!0, !1}
!1 = !{!"llvm.loop.vectorize.enable", i1 true}
!2 = distinct !{!2, !3}
!3 = !{!"llvm.loop.vectorize.width", i32 2}
