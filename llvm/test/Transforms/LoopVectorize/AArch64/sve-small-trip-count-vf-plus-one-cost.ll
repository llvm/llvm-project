; REQUIRES: asserts
; RUN: opt -passes=loop-vectorize -mtriple=aarch64-linux-gnu -mattr=+sve -S %s | FileCheck %s --check-prefix=IR
; RUN: opt -passes=loop-vectorize -mtriple=aarch64-linux-gnu -mattr=+sve -debug-only=loop-vectorize -disable-output %s 2>&1 | FileCheck %s --check-prefix=DBG

target triple = "aarch64-unknown-linux-gnu"

define void @tc3_udiv_i8_reject(ptr noalias %a, ptr noalias %b,
                                ptr noalias %c) #0 {
; IR-LABEL: define void @tc3_udiv_i8_reject(
; IR-NOT: vector.body:
;
; DBG-LABEL: LV: Checking a loop in 'tc3_udiv_i8_reject'
; DBG: LV: Picking MaxVF=2 with 1 scalar iteration remaining.
; DBG: LV: Scalar loop costs: 9.
; DBG: Cost for VF 2: 19
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

define void @tc3_smin_i8_accept(ptr noalias %a, ptr noalias %b) #0 {
; IR-LABEL: define void @tc3_smin_i8_accept(
; IR: vector.body

; DBG-LABEL: LV: Checking a loop in 'tc3_smin_i8_accept'
; DBG: LV: Picking MaxVF=2 with 1 scalar iteration remaining.
; DBG: LV: Scalar loop costs: 10.
; DBG: Cost for VF 2: 19
; DBG: LV: Selecting VF: 2.
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
; IR-LABEL: define void @tc3_udiv_i8_forced(
; IR: vector.body

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

define void @tc5_udiv_i8_reject_ic2(ptr noalias %a, ptr noalias %b, ptr noalias %c) #0{
; IR-LABEL: define void @tc5_udiv_i8_reject_ic2(
; IR: vector.body

; DBG-LABEL: LV: Checking a loop in 'tc5_udiv_i8_reject_ic2'
; DBG-NOT: LV: Selecting VF: 2.
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
  %exitcond = icmp eq i64 %iv.next, 5
  br i1 %exitcond, label %exit, label %loop, !llvm.loop !2

exit:
  ret void
}

; Ensure that the LoopVectorizer will select a VF that will
; ensure TC == (VF * IC) + 1.
define void @tc5_sin_f32_dont_select_smaller_vf(ptr noalias %a,
                                             ptr noalias %c) #0 {
; IR-LABEL: define void @tc5_sin_f32_dont_select_smaller_vf(
; IR: vector.body

; DBG-LABEL: LV: Checking a loop in 'tc5_sin_f32_dont_select_smaller_vf' 
; DBG: Picking MaxVF=4 with 1 scalar iteration remaining.
; DBG-NOT: Selecting VF: 2.
; DBG: Selecting VF: 4


entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %pa = getelementptr inbounds float, ptr %a, i64 %iv
  %pc = getelementptr inbounds float, ptr %c, i64 %iv
  %va = load float, ptr %pa, align 4
  %sin = tail call float @llvm.sin.f32(float %va)
  store float %sin, ptr %pc, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 5
  br i1 %exitcond, label %exit, label %loop

exit:
  ret void
}

declare i8 @llvm.smin.i8(i8, i8)

declare float @llvm.sin.f32(float)

attributes #0 = { vscale_range(1,16) "target-features"="+sve" }

!0 = distinct !{!0, !1}
!1 = !{!"llvm.loop.vectorize.enable"}
!2 = distinct !{!2, !3}
!3 = !{!"llvm.loop.interleave.count", i32 2}
