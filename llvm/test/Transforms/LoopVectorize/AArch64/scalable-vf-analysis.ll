; REQUIRES: asserts
; RUN: opt -mtriple=aarch64-none-linux-gnu -mattr=+sve -force-target-instruction-cost=1 -loop-vectorize -S -debug-only=loop-vectorize < %s 2>&1 | FileCheck %s --check-prefixes=CHECK,CHECK_SCALABLE_ON
; RUN: opt -mtriple=aarch64-none-linux-gnu -mattr=+sve -force-target-instruction-cost=1 -loop-vectorize -S -debug-only=loop-vectorize -vectorizer-maximize-bandwidth < %s 2>&1 | FileCheck %s --check-prefixes=CHECK,CHECK_SCALABLE_ON_MAXBW

; Test that the MaxVF for the following loop, that has no dependence distances,
; is calculated as vscale x 4 (max legal SVE vector size) or vscale x 16
; (maximized bandwidth for i8 in the loop).
define void @test0(i32* %a, i8* %b, i32* %c) {
; CHECK: LV: Checking a loop in "test0"
; CHECK_SCALABLE_ON: LV: Found feasible scalable VF = vscale x 4
; CHECK_SCALABLE_ON_MAXBW: LV: Found feasible scalable VF = vscale x 16
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %arrayidx = getelementptr inbounds i32, i32* %c, i64 %iv
  %0 = load i32, i32* %arrayidx, align 4
  %arrayidx2 = getelementptr inbounds i8, i8* %b, i64 %iv
  %1 = load i8, i8* %arrayidx2, align 4
  %zext = zext i8 %1 to i32
  %add = add nsw i32 %zext, %0
  %arrayidx5 = getelementptr inbounds i32, i32* %a, i64 %iv
  store i32 %add, i32* %arrayidx5, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond.not = icmp eq i64 %iv.next, 1024
  br i1 %exitcond.not, label %exit, label %loop, !llvm.loop !0

exit:
  ret void
}

; Test that the MaxVF for the following loop, with a dependence distance
; of 64 elements, is calculated as (maxvscale = 16) * 4.
define void @test1(i32* %a, i8* %b) {
; CHECK: LV: Checking a loop in "test1"
; CHECK_SCALABLE_ON: LV: Found feasible scalable VF = vscale x 4
; CHECK_SCALABLE_ON_MAXBW: LV: Found feasible scalable VF = vscale x 4
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %arrayidx = getelementptr inbounds i32, i32* %a, i64 %iv
  %0 = load i32, i32* %arrayidx, align 4
  %arrayidx2 = getelementptr inbounds i8, i8* %b, i64 %iv
  %1 = load i8, i8* %arrayidx2, align 4
  %zext = zext i8 %1 to i32
  %add = add nsw i32 %zext, %0
  %2 = add nuw nsw i64 %iv, 64
  %arrayidx5 = getelementptr inbounds i32, i32* %a, i64 %2
  store i32 %add, i32* %arrayidx5, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond.not = icmp eq i64 %iv.next, 1024
  br i1 %exitcond.not, label %exit, label %loop, !llvm.loop !0

exit:
  ret void
}

; Test that the MaxVF for the following loop, with a dependence distance
; of 32 elements, is calculated as (maxvscale = 16) * 2.
define void @test2(i32* %a, i8* %b) {
; CHECK: LV: Checking a loop in "test2"
; CHECK_SCALABLE_ON: LV: Found feasible scalable VF = vscale x 2
; CHECK_SCALABLE_ON_MAXBW: LV: Found feasible scalable VF = vscale x 2
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %arrayidx = getelementptr inbounds i32, i32* %a, i64 %iv
  %0 = load i32, i32* %arrayidx, align 4
  %arrayidx2 = getelementptr inbounds i8, i8* %b, i64 %iv
  %1 = load i8, i8* %arrayidx2, align 4
  %zext = zext i8 %1 to i32
  %add = add nsw i32 %zext, %0
  %2 = add nuw nsw i64 %iv, 32
  %arrayidx5 = getelementptr inbounds i32, i32* %a, i64 %2
  store i32 %add, i32* %arrayidx5, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond.not = icmp eq i64 %iv.next, 1024
  br i1 %exitcond.not, label %exit, label %loop, !llvm.loop !0

exit:
  ret void
}

; Test that the MaxVF for the following loop, with a dependence distance
; of 16 elements, is calculated as (maxvscale = 16) * 1.
define void @test3(i32* %a, i8* %b) {
; CHECK: LV: Checking a loop in "test3"
; CHECK_SCALABLE_ON: LV: Found feasible scalable VF = vscale x 1
; CHECK_SCALABLE_ON_MAXBW: LV: Found feasible scalable VF = vscale x 1
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %arrayidx = getelementptr inbounds i32, i32* %a, i64 %iv
  %0 = load i32, i32* %arrayidx, align 4
  %arrayidx2 = getelementptr inbounds i8, i8* %b, i64 %iv
  %1 = load i8, i8* %arrayidx2, align 4
  %zext = zext i8 %1 to i32
  %add = add nsw i32 %zext, %0
  %2 = add nuw nsw i64 %iv, 16
  %arrayidx5 = getelementptr inbounds i32, i32* %a, i64 %2
  store i32 %add, i32* %arrayidx5, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond.not = icmp eq i64 %iv.next, 1024
  br i1 %exitcond.not, label %exit, label %loop, !llvm.loop !0

exit:
  ret void
}

; Test the fallback mechanism when scalable vectors are not feasible due
; to e.g. dependence distance. For the '-scalable-vectorization=exclusive'
; it shouldn't try to vectorize with fixed-width vectors.
define void @test4(i32* %a, i32* %b) {
; CHECK: LV: Checking a loop in "test4"
; CHECK_SCALABLE_ON-NOT: LV: Found feasible scalable VF
; CHECK_SCALABLE_ON_MAXBW-NOT: LV: Found feasible scalable VF
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %arrayidx = getelementptr inbounds i32, i32* %a, i64 %iv
  %0 = load i32, i32* %arrayidx, align 4
  %arrayidx2 = getelementptr inbounds i32, i32* %b, i64 %iv
  %1 = load i32, i32* %arrayidx2, align 4
  %add = add nsw i32 %1, %0
  %2 = add nuw nsw i64 %iv, 8
  %arrayidx5 = getelementptr inbounds i32, i32* %a, i64 %2
  store i32 %add, i32* %arrayidx5, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond.not = icmp eq i64 %iv.next, 1024
  br i1 %exitcond.not, label %exit, label %loop, !llvm.loop !2

exit:
  ret void
}

!0 = distinct !{!0, !1}
!1 = !{!"llvm.loop.vectorize.enable", i1 true}
!2 = distinct !{!2, !3, !4}
!3 = !{!"llvm.loop.vectorize.enable", i1 true}
!4 = !{!"llvm.loop.vectorize.scalable.enable", i1 true}
