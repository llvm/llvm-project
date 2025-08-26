; REQUIRES: asserts
; RUN: opt -mtriple=aarch64-none-linux-gnu -mattr=+sve -force-target-instruction-cost=1 -passes=loop-vectorize -S -debug-only=loop-vectorize --disable-output -scalable-vectorization=off < %s 2>&1 | FileCheck %s --check-prefixes=CHECK,CHECK_SCALABLE_DISABLED
; RUN: opt -mtriple=aarch64-none-linux-gnu -mattr=+sve -force-target-instruction-cost=1 -passes=loop-vectorize -S -debug-only=loop-vectorize --disable-output -scalable-vectorization=on < %s 2>&1 | FileCheck %s --check-prefixes=CHECK,CHECK_SCALABLE_ON
; RUN: opt -mtriple=aarch64-none-linux-gnu -mattr=+sve -force-target-instruction-cost=1 -passes=loop-vectorize -S -debug-only=loop-vectorize --disable-output -vectorizer-maximize-bandwidth -scalable-vectorization=on < %s 2>&1 | FileCheck %s --check-prefixes=CHECK,CHECK_SCALABLE_ON_MAXBW

; Test that the MaxVF for the following loop, that has no dependence distances,
; is calculated as vscale x 4 (max legal SVE vector size) or vscale x 16
; (maximized bandwidth for i8 in the loop).
define void @test0(ptr %a, ptr %b, ptr %c) #0 {
; CHECK: LV: Checking a loop in 'test0'
; CHECK_SCALABLE_ON: LV: Found feasible scalable VF = vscale x 4
; CHECK_SCALABLE_ON: LV: Selecting VF: 16
; CHECK_SCALABLE_DISABLED-NOT: LV: Found feasible scalable VF
; CHECK_SCALABLE_DISABLED: LV: Selecting VF: 16
; CHECK_SCALABLE_ON_MAXBW: LV: Found feasible scalable VF = vscale x 16
; CHECK_SCALABLE_ON_MAXBW: LV: Selecting VF: vscale x 16
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %arrayidx = getelementptr inbounds i32, ptr %c, i64 %iv
  %0 = load i32, ptr %arrayidx, align 4
  %arrayidx2 = getelementptr inbounds i8, ptr %b, i64 %iv
  %1 = load i8, ptr %arrayidx2, align 4
  %zext = zext i8 %1 to i32
  %add = add nsw i32 %zext, %0
  %arrayidx5 = getelementptr inbounds i32, ptr %a, i64 %iv
  store i32 %add, ptr %arrayidx5, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond.not = icmp eq i64 %iv.next, 1024
  br i1 %exitcond.not, label %exit, label %loop

exit:
  ret void
}

; Test that the MaxVF for the following loop, with a dependence distance
; of 64 elements, is calculated as (maxvscale = 16) * 4.
define void @test1(ptr %a, ptr %b) #0 {
; CHECK: LV: Checking a loop in 'test1'
; CHECK_SCALABLE_ON: LV: Found feasible scalable VF = vscale x 4
; CHECK_SCALABLE_ON: LV: Selecting VF: 16
; CHECK_SCALABLE_DISABLED-NOT: LV: Found feasible scalable VF
; CHECK_SCALABLE_DISABLED: LV: Selecting VF: 16
; CHECK_SCALABLE_ON_MAXBW: LV: Found feasible scalable VF = vscale x 4
; CHECK_SCALABLE_ON_MAXBW: LV: Selecting VF: 16
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %arrayidx = getelementptr inbounds i32, ptr %a, i64 %iv
  %0 = load i32, ptr %arrayidx, align 4
  %arrayidx2 = getelementptr inbounds i8, ptr %b, i64 %iv
  %1 = load i8, ptr %arrayidx2, align 4
  %zext = zext i8 %1 to i32
  %add = add nsw i32 %zext, %0
  %2 = add nuw nsw i64 %iv, 64
  %arrayidx5 = getelementptr inbounds i32, ptr %a, i64 %2
  store i32 %add, ptr %arrayidx5, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond.not = icmp eq i64 %iv.next, 1024
  br i1 %exitcond.not, label %exit, label %loop

exit:
  ret void
}

; Test that the MaxVF for the following loop, with a dependence distance
; of 32 elements, is calculated as (maxvscale = 16) * 2.
define void @test2(ptr %a, ptr %b) #0 {
; CHECK: LV: Checking a loop in 'test2'
; CHECK_SCALABLE_ON: LV: Found feasible scalable VF = vscale x 2
; CHECK_SCALABLE_ON: LV: Selecting VF: 16
; CHECK_SCALABLE_DISABLED-NOT: LV: Found feasible scalable VF
; CHECK_SCALABLE_DISABLED: LV: Selecting VF: 16
; CHECK_SCALABLE_ON_MAXBW: LV: Found feasible scalable VF = vscale x 2
; CHECK_SCALABLE_ON_MAXBW: LV: Selecting VF: 16
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %arrayidx = getelementptr inbounds i32, ptr %a, i64 %iv
  %0 = load i32, ptr %arrayidx, align 4
  %arrayidx2 = getelementptr inbounds i8, ptr %b, i64 %iv
  %1 = load i8, ptr %arrayidx2, align 4
  %zext = zext i8 %1 to i32
  %add = add nsw i32 %zext, %0
  %2 = add nuw nsw i64 %iv, 32
  %arrayidx5 = getelementptr inbounds i32, ptr %a, i64 %2
  store i32 %add, ptr %arrayidx5, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond.not = icmp eq i64 %iv.next, 1024
  br i1 %exitcond.not, label %exit, label %loop

exit:
  ret void
}

; Test that the MaxVF for the following loop, with a dependence distance
; of 16 elements, is calculated as (maxvscale = 16) * 1.
define void @test3(ptr %a, ptr %b) #0 {
; CHECK: LV: Checking a loop in 'test3'
; CHECK_SCALABLE_ON: LV: Found feasible scalable VF = vscale x 1
; CHECK_SCALABLE_ON: LV: Selecting VF: 16
; CHECK_SCALABLE_DISABLED-NOT: LV: Found feasible scalable VF
; CHECK_SCALABLE_DISABLED: LV: Selecting VF: 16
; CHECK_SCALABLE_ON_MAXBW: LV: Found feasible scalable VF = vscale x 1
; CHECK_SCALABLE_ON_MAXBW: LV: Selecting VF: 16
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %arrayidx = getelementptr inbounds i32, ptr %a, i64 %iv
  %0 = load i32, ptr %arrayidx, align 4
  %arrayidx2 = getelementptr inbounds i8, ptr %b, i64 %iv
  %1 = load i8, ptr %arrayidx2, align 4
  %zext = zext i8 %1 to i32
  %add = add nsw i32 %zext, %0
  %2 = add nuw nsw i64 %iv, 16
  %arrayidx5 = getelementptr inbounds i32, ptr %a, i64 %2
  store i32 %add, ptr %arrayidx5, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond.not = icmp eq i64 %iv.next, 1024
  br i1 %exitcond.not, label %exit, label %loop

exit:
  ret void
}

; Test the fallback mechanism when scalable vectors are not feasible due
; to e.g. dependence distance.
define void @test4(ptr %a, ptr %b) #0 {
; CHECK: LV: Checking a loop in 'test4'
; CHECK_SCALABLE_ON-NOT: LV: Found feasible scalable VF
; CHECK_SCALABLE_ON-NOT: LV: Found feasible scalable VF
; CHECK_SCALABLE_ON: LV: Selecting VF: 4
; CHECK_SCALABLE_DISABLED-NOT: LV: Found feasible scalable VF
; CHECK_SCALABLE_DISABLED: LV: Selecting VF: 4
; CHECK_SCALABLE_ON_MAXBW-NOT: LV: Found feasible scalable VF
; CHECK_SCALABLE_ON_MAXBW: LV: Selecting VF: 4
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %arrayidx = getelementptr inbounds i32, ptr %a, i64 %iv
  %0 = load i32, ptr %arrayidx, align 4
  %arrayidx2 = getelementptr inbounds i32, ptr %b, i64 %iv
  %1 = load i32, ptr %arrayidx2, align 4
  %add = add nsw i32 %1, %0
  %2 = add nuw nsw i64 %iv, 8
  %arrayidx5 = getelementptr inbounds i32, ptr %a, i64 %2
  store i32 %add, ptr %arrayidx5, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond.not = icmp eq i64 %iv.next, 1024
  br i1 %exitcond.not, label %exit, label %loop

exit:
  ret void
}

attributes #0 = { vscale_range(1, 16) }
