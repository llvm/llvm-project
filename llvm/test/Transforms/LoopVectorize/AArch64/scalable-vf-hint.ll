; REQUIRES: asserts
; RUN: opt -mtriple=aarch64-none-linux-gnu -mattr=+sve -loop-vectorize -S < %s 2>&1 | FileCheck %s
; RUN: opt -mtriple=aarch64-none-linux-gnu -mattr=+sve -loop-vectorize -pass-remarks-analysis=loop-vectorize -debug-only=loop-vectorize -S < %s 2>&1 | FileCheck --check-prefix=CHECK-DBG %s
; RUN: opt -mtriple=aarch64-none-linux-gnu -loop-vectorize -pass-remarks-analysis=loop-vectorize -debug-only=loop-vectorize -S < %s 2>%t | FileCheck --check-prefix=CHECK-NO-SVE %s
; RUN: cat %t | FileCheck %s -check-prefix=CHECK-NO-SVE-REMARKS

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"

; These tests validate the behaviour of scalable vectorization factor hints,
; where the following applies:
;
; * If the backend does not support scalable vectors, ignore the hint and let
;   the vectorizer pick a VF.
; * If there are no dependencies and assuming the VF is a power of 2 the VF
;   should be accepted. This applies to both fixed and scalable VFs.
; * If the dependency is too small to use scalable vectors, change the VF to
;   fixed, where existing behavior applies (clamping).
; * If scalable vectorization is feasible given the dependency and the VF is
;   valid, accept it. Otherwise, clamp to the max scalable VF.

; test1
;
; Scalable vectorization unfeasible, clamp VF from (4, scalable) -> (4, fixed).
;
; The pragma applied to this loop implies a scalable vector <vscale x 4 x i32>
; be used for vectorization. For fixed vectors the MaxVF=8, otherwise there
; would be a dependence between vector lanes for vectors greater than 256 bits.
;
; void test1(int *a, int *b, int N) {
;   #pragma clang loop vectorize(enable) vectorize_width(4, scalable)
;   for (int i=0; i<N; ++i) {
;     a[i + 8] = a[i] + b[i];
;   }
; }
;
; For scalable vectorization 'vscale' has to be considered, for this example
; unless max(vscale)=2 it's unsafe to vectorize. For SVE max(vscale)=16, check
; fixed-width vectorization is used instead.

; CHECK-DBG: LV: Checking a loop in "test1"
; CHECK-DBG: LV: Scalable vectorization is available
; CHECK-DBG: LV: Max legal vector width too small, scalable vectorization unfeasible.
; CHECK-DBG: remark: <unknown>:0:0: Max legal vector width too small, scalable vectorization unfeasible.
; CHECK-DBG: LV: The max safe fixed VF is: 8.
; CHECK-DBG: LV: Selecting VF: 4.
; CHECK-LABEL: @test1
; CHECK: <4 x i32>
define void @test1(i32* %a, i32* %b) #0 {
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
  br i1 %exitcond.not, label %exit, label %loop, !llvm.loop !0

exit:
  ret void
}

!0 = !{!0, !1, !2}
!1 = !{!"llvm.loop.vectorize.width", i32 4}
!2 = !{!"llvm.loop.vectorize.scalable.enable", i1 true}

; test2
;
; Scalable vectorization unfeasible, clamp VF from (8, scalable) -> (4, fixed).
;
; void test2(int *a, int *b, int N) {
;   #pragma clang loop vectorize(enable) vectorize_width(8, scalable)
;   for (int i=0; i<N; ++i) {
;     a[i + 4] = a[i] + b[i];
;   }
; }

; CHECK-DBG: LV: Checking a loop in "test2"
; CHECK-DBG: LV: Scalable vectorization is available
; CHECK-DBG: LV: Max legal vector width too small, scalable vectorization unfeasible.
; CHECK-DBG: LV: The max safe fixed VF is: 4.
; CHECK-DBG: LV: User VF=vscale x 8 is unsafe. Ignoring scalable UserVF.
; CHECK-DBG: LV: Selecting VF: 4.
; CHECK-LABEL: @test2
; CHECK: <4 x i32>
define void @test2(i32* %a, i32* %b) #0 {
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %arrayidx = getelementptr inbounds i32, i32* %a, i64 %iv
  %0 = load i32, i32* %arrayidx, align 4
  %arrayidx2 = getelementptr inbounds i32, i32* %b, i64 %iv
  %1 = load i32, i32* %arrayidx2, align 4
  %add = add nsw i32 %1, %0
  %2 = add nuw nsw i64 %iv, 4
  %arrayidx5 = getelementptr inbounds i32, i32* %a, i64 %2
  store i32 %add, i32* %arrayidx5, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond.not = icmp eq i64 %iv.next, 1024
  br i1 %exitcond.not, label %exit, label %loop, !llvm.loop !3

exit:
  ret void
}

!3 = !{!3, !4, !5}
!4 = !{!"llvm.loop.vectorize.width", i32 8}
!5 = !{!"llvm.loop.vectorize.scalable.enable", i1 true}

; test3
;
; Scalable vectorization feasible and the VF is valid.
;
; Specifies a vector of <vscale x 2 x i32>, i.e. maximum of 32 x i32 with 2
; words per 128-bits (unpacked).
;
; void test3(int *a, int *b, int N) {
;   #pragma clang loop vectorize(enable) vectorize_width(2, scalable)
;   for (int i=0; i<N; ++i) {
;     a[i + 32] = a[i] + b[i];
;   }
; }
;
; Max fixed VF=32, Max scalable VF=2, safe to vectorize.

; CHECK-DBG-LABEL: LV: Checking a loop in "test3"
; CHECK-DBG: LV: Scalable vectorization is available
; CHECK-DBG: LV: The max safe scalable VF is: vscale x 2.
; CHECK-DBG: LV: Using user VF vscale x 2.
; CHECK-LABEL: @test3
; CHECK: <vscale x 2 x i32>
define void @test3(i32* %a, i32* %b) #0 {
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %arrayidx = getelementptr inbounds i32, i32* %a, i64 %iv
  %0 = load i32, i32* %arrayidx, align 4
  %arrayidx2 = getelementptr inbounds i32, i32* %b, i64 %iv
  %1 = load i32, i32* %arrayidx2, align 4
  %add = add nsw i32 %1, %0
  %2 = add nuw nsw i64 %iv, 32
  %arrayidx5 = getelementptr inbounds i32, i32* %a, i64 %2
  store i32 %add, i32* %arrayidx5, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond.not = icmp eq i64 %iv.next, 1024
  br i1 %exitcond.not, label %exit, label %loop, !llvm.loop !6

exit:
  ret void
}

!6 = !{!6, !7, !8}
!7 = !{!"llvm.loop.vectorize.width", i32 2}
!8 = !{!"llvm.loop.vectorize.scalable.enable", i1 true}

; test4
;
; Scalable vectorization feasible, but the given VF is unsafe. Should ignore
; the hint and leave it to the vectorizer to pick a more suitable VF.
;
; Specifies a vector of <vscale x 4 x i32>, i.e. maximum of 64 x i32 with 4
; words per 128-bits (packed).
;
; void test4(int *a, int *b, int N) {
;   #pragma clang loop vectorize(enable) vectorize_width(4, scalable)
;   for (int i=0; i<N; ++i) {
;     a[i + 32] = a[i] + b[i];
;   }
; }
;
; Max fixed VF=32, Max scalable VF=2, unsafe to vectorize.

; CHECK-DBG-LABEL: LV: Checking a loop in "test4"
; CHECK-DBG: LV: Scalable vectorization is available
; CHECK-DBG: LV: The max safe scalable VF is: vscale x 2.
; CHECK-DBG: LV: User VF=vscale x 4 is unsafe. Ignoring scalable UserVF.
; CHECK-DBG: remark: <unknown>:0:0: User-specified vectorization factor vscale x 4 is unsafe. Ignoring the hint to let the compiler pick a more suitable value.
; CHECK-DBG: Found feasible scalable VF = vscale x 2
; CHECK-DBG: LV: Selecting VF: vscale x 2.
; CHECK-LABEL: @test4
; CHECK: <vscale x 2 x i32>
define void @test4(i32* %a, i32* %b) #0 {
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %arrayidx = getelementptr inbounds i32, i32* %a, i64 %iv
  %0 = load i32, i32* %arrayidx, align 4
  %arrayidx2 = getelementptr inbounds i32, i32* %b, i64 %iv
  %1 = load i32, i32* %arrayidx2, align 4
  %add = add nsw i32 %1, %0
  %2 = add nuw nsw i64 %iv, 32
  %arrayidx5 = getelementptr inbounds i32, i32* %a, i64 %2
  store i32 %add, i32* %arrayidx5, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond.not = icmp eq i64 %iv.next, 1024
  br i1 %exitcond.not, label %exit, label %loop, !llvm.loop !9

exit:
  ret void
}

!9 = !{!9, !10, !11}
!10 = !{!"llvm.loop.vectorize.width", i32 4}
!11 = !{!"llvm.loop.vectorize.scalable.enable", i1 true}

; test5
;
; Scalable vectorization feasible and the VF is valid.
;
; Specifies a vector of <vscale x 4 x i32>, i.e. maximum of 64 x i32 with 4
; words per 128-bits (packed).
;
; void test5(int *a, int *b, int N) {
;   #pragma clang loop vectorize(enable) vectorize_width(4, scalable)
;   for (int i=0; i<N; ++i) {
;     a[i + 128] = a[i] + b[i];
;   }
; }
;
; Max fixed VF=128, Max scalable VF=8, safe to vectorize.

; CHECK-DBG-LABEL: LV: Checking a loop in "test5"
; CHECK-DBG: LV: Scalable vectorization is available
; CHECK-DBG: LV: The max safe scalable VF is: vscale x 8.
; CHECK-DBG: LV: Using user VF vscale x 4
; CHECK-LABEL: @test5
; CHECK: <vscale x 4 x i32>
define void @test5(i32* %a, i32* %b) #0 {
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %arrayidx = getelementptr inbounds i32, i32* %a, i64 %iv
  %0 = load i32, i32* %arrayidx, align 4
  %arrayidx2 = getelementptr inbounds i32, i32* %b, i64 %iv
  %1 = load i32, i32* %arrayidx2, align 4
  %add = add nsw i32 %1, %0
  %2 = add nuw nsw i64 %iv, 128
  %arrayidx5 = getelementptr inbounds i32, i32* %a, i64 %2
  store i32 %add, i32* %arrayidx5, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond.not = icmp eq i64 %iv.next, 1024
  br i1 %exitcond.not, label %exit, label %loop, !llvm.loop !12

exit:
  ret void
}

!12 = !{!12, !13, !14}
!13 = !{!"llvm.loop.vectorize.width", i32 4}
!14 = !{!"llvm.loop.vectorize.scalable.enable", i1 true}

; test6
;
; Scalable vectorization feasible, but the VF is unsafe. Should ignore
; the hint and leave it to the vectorizer to pick a more suitable VF.
;
; Specifies a vector of <vscale x 16 x i32>, i.e. maximum of 256 x i32.
;
; void test6(int *a, int *b, int N) {
;   #pragma clang loop vectorize(enable) vectorize_width(16, scalable)
;   for (int i=0; i<N; ++i) {
;     a[i + 128] = a[i] + b[i];
;   }
; }
;
; Max fixed VF=128, Max scalable VF=8, unsafe to vectorize.

; CHECK-DBG-LABEL: LV: Checking a loop in "test6"
; CHECK-DBG: LV: Scalable vectorization is available
; CHECK-DBG: LV: The max safe scalable VF is: vscale x 8.
; CHECK-DBG: LV: User VF=vscale x 16 is unsafe. Ignoring scalable UserVF.
; CHECK-DBG: remark: <unknown>:0:0: User-specified vectorization factor vscale x 16 is unsafe. Ignoring the hint to let the compiler pick a more suitable value.
; CHECK-DBG: LV: Found feasible scalable VF = vscale x 4
; CHECK-DBG: Selecting VF: vscale x 4.
; CHECK-LABEL: @test6
; CHECK: <vscale x 4 x i32>
define void @test6(i32* %a, i32* %b) #0 {
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %arrayidx = getelementptr inbounds i32, i32* %a, i64 %iv
  %0 = load i32, i32* %arrayidx, align 4
  %arrayidx2 = getelementptr inbounds i32, i32* %b, i64 %iv
  %1 = load i32, i32* %arrayidx2, align 4
  %add = add nsw i32 %1, %0
  %2 = add nuw nsw i64 %iv, 128
  %arrayidx5 = getelementptr inbounds i32, i32* %a, i64 %2
  store i32 %add, i32* %arrayidx5, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond.not = icmp eq i64 %iv.next, 1024
  br i1 %exitcond.not, label %exit, label %loop, !llvm.loop !15

exit:
  ret void
}

!15 = !{!15, !16, !17}
!16 = !{!"llvm.loop.vectorize.width", i32 16}
!17 = !{!"llvm.loop.vectorize.scalable.enable", i1 true}

; CHECK-NO-SVE-REMARKS-LABEL: LV: Checking a loop in "test_no_sve"
; CHECK-NO-SVE-REMARKS: LV: User VF=vscale x 4 is ignored because scalable vectors are not available.
; CHECK-NO-SVE-REMARKS: remark: <unknown>:0:0: User-specified vectorization factor vscale x 4 is ignored because the target does not support scalable vectors. The compiler will pick a more suitable value.
; CHECK-NO-SVE-REMARKS: LV: Selecting VF: 4.
; CHECK-NO-SVE-LABEL: @test_no_sve
; CHECK-NO-SVE: <4 x i32>
; CHECK-NO-SVE-NOT: <vscale x 4 x i32>
define void @test_no_sve(i32* %a, i32* %b) #0 {
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %arrayidx = getelementptr inbounds i32, i32* %a, i64 %iv
  %0 = load i32, i32* %arrayidx, align 4
  %arrayidx2 = getelementptr inbounds i32, i32* %b, i64 %iv
  %1 = load i32, i32* %arrayidx2, align 4
  %add = add nsw i32 %1, %0
  store i32 %add, i32* %arrayidx, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond.not = icmp eq i64 %iv.next, 1024
  br i1 %exitcond.not, label %exit, label %loop, !llvm.loop !18

exit:
  ret void
}

!18 = !{!18, !19, !20}
!19 = !{!"llvm.loop.vectorize.width", i32 4}
!20 = !{!"llvm.loop.vectorize.scalable.enable", i1 true}

; Test the LV falls back to fixed-width vectorization if scalable vectors are
; supported but max vscale is undefined.
;
; CHECK-DBG-LABEL: LV: Checking a loop in "test_no_max_vscale"
; CHECK-DBG: LV: Scalable vectorization is available
; CHECK-DBG: The max safe fixed VF is: 4.
; CHECK-DBG: LV: User VF=vscale x 4 is unsafe. Ignoring scalable UserVF.
; CHECK-DBG: LV: Selecting VF: 4.
; CHECK-LABEL: @test_no_max_vscale
; CHECK: <4 x i32>
define void @test_no_max_vscale(i32* %a, i32* %b) #0 {
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %arrayidx = getelementptr inbounds i32, i32* %a, i64 %iv
  %0 = load i32, i32* %arrayidx, align 4
  %arrayidx2 = getelementptr inbounds i32, i32* %b, i64 %iv
  %1 = load i32, i32* %arrayidx2, align 4
  %add = add nsw i32 %1, %0
  %2 = add nuw nsw i64 %iv, 4
  %arrayidx5 = getelementptr inbounds i32, i32* %a, i64 %2
  store i32 %add, i32* %arrayidx5, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond.not = icmp eq i64 %iv.next, 1024
  br i1 %exitcond.not, label %exit, label %loop, !llvm.loop !21

exit:
  ret void
}

attributes #0 = { vscale_range(1, 16) }
!21 = !{!21, !22, !23}
!22 = !{!"llvm.loop.vectorize.width", i32 4}
!23 = !{!"llvm.loop.vectorize.scalable.enable", i1 true}
