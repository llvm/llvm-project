; RUN: opt -loop-vectorize -vectorizer-maximize-bandwidth -mcpu=corei7-avx -debug-only=loop-vectorize -S < %s 2>&1 | FileCheck %s --check-prefix=CHECK-AVX1
; RUN: opt -loop-vectorize -vectorizer-maximize-bandwidth -mcpu=core-avx2 -debug-only=loop-vectorize -S < %s 2>&1 | FileCheck %s --check-prefix=CHECK-AVX2
; REQUIRES: asserts

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@a = global [1000 x i8] zeroinitializer, align 16
@b = global [1000 x i8] zeroinitializer, align 16
@c = global [1000 x i8] zeroinitializer, align 16
@u = global [1000 x i32] zeroinitializer, align 16
@v = global [1000 x i32] zeroinitializer, align 16
@w = global [1000 x i32] zeroinitializer, align 16

; Tests that the vectorization factor is determined by the smallest instead of
; widest type in the loop for maximum bandwidth when
; -vectorizer-maximize-bandwidth is indicated.
;
; CHECK-label: foo
; CHECK-AVX1: LV: Selecting VF: 16.
; CHECK-AVX2: LV: Selecting VF: 32.
define void @foo() {
entry:
  br label %for.body

for.cond.cleanup:
  ret void

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds [1000 x i8], [1000 x i8]* @b, i64 0, i64 %indvars.iv
  %0 = load i8, i8* %arrayidx, align 1
  %arrayidx2 = getelementptr inbounds [1000 x i8], [1000 x i8]* @c, i64 0, i64 %indvars.iv
  %1 = load i8, i8* %arrayidx2, align 1
  %add = add i8 %1, %0
  %arrayidx6 = getelementptr inbounds [1000 x i8], [1000 x i8]* @a, i64 0, i64 %indvars.iv
  store i8 %add, i8* %arrayidx6, align 1
  %arrayidx8 = getelementptr inbounds [1000 x i32], [1000 x i32]* @v, i64 0, i64 %indvars.iv
  %2 = load i32, i32* %arrayidx8, align 4
  %arrayidx10 = getelementptr inbounds [1000 x i32], [1000 x i32]* @w, i64 0, i64 %indvars.iv
  %3 = load i32, i32* %arrayidx10, align 4
  %add11 = add nsw i32 %3, %2
  %arrayidx13 = getelementptr inbounds [1000 x i32], [1000 x i32]* @u, i64 0, i64 %indvars.iv
  store i32 %add11, i32* %arrayidx13, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 1000
  br i1 %exitcond, label %for.cond.cleanup, label %for.body
}

; We should not choose a VF larger than the constant TC.
; VF chosen should be atmost 16 (not the max possible vector width = 32 for AVX2)
define void @not_too_small_tc(i8* noalias nocapture %A, i8* noalias nocapture readonly %B) {
; CHECK-LABEL: not_too_small_tc
; CHECK-AVX2: LV: Selecting VF: 16.
entry:
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds i8, i8* %B, i64 %indvars.iv
  %l1 = load i8, i8* %arrayidx, align 4, !llvm.mem.parallel_loop_access !3
  %arrayidx2 = getelementptr inbounds i8, i8* %A, i64 %indvars.iv
  %l2 = load i8, i8* %arrayidx2, align 4, !llvm.mem.parallel_loop_access !3
  %add = add i8 %l1, %l2
  store i8 %add, i8* %arrayidx2, align 4, !llvm.mem.parallel_loop_access !3
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 16
  br i1 %exitcond, label %for.end, label %for.body, !llvm.loop !4

for.end:
  ret void
}
!3 = !{!3}
!4 = !{!4}
