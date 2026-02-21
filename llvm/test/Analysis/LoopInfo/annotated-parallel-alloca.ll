; RUN: opt -passes='print<loops>' -disable-output %s 2>&1 | FileCheck %s
;
; void func(long n, long *A) {
;   #pragma clang loop vectorize(assume_safety)
;   for (long i = 0; i < n; i += 1) {
;     long t[32];
;     for (long j = 0; j < 32; j += 1)
;       t[j] = i;
;     A[i] = t[i];
;   }
; }
;
; The alloca for `t` usually gets hoisted outside of the loop (either by Clang
; itself, or by an inlining pass if the loop body is in a function, etc.) and
; gets incorrectly shared between iterations. Check that isAnnotatedParallel is
; blocking this kind of usage, as it will not get vectorized correctly unless
; mem2reg converts the array.
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @func(i64 %n, ptr noalias nonnull %A) {
entry:
  %t = alloca [32 x i64], align 16
  %cmp17 = icmp sgt i64 %n, 0
  br i1 %cmp17, label %for.body, label %for.cond.cleanup

for.body:
  %i.018 = phi i64 [ %add8, %for.cond.cleanup3 ], [ 0, %entry ]
  br label %for.body4

for.body4:
  %j.016 = phi i64 [ 0, %for.body ], [ %add, %for.body4 ]
  %arrayidx = getelementptr inbounds nuw i64, ptr %t, i64 %j.016
  store i64 %i.018, ptr %arrayidx, align 8, !llvm.access.group !9
  %add = add nuw nsw i64 %j.016, 1
  %exitcond.not = icmp eq i64 %add, 32
  br i1 %exitcond.not, label %for.cond.cleanup3, label %for.body4

for.cond.cleanup3:
  %arrayidx5 = getelementptr inbounds nuw i64, ptr %t, i64 %i.018
  %0 = load i64, ptr %arrayidx5, align 8, !llvm.access.group !9
  %arrayidx6 = getelementptr inbounds nuw i64, ptr %A, i64 %i.018
  store i64 %0, ptr %arrayidx6, align 8, !llvm.access.group !9
  %add8 = add nuw nsw i64 %i.018, 1
  %exitcond19.not = icmp eq i64 %add8, %n
  br i1 %exitcond19.not, label %for.cond.cleanup, label %for.body, !llvm.loop !10

for.cond.cleanup:
  ret void
}

!9 = distinct !{}
!10 = distinct !{!10, !11}
!11 = !{!"llvm.loop.parallel_accesses", !9}

; CHECK: Loop info for function 'func':
; CHECK-NOT: Parallel Loop at depth 1 containing:
