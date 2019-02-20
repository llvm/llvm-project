; RUN: opt -loops -analyze < %s | FileCheck %s
;
; void func(long n, double A[static const restrict n]) {
;   for (long i = 0; i < n; i += 1)
;     A[i] = 21;
; }
;
; Check that isAnnotatedParallel is working as expected.
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @func(i64 %n, double* noalias nonnull %A) {
entry:
  br label %for.cond

for.cond:
  %i.0 = phi i64 [ 0, %entry ], [ %add, %for.body ]
  %cmp = icmp slt i64 %i.0, %n
  br i1 %cmp, label %for.body, label %for.end

for.body:
  %arrayidx = getelementptr inbounds double, double* %A, i64 %i.0
  store double 2.100000e+01, double* %arrayidx, align 8, !llvm.access.group !6
  %add = add nuw nsw i64 %i.0, 1
  br label %for.cond, !llvm.loop !7

for.end:
  ret void
}

!6 = distinct !{} ; access group

!7 = distinct !{!7, !9} ; LoopID
!9 = !{!"llvm.loop.parallel_accesses", !6}


; CHECK: Parallel Loop
