; RUN: opt -mcpu=a2 -passes=loop-data-prefetch -mtriple=powerpc64le-unknown-linux -enable-ppc-prefetching -S < %s | FileCheck %s
target datalayout = "E-m:e-i64:64-n32:64"

define void @foo(double* nocapture %a, double* nocapture readonly %b) {
entry:
  br label %for.body

; CHECK: for.body:
for.body:                                         ; preds = %for.body, %entry
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds double, double* %b, i64 %indvars.iv
; CHECK: call void @llvm.prefetch
  %0 = load double, double* %arrayidx, align 8
  %add = fadd double %0, 1.000000e+00
  %arrayidx2 = getelementptr inbounds double, double* %a, i64 %indvars.iv
  store double %add, double* %arrayidx2, align 8
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 1600
  br i1 %exitcond, label %for.end, label %for.body

; CHECK: for.end:
for.end:                                          ; preds = %for.body
  ret void
}
