; RUN: opt -S -loop-vectorize < %s | FileCheck %s
target datalayout = "E-m:e-i64:64-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

; Function Attrs: nounwind
define void @foo(double* noalias nocapture %a, double* noalias nocapture readonly %b) #0 {
entry:
  br label %for.body

; CHECK-LABEL: @foo
; CHECK: <2 x double>

for.cond.cleanup:                                 ; preds = %for.body
  ret void

for.body:                                         ; preds = %for.body, %entry
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %0 = shl nsw i64 %indvars.iv, 1
  %odd.idx = add nsw i64 %0, 1

  %arrayidx = getelementptr inbounds double, double* %b, i64 %0
  %arrayidx.odd = getelementptr inbounds double, double* %b, i64 %odd.idx

  %1 = load double, double* %arrayidx, align 8
  %2 = load double, double* %arrayidx.odd, align 8

  %add = fadd double %1, %2
  %arrayidx2 = getelementptr inbounds double, double* %a, i64 %indvars.iv
  store double %add, double* %arrayidx2, align 8
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 1600
  br i1 %exitcond, label %for.cond.cleanup, label %for.body
}

attributes #0 = { nounwind "target-cpu"="pwr8" }

