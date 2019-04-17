; RUN: opt -S -basicaa -loop-vectorize < %s | FileCheck %s
target datalayout = "E-m:e-i64:64-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

; Function Attrs: nounwind
define void @foo(double* noalias nocapture %a, double* noalias nocapture readonly %b, double* noalias nocapture readonly %c) #0 {
entry:
  br label %for.body

; CHECK-LABEL: @foo
; CHECK: fmul <4 x double> %{{[^,]+}}, <double 2.000000e+00, double 2.000000e+00, double 2.000000e+00, double 2.000000e+00>
; CHECK-NEXT: fmul <4 x double> %{{[^,]+}}, <double 2.000000e+00, double 2.000000e+00, double 2.000000e+00, double 2.000000e+00>

for.cond.cleanup:                                 ; preds = %for.body
  ret void

for.body:                                         ; preds = %for.body, %entry
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds double, double* %b, i64 %indvars.iv
  %0 = load double, double* %arrayidx, align 8
  %mul = fmul double %0, 2.000000e+00
  %mul3 = fmul double %0, %mul
  %arrayidx5 = getelementptr inbounds double, double* %c, i64 %indvars.iv
  %1 = load double, double* %arrayidx5, align 8
  %mul6 = fmul double %1, 3.000000e+00
  %mul9 = fmul double %1, %mul6
  %add = fadd double %mul3, %mul9
  %mul12 = fmul double %0, 4.000000e+00
  %mul15 = fmul double %mul12, %1
  %add16 = fadd double %mul15, %add
  %add17 = fadd double %add16, 1.000000e+00
  %arrayidx19 = getelementptr inbounds double, double* %a, i64 %indvars.iv
  store double %add17, double* %arrayidx19, align 8
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 1600
  br i1 %exitcond, label %for.cond.cleanup, label %for.body
}

attributes #0 = { nounwind "target-cpu"="a2q" }

