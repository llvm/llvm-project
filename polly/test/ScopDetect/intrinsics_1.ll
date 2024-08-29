; RUN: opt %loadNPMPolly -aa-pipeline=basic-aa '-passes=print<polly-detect>' -disable-output < %s 2>&1 | FileCheck %s
;
; CHECK: Valid Region for Scop: for.cond => for.end
;
;    #include "math.h"
;
;    void jd(int *restrict A, float *restrict B) {
;      for (int i = 0; i < 1024; i++) {
;        A[i] = pow(ceil(log10(sqrt(i))), floor(log2(i)));
;        B[i] = fabs(log(sin(i)) + exp2(cos(i))) + exp(i);
;      }
;    }
;
; ModuleID = '/home/johannes/repos/polly/test/ScopDetect/intrinsics.ll'
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

; Function Attrs: nounwind uwtable
define void @jd(ptr noalias %A, ptr noalias %B) #0 {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.inc ], [ 0, %entry ]
  %exitcond = icmp ne i64 %indvars.iv, 1024
  br i1 %exitcond, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %tmp = trunc i64 %indvars.iv to i32
  %conv = sitofp i32 %tmp to double
  %tmp1 = call double @llvm.sqrt.f64(double %conv)
  %call = call double @__log10_finite(double %tmp1) #2
  %call1 = call double @ceil(double %call) #2
  %tmp2 = trunc i64 %indvars.iv to i32
  %conv2 = sitofp i32 %tmp2 to double
  %call3 = call double @__log2_finite(double %conv2) #2
  %call4 = call double @floor(double %call3) #2
  %tmp3 = call double @llvm.pow.f64(double %call1, double %call4)
  %conv5 = fptosi double %tmp3 to i32
  %arrayidx = getelementptr inbounds i32, ptr %A, i64 %indvars.iv
  store i32 %conv5, ptr %arrayidx, align 4
  %tmp4 = trunc i64 %indvars.iv to i32
  %conv6 = sitofp i32 %tmp4 to double
  %call7 = call double @sin(double %conv6) #2
  %call8 = call double @__log_finite(double %call7) #2
  %tmp5 = trunc i64 %indvars.iv to i32
  %conv9 = sitofp i32 %tmp5 to double
  %call10 = call double @cos(double %conv9) #2
  %call11 = call double @__exp2_finite(double %call10) #2
  %add = fadd fast double %call8, %call11
  %call12 = call double @fabs(double %add) #2
  %tmp6 = trunc i64 %indvars.iv to i32
  %conv13 = sitofp i32 %tmp6 to double
  %call14 = call double @__exp_finite(double %conv13) #2
  %add15 = fadd fast double %call12, %call14
  %conv16 = fptrunc double %add15 to float
  %arrayidx18 = getelementptr inbounds float, ptr %B, i64 %indvars.iv
  store float %conv16, ptr %arrayidx18, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret void
}

; Function Attrs: nounwind readnone
declare double @ceil(double) #1

; Function Attrs: nounwind readnone
declare double @__log10_finite(double) #1

; Function Attrs: nounwind readnone
declare double @llvm.sqrt.f64(double) #2

; Function Attrs: nounwind readnone
declare double @floor(double) #1

; Function Attrs: nounwind readnone
declare double @__log2_finite(double) #1

; Function Attrs: nounwind readnone
declare double @llvm.pow.f64(double, double) #2

; Function Attrs: nounwind readnone
declare double @fabs(double) #1

; Function Attrs: nounwind readnone
declare double @__log_finite(double) #1

; Function Attrs: nounwind readnone
declare double @sin(double) #1

; Function Attrs: nounwind readnone
declare double @__exp2_finite(double) #1

; Function Attrs: nounwind readnone
declare double @cos(double) #1

; Function Attrs: nounwind readnone
declare double @__exp_finite(double) #1

attributes #0 = { nounwind uwtable "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #1 = { nounwind readnone "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #2 = { nounwind readnone }
