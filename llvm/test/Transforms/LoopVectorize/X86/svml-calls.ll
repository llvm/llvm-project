; RUN: opt -vector-library=SVML -passes=inject-tli-mappings,loop-vectorize -force-vector-width=4 -force-vector-interleave=1 -mattr=avx -S < %s | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare double @sin(double) #0
declare float @sinf(float) #0
declare double @llvm.sin.f64(double) #0
declare float @llvm.sin.f32(float) #0

declare double @cos(double) #0
declare float @cosf(float) #0
declare double @llvm.cos.f64(double) #0
declare float @llvm.cos.f32(float) #0

declare double @pow(double, double) #0
declare float @powf(float, float) #0
declare double @llvm.pow.f64(double, double) #0
declare float @llvm.pow.f32(float, float) #0

declare double @exp(double) #0
declare float @expf(float) #0
declare double @llvm.exp.f64(double) #0
declare float @llvm.exp.f32(float) #0

declare double @log(double) #0
declare float @logf(float) #0
declare double @llvm.log.f64(double) #0
declare float @llvm.log.f32(float) #0

declare double @log2(double) #0
declare float @log2f(float) #0
declare double @llvm.log2.f64(double) #0
declare float @llvm.log2.f32(float) #0

declare double @log10(double) #0
declare float @log10f(float) #0
declare double @llvm.log10.f64(double) #0
declare float @llvm.log10.f32(float) #0

declare double @sqrt(double) #0
declare float @sqrtf(float) #0

declare double @exp2(double) #0
declare float @exp2f(float) #0
declare double @llvm.exp2.f64(double) #0
declare float @llvm.exp2.f32(float) #0

define void @sin_f64(double* nocapture %varray) {
; CHECK-LABEL: @sin_f64(
; CHECK:    [[TMP5:%.*]] = call <4 x double> @__svml_sin4(<4 x double> [[TMP4:%.*]])
; CHECK:    ret void
;
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %tmp = trunc i64 %iv to i32
  %conv = sitofp i32 %tmp to double
  %call = tail call double @sin(double %conv)
  %arrayidx = getelementptr inbounds double, double* %varray, i64 %iv
  store double %call, double* %arrayidx, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

define void @sin_f32(float* nocapture %varray) {
; CHECK-LABEL: @sin_f32(
; CHECK:    [[TMP5:%.*]] = call <4 x float> @__svml_sinf4(<4 x float> [[TMP4:%.*]])
; CHECK:    ret void
;
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %tmp = trunc i64 %iv to i32
  %conv = sitofp i32 %tmp to float
  %call = tail call float @sinf(float %conv)
  %arrayidx = getelementptr inbounds float, float* %varray, i64 %iv
  store float %call, float* %arrayidx, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

define void @sin_f64_intrinsic(double* nocapture %varray) {
; CHECK-LABEL: @sin_f64_intrinsic(
; CHECK:    [[TMP5:%.*]] = call <4 x double> @__svml_sin4(<4 x double> [[TMP4:%.*]])
; CHECK:    ret void
;
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %tmp = trunc i64 %iv to i32
  %conv = sitofp i32 %tmp to double
  %call = tail call double @llvm.sin.f64(double %conv)
  %arrayidx = getelementptr inbounds double, double* %varray, i64 %iv
  store double %call, double* %arrayidx, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

define void @sin_f32_intrinsic(float* nocapture %varray) {
; CHECK-LABEL: @sin_f32_intrinsic(
; CHECK:    [[TMP5:%.*]] = call <4 x float> @__svml_sinf4(<4 x float> [[TMP4:%.*]])
; CHECK:    ret void
;
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %tmp = trunc i64 %iv to i32
  %conv = sitofp i32 %tmp to float
  %call = tail call float @llvm.sin.f32(float %conv)
  %arrayidx = getelementptr inbounds float, float* %varray, i64 %iv
  store float %call, float* %arrayidx, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

define void @cos_f64(double* nocapture %varray) {
; CHECK-LABEL: @cos_f64(
; CHECK:    [[TMP5:%.*]] = call <4 x double> @__svml_cos4(<4 x double> [[TMP4:%.*]])
; CHECK:    ret void
;
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %tmp = trunc i64 %iv to i32
  %conv = sitofp i32 %tmp to double
  %call = tail call double @cos(double %conv)
  %arrayidx = getelementptr inbounds double, double* %varray, i64 %iv
  store double %call, double* %arrayidx, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

define void @cos_f32(float* nocapture %varray) {
; CHECK-LABEL: @cos_f32(
; CHECK:    [[TMP5:%.*]] = call <4 x float> @__svml_cosf4(<4 x float> [[TMP4:%.*]])
; CHECK:    ret void
;
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %tmp = trunc i64 %iv to i32
  %conv = sitofp i32 %tmp to float
  %call = tail call float @cosf(float %conv)
  %arrayidx = getelementptr inbounds float, float* %varray, i64 %iv
  store float %call, float* %arrayidx, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

define void @cos_f64_intrinsic(double* nocapture %varray) {
; CHECK-LABEL: @cos_f64_intrinsic(
; CHECK:    [[TMP5:%.*]] = call <4 x double> @__svml_cos4(<4 x double> [[TMP4:%.*]])
; CHECK:    ret void
;
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %tmp = trunc i64 %iv to i32
  %conv = sitofp i32 %tmp to double
  %call = tail call double @llvm.cos.f64(double %conv)
  %arrayidx = getelementptr inbounds double, double* %varray, i64 %iv
  store double %call, double* %arrayidx, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

define void @cos_f32_intrinsic(float* nocapture %varray) {
; CHECK-LABEL: @cos_f32_intrinsic(
; CHECK:    [[TMP5:%.*]] = call <4 x float> @__svml_cosf4(<4 x float> [[TMP4:%.*]])
; CHECK:    ret void
;
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %tmp = trunc i64 %iv to i32
  %conv = sitofp i32 %tmp to float
  %call = tail call float @llvm.cos.f32(float %conv)
  %arrayidx = getelementptr inbounds float, float* %varray, i64 %iv
  store float %call, float* %arrayidx, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

define void @pow_f64(double* nocapture %varray, double* nocapture readonly %exp) {
; CHECK-LABEL: @pow_f64(
; CHECK:    [[TMP8:%.*]] = call <4 x double> @__svml_pow4(<4 x double> [[TMP4:%.*]], <4 x double> [[WIDE_LOAD:%.*]])
; CHECK:    ret void
;
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %tmp = trunc i64 %iv to i32
  %conv = sitofp i32 %tmp to double
  %arrayidx = getelementptr inbounds double, double* %exp, i64 %iv
  %tmp1 = load double, double* %arrayidx, align 4
  %tmp2 = tail call double @pow(double %conv, double %tmp1)
  %arrayidx2 = getelementptr inbounds double, double* %varray, i64 %iv
  store double %tmp2, double* %arrayidx2, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

define void @pow_f64_intrinsic(double* nocapture %varray, double* nocapture readonly %exp) {
; CHECK-LABEL: @pow_f64_intrinsic(
; CHECK:    [[TMP8:%.*]] = call <4 x double> @__svml_pow4(<4 x double> [[TMP4:%.*]], <4 x double> [[WIDE_LOAD:%.*]])
; CHECK:    ret void
;
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %tmp = trunc i64 %iv to i32
  %conv = sitofp i32 %tmp to double
  %arrayidx = getelementptr inbounds double, double* %exp, i64 %iv
  %tmp1 = load double, double* %arrayidx, align 4
  %tmp2 = tail call double @llvm.pow.f64(double %conv, double %tmp1)
  %arrayidx2 = getelementptr inbounds double, double* %varray, i64 %iv
  store double %tmp2, double* %arrayidx2, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

define void @pow_f32(float* nocapture %varray, float* nocapture readonly %exp) {
; CHECK-LABEL: @pow_f32(
; CHECK:    [[TMP8:%.*]] = call <4 x float> @__svml_powf4(<4 x float> [[TMP4:%.*]], <4 x float> [[WIDE_LOAD:%.*]])
; CHECK:    ret void
;
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %tmp = trunc i64 %iv to i32
  %conv = sitofp i32 %tmp to float
  %arrayidx = getelementptr inbounds float, float* %exp, i64 %iv
  %tmp1 = load float, float* %arrayidx, align 4
  %tmp2 = tail call float @powf(float %conv, float %tmp1)
  %arrayidx2 = getelementptr inbounds float, float* %varray, i64 %iv
  store float %tmp2, float* %arrayidx2, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

define void @pow_f32_intrinsic(float* nocapture %varray, float* nocapture readonly %exp) {
; CHECK-LABEL: @pow_f32_intrinsic(
; CHECK:    [[TMP8:%.*]] = call <4 x float> @__svml_powf4(<4 x float> [[TMP4:%.*]], <4 x float> [[WIDE_LOAD:%.*]])
; CHECK:    ret void
;
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %tmp = trunc i64 %iv to i32
  %conv = sitofp i32 %tmp to float
  %arrayidx = getelementptr inbounds float, float* %exp, i64 %iv
  %tmp1 = load float, float* %arrayidx, align 4
  %tmp2 = tail call float @llvm.pow.f32(float %conv, float %tmp1)
  %arrayidx2 = getelementptr inbounds float, float* %varray, i64 %iv
  store float %tmp2, float* %arrayidx2, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

define void @exp_f64(double* nocapture %varray) {
; CHECK-LABEL: @exp_f64(
; CHECK:    [[TMP5:%.*]] = call <4 x double> @__svml_exp4(<4 x double> [[TMP4:%.*]])
; CHECK:    ret void
;
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %tmp = trunc i64 %iv to i32
  %conv = sitofp i32 %tmp to double
  %call = tail call double @exp(double %conv)
  %arrayidx = getelementptr inbounds double, double* %varray, i64 %iv
  store double %call, double* %arrayidx, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

define void @exp_f32(float* nocapture %varray) {
; CHECK-LABEL: @exp_f32(
; CHECK:    [[TMP5:%.*]] = call <4 x float> @__svml_expf4(<4 x float> [[TMP4:%.*]])
; CHECK:    ret void
;
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %tmp = trunc i64 %iv to i32
  %conv = sitofp i32 %tmp to float
  %call = tail call float @expf(float %conv)
  %arrayidx = getelementptr inbounds float, float* %varray, i64 %iv
  store float %call, float* %arrayidx, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

define void @exp_f64_intrinsic(double* nocapture %varray) {
; CHECK-LABEL: @exp_f64_intrinsic(
; CHECK:    [[TMP5:%.*]] = call <4 x double> @__svml_exp4(<4 x double> [[TMP4:%.*]])
; CHECK:    ret void
;
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %tmp = trunc i64 %iv to i32
  %conv = sitofp i32 %tmp to double
  %call = tail call double @llvm.exp.f64(double %conv)
  %arrayidx = getelementptr inbounds double, double* %varray, i64 %iv
  store double %call, double* %arrayidx, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

define void @exp_f32_intrinsic(float* nocapture %varray) {
; CHECK-LABEL: @exp_f32_intrinsic(
; CHECK:    [[TMP5:%.*]] = call <4 x float> @__svml_expf4(<4 x float> [[TMP4:%.*]])
; CHECK:    ret void
;
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %tmp = trunc i64 %iv to i32
  %conv = sitofp i32 %tmp to float
  %call = tail call float @llvm.exp.f32(float %conv)
  %arrayidx = getelementptr inbounds float, float* %varray, i64 %iv
  store float %call, float* %arrayidx, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

define void @log_f64(double* nocapture %varray) {
; CHECK-LABEL: @log_f64(
; CHECK:    [[TMP5:%.*]] = call <4 x double> @__svml_log4(<4 x double> [[TMP4:%.*]])
; CHECK:    ret void
;
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %tmp = trunc i64 %iv to i32
  %conv = sitofp i32 %tmp to double
  %call = tail call double @log(double %conv)
  %arrayidx = getelementptr inbounds double, double* %varray, i64 %iv
  store double %call, double* %arrayidx, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

define void @log_f32(float* nocapture %varray) {
; CHECK-LABEL: @log_f32(
; CHECK:    [[TMP5:%.*]] = call <4 x float> @__svml_logf4(<4 x float> [[TMP4:%.*]])
; CHECK:    ret void
;
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %tmp = trunc i64 %iv to i32
  %conv = sitofp i32 %tmp to float
  %call = tail call float @logf(float %conv)
  %arrayidx = getelementptr inbounds float, float* %varray, i64 %iv
  store float %call, float* %arrayidx, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

define void @log_f64_intrinsic(double* nocapture %varray) {
; CHECK-LABEL: @log_f64_intrinsic(
; CHECK:    [[TMP5:%.*]] = call <4 x double> @__svml_log4(<4 x double> [[TMP4:%.*]])
; CHECK:    ret void
;
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %tmp = trunc i64 %iv to i32
  %conv = sitofp i32 %tmp to double
  %call = tail call double @llvm.log.f64(double %conv)
  %arrayidx = getelementptr inbounds double, double* %varray, i64 %iv
  store double %call, double* %arrayidx, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

define void @log_f32_intrinsic(float* nocapture %varray) {
; CHECK-LABEL: @log_f32_intrinsic(
; CHECK:    [[TMP5:%.*]] = call <4 x float> @__svml_logf4(<4 x float> [[TMP4:%.*]])
; CHECK:    ret void
;
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %tmp = trunc i64 %iv to i32
  %conv = sitofp i32 %tmp to float
  %call = tail call float @llvm.log.f32(float %conv)
  %arrayidx = getelementptr inbounds float, float* %varray, i64 %iv
  store float %call, float* %arrayidx, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

define void @log2_f64(double* nocapture %varray) {
; CHECK-LABEL: @log2_f64(
; CHECK:    [[TMP5:%.*]] = call <4 x double> @__svml_log24(<4 x double> [[TMP4:%.*]])
; CHECK:    ret void
;
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %tmp = trunc i64 %iv to i32
  %conv = sitofp i32 %tmp to double
  %call = tail call double @log2(double %conv)
  %arrayidx = getelementptr inbounds double, double* %varray, i64 %iv
  store double %call, double* %arrayidx, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

define void @log2_f32(float* nocapture %varray) {
; CHECK-LABEL: @log2_f32(
; CHECK:    [[TMP5:%.*]] = call <4 x float> @__svml_log2f4(<4 x float> [[TMP4:%.*]])
; CHECK:    ret void
;
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %tmp = trunc i64 %iv to i32
  %conv = sitofp i32 %tmp to float
  %call = tail call float @log2f(float %conv)
  %arrayidx = getelementptr inbounds float, float* %varray, i64 %iv
  store float %call, float* %arrayidx, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

define void @log2_f64_intrinsic(double* nocapture %varray) {
; CHECK-LABEL: @log2_f64_intrinsic(
; CHECK:    [[TMP5:%.*]] = call <4 x double> @__svml_log24(<4 x double> [[TMP4:%.*]])
; CHECK:    ret void
;
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %tmp = trunc i64 %iv to i32
  %conv = sitofp i32 %tmp to double
  %call = tail call double @llvm.log2.f64(double %conv)
  %arrayidx = getelementptr inbounds double, double* %varray, i64 %iv
  store double %call, double* %arrayidx, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

define void @log2_f32_intrinsic(float* nocapture %varray) {
; CHECK-LABEL: @log2_f32_intrinsic(
; CHECK:    [[TMP5:%.*]] = call <4 x float> @__svml_log2f4(<4 x float> [[TMP4:%.*]])
; CHECK:    ret void
;
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %tmp = trunc i64 %iv to i32
  %conv = sitofp i32 %tmp to float
  %call = tail call float @llvm.log2.f32(float %conv)
  %arrayidx = getelementptr inbounds float, float* %varray, i64 %iv
  store float %call, float* %arrayidx, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

define void @log10_f64(double* nocapture %varray) {
; CHECK-LABEL: @log10_f64(
; CHECK:    [[TMP5:%.*]] = call <4 x double> @__svml_log104(<4 x double> [[TMP4:%.*]])
; CHECK:    ret void
;
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %tmp = trunc i64 %iv to i32
  %conv = sitofp i32 %tmp to double
  %call = tail call double @log10(double %conv)
  %arrayidx = getelementptr inbounds double, double* %varray, i64 %iv
  store double %call, double* %arrayidx, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

define void @log10_f32(float* nocapture %varray) {
; CHECK-LABEL: @log10_f32(
; CHECK:    [[TMP5:%.*]] = call <4 x float> @__svml_log10f4(<4 x float> [[TMP4:%.*]])
; CHECK:    ret void
;
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %tmp = trunc i64 %iv to i32
  %conv = sitofp i32 %tmp to float
  %call = tail call float @log10f(float %conv)
  %arrayidx = getelementptr inbounds float, float* %varray, i64 %iv
  store float %call, float* %arrayidx, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

define void @log10_f64_intrinsic(double* nocapture %varray) {
; CHECK-LABEL: @log10_f64_intrinsic(
; CHECK:    [[TMP5:%.*]] = call <4 x double> @__svml_log104(<4 x double> [[TMP4:%.*]])
; CHECK:    ret void
;
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %tmp = trunc i64 %iv to i32
  %conv = sitofp i32 %tmp to double
  %call = tail call double @llvm.log10.f64(double %conv)
  %arrayidx = getelementptr inbounds double, double* %varray, i64 %iv
  store double %call, double* %arrayidx, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

define void @log10_f32_intrinsic(float* nocapture %varray) {
; CHECK-LABEL: @log10_f32_intrinsic(
; CHECK:    [[TMP5:%.*]] = call <4 x float> @__svml_log10f4(<4 x float> [[TMP4:%.*]])
; CHECK:    ret void
;
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %tmp = trunc i64 %iv to i32
  %conv = sitofp i32 %tmp to float
  %call = tail call float @llvm.log10.f32(float %conv)
  %arrayidx = getelementptr inbounds float, float* %varray, i64 %iv
  store float %call, float* %arrayidx, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

define void @sqrt_f64(double* nocapture %varray) {
; CHECK-LABEL: @sqrt_f64(
; CHECK:    [[TMP5:%.*]] = call <4 x double> @__svml_sqrt4(<4 x double> [[TMP4:%.*]])
; CHECK:    ret void
;
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %tmp = trunc i64 %iv to i32
  %conv = sitofp i32 %tmp to double
  %call = tail call double @sqrt(double %conv)
  %arrayidx = getelementptr inbounds double, double* %varray, i64 %iv
  store double %call, double* %arrayidx, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

define void @sqrt_f32(float* nocapture %varray) {
; CHECK-LABEL: @sqrt_f32(
; CHECK:    [[TMP5:%.*]] = call <4 x float> @__svml_sqrtf4(<4 x float> [[TMP4:%.*]])
; CHECK:    ret void
;
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %tmp = trunc i64 %iv to i32
  %conv = sitofp i32 %tmp to float
  %call = tail call float @sqrtf(float %conv)
  %arrayidx = getelementptr inbounds float, float* %varray, i64 %iv
  store float %call, float* %arrayidx, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

define void @exp2_f64(double* nocapture %varray) {
; CHECK-LABEL: @exp2_f64(
; CHECK:    [[TMP5:%.*]] = call <4 x double> @__svml_exp24(<4 x double> [[TMP4:%.*]])
; CHECK:    ret void
;
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %tmp = trunc i64 %iv to i32
  %conv = sitofp i32 %tmp to double
  %call = tail call double @exp2(double %conv)
  %arrayidx = getelementptr inbounds double, double* %varray, i64 %iv
  store double %call, double* %arrayidx, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

define void @exp2_f32(float* nocapture %varray) {
; CHECK-LABEL: @exp2_f32(
; CHECK:    [[TMP5:%.*]] = call <4 x float> @__svml_exp2f4(<4 x float> [[TMP4:%.*]])
; CHECK:    ret void
;
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %tmp = trunc i64 %iv to i32
  %conv = sitofp i32 %tmp to float
  %call = tail call float @exp2f(float %conv)
  %arrayidx = getelementptr inbounds float, float* %varray, i64 %iv
  store float %call, float* %arrayidx, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

define void @exp2_f64_intrinsic(double* nocapture %varray) {
; CHECK-LABEL: @exp2_f64_intrinsic(
; CHECK:    [[TMP5:%.*]] = call <4 x double> @__svml_exp24(<4 x double> [[TMP4:%.*]])
; CHECK:    ret void
;
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %tmp = trunc i64 %iv to i32
  %conv = sitofp i32 %tmp to double
  %call = tail call double @llvm.exp2.f64(double %conv)
  %arrayidx = getelementptr inbounds double, double* %varray, i64 %iv
  store double %call, double* %arrayidx, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

define void @exp2_f32_intrinsic(float* nocapture %varray) {
; CHECK-LABEL: @exp2_f32_intrinsic(
; CHECK:    [[TMP5:%.*]] = call <4 x float> @__svml_exp2f4(<4 x float> [[TMP4:%.*]])
; CHECK:    ret void
;
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %tmp = trunc i64 %iv to i32
  %conv = sitofp i32 %tmp to float
  %call = tail call float @llvm.exp2.f32(float %conv)
  %arrayidx = getelementptr inbounds float, float* %varray, i64 %iv
  store float %call, float* %arrayidx, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

attributes #0 = { nounwind readnone }
