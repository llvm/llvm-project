; Do NOT use -O3. It will lower exp2 to ldexp, and the test will fail.
; RUN: opt -vector-library=sleefgnuabi -replace-with-veclib < %s | opt -vector-library=sleefgnuabi -passes=inject-tli-mappings,loop-unroll,loop-vectorize -S | FileCheck %s --check-prefixes=CHECK,NEON
; RUN: opt -mattr=+sve -vector-library=sleefgnuabi -replace-with-veclib < %s | opt -vector-library=sleefgnuabi -passes=inject-tli-mappings,loop-unroll,loop-vectorize -S | FileCheck %s --check-prefixes=CHECK,SVE

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64-unknown-linux-gnu"

declare double @acos(double) #0
declare float @acosf(float) #0
declare double @llvm.acos.f64(double) #0
declare float @llvm.acos.f32(float) #0

define void @acos_f64(double* nocapture %varray) {
  ; CHECK-LABEL: @acos_f64(
  ; NEON:     [[TMP5:%.*]] = call <2 x double> @_ZGVnN2v_acos(<2 x double> [[TMP4:%.*]])
  ; SVE:      [[TMP5:%.*]] = call <vscale x 2 x double> @_ZGVsMxv_acos(<vscale x 2 x double> [[TMP4:%.*]], <vscale x 2 x i1> {{.*}})
  ; CHECK:    ret void
  ;
  entry:
  br label %for.body

  for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %tmp = trunc i64 %iv to i32
  %conv = sitofp i32 %tmp to double
  %call = tail call double @acos(double %conv)
  %arrayidx = getelementptr inbounds double, double* %varray, i64 %iv
  store double %call, double* %arrayidx, align 8
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

  for.end:
  ret void
}

define void @acos_f32(float* nocapture %varray) {
  ; CHECK-LABEL: @acos_f32(
  ; NEON:    [[TMP5:%.*]] = call <4 x float> @_ZGVnN4v_acosf(<4 x float> [[TMP4:%.*]])
  ; SVE:     [[TMP5:%.*]] = call <vscale x 4 x float> @_ZGVsMxv_acosf(<vscale x 4 x float> [[TMP4:%.*]], <vscale x 4 x i1> {{.*}})
  ; CHECK:    ret void
  ;
  entry:
  br label %for.body

  for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %tmp = trunc i64 %iv to i32
  %conv = sitofp i32 %tmp to float
  %call = tail call float @acosf(float %conv)
  %arrayidx = getelementptr inbounds float, float* %varray, i64 %iv
  store float %call, float* %arrayidx, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

  for.end:
  ret void
}

declare double @asin(double) #0
declare float @asinf(float) #0
declare double @llvm.asin.f64(double) #0
declare float @llvm.asin.f32(float) #0

define void @asin_f64(double* nocapture %varray) {
  ; CHECK-LABEL: @asin_f64(
  ; NEON:     [[TMP5:%.*]] = call <2 x double> @_ZGVnN2v_asin(<2 x double> [[TMP4:%.*]])
  ; SVE:      [[TMP5:%.*]] = call <vscale x 2 x double> @_ZGVsMxv_asin(<vscale x 2 x double> [[TMP4:%.*]], <vscale x 2 x i1> {{.*}})
  ; CHECK:    ret void
  ;
  entry:
  br label %for.body

  for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %tmp = trunc i64 %iv to i32
  %conv = sitofp i32 %tmp to double
  %call = tail call double @asin(double %conv)
  %arrayidx = getelementptr inbounds double, double* %varray, i64 %iv
  store double %call, double* %arrayidx, align 8
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

  for.end:
  ret void
}

define void @asin_f32(float* nocapture %varray) {
  ; CHECK-LABEL: @asin_f32(
  ; NEON:     [[TMP5:%.*]] = call <4 x float> @_ZGVnN4v_asinf(<4 x float> [[TMP4:%.*]])
  ; SVE:      [[TMP5:%.*]] = call <vscale x 4 x float> @_ZGVsMxv_asinf(<vscale x 4 x float> [[TMP4:%.*]], <vscale x 4 x i1> {{.*}})
  ; CHECK:    ret void
  ;
  entry:
  br label %for.body

  for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %tmp = trunc i64 %iv to i32
  %conv = sitofp i32 %tmp to float
  %call = tail call float @asinf(float %conv)
  %arrayidx = getelementptr inbounds float, float* %varray, i64 %iv
  store float %call, float* %arrayidx, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

  for.end:
  ret void
}

declare double @atan(double) #0
declare float @atanf(float) #0
declare double @llvm.atan.f64(double) #0
declare float @llvm.atan.f32(float) #0

define void @atan_f64(double* nocapture %varray) {
  ; CHECK-LABEL: @atan_f64(
  ; NEON:     [[TMP5:%.*]] = call <2 x double> @_ZGVnN2v_atan(<2 x double> [[TMP4:%.*]])
  ; SVE:      [[TMP5:%.*]] = call <vscale x 2 x double> @_ZGVsMxv_atan(<vscale x 2 x double> [[TMP4:%.*]], <vscale x 2 x i1> {{.*}})
  ; CHECK:    ret void
  ;
  entry:
  br label %for.body

  for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %tmp = trunc i64 %iv to i32
  %conv = sitofp i32 %tmp to double
  %call = tail call double @atan(double %conv)
  %arrayidx = getelementptr inbounds double, double* %varray, i64 %iv
  store double %call, double* %arrayidx, align 8
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

  for.end:
  ret void
}

define void @atan_f32(float* nocapture %varray) {
  ; CHECK-LABEL: @atan_f32(
  ; NEON:     [[TMP5:%.*]] = call <4 x float> @_ZGVnN4v_atanf(<4 x float> [[TMP4:%.*]])
  ; SVE:      [[TMP5:%.*]] = call <vscale x 4 x float> @_ZGVsMxv_atanf(<vscale x 4 x float> [[TMP4:%.*]], <vscale x 4 x i1> {{.*}})
  ; CHECK:    ret void
  ;
  entry:
  br label %for.body

  for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %tmp = trunc i64 %iv to i32
  %conv = sitofp i32 %tmp to float
  %call = tail call float @atanf(float %conv)
  %arrayidx = getelementptr inbounds float, float* %varray, i64 %iv
  store float %call, float* %arrayidx, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

  for.end:
  ret void
}

declare double @atan2(double, double) #0
declare float @atan2f(float, float) #0
declare double @llvm.atan2.f64(double, double) #0
declare float @llvm.atan2.f32(float, float) #0

define void @atan2_f64(double* nocapture %varray) {
  ; CHECK-LABEL: @atan2_f64(
  ; NEON:     [[TMP5:%.*]] = call <2 x double> @_ZGVnN2vv_atan2(<2 x double> [[TMP4:%.*]], <2 x double> [[TMP4:%.*]])
  ; SVE:      [[TMP5:%.*]] = call <vscale x 2 x double> @_ZGVsMxvv_atan2(<vscale x 2 x double> [[TMP4:%.*]], <vscale x 2 x double> [[TMP4:%.*]], <vscale x 2 x i1> {{.*}})
  ; CHECK:    ret void
  ;
  entry:
  br label %for.body

  for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %tmp = trunc i64 %iv to i32
  %conv = sitofp i32 %tmp to double
  %call = tail call double @atan2(double %conv, double %conv)
  %arrayidx = getelementptr inbounds double, double* %varray, i64 %iv
  store double %call, double* %arrayidx, align 8
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

  for.end:
  ret void
}

define void @atan2_f32(float* nocapture %varray) {
  ; CHECK-LABEL: @atan2_f32(
  ; NEON:     [[TMP5:%.*]] = call <4 x float> @_ZGVnN4vv_atan2f(<4 x float> [[TMP4:%.*]], <4 x float> [[TMP4:%.*]])
  ; SVE:      [[TMP5:%.*]] = call <vscale x 4 x float> @_ZGVsMxvv_atan2f(<vscale x 4 x float> [[TMP4:%.*]], <vscale x 4 x float> [[TMP4:%.*]], <vscale x 4 x i1> {{.*}})
  ; CHECK:    ret void
  ;
  entry:
  br label %for.body

  for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %tmp = trunc i64 %iv to i32
  %conv = sitofp i32 %tmp to float
  %call = tail call float @atan2f(float %conv, float %conv)
  %arrayidx = getelementptr inbounds float, float* %varray, i64 %iv
  store float %call, float* %arrayidx, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

  for.end:
  ret void
}

declare double @atanh(double) #0
declare float @atanhf(float) #0
declare double @llvm.atanh.f64(double) #0
declare float @llvm.atanh.f32(float) #0

define void @atanh_f64(double* nocapture %varray) {
  ; CHECK-LABEL: @atanh_f64(
  ; NEON:     [[TMP5:%.*]] = call <2 x double> @_ZGVnN2v_atanh(<2 x double> [[TMP4:%.*]])
  ; SVE:      [[TMP5:%.*]] = call <vscale x 2 x double> @_ZGVsMxv_atanh(<vscale x 2 x double> [[TMP4:%.*]], <vscale x 2 x i1> {{.*}})
  ; CHECK:    ret void
  ;
  entry:
  br label %for.body

  for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %tmp = trunc i64 %iv to i32
  %conv = sitofp i32 %tmp to double
  %call = tail call double @atanh(double %conv)
  %arrayidx = getelementptr inbounds double, double* %varray, i64 %iv
  store double %call, double* %arrayidx, align 8
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

  for.end:
  ret void
}

define void @atanh_f32(float* nocapture %varray) {
  ; CHECK-LABEL: @atanh_f32(
  ; NEON:     [[TMP5:%.*]] = call <4 x float> @_ZGVnN4v_atanhf(<4 x float> [[TMP4:%.*]])
  ; SVE:      [[TMP5:%.*]] = call <vscale x 4 x float> @_ZGVsMxv_atanhf(<vscale x 4 x float> [[TMP4:%.*]], <vscale x 4 x i1> {{.*}})
  ; CHECK:    ret void
  ;
  entry:
  br label %for.body

  for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %tmp = trunc i64 %iv to i32
  %conv = sitofp i32 %tmp to float
  %call = tail call float @atanhf(float %conv)
  %arrayidx = getelementptr inbounds float, float* %varray, i64 %iv
  store float %call, float* %arrayidx, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

  for.end:
  ret void
}

declare double @cos(double) #0
declare float @cosf(float) #0
declare double @llvm.cos.f64(double) #0
declare float @llvm.cos.f32(float) #0

define void @cos_f64(double* nocapture %varray) {
  ; CHECK-LABEL: @cos_f64(
  ; NEON:     [[TMP5:%.*]] = call <2 x double> @_ZGVnN2v_cos(<2 x double> [[TMP4:%.*]])
  ; SVE:      [[TMP5:%.*]] = call <vscale x 2 x double> @_ZGVsMxv_cos(<vscale x 2 x double> [[TMP4:%.*]], <vscale x 2 x i1> {{.*}})
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
  store double %call, double* %arrayidx, align 8
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

  for.end:
  ret void
}

define void @cos_f32(float* nocapture %varray) {
  ; CHECK-LABEL: @cos_f32(
  ; NEON:     [[TMP5:%.*]] = call <4 x float> @_ZGVnN4v_cosf(<4 x float> [[TMP4:%.*]])
  ; SVE:      [[TMP5:%.*]] = call <vscale x 4 x float> @_ZGVsMxv_cosf(<vscale x 4 x float> [[TMP4:%.*]], <vscale x 4 x i1> {{.*}})
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

declare double @cosh(double) #0
declare float @coshf(float) #0
declare double @llvm.cosh.f64(double) #0
declare float @llvm.cosh.f32(float) #0

define void @cosh_f64(double* nocapture %varray) {
  ; CHECK-LABEL: @cosh_f64(
  ; NEON:     [[TMP5:%.*]] = call <2 x double> @_ZGVnN2v_cosh(<2 x double> [[TMP4:%.*]])
  ; SVE:      [[TMP5:%.*]] = call <vscale x 2 x double> @_ZGVsMxv_cosh(<vscale x 2 x double> [[TMP4:%.*]], <vscale x 2 x i1> {{.*}})
  ; CHECK:    ret void
  ;
  entry:
  br label %for.body

  for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %tmp = trunc i64 %iv to i32
  %conv = sitofp i32 %tmp to double
  %call = tail call double @cosh(double %conv)
  %arrayidx = getelementptr inbounds double, double* %varray, i64 %iv
  store double %call, double* %arrayidx, align 8
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

  for.end:
  ret void
}

define void @cosh_f32(float* nocapture %varray) {
  ; CHECK-LABEL: @cosh_f32(
  ; NEON:     [[TMP5:%.*]] = call <4 x float> @_ZGVnN4v_coshf(<4 x float> [[TMP4:%.*]])
  ; SVE:      [[TMP5:%.*]] = call <vscale x 4 x float> @_ZGVsMxv_coshf(<vscale x 4 x float> [[TMP4:%.*]], <vscale x 4 x i1> {{.*}})
  ; CHECK:    ret void
  ;
  entry:
  br label %for.body

  for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %tmp = trunc i64 %iv to i32
  %conv = sitofp i32 %tmp to float
  %call = tail call float @coshf(float %conv)
  %arrayidx = getelementptr inbounds float, float* %varray, i64 %iv
  store float %call, float* %arrayidx, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

  for.end:
  ret void
}

declare double @exp(double) #0
declare float @expf(float) #0
declare double @llvm.exp.f64(double) #0
declare float @llvm.exp.f32(float) #0

define void @exp_f64(double* nocapture %varray) {
  ; CHECK-LABEL: @exp_f64(
  ; NEON:     [[TMP5:%.*]] = call <2 x double> @_ZGVnN2v_exp(<2 x double> [[TMP4:%.*]])
  ; SVE:      [[TMP5:%.*]] = call <vscale x 2 x double> @_ZGVsMxv_exp(<vscale x 2 x double> [[TMP4:%.*]], <vscale x 2 x i1> {{.*}})
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
  store double %call, double* %arrayidx, align 8
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

  for.end:
  ret void
}

define void @exp_f32(float* nocapture %varray) {
  ; CHECK-LABEL: @exp_f32(
  ; NEON:     [[TMP5:%.*]] = call <4 x float> @_ZGVnN4v_expf(<4 x float> [[TMP4:%.*]])
  ; SVE:      [[TMP5:%.*]] = call <vscale x 4 x float> @_ZGVsMxv_expf(<vscale x 4 x float> [[TMP4:%.*]], <vscale x 4 x i1> {{.*}})
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

declare double @exp2(double) #0
declare float @exp2f(float) #0
declare double @llvm.exp2.f64(double) #0
declare float @llvm.exp2.f32(float) #0

define void @exp2_f64(double* nocapture %varray) {
  ; CHECK-LABEL: @exp2_f64(
  ; NEON:     [[TMP5:%.*]] = call <2 x double> @_ZGVnN2v_exp2(<2 x double> [[TMP4:%.*]])
  ; SVE:      [[TMP5:%.*]] = call <vscale x 2 x double> @_ZGVsMxv_exp2(<vscale x 2 x double> [[TMP4:%.*]], <vscale x 2 x i1> {{.*}})
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
  store double %call, double* %arrayidx, align 8
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

  for.end:
  ret void
}

define void @exp2_f32(float* nocapture %varray) {
  ; CHECK-LABEL: @exp2_f32(
  ; NEON:     [[TMP5:%.*]] = call <4 x float> @_ZGVnN4v_exp2f(<4 x float> [[TMP4:%.*]])
  ; SVE:      [[TMP5:%.*]] = call <vscale x 4 x float> @_ZGVsMxv_exp2f(<vscale x 4 x float> [[TMP4:%.*]], <vscale x 4 x i1> {{.*}})
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

declare double @exp10(double) #0
declare float @exp10f(float) #0
declare double @llvm.exp10.f64(double) #0
declare float @llvm.exp10.f32(float) #0

define void @exp10_f64(double* nocapture %varray) {
  ; CHECK-LABEL: @exp10_f64(
  ; NEON:     [[TMP5:%.*]] = call <2 x double> @_ZGVnN2v_exp10(<2 x double> [[TMP4:%.*]])
  ; SVE:      [[TMP5:%.*]] = call <vscale x 2 x double> @_ZGVsMxv_exp10(<vscale x 2 x double> [[TMP4:%.*]], <vscale x 2 x i1> {{.*}})
  ; CHECK:    ret void
  ;
  entry:
  br label %for.body

  for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %tmp = trunc i64 %iv to i32
  %conv = sitofp i32 %tmp to double
  %call = tail call double @exp10(double %conv)
  %arrayidx = getelementptr inbounds double, double* %varray, i64 %iv
  store double %call, double* %arrayidx, align 8
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

  for.end:
  ret void
}

define void @exp10_f32(float* nocapture %varray) {
  ; CHECK-LABEL: @exp10_f32(
  ; NEON:     [[TMP5:%.*]] = call <4 x float> @_ZGVnN4v_exp10f(<4 x float> [[TMP4:%.*]])
  ; SVE:      [[TMP5:%.*]] = call <vscale x 4 x float> @_ZGVsMxv_exp10f(<vscale x 4 x float> [[TMP4:%.*]], <vscale x 4 x i1> {{.*}})
  ; CHECK:    ret void
  ;
  entry:
  br label %for.body

  for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %tmp = trunc i64 %iv to i32
  %conv = sitofp i32 %tmp to float
  %call = tail call float @exp10f(float %conv)
  %arrayidx = getelementptr inbounds float, float* %varray, i64 %iv
  store float %call, float* %arrayidx, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

  for.end:
  ret void
}

declare double @lgamma(double) #0
declare float @lgammaf(float) #0
declare double @llvm.lgamma.f64(double) #0
declare float @llvm.lgamma.f32(float) #0

define void @lgamma_f64(double* nocapture %varray) {
  ; CHECK-LABEL: @lgamma_f64(
  ; NEON:     [[TMP5:%.*]] = call <2 x double> @_ZGVnN2v_lgamma(<2 x double> [[TMP4:%.*]])
  ; SVE:      [[TMP5:%.*]] = call <vscale x 2 x double> @_ZGVsMxv_lgamma(<vscale x 2 x double> [[TMP4:%.*]], <vscale x 2 x i1> {{.*}})
  ; CHECK:    ret void
  ;
  entry:
  br label %for.body

  for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %tmp = trunc i64 %iv to i32
  %conv = sitofp i32 %tmp to double
  %call = tail call double @lgamma(double %conv)
  %arrayidx = getelementptr inbounds double, double* %varray, i64 %iv
  store double %call, double* %arrayidx, align 8
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

  for.end:
  ret void
}

define void @lgamma_f32(float* nocapture %varray) {
  ; CHECK-LABEL: @lgamma_f32(
  ; NEON:     [[TMP5:%.*]] = call <4 x float> @_ZGVnN4v_lgammaf(<4 x float> [[TMP4:%.*]])
  ; SVE:      [[TMP5:%.*]] = call <vscale x 4 x float> @_ZGVsMxv_lgammaf(<vscale x 4 x float> [[TMP4:%.*]], <vscale x 4 x i1> {{.*}})
  ; CHECK:    ret void
  ;
  entry:
  br label %for.body

  for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %tmp = trunc i64 %iv to i32
  %conv = sitofp i32 %tmp to float
  %call = tail call float @lgammaf(float %conv)
  %arrayidx = getelementptr inbounds float, float* %varray, i64 %iv
  store float %call, float* %arrayidx, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

  for.end:
  ret void
}

declare double @log10(double) #0
declare float @log10f(float) #0
declare double @llvm.log10.f64(double) #0
declare float @llvm.log10.f32(float) #0

define void @log10_f64(double* nocapture %varray) {
  ; CHECK-LABEL: @log10_f64(
  ; NEON:     [[TMP5:%.*]] = call <2 x double> @_ZGVnN2v_log10(<2 x double> [[TMP4:%.*]])
  ; SVE:      [[TMP5:%.*]] = call <vscale x 2 x double> @_ZGVsMxv_log10(<vscale x 2 x double> [[TMP4:%.*]], <vscale x 2 x i1> {{.*}})
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
  store double %call, double* %arrayidx, align 8
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

  for.end:
  ret void
}

define void @log10_f32(float* nocapture %varray) {
  ; CHECK-LABEL: @log10_f32(
  ; NEON:     [[TMP5:%.*]] = call <4 x float> @_ZGVnN4v_log10f(<4 x float> [[TMP4:%.*]])
  ; SVE:      [[TMP5:%.*]] = call <vscale x 4 x float> @_ZGVsMxv_log10f(<vscale x 4 x float> [[TMP4:%.*]], <vscale x 4 x i1> {{.*}})
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

declare double @log2(double) #0
declare float @log2f(float) #0
declare double @llvm.log2.f64(double) #0
declare float @llvm.log2.f32(float) #0

define void @log2_f64(double* nocapture %varray) {
  ; CHECK-LABEL: @log2_f64(
  ; CHECK:    [[TMP5:%.*]] = call <2 x double> @_ZGVnN2v_log2(<2 x double> [[TMP4:%.*]])
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
  store double %call, double* %arrayidx, align 8
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

  for.end:
  ret void
}

define void @log2_f32(float* nocapture %varray) {
  ; CHECK-LABEL: @log2_f32(
  ; CHECK:    [[TMP5:%.*]] = call <4 x float> @_ZGVnN4v_log2f(<4 x float> [[TMP4:%.*]])
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

declare double @log(double) #0
declare float @logf(float) #0
declare double @llvm.log.f64(double) #0
declare float @llvm.log.f32(float) #0

define void @log_f64(double* nocapture %varray) {
  ; CHECK-LABEL: @log_f64(
  ; NEON:     [[TMP5:%.*]] = call <2 x double> @_ZGVnN2v_log(<2 x double> [[TMP4:%.*]])
  ; SVE:      [[TMP5:%.*]] = call <vscale x 2 x double> @_ZGVsMxv_log(<vscale x 2 x double> [[TMP4:%.*]], <vscale x 2 x i1> {{.*}})
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
  store double %call, double* %arrayidx, align 8
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

  for.end:
  ret void
}

define void @log_f32(float* nocapture %varray) {
  ; CHECK-LABEL: @log_f32(
  ; NEON:     [[TMP5:%.*]] = call <4 x float> @_ZGVnN4v_logf(<4 x float> [[TMP4:%.*]])
  ; SVE:      [[TMP5:%.*]] = call <vscale x 4 x float> @_ZGVsMxv_logf(<vscale x 4 x float> [[TMP4:%.*]], <vscale x 4 x i1> {{.*}})
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

declare double @pow(double, double) #0
declare float @powf(float, float) #0
declare double @llvm.pow.f64(double, double) #0
declare float @llvm.pow.f32(float, float) #0

define void @pow_f64(double* nocapture %varray) {
  ; CHECK-LABEL: @pow_f64(
  ; NEON:     [[TMP5:%.*]] = call <2 x double> @_ZGVnN2vv_pow(<2 x double> [[TMP4:%.*]], <2 x double> [[TMP4:%.*]])
  ; SVE:      [[TMP5:%.*]] = call <vscale x 2 x double> @_ZGVsMxvv_pow(<vscale x 2 x double> [[TMP4:%.*]], <vscale x 2 x double> [[TMP4:%.*]], <vscale x 2 x i1> {{.*}})
  ; CHECK:    ret void
  ;
  entry:
  br label %for.body

  for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %tmp = trunc i64 %iv to i32
  %conv = sitofp i32 %tmp to double
  %call = tail call double @pow(double %conv, double %conv)
  %arrayidx = getelementptr inbounds double, double* %varray, i64 %iv
  store double %call, double* %arrayidx, align 8
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

  for.end:
  ret void
}

define void @pow_f32(float* nocapture %varray) {
  ; CHECK-LABEL: @pow_f32(
  ; NEON:     [[TMP5:%.*]] = call <4 x float> @_ZGVnN4vv_powf(<4 x float> [[TMP4:%.*]], <4 x float> [[TMP4:%.*]])
  ; SVE:      [[TMP5:%.*]] = call <vscale x 4 x float> @_ZGVsMxvv_powf(<vscale x 4 x float> [[TMP4:%.*]], <vscale x 4 x float> [[TMP4:%.*]], <vscale x 4 x i1> {{.*}})
  ; CHECK:    ret void
  ;
  entry:
  br label %for.body

  for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %tmp = trunc i64 %iv to i32
  %conv = sitofp i32 %tmp to float
  %call = tail call float @powf(float %conv, float %conv)
  %arrayidx = getelementptr inbounds float, float* %varray, i64 %iv
  store float %call, float* %arrayidx, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

  for.end:
  ret void
}

declare double @sin(double) #0
declare float @sinf(float) #0
declare double @llvm.sin.f64(double) #0
declare float @llvm.sin.f32(float) #0

define void @sin_f64(double* nocapture %varray) {
  ; CHECK-LABEL: @sin_f64(
  ; NEON:     [[TMP5:%.*]] = call <2 x double> @_ZGVnN2v_sin(<2 x double> [[TMP4:%.*]])
  ; SVE:      [[TMP5:%.*]] = call <vscale x 2 x double> @_ZGVsMxv_sin(<vscale x 2 x double> [[TMP4:%.*]], <vscale x 2 x i1> {{.*}})
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
  store double %call, double* %arrayidx, align 8
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

  for.end:
  ret void
}

define void @sin_f32(float* nocapture %varray) {
  ; CHECK-LABEL: @sin_f32(
  ; NEON:     [[TMP5:%.*]] = call <4 x float> @_ZGVnN4v_sinf(<4 x float> [[TMP4:%.*]])
  ; SVE:      [[TMP5:%.*]] = call <vscale x 4 x float> @_ZGVsMxv_sinf(<vscale x 4 x float> [[TMP4:%.*]], <vscale x 4 x i1> {{.*}})
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

declare double @sinh(double) #0
declare float @sinhf(float) #0
declare double @llvm.sinh.f64(double) #0
declare float @llvm.sinh.f32(float) #0

define void @sinh_f64(double* nocapture %varray) {
  ; CHECK-LABEL: @sinh_f64(
  ; NEON:     [[TMP5:%.*]] = call <2 x double> @_ZGVnN2v_sinh(<2 x double> [[TMP4:%.*]])
  ; SVE:      [[TMP5:%.*]] = call <vscale x 2 x double> @_ZGVsMxv_sinh(<vscale x 2 x double> [[TMP4:%.*]], <vscale x 2 x i1> {{.*}})
  ; CHECK:    ret void
  ;
  entry:
  br label %for.body

  for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %tmp = trunc i64 %iv to i32
  %conv = sitofp i32 %tmp to double
  %call = tail call double @sinh(double %conv)
  %arrayidx = getelementptr inbounds double, double* %varray, i64 %iv
  store double %call, double* %arrayidx, align 8
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

  for.end:
  ret void
}

define void @sinh_f32(float* nocapture %varray) {
  ; CHECK-LABEL: @sinh_f32(
  ; NEON:     [[TMP5:%.*]] = call <4 x float> @_ZGVnN4v_sinhf(<4 x float> [[TMP4:%.*]])
  ; SVE:      [[TMP5:%.*]] = call <vscale x 4 x float> @_ZGVsMxv_sinhf(<vscale x 4 x float> [[TMP4:%.*]], <vscale x 4 x i1> {{.*}})
  ; CHECK:    ret void
  ;
  entry:
  br label %for.body

  for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %tmp = trunc i64 %iv to i32
  %conv = sitofp i32 %tmp to float
  %call = tail call float @sinhf(float %conv)
  %arrayidx = getelementptr inbounds float, float* %varray, i64 %iv
  store float %call, float* %arrayidx, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

  for.end:
  ret void
}

declare double @sqrt(double) #0
declare float @sqrtf(float) #0
declare double @llvm.sqrt.f64(double) #0
declare float @llvm.sqrt.f32(float) #0

define void @sqrt_f64(double* nocapture %varray) {
  ; CHECK-LABEL: @sqrt_f64(
  ; NEON:     [[TMP5:%.*]] = call <2 x double> @_ZGVnN2v_sqrt(<2 x double> [[TMP4:%.*]])
  ; SVE:      [[TMP5:%.*]] = call <vscale x 2 x double> @_ZGVsMxv_sqrt(<vscale x 2 x double> [[TMP4:%.*]], <vscale x 2 x i1> {{.*}})
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
  store double %call, double* %arrayidx, align 8
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

  for.end:
  ret void
}

define void @sqrt_f32(float* nocapture %varray) {
  ; CHECK-LABEL: @sqrt_f32(
  ; NEON:     [[TMP5:%.*]] = call <4 x float> @_ZGVnN4v_sqrtf(<4 x float> [[TMP4:%.*]])
  ; SVE:      [[TMP5:%.*]] = call <vscale x 4 x float> @_ZGVsMxv_sqrtf(<vscale x 4 x float> [[TMP4:%.*]], <vscale x 4 x i1> {{.*}})
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

define void @llvm_sqrt_f64(double* nocapture %varray) {
  ; CHECK-LABEL: @llvm_sqrt_f64(
  ; NEON:     [[TMP5:%.*]] = call fast <2 x double> @llvm.sqrt.v2f64(<2 x double> [[TMP4:%.*]])
  ; SVE:      [[TMP5:%.*]] = call fast <vscale x 2 x double> @llvm.sqrt.nxv2f64(<vscale x 2 x double> [[TMP4:%.*]])
  ; CHECK:    ret void
  ;
  entry:
  br label %for.body

  for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %tmp = trunc i64 %iv to i32
  %conv = sitofp i32 %tmp to double
  %call = tail call fast double @llvm.sqrt.f64(double %conv)
  %arrayidx = getelementptr inbounds double, double* %varray, i64 %iv
  store double %call, double* %arrayidx, align 8
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

  for.end:
  ret void
}

define void @llvm_sqrt_f32(float* nocapture %varray) {
  ; CHECK-LABEL: @llvm_sqrt_f32(
  ; NEON:     [[TMP5:%.*]] = call fast <4 x float> @llvm.sqrt.v4f32(<4 x float> [[TMP4:%.*]])
  ; SVE:      [[TMP5:%.*]] = call fast <vscale x 4 x float> @llvm.sqrt.nxv4f32(<vscale x 4 x float> [[TMP4:%.*]])
  ; CHECK:    ret void
  ;
  entry:
  br label %for.body

  for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %tmp = trunc i64 %iv to i32
  %conv = sitofp i32 %tmp to float
  %call = tail call fast float @llvm.sqrt.f32(float %conv)
  %arrayidx = getelementptr inbounds float, float* %varray, i64 %iv
  store float %call, float* %arrayidx, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

  for.end:
  ret void
}

declare double @tan(double) #0
declare float @tanf(float) #0
declare double @llvm.tan.f64(double) #0
declare float @llvm.tan.f32(float) #0

define void @tan_f64(double* nocapture %varray) {
  ; CHECK-LABEL: @tan_f64(
  ; NEON:     [[TMP5:%.*]] = call <2 x double> @_ZGVnN2v_tan(<2 x double> [[TMP4:%.*]])
  ; SVE:      [[TMP5:%.*]] = call <vscale x 2 x double> @_ZGVsMxv_tan(<vscale x 2 x double> [[TMP4:%.*]], <vscale x 2 x i1> {{.*}})
  ; CHECK:    ret void
  ;
  entry:
  br label %for.body

  for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %tmp = trunc i64 %iv to i32
  %conv = sitofp i32 %tmp to double
  %call = tail call double @tan(double %conv)
  %arrayidx = getelementptr inbounds double, double* %varray, i64 %iv
  store double %call, double* %arrayidx, align 8
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

  for.end:
  ret void
}

define void @tan_f32(float* nocapture %varray) {
  ; CHECK-LABEL: @tan_f32(
  ; NEON:     [[TMP5:%.*]] = call <4 x float> @_ZGVnN4v_tanf(<4 x float> [[TMP4:%.*]])
  ; SVE:      [[TMP5:%.*]] = call <vscale x 4 x float> @_ZGVsMxv_tanf(<vscale x 4 x float> [[TMP4:%.*]], <vscale x 4 x i1> {{.*}})
  ; CHECK:    ret void
  ;
  entry:
  br label %for.body

  for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %tmp = trunc i64 %iv to i32
  %conv = sitofp i32 %tmp to float
  %call = tail call float @tanf(float %conv)
  %arrayidx = getelementptr inbounds float, float* %varray, i64 %iv
  store float %call, float* %arrayidx, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

  for.end:
  ret void
}

declare double @tanh(double) #0
declare float @tanhf(float) #0
declare double @llvm.tanh.f64(double) #0
declare float @llvm.tanh.f32(float) #0

define void @tanh_f64(double* nocapture %varray) {
  ; CHECK-LABEL: @tanh_f64(
  ; NEON:     [[TMP5:%.*]] = call <2 x double> @_ZGVnN2v_tanh(<2 x double> [[TMP4:%.*]])
  ; SVE:      [[TMP5:%.*]] = call <vscale x 2 x double> @_ZGVsMxv_tanh(<vscale x 2 x double> [[TMP4:%.*]], <vscale x 2 x i1> {{.*}})
  ; CHECK:    ret void
  ;
  entry:
  br label %for.body

  for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %tmp = trunc i64 %iv to i32
  %conv = sitofp i32 %tmp to double
  %call = tail call double @tanh(double %conv)
  %arrayidx = getelementptr inbounds double, double* %varray, i64 %iv
  store double %call, double* %arrayidx, align 8
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

  for.end:
  ret void
}

define void @tanh_f32(float* nocapture %varray) {
  ; CHECK-LABEL: @tanh_f32(
  ; NEON:     [[TMP5:%.*]] = call <4 x float> @_ZGVnN4v_tanhf(<4 x float> [[TMP4:%.*]])
  ; SVE:      [[TMP5:%.*]] = call <vscale x 4 x float> @_ZGVsMxv_tanhf(<vscale x 4 x float> [[TMP4:%.*]], <vscale x 4 x i1> {{.*}})
  ; CHECK:    ret void
  ;
  entry:
  br label %for.body

  for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %tmp = trunc i64 %iv to i32
  %conv = sitofp i32 %tmp to float
  %call = tail call float @tanhf(float %conv)
  %arrayidx = getelementptr inbounds float, float* %varray, i64 %iv
  store float %call, float* %arrayidx, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

  for.end:
  ret void
}

declare double @tgamma(double) #0
declare float @tgammaf(float) #0
declare double @llvm.tgamma.f64(double) #0
declare float @llvm.tgamma.f32(float) #0

define void @tgamma_f64(double* nocapture %varray) {
  ; CHECK-LABEL: @tgamma_f64(
  ; NEON:     [[TMP5:%.*]] = call <2 x double> @_ZGVnN2v_tgamma(<2 x double> [[TMP4:%.*]])
  ; SVE:      [[TMP5:%.*]] = call <vscale x 2 x double> @_ZGVsMxv_tgamma(<vscale x 2 x double> [[TMP4:%.*]], <vscale x 2 x i1> {{.*}})
  ; CHECK:    ret void
  ;
  entry:
  br label %for.body

  for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %tmp = trunc i64 %iv to i32
  %conv = sitofp i32 %tmp to double
  %call = tail call double @tgamma(double %conv)
  %arrayidx = getelementptr inbounds double, double* %varray, i64 %iv
  store double %call, double* %arrayidx, align 8
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

  for.end:
  ret void
}

define void @tgamma_f32(float* nocapture %varray) {
  ; CHECK-LABEL: @tgamma_f32(
  ; NEON:     [[TMP5:%.*]] = call <4 x float> @_ZGVnN4v_tgammaf(<4 x float> [[TMP4:%.*]])
  ; SVE:      [[TMP5:%.*]] = call <vscale x 4 x float> @_ZGVsMxv_tgammaf(<vscale x 4 x float> [[TMP4:%.*]], <vscale x 4 x i1> {{.*}})
  ; CHECK:    ret void
  ;
  entry:
  br label %for.body

  for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %tmp = trunc i64 %iv to i32
  %conv = sitofp i32 %tmp to float
  %call = tail call float @tgammaf(float %conv)
  %arrayidx = getelementptr inbounds float, float* %varray, i64 %iv
  store float %call, float* %arrayidx, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

  for.end:
  ret void
}
