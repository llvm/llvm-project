; RUN: opt -vector-library=AMDLIBM -passes=inject-tli-mappings,loop-vectorize -force-vector-width=4 -force-vector-interleave=1 -mattr=avx -S < %s | FileCheck %s
; RUN: opt -vector-library=AMDLIBM -passes=inject-tli-mappings,loop-vectorize -force-vector-width=8 -force-vector-interleave=1 -mattr=+avx512f -S < %s | FileCheck %s --check-prefix=CHECK-AVX512-VF8
; RUN: opt -vector-library=AMDLIBM -passes=inject-tli-mappings,loop-vectorize -force-vector-width=16 -force-vector-interleave=1 -mattr=+avx512f -S < %s | FileCheck %s --check-prefix=CHECK-AVX512-VF16

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

define void @sin_f64(ptr nocapture %varray) {
; CHECK-LABEL: @sin_f64(
; CHECK:    [[TMP5:%.*]] = call <4 x double> @amd_vrd4_sin(<4 x double> [[TMP4:%.*]])
; CHECK:    ret void
;
; CHECK-AVX512-VF8-LABEL: @sin_f64(
; CHECK-AVX512-VF8:    [[TMP5:%.*]] = call <8 x double> @amd_vrd8_sin(<8 x double> [[TMP4:%.*]])
; CHECK-AVX512-VF8:    ret void
;
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %tmp = trunc i64 %iv to i32
  %conv = sitofp i32 %tmp to double
  %call = tail call double @sin(double %conv)
  %arrayidx = getelementptr inbounds double, ptr %varray, i64 %iv
  store double %call, ptr %arrayidx, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

define void @sin_f32(ptr nocapture %varray) {
; CHECK-LABEL: @sin_f32(
; CHECK:    [[TMP5:%.*]] = call <4 x float> @amd_vrs4_sinf(<4 x float> [[TMP4:%.*]])
; CHECK:    ret void
;
; CHECK-AVX512-VF16-LABEL: @sin_f32(
; CHECK-AVX512-VF16:    [[TMP5:%.*]] = call <16 x float> @amd_vrs16_sinf(<16 x float> [[TMP4:%.*]])
; CHECK-AVX512-VF16:    ret void
;
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %tmp = trunc i64 %iv to i32
  %conv = sitofp i32 %tmp to float
  %call = tail call float @sinf(float %conv)
  %arrayidx = getelementptr inbounds float, ptr %varray, i64 %iv
  store float %call, ptr %arrayidx, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

define void @sin_f64_intrinsic(ptr nocapture %varray) {
; CHECK-LABEL: @sin_f64_intrinsic(
; CHECK:    [[TMP5:%.*]] = call <4 x double> @amd_vrd4_sin(<4 x double> [[TMP4:%.*]])
; CHECK:    ret void
;
; CHECK-AVX512-VF8-LABEL: @sin_f64_intrinsic(
; CHECK-AVX512-VF8:    [[TMP5:%.*]] = call <8 x double> @amd_vrd8_sin(<8 x double> [[TMP4:%.*]])
; CHECK-AVX512-VF8:    ret void
;
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %tmp = trunc i64 %iv to i32
  %conv = sitofp i32 %tmp to double
  %call = tail call double @llvm.sin.f64(double %conv)
  %arrayidx = getelementptr inbounds double, ptr %varray, i64 %iv
  store double %call, ptr %arrayidx, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

define void @sin_f32_intrinsic(ptr nocapture %varray) {
; CHECK-LABEL: @sin_f32_intrinsic(
; CHECK:    [[TMP5:%.*]] = call <4 x float> @amd_vrs4_sinf(<4 x float> [[TMP4:%.*]])
; CHECK:    ret void
;
; CHECK-AVX512-VF16-LABEL: @sin_f32_intrinsic(
; CHECK-AVX512-VF16:    [[TMP5:%.*]] = call <16 x float> @amd_vrs16_sinf(<16 x float> [[TMP4:%.*]])
; CHECK-AVX512-VF16:    ret void
;
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %tmp = trunc i64 %iv to i32
  %conv = sitofp i32 %tmp to float
  %call = tail call float @llvm.sin.f32(float %conv)
  %arrayidx = getelementptr inbounds float, ptr %varray, i64 %iv
  store float %call, ptr %arrayidx, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

define void @cos_f64(ptr nocapture %varray) {
; CHECK-LABEL: @cos_f64(
; CHECK:    [[TMP5:%.*]] = call <4 x double> @amd_vrd4_cos(<4 x double> [[TMP4:%.*]])
; CHECK:    ret void
;
; CHECK-AVX512-VF8-LABEL: @cos_f64(
; CHECK-AVX512-VF8:    [[TMP5:%.*]] = call <8 x double> @amd_vrd8_cos(<8 x double> [[TMP4:%.*]])
; CHECK-AVX512-VF8:    ret void
;
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %tmp = trunc i64 %iv to i32
  %conv = sitofp i32 %tmp to double
  %call = tail call double @cos(double %conv)
  %arrayidx = getelementptr inbounds double, ptr %varray, i64 %iv
  store double %call, ptr %arrayidx, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

define void @cos_f32(ptr nocapture %varray) {
; CHECK-LABEL: @cos_f32(
; CHECK:    [[TMP5:%.*]] = call <4 x float> @amd_vrs4_cosf(<4 x float> [[TMP4:%.*]])
; CHECK:    ret void
;
; CHECK-AVX512-VF16-LABEL: @cos_f32(
; CHECK-AVX512-VF16:    [[TMP5:%.*]] = call <16 x float> @amd_vrs16_cosf(<16 x float> [[TMP4:%.*]])
; CHECK-AVX512-VF16:    ret void
;
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %tmp = trunc i64 %iv to i32
  %conv = sitofp i32 %tmp to float
  %call = tail call float @cosf(float %conv)
  %arrayidx = getelementptr inbounds float, ptr %varray, i64 %iv
  store float %call, ptr %arrayidx, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

define void @cos_f64_intrinsic(ptr nocapture %varray) {
; CHECK-LABEL: @cos_f64_intrinsic(
; CHECK:    [[TMP5:%.*]] = call <4 x double> @amd_vrd4_cos(<4 x double> [[TMP4:%.*]])
; CHECK:    ret void
;
; CHECK-AVX512-VF8-LABEL: @cos_f64_intrinsic(
; CHECK-AVX512-VF8:    [[TMP5:%.*]] = call <8 x double> @amd_vrd8_cos(<8 x double> [[TMP4:%.*]])
; CHECK-AVX512-VF8:    ret void
;
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %tmp = trunc i64 %iv to i32
  %conv = sitofp i32 %tmp to double
  %call = tail call double @llvm.cos.f64(double %conv)
  %arrayidx = getelementptr inbounds double, ptr %varray, i64 %iv
  store double %call, ptr %arrayidx, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

define void @cos_f32_intrinsic(ptr nocapture %varray) {
; CHECK-LABEL: @cos_f32_intrinsic(
; CHECK:    [[TMP5:%.*]] = call <4 x float> @amd_vrs4_cosf(<4 x float> [[TMP4:%.*]])
; CHECK:    ret void
;
; CHECK-AVX512-VF16-LABEL: @cos_f32_intrinsic(
; CHECK-AVX512-VF16:    [[TMP5:%.*]] = call <16 x float> @amd_vrs16_cosf(<16 x float> [[TMP4:%.*]])
; CHECK-AVX512-VF16:    ret void
;
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %tmp = trunc i64 %iv to i32
  %conv = sitofp i32 %tmp to float
  %call = tail call float @llvm.cos.f32(float %conv)
  %arrayidx = getelementptr inbounds float, ptr %varray, i64 %iv
  store float %call, ptr %arrayidx, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

define void @pow_f64(ptr nocapture %varray, ptr nocapture readonly %exp) {
; CHECK-LABEL: @pow_f64(
; CHECK:    [[TMP8:%.*]] = call <4 x double> @amd_vrd4_pow(<4 x double> [[TMP4:%.*]], <4 x double> [[WIDE_LOAD:%.*]])
; CHECK:    ret void
;
; CHECK-AVX512-VF8-LABEL: @pow_f64(
; CHECK-AVX512-VF8:    [[TMP8:%.*]] = call <8 x double> @amd_vrd8_pow(<8 x double> [[TMP4:%.*]], <8 x double> [[WIDE_LOAD:%.*]])
; CHECK-AVX512-VF8:    ret void
;
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %tmp = trunc i64 %iv to i32
  %conv = sitofp i32 %tmp to double
  %arrayidx = getelementptr inbounds double, ptr %exp, i64 %iv
  %tmp1 = load double, ptr %arrayidx, align 4
  %tmp2 = tail call double @pow(double %conv, double %tmp1)
  %arrayidx2 = getelementptr inbounds double, ptr %varray, i64 %iv
  store double %tmp2, ptr %arrayidx2, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

define void @pow_f64_intrinsic(ptr nocapture %varray, ptr nocapture readonly %exp) {
; CHECK-LABEL: @pow_f64_intrinsic(
; CHECK:    [[TMP8:%.*]] = call <4 x double> @amd_vrd4_pow(<4 x double> [[TMP4:%.*]], <4 x double> [[WIDE_LOAD:%.*]])
; CHECK:    ret void
;
; CHECK-AVX512-VF8-LABEL: @pow_f64_intrinsic(
; CHECK-AVX512-VF8:    [[TMP8:%.*]] = call <8 x double> @amd_vrd8_pow(<8 x double> [[TMP4:%.*]], <8 x double> [[WIDE_LOAD:%.*]])
; CHECK-AVX512-VF8:    ret void
;
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %tmp = trunc i64 %iv to i32
  %conv = sitofp i32 %tmp to double
  %arrayidx = getelementptr inbounds double, ptr %exp, i64 %iv
  %tmp1 = load double, ptr %arrayidx, align 4
  %tmp2 = tail call double @llvm.pow.f64(double %conv, double %tmp1)
  %arrayidx2 = getelementptr inbounds double, ptr %varray, i64 %iv
  store double %tmp2, ptr %arrayidx2, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

define void @pow_f32(ptr nocapture %varray, ptr nocapture readonly %exp) {
; CHECK-LABEL: @pow_f32(
; CHECK:    [[TMP8:%.*]] = call <4 x float> @amd_vrs4_powf(<4 x float> [[TMP4:%.*]], <4 x float> [[WIDE_LOAD:%.*]])
; CHECK:    ret void
;
; CHECK-AVX512-VF16-LABEL: @pow_f32(
; CHECK-AVX512-VF16:    [[TMP8:%.*]] = call <16 x float> @amd_vrs16_powf(<16 x float> [[TMP4:%.*]], <16 x float> [[WIDE_LOAD:%.*]])
; CHECK-AVX512-VF16:    ret void
;
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %tmp = trunc i64 %iv to i32
  %conv = sitofp i32 %tmp to float
  %arrayidx = getelementptr inbounds float, ptr %exp, i64 %iv
  %tmp1 = load float, ptr %arrayidx, align 4
  %tmp2 = tail call float @powf(float %conv, float %tmp1)
  %arrayidx2 = getelementptr inbounds float, ptr %varray, i64 %iv
  store float %tmp2, ptr %arrayidx2, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

define void @pow_f32_intrinsic(ptr nocapture %varray, ptr nocapture readonly %exp) {
; CHECK-LABEL: @pow_f32_intrinsic(
; CHECK:    [[TMP8:%.*]] = call <4 x float> @amd_vrs4_powf(<4 x float> [[TMP4:%.*]], <4 x float> [[WIDE_LOAD:%.*]])
; CHECK:    ret void
;
; CHECK-AVX512-VF16-LABEL: @pow_f32_intrinsic(
; CHECK-AVX512-VF16:    [[TMP8:%.*]] = call <16 x float> @amd_vrs16_powf(<16 x float> [[TMP4:%.*]], <16 x float> [[WIDE_LOAD:%.*]])
; CHECK-AVX512-VF16:    ret void
;
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %tmp = trunc i64 %iv to i32
  %conv = sitofp i32 %tmp to float
  %arrayidx = getelementptr inbounds float, ptr %exp, i64 %iv
  %tmp1 = load float, ptr %arrayidx, align 4
  %tmp2 = tail call float @llvm.pow.f32(float %conv, float %tmp1)
  %arrayidx2 = getelementptr inbounds float, ptr %varray, i64 %iv
  store float %tmp2, ptr %arrayidx2, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

define void @exp_f64(ptr nocapture %varray) {
; CHECK-LABEL: @exp_f64(
; CHECK:    [[TMP5:%.*]] = call <4 x double> @amd_vrd4_exp(<4 x double> [[TMP4:%.*]])
; CHECK:    ret void
;
; CHECK-AVX512-VF8-LABEL: @exp_f64(
; CHECK-AVX512-VF8:    [[TMP5:%.*]] = call <8 x double> @amd_vrd8_exp(<8 x double> [[TMP4:%.*]])
; CHECK-AVX512-VF8:    ret void
;
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %tmp = trunc i64 %iv to i32
  %conv = sitofp i32 %tmp to double
  %call = tail call double @exp(double %conv)
  %arrayidx = getelementptr inbounds double, ptr %varray, i64 %iv
  store double %call, ptr %arrayidx, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

define void @exp_f32(ptr nocapture %varray) {
; CHECK-LABEL: @exp_f32(
; CHECK:    [[TMP5:%.*]] = call <4 x float> @amd_vrs4_expf(<4 x float> [[TMP4:%.*]])
; CHECK:    ret void
;
; CHECK-AVX512-VF16-LABEL: @exp_f32(
; CHECK-AVX512-VF16:    [[TMP5:%.*]] = call <16 x float> @amd_vrs16_expf(<16 x float> [[TMP4:%.*]])
; CHECK-AVX512-VF16:    ret void
;
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %tmp = trunc i64 %iv to i32
  %conv = sitofp i32 %tmp to float
  %call = tail call float @expf(float %conv)
  %arrayidx = getelementptr inbounds float, ptr %varray, i64 %iv
  store float %call, ptr %arrayidx, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

define void @exp_f64_intrinsic(ptr nocapture %varray) {
; CHECK-LABEL: @exp_f64_intrinsic(
; CHECK:    [[TMP5:%.*]] = call <4 x double> @amd_vrd4_exp(<4 x double> [[TMP4:%.*]])
; CHECK:    ret void
;
; CHECK-AVX512-VF8-LABEL: @exp_f64_intrinsic(
; CHECK-AVX512-VF8:    [[TMP5:%.*]] = call <8 x double> @amd_vrd8_exp(<8 x double> [[TMP4:%.*]])
; CHECK-AVX512-VF8:    ret void
;
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %tmp = trunc i64 %iv to i32
  %conv = sitofp i32 %tmp to double
  %call = tail call double @llvm.exp.f64(double %conv)
  %arrayidx = getelementptr inbounds double, ptr %varray, i64 %iv
  store double %call, ptr %arrayidx, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

define void @exp_f32_intrinsic(ptr nocapture %varray) {
; CHECK-LABEL: @exp_f32_intrinsic(
; CHECK:    [[TMP5:%.*]] = call <4 x float> @amd_vrs4_expf(<4 x float> [[TMP4:%.*]])
; CHECK:    ret void
;
; CHECK-AVX512-VF16-LABEL: @exp_f32_intrinsic(
; CHECK-AVX512-VF16:    [[TMP5:%.*]] = call <16 x float> @amd_vrs16_expf(<16 x float> [[TMP4:%.*]])
; CHECK-AVX512-VF16:    ret void
;
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %tmp = trunc i64 %iv to i32
  %conv = sitofp i32 %tmp to float
  %call = tail call float @llvm.exp.f32(float %conv)
  %arrayidx = getelementptr inbounds float, ptr %varray, i64 %iv
  store float %call, ptr %arrayidx, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

define void @log_f64(ptr nocapture %varray) {
; CHECK-LABEL: @log_f64(
; CHECK:    [[TMP5:%.*]] = call <4 x double> @amd_vrd4_log(<4 x double> [[TMP4:%.*]])
; CHECK:    ret void
;
; CHECK-AVX512-VF8-LABEL: @log_f64(
; CHECK-AVX512-VF8:    [[TMP5:%.*]] = call <8 x double> @amd_vrd8_log(<8 x double> [[TMP4:%.*]])
; CHECK-AVX512-VF8:    ret void
;
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %tmp = trunc i64 %iv to i32
  %conv = sitofp i32 %tmp to double
  %call = tail call double @log(double %conv)
  %arrayidx = getelementptr inbounds double, ptr %varray, i64 %iv
  store double %call, ptr %arrayidx, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

define void @log_f32(ptr nocapture %varray) {
; CHECK-LABEL: @log_f32(
; CHECK:    [[TMP5:%.*]] = call <4 x float> @amd_vrs4_logf(<4 x float> [[TMP4:%.*]])
; CHECK:    ret void
;
; CHECK-AVX512-VF16-LABEL: @log_f32(
; CHECK-AVX512-VF16:    [[TMP5:%.*]] = call <16 x float> @amd_vrs16_logf(<16 x float> [[TMP4:%.*]])
; CHECK-AVX512-VF16:    ret void
;
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %tmp = trunc i64 %iv to i32
  %conv = sitofp i32 %tmp to float
  %call = tail call float @logf(float %conv)
  %arrayidx = getelementptr inbounds float, ptr %varray, i64 %iv
  store float %call, ptr %arrayidx, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

define void @log_f64_intrinsic(ptr nocapture %varray) {
; CHECK-LABEL: @log_f64_intrinsic(
; CHECK:    [[TMP5:%.*]] = call <4 x double> @amd_vrd4_log(<4 x double> [[TMP4:%.*]])
; CHECK:    ret void
;
; CHECK-AVX512-VF8-LABEL: @log_f64_intrinsic(
; CHECK-AVX512-VF8:    [[TMP5:%.*]] = call <8 x double> @amd_vrd8_log(<8 x double> [[TMP4:%.*]])
; CHECK-AVX512-VF8:    ret void
;
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %tmp = trunc i64 %iv to i32
  %conv = sitofp i32 %tmp to double
  %call = tail call double @llvm.log.f64(double %conv)
  %arrayidx = getelementptr inbounds double, ptr %varray, i64 %iv
  store double %call, ptr %arrayidx, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

define void @log_f32_intrinsic(ptr nocapture %varray) {
; CHECK-LABEL: @log_f32_intrinsic(
; CHECK:    [[TMP5:%.*]] = call <4 x float> @amd_vrs4_logf(<4 x float> [[TMP4:%.*]])
; CHECK:    ret void
;
; CHECK-AVX512-VF16-LABEL: @log_f32_intrinsic(
; CHECK-AVX512-VF16:    [[TMP5:%.*]] = call <16 x float> @amd_vrs16_logf(<16 x float> [[TMP4:%.*]])
; CHECK-AVX512-VF16:    ret void
;
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %tmp = trunc i64 %iv to i32
  %conv = sitofp i32 %tmp to float
  %call = tail call float @llvm.log.f32(float %conv)
  %arrayidx = getelementptr inbounds float, ptr %varray, i64 %iv
  store float %call, ptr %arrayidx, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

define void @log2_f64(ptr nocapture %varray) {
; CHECK-LABEL: @log2_f64(
; CHECK:    [[TMP5:%.*]] = call <4 x double> @amd_vrd4_log2(<4 x double> [[TMP4:%.*]])
; CHECK:    ret void
;
; CHECK-AVX512-VF8-LABEL: @log2_f64(
; CHECK-AVX512-VF8:    [[TMP5:%.*]] = call <8 x double> @amd_vrd8_log2(<8 x double> [[TMP4:%.*]])
; CHECK-AVX512-VF8:    ret void
;
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %tmp = trunc i64 %iv to i32
  %conv = sitofp i32 %tmp to double
  %call = tail call double @log2(double %conv)
  %arrayidx = getelementptr inbounds double, ptr %varray, i64 %iv
  store double %call, ptr %arrayidx, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

define void @log2_f32(ptr nocapture %varray) {
; CHECK-LABEL: @log2_f32(
; CHECK:    [[TMP5:%.*]] = call <4 x float> @amd_vrs4_log2f(<4 x float> [[TMP4:%.*]])
; CHECK:    ret void
;
; CHECK-AVX512-VF16-LABEL: @log2_f32(
; CHECK-AVX512-VF16:    [[TMP5:%.*]] = call <16 x float> @amd_vrs16_log2f(<16 x float> [[TMP4:%.*]])
; CHECK-AVX512-VF16:    ret void
;
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %tmp = trunc i64 %iv to i32
  %conv = sitofp i32 %tmp to float
  %call = tail call float @log2f(float %conv)
  %arrayidx = getelementptr inbounds float, ptr %varray, i64 %iv
  store float %call, ptr %arrayidx, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

define void @log2_f64_intrinsic(ptr nocapture %varray) {
; CHECK-LABEL: @log2_f64_intrinsic(
; CHECK:    [[TMP5:%.*]] = call <4 x double> @amd_vrd4_log2(<4 x double> [[TMP4:%.*]])
; CHECK:    ret void
;
; CHECK-AVX512-VF8-LABEL: @log2_f64_intrinsic(
; CHECK-AVX512-VF8:    [[TMP5:%.*]] = call <8 x double> @amd_vrd8_log2(<8 x double> [[TMP4:%.*]])
; CHECK-AVX512-VF8:    ret void
;
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %tmp = trunc i64 %iv to i32
  %conv = sitofp i32 %tmp to double
  %call = tail call double @llvm.log2.f64(double %conv)
  %arrayidx = getelementptr inbounds double, ptr %varray, i64 %iv
  store double %call, ptr %arrayidx, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

define void @log2_f32_intrinsic(ptr nocapture %varray) {
; CHECK-LABEL: @log2_f32_intrinsic(
; CHECK:    [[TMP5:%.*]] = call <4 x float> @amd_vrs4_log2f(<4 x float> [[TMP4:%.*]])
; CHECK:    ret void
;
; CHECK-AVX512-VF16-LABEL: @log2_f32_intrinsic(
; CHECK-AVX512-VF16:    [[TMP5:%.*]] = call <16 x float> @amd_vrs16_log2f(<16 x float> [[TMP4:%.*]])
; CHECK-AVX512-VF16:    ret void
;
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %tmp = trunc i64 %iv to i32
  %conv = sitofp i32 %tmp to float
  %call = tail call float @llvm.log2.f32(float %conv)
  %arrayidx = getelementptr inbounds float, ptr %varray, i64 %iv
  store float %call, ptr %arrayidx, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

define void @log10_f32(ptr nocapture %varray) {
; CHECK-LABEL: @log10_f32(
; CHECK:    [[TMP5:%.*]] = call <4 x float> @amd_vrs4_log10f(<4 x float> [[TMP4:%.*]])
; CHECK:    ret void
;
; CHECK-AVX512-VF16-LABEL: @log10_f32(
; CHECK-AVX512-VF16:    [[TMP5:%.*]] = call <16 x float> @amd_vrs16_log10f(<16 x float> [[TMP4:%.*]])
; CHECK-AVX512-VF16:    ret void
;
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %tmp = trunc i64 %iv to i32
  %conv = sitofp i32 %tmp to float
  %call = tail call float @log10f(float %conv)
  %arrayidx = getelementptr inbounds float, ptr %varray, i64 %iv
  store float %call, ptr %arrayidx, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

define void @log10_f32_intrinsic(ptr nocapture %varray) {
; CHECK-LABEL: @log10_f32_intrinsic(
; CHECK:    [[TMP5:%.*]] = call <4 x float> @amd_vrs4_log10f(<4 x float> [[TMP4:%.*]])
; CHECK:    ret void
;
; CHECK-AVX512-VF16-LABEL: @log10_f32_intrinsic(
; CHECK-AVX512-VF16:    [[TMP5:%.*]] = call <16 x float> @amd_vrs16_log10f(<16 x float> [[TMP4:%.*]])
; CHECK-AVX512-VF16:    ret void
;
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %tmp = trunc i64 %iv to i32
  %conv = sitofp i32 %tmp to float
  %call = tail call float @llvm.log10.f32(float %conv)
  %arrayidx = getelementptr inbounds float, ptr %varray, i64 %iv
  store float %call, ptr %arrayidx, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

define void @exp2_f64(ptr nocapture %varray) {
; CHECK-LABEL: @exp2_f64(
; CHECK:    [[TMP5:%.*]] = call <4 x double> @amd_vrd4_exp2(<4 x double> [[TMP4:%.*]])
; CHECK:    ret void
;
; CHECK-AVX512-VF8-LABEL: @exp2_f64(
; CHECK-AVX512-VF8:    [[TMP5:%.*]] = call <8 x double> @amd_vrd8_exp2(<8 x double> [[TMP4:%.*]])
; CHECK-AVX512-VF8:    ret void
;
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %tmp = trunc i64 %iv to i32
  %conv = sitofp i32 %tmp to double
  %call = tail call double @exp2(double %conv)
  %arrayidx = getelementptr inbounds double, ptr %varray, i64 %iv
  store double %call, ptr %arrayidx, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

define void @exp2_f32(ptr nocapture %varray) {
; CHECK-LABEL: @exp2_f32(
; CHECK:    [[TMP5:%.*]] = call <4 x float> @amd_vrs4_exp2f(<4 x float> [[TMP4:%.*]])
; CHECK:    ret void
;
; CHECK-AVX512-VF16-LABEL: @exp2_f32(
; CHECK-AVX512-VF16:    [[TMP5:%.*]] = call <16 x float> @amd_vrs16_exp2f(<16 x float> [[TMP4:%.*]])
; CHECK-AVX512-VF16:    ret void
;
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %tmp = trunc i64 %iv to i32
  %conv = sitofp i32 %tmp to float
  %call = tail call float @exp2f(float %conv)
  %arrayidx = getelementptr inbounds float, ptr %varray, i64 %iv
  store float %call, ptr %arrayidx, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

define void @exp2_f64_intrinsic(ptr nocapture %varray) {
; CHECK-LABEL: @exp2_f64_intrinsic(
; CHECK:    [[TMP5:%.*]] = call <4 x double> @amd_vrd4_exp2(<4 x double> [[TMP4:%.*]])
; CHECK:    ret void
;
; CHECK-AVX512-VF8-LABEL: @exp2_f64_intrinsic(
; CHECK-AVX512-VF8:    [[TMP5:%.*]] = call <8 x double> @amd_vrd8_exp2(<8 x double> [[TMP4:%.*]])
; CHECK-AVX512-VF8:    ret void
;
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %tmp = trunc i64 %iv to i32
  %conv = sitofp i32 %tmp to double
  %call = tail call double @llvm.exp2.f64(double %conv)
  %arrayidx = getelementptr inbounds double, ptr %varray, i64 %iv
  store double %call, ptr %arrayidx, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

define void @exp2_f32_intrinsic(ptr nocapture %varray) {
; CHECK-LABEL: @exp2_f32_intrinsic(
; CHECK:    [[TMP5:%.*]] = call <4 x float> @amd_vrs4_exp2f(<4 x float> [[TMP4:%.*]])
; CHECK:    ret void
;
; CHECK-AVX512-VF16-LABEL: @exp2_f32_intrinsic(
; CHECK-AVX512-VF16:    [[TMP5:%.*]] = call <16 x float> @amd_vrs16_exp2f(<16 x float> [[TMP4:%.*]])
; CHECK-AVX512-VF16:    ret void
;
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %tmp = trunc i64 %iv to i32
  %conv = sitofp i32 %tmp to float
  %call = tail call float @llvm.exp2.f32(float %conv)
  %arrayidx = getelementptr inbounds float, ptr %varray, i64 %iv
  store float %call, ptr %arrayidx, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

attributes #0 = { nounwind readnone }
