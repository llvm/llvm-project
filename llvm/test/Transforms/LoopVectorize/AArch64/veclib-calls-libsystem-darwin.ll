; RUN: opt < %s -vector-library=Darwin_libsystem_m -passes=inject-tli-mappings,loop-vectorize -S | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "arm64-apple-darwin"

declare float @expf(float) nounwind readnone
define void @expf_v4f32(i64 %n, ptr noalias %y, ptr noalias %x) {
; CHECK-LABEL: @expf_v4f32(
; CHECK: call <4 x float> @_simd_exp_f4(
; CHECK: ret void

entry:
  br label %for.body

for.body:
  %iv = phi i64 [ %iv.next, %for.body ], [ 0, %entry ]
  %gep.y = getelementptr inbounds float, ptr %y, i64 %iv
  %lv = load float, ptr %gep.y, align 4
  %call = tail call float @expf(float %lv)
  %gep.x = getelementptr inbounds float, ptr %x, i64 %iv
  store float %call, ptr %gep.x, align 4
  %iv.next = add i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

declare double @exp(double) nounwind readnone
define void @exp_v2f64(i64 %n, ptr noalias %y, ptr noalias %x) {
; CHECK-LABEL: @exp_v2f64(
; CHECK: call <2 x double> @_simd_exp_d2(
; CHECK: ret void

entry:
  br label %for.body

for.body:
  %iv = phi i64 [ %iv.next, %for.body ], [ 0, %entry ]
  %gep.y = getelementptr inbounds double, ptr %y, i64 %iv
  %lv = load double, ptr %gep.y, align 4
  %call = tail call double @exp(double %lv)
  %gep.x = getelementptr inbounds double, ptr %x, i64 %iv
  store double %call, ptr %gep.x, align 4
  %iv.next = add i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

declare float @exp2f(float) nounwind readnone
define void @exp2f_v4f32(i64 %n, ptr noalias %y, ptr noalias %x) {
; CHECK-LABEL: @exp2f_v4f32(
; CHECK: call <4 x float> @_simd_exp2_f4(
; CHECK: ret void

entry:
  br label %for.body

for.body:
  %iv = phi i64 [ %iv.next, %for.body ], [ 0, %entry ]
  %gep.y = getelementptr inbounds float, ptr %y, i64 %iv
  %lv = load float, ptr %gep.y, align 4
  %call = tail call float @exp2f(float %lv)
  %gep.x = getelementptr inbounds float, ptr %x, i64 %iv
  store float %call, ptr %gep.x, align 4
  %iv.next = add i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

declare double @exp2(double) nounwind readnone
define void @exp2_v2f64(i64 %n, ptr noalias %y, ptr noalias %x) {
; CHECK-LABEL: @exp2_v2f64(
; CHECK: call <2 x double> @_simd_exp2_d2(
; CHECK: ret void

entry:
  br label %for.body

for.body:
  %iv = phi i64 [ %iv.next, %for.body ], [ 0, %entry ]
  %gep.y = getelementptr inbounds double, ptr %y, i64 %iv
  %lv = load double, ptr %gep.y, align 4
  %call = tail call double @exp2(double %lv)
  %gep.x = getelementptr inbounds double, ptr %x, i64 %iv
  store double %call, ptr %gep.x, align 4
  %iv.next = add i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

declare float @__exp10f(float) nounwind readnone
define void @__exp10f_v4f32(i64 %n, ptr noalias %y, ptr noalias %x) {
; CHECK-LABEL: @__exp10f_v4f32(
; CHECK: call <4 x float> @_simd_exp10_f4(
; CHECK: ret void

entry:
  br label %for.body

for.body:
  %iv = phi i64 [ %iv.next, %for.body ], [ 0, %entry ]
  %gep.y = getelementptr inbounds float, ptr %y, i64 %iv
  %lv = load float, ptr %gep.y, align 4
  %call = tail call float @__exp10f(float %lv)
  %gep.x = getelementptr inbounds float, ptr %x, i64 %iv
  store float %call, ptr %gep.x, align 4
  %iv.next = add i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

declare double @__exp10(double) nounwind readnone
define void @__exp10_v2f64(i64 %n, ptr noalias %y, ptr noalias %x) {
; CHECK-LABEL: @__exp10_v2f64(
; CHECK: call <2 x double> @_simd_exp10_d2(
; CHECK: ret void

entry:
  br label %for.body

for.body:
  %iv = phi i64 [ %iv.next, %for.body ], [ 0, %entry ]
  %gep.y = getelementptr inbounds double, ptr %y, i64 %iv
  %lv = load double, ptr %gep.y, align 4
  %call = tail call double @__exp10(double %lv)
  %gep.x = getelementptr inbounds double, ptr %x, i64 %iv
  store double %call, ptr %gep.x, align 4
  %iv.next = add i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

declare float @expm1f(float) nounwind readnone
define void @expm1f_v4f32(i64 %n, ptr noalias %y, ptr noalias %x) {
; CHECK-LABEL: @expm1f_v4f32(
; CHECK: call <4 x float> @_simd_expm1_f4(
; CHECK: ret void

entry:
  br label %for.body

for.body:
  %iv = phi i64 [ %iv.next, %for.body ], [ 0, %entry ]
  %gep.y = getelementptr inbounds float, ptr %y, i64 %iv
  %lv = load float, ptr %gep.y, align 4
  %call = tail call float @expm1f(float %lv)
  %gep.x = getelementptr inbounds float, ptr %x, i64 %iv
  store float %call, ptr %gep.x, align 4
  %iv.next = add i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

declare double @expm1(double) nounwind readnone
define void @expm1_v2f64(i64 %n, ptr noalias %y, ptr noalias %x) {
; CHECK-LABEL: @expm1_v2f64(
; CHECK: call <2 x double> @_simd_expm1_d2(
; CHECK: ret void

entry:
  br label %for.body

for.body:
  %iv = phi i64 [ %iv.next, %for.body ], [ 0, %entry ]
  %gep.y = getelementptr inbounds double, ptr %y, i64 %iv
  %lv = load double, ptr %gep.y, align 4
  %call = tail call double @expm1(double %lv)
  %gep.x = getelementptr inbounds double, ptr %x, i64 %iv
  store double %call, ptr %gep.x, align 4
  %iv.next = add i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

declare float @logf(float) nounwind readnone
define void @logf_v4f32(i64 %n, ptr noalias %y, ptr noalias %x) {
; CHECK-LABEL: @logf_v4f32(
; CHECK: call <4 x float> @_simd_log_f4(
; CHECK: ret void

entry:
  br label %for.body

for.body:
  %iv = phi i64 [ %iv.next, %for.body ], [ 0, %entry ]
  %gep.y = getelementptr inbounds float, ptr %y, i64 %iv
  %lv = load float, ptr %gep.y, align 4
  %call = tail call float @logf(float %lv)
  %gep.x = getelementptr inbounds float, ptr %x, i64 %iv
  store float %call, ptr %gep.x, align 4
  %iv.next = add i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

declare double @log(double) nounwind readnone
define void @log_v2f64(i64 %n, ptr noalias %y, ptr noalias %x) {
; CHECK-LABEL: @log_v2f64(
; CHECK: call <2 x double> @_simd_log_d2(
; CHECK: ret void

entry:
  br label %for.body

for.body:
  %iv = phi i64 [ %iv.next, %for.body ], [ 0, %entry ]
  %gep.y = getelementptr inbounds double, ptr %y, i64 %iv
  %lv = load double, ptr %gep.y, align 4
  %call = tail call double @log(double %lv)
  %gep.x = getelementptr inbounds double, ptr %x, i64 %iv
  store double %call, ptr %gep.x, align 4
  %iv.next = add i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

declare float @log10f(float) nounwind readnone
define void @log10f_v4f32(i64 %n, ptr noalias %y, ptr noalias %x) {
; CHECK-LABEL: @log10f_v4f32(
; CHECK: call <4 x float> @_simd_log10_f4(
; CHECK: ret void

entry:
  br label %for.body

for.body:
  %iv = phi i64 [ %iv.next, %for.body ], [ 0, %entry ]
  %gep.y = getelementptr inbounds float, ptr %y, i64 %iv
  %lv = load float, ptr %gep.y, align 4
  %call = tail call float @log10f(float %lv)
  %gep.x = getelementptr inbounds float, ptr %x, i64 %iv
  store float %call, ptr %gep.x, align 4
  %iv.next = add i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

declare double @log10(double) nounwind readnone
define void @log10_v2f64(i64 %n, ptr noalias %y, ptr noalias %x) {
; CHECK-LABEL: @log10_v2f64(
; CHECK: call <2 x double> @_simd_log10_d2(
; CHECK: ret void

entry:
  br label %for.body

for.body:
  %iv = phi i64 [ %iv.next, %for.body ], [ 0, %entry ]
  %gep.y = getelementptr inbounds double, ptr %y, i64 %iv
  %lv = load double, ptr %gep.y, align 4
  %call = tail call double @log10(double %lv)
  %gep.x = getelementptr inbounds double, ptr %x, i64 %iv
  store double %call, ptr %gep.x, align 4
  %iv.next = add i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

declare float @log1pf(float) nounwind readnone
define void @log1pf_v4f32(i64 %n, ptr noalias %y, ptr noalias %x) {
; CHECK-LABEL: @log1pf_v4f32(
; CHECK: call <4 x float> @_simd_log1p_f4(
; CHECK: ret void

entry:
  br label %for.body

for.body:
  %iv = phi i64 [ %iv.next, %for.body ], [ 0, %entry ]
  %gep.y = getelementptr inbounds float, ptr %y, i64 %iv
  %lv = load float, ptr %gep.y, align 4
  %call = tail call float @log1pf(float %lv)
  %gep.x = getelementptr inbounds float, ptr %x, i64 %iv
  store float %call, ptr %gep.x, align 4
  %iv.next = add i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

declare double @log1p(double) nounwind readnone
define void @log1p_v2f64(i64 %n, ptr noalias %y, ptr noalias %x) {
; CHECK-LABEL: @log1p_v2f64(
; CHECK: call <2 x double> @_simd_log1p_d2(
; CHECK: ret void

entry:
  br label %for.body

for.body:
  %iv = phi i64 [ %iv.next, %for.body ], [ 0, %entry ]
  %gep.y = getelementptr inbounds double, ptr %y, i64 %iv
  %lv = load double, ptr %gep.y, align 4
  %call = tail call double @log1p(double %lv)
  %gep.x = getelementptr inbounds double, ptr %x, i64 %iv
  store double %call, ptr %gep.x, align 4
  %iv.next = add i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

declare float @log2f(float) nounwind readnone
define void @log2f_v4f32(i64 %n, ptr noalias %y, ptr noalias %x) {
; CHECK-LABEL: @log2f_v4f32(
; CHECK: call <4 x float> @_simd_log2_f4(
; CHECK: ret void

entry:
  br label %for.body

for.body:
  %iv = phi i64 [ %iv.next, %for.body ], [ 0, %entry ]
  %gep.y = getelementptr inbounds float, ptr %y, i64 %iv
  %lv = load float, ptr %gep.y, align 4
  %call = tail call float @log2f(float %lv)
  %gep.x = getelementptr inbounds float, ptr %x, i64 %iv
  store float %call, ptr %gep.x, align 4
  %iv.next = add i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

declare double @log2(double) nounwind readnone
define void @log2_v2f64(i64 %n, ptr noalias %y, ptr noalias %x) {
; CHECK-LABEL: @log2_v2f64(
; CHECK: call <2 x double> @_simd_log2_d2(
; CHECK: ret void

entry:
  br label %for.body

for.body:
  %iv = phi i64 [ %iv.next, %for.body ], [ 0, %entry ]
  %gep.y = getelementptr inbounds double, ptr %y, i64 %iv
  %lv = load double, ptr %gep.y, align 4
  %call = tail call double @log2(double %lv)
  %gep.x = getelementptr inbounds double, ptr %x, i64 %iv
  store double %call, ptr %gep.x, align 4
  %iv.next = add i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

declare float @acosf(float) nounwind readnone
define void @acos_v4f32(i64 %n, ptr noalias %y, ptr noalias %x) {
; CHECK-LABEL: @acos_v4f32(
; CHECK: call <4 x float> @_simd_acos_f4(
; CHECK: ret void

entry:
  br label %for.body

for.body:
  %iv = phi i64 [ %iv.next, %for.body ], [ 0, %entry ]
  %gep.y = getelementptr inbounds float, ptr %y, i64 %iv
  %lv = load float, ptr %gep.y, align 4
  %call = tail call float @acosf(float %lv)
  %gep.x = getelementptr inbounds float, ptr %x, i64 %iv
  store float %call, ptr %gep.x, align 4
  %iv.next = add i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

declare double @acos(double) nounwind readnone
define void @acos_v2f64(i64 %n, ptr noalias %y, ptr noalias %x) {
; CHECK-LABEL: @acos_v2f64(
; CHECK: call <2 x double> @_simd_acos_d2(
; CHECK: ret void

entry:
  br label %for.body

for.body:
  %iv = phi i64 [ %iv.next, %for.body ], [ 0, %entry ]
  %gep.y = getelementptr inbounds double, ptr %y, i64 %iv
  %lv = load double, ptr %gep.y, align 4
  %call = tail call double @acos(double %lv)
  %gep.x = getelementptr inbounds double, ptr %x, i64 %iv
  store double %call, ptr %gep.x, align 4
  %iv.next = add i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

declare float @asinf(float) nounwind readnone
define void @asinf_v4f32(i64 %n, ptr noalias %y, ptr noalias %x) {
; CHECK-LABEL: @asinf_v4f32(
; CHECK: call <4 x float> @_simd_asin_f4(
; CHECK: ret void

entry:
  br label %for.body

for.body:
  %iv = phi i64 [ %iv.next, %for.body ], [ 0, %entry ]
  %gep.y = getelementptr inbounds float, ptr %y, i64 %iv
  %lv = load float, ptr %gep.y, align 4
  %call = tail call float @asinf(float %lv)
  %gep.x = getelementptr inbounds float, ptr %x, i64 %iv
  store float %call, ptr %gep.x, align 4
  %iv.next = add i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

declare double @asin(double) nounwind readnone
define void @asin_v2f64(i64 %n, ptr noalias %y, ptr noalias %x) {
; CHECK-LABEL: @asin_v2f64(
; CHECK: call <2 x double> @_simd_asin_d2(
; CHECK: ret void

entry:
  br label %for.body

for.body:
  %iv = phi i64 [ %iv.next, %for.body ], [ 0, %entry ]
  %gep.y = getelementptr inbounds double, ptr %y, i64 %iv
  %lv = load double, ptr %gep.y, align 4
  %call = tail call double @asin(double %lv)
  %gep.x = getelementptr inbounds double, ptr %x, i64 %iv
  store double %call, ptr %gep.x, align 4
  %iv.next = add i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

 declare float @atanf(float) nounwind readnone
define void @atanf_v4f32(i64 %n, ptr noalias %y, ptr noalias %x) {
; CHECK-LABEL: @atanf_v4f32(
; CHECK: call <4 x float> @_simd_atan_f4(
; CHECK: ret void

entry:
  br label %for.body

for.body:
  %iv = phi i64 [ %iv.next, %for.body ], [ 0, %entry ]
  %gep.y = getelementptr inbounds float, ptr %y, i64 %iv
  %lv = load float, ptr %gep.y, align 4
  %call = tail call float @atanf(float %lv)
  %gep.x = getelementptr inbounds float, ptr %x, i64 %iv
  store float %call, ptr %gep.x, align 4
  %iv.next = add i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

declare double @atan(double) nounwind readnone
define void @atan_v2f64(i64 %n, ptr noalias %y, ptr noalias %x) {
; CHECK-LABEL: @atan_v2f64(
; CHECK: call <2 x double> @_simd_atan_d2(
; CHECK: ret void

entry:
  br label %for.body

for.body:
  %iv = phi i64 [ %iv.next, %for.body ], [ 0, %entry ]
  %gep.y = getelementptr inbounds double, ptr %y, i64 %iv
  %lv = load double, ptr %gep.y, align 4
  %call = tail call double @atan(double %lv)
  %gep.x = getelementptr inbounds double, ptr %x, i64 %iv
  store double %call, ptr %gep.x, align 4
  %iv.next = add i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

declare float @atan2f(float, float) nounwind readnone
define void @atan2f_v4f32(i64 %n, ptr noalias %y, ptr noalias %x) {
; CHECK-LABEL: @atan2f_v4f32(
; CHECK: call <4 x float> @_simd_atan2_f4(
; CHECK: ret void

entry:
  br label %for.body

for.body:
  %iv = phi i64 [ %iv.next, %for.body ], [ 0, %entry ]
  %gep.y = getelementptr inbounds float, ptr %y, i64 %iv
  %lv = load float, ptr %gep.y, align 4
  %call = tail call float @atan2f(float %lv, float %lv)
  %gep.x = getelementptr inbounds float, ptr %x, i64 %iv
  store float %call, ptr %gep.x, align 4
  %iv.next = add i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

declare double @atan2(double, double) nounwind readnone
define void @atan2_v2f64(i64 %n, ptr noalias %y, ptr noalias %x) {
; CHECK-LABEL: @atan2_v2f64(
; CHECK: call <2 x double> @_simd_atan2_d2(
; CHECK: ret void

entry:
  br label %for.body

for.body:
  %iv = phi i64 [ %iv.next, %for.body ], [ 0, %entry ]
  %gep.y = getelementptr inbounds double, ptr %y, i64 %iv
  %lv = load double, ptr %gep.y, align 4
  %call = tail call double @atan2(double %lv, double %lv)
  %gep.x = getelementptr inbounds double, ptr %x, i64 %iv
  store double %call, ptr %gep.x, align 4
  %iv.next = add i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

declare float @cosf(float) nounwind readnone
define void @cosf_v4f32(i64 %n, ptr noalias %y, ptr noalias %x) {
; CHECK-LABEL: @cosf_v4f32(
; CHECK: call <4 x float> @_simd_cos_f4(
; CHECK: ret void

entry:
  br label %for.body

for.body:
  %iv = phi i64 [ %iv.next, %for.body ], [ 0, %entry ]
  %gep.y = getelementptr inbounds float, ptr %y, i64 %iv
  %lv = load float, ptr %gep.y, align 4
  %call = tail call float @cosf(float %lv)
  %gep.x = getelementptr inbounds float, ptr %x, i64 %iv
  store float %call, ptr %gep.x, align 4
  %iv.next = add i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

declare double @cos(double) nounwind readnone
define void @cos_v2f64(i64 %n, ptr noalias %y, ptr noalias %x) {
; CHECK-LABEL: @cos_v2f64(
; CHECK: call <2 x double> @_simd_cos_d2(
; CHECK: ret void

entry:
  br label %for.body

for.body:
  %iv = phi i64 [ %iv.next, %for.body ], [ 0, %entry ]
  %gep.y = getelementptr inbounds double, ptr %y, i64 %iv
  %lv = load double, ptr %gep.y, align 4
  %call = tail call double @cos(double %lv)
  %gep.x = getelementptr inbounds double, ptr %x, i64 %iv
  store double %call, ptr %gep.x, align 4
  %iv.next = add i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

declare float @sinf(float) nounwind readnone
define void @sinf_v4f32(i64 %n, ptr noalias %y, ptr noalias %x) {
; CHECK-LABEL: @sinf_v4f32(
; CHECK: call <4 x float> @_simd_sin_f4(
; CHECK: ret void

entry:
  br label %for.body

for.body:
  %iv = phi i64 [ %iv.next, %for.body ], [ 0, %entry ]
  %gep.y = getelementptr inbounds float, ptr %y, i64 %iv
  %lv = load float, ptr %gep.y, align 4
  %call = tail call float @sinf(float %lv)
  %gep.x = getelementptr inbounds float, ptr %x, i64 %iv
  store float %call, ptr %gep.x, align 4
  %iv.next = add i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

declare double @sin(double) nounwind readnone
define void @sin_v2f64(i64 %n, ptr noalias %y, ptr noalias %x) {
; CHECK-LABEL: @sin_v2f64(
; CHECK: call <2 x double> @_simd_sin_d2(
; CHECK: ret void

entry:
  br label %for.body

for.body:
  %iv = phi i64 [ %iv.next, %for.body ], [ 0, %entry ]
  %gep.y = getelementptr inbounds double, ptr %y, i64 %iv
  %lv = load double, ptr %gep.y, align 4
  %call = tail call double @sin(double %lv)
  %gep.x = getelementptr inbounds double, ptr %x, i64 %iv
  store double %call, ptr %gep.x, align 4
  %iv.next = add i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

declare float @tanf(float) nounwind readnone
define void @tanf_v4f32(i64 %n, ptr noalias %y, ptr noalias %x) {
; CHECK-LABEL: @tanf_v4f32(
; CHECK: call <4 x float> @_simd_tan_f4(
; CHECK: ret void

entry:
  br label %for.body

for.body:
  %iv = phi i64 [ %iv.next, %for.body ], [ 0, %entry ]
  %gep.y = getelementptr inbounds float, ptr %y, i64 %iv
  %lv = load float, ptr %gep.y, align 4
  %call = tail call float @tanf(float %lv)
  %gep.x = getelementptr inbounds float, ptr %x, i64 %iv
  store float %call, ptr %gep.x, align 4
  %iv.next = add i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

declare double @tan(double) nounwind readnone
define void @tan_v2f64(i64 %n, ptr noalias %y, ptr noalias %x) {
; CHECK-LABEL: @tan_v2f64(
; CHECK: call <2 x double> @_simd_tan_d2(
; CHECK: ret void

entry:
  br label %for.body

for.body:
  %iv = phi i64 [ %iv.next, %for.body ], [ 0, %entry ]
  %gep.y = getelementptr inbounds double, ptr %y, i64 %iv
  %lv = load double, ptr %gep.y, align 4
  %call = tail call double @tan(double %lv)
  %gep.x = getelementptr inbounds double, ptr %x, i64 %iv
  store double %call, ptr %gep.x, align 4
  %iv.next = add i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

define void @tan_v4f32_intrinsic(i64 %n, ptr noalias %y, ptr noalias %x) {
; CHECK-LABEL: @tan_v4f32_intrinsic(
; CHECK: call <4 x float> @_simd_tan_f4(<4 x float>
; CHECK: ret void

entry:
  br label %for.body

for.body:
  %iv = phi i64 [ %iv.next, %for.body ], [ 0, %entry ]
  %gep.y = getelementptr inbounds float, ptr %y, i64 %iv
  %lv = load float, ptr %gep.y, align 4
  %call = tail call float @llvm.tan.f32(float %lv)
  %gep.x = getelementptr inbounds float, ptr %x, i64 %iv
  store float %call, ptr %gep.x, align 4
  %iv.next = add i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

define void @tan_v2f64_intrinsic(i64 %n, ptr noalias %y, ptr noalias %x) {
; CHECK-LABEL: @tan_v2f64_intrinsic(
; CHECK: call <2 x double> @_simd_tan_d2(<2 x double>
; CHECK: ret void

entry:
  br label %for.body

for.body:
  %iv = phi i64 [ %iv.next, %for.body ], [ 0, %entry ]
  %gep.y = getelementptr inbounds double, ptr %y, i64 %iv
  %lv = load double, ptr %gep.y, align 4
  %call = tail call double @llvm.tan.f64(double %lv)
  %gep.x = getelementptr inbounds double, ptr %x, i64 %iv
  store double %call, ptr %gep.x, align 4
  %iv.next = add i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

define void @acos_v4f32_intrinsic(i64 %n, ptr noalias %y, ptr noalias %x) {
; CHECK-LABEL: @acos_v4f32_intrinsic(
; CHECK: call <4 x float> @_simd_acos_f4(<4 x float>
; CHECK: ret void

entry:
  br label %for.body

for.body:
  %iv = phi i64 [ %iv.next, %for.body ], [ 0, %entry ]
  %gep.y = getelementptr inbounds float, ptr %y, i64 %iv
  %lv = load float, ptr %gep.y, align 4
  %call = tail call float @llvm.acos.f32(float %lv)
  %gep.x = getelementptr inbounds float, ptr %x, i64 %iv
  store float %call, ptr %gep.x, align 4
  %iv.next = add i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

define void @acos_v2f64_intrinsic(i64 %n, ptr noalias %y, ptr noalias %x) {
; CHECK-LABEL: @acos_v2f64_intrinsic(
; CHECK: call <2 x double> @_simd_acos_d2(<2 x double>
; CHECK: ret void

entry:
  br label %for.body

for.body:
  %iv = phi i64 [ %iv.next, %for.body ], [ 0, %entry ]
  %gep.y = getelementptr inbounds double, ptr %y, i64 %iv
  %lv = load double, ptr %gep.y, align 4
  %call = tail call double @llvm.acos.f64(double %lv)
  %gep.x = getelementptr inbounds double, ptr %x, i64 %iv
  store double %call, ptr %gep.x, align 4
  %iv.next = add i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

define void @asin_v4f32_intrinsic(i64 %n, ptr noalias %y, ptr noalias %x) {
; CHECK-LABEL: @asin_v4f32_intrinsic(
; CHECK: call <4 x float> @_simd_asin_f4(<4 x float>
; CHECK: ret void

entry:
  br label %for.body

for.body:
  %iv = phi i64 [ %iv.next, %for.body ], [ 0, %entry ]
  %gep.y = getelementptr inbounds float, ptr %y, i64 %iv
  %lv = load float, ptr %gep.y, align 4
  %call = tail call float @llvm.asin.f32(float %lv)
  %gep.x = getelementptr inbounds float, ptr %x, i64 %iv
  store float %call, ptr %gep.x, align 4
  %iv.next = add i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

define void @asin_v2f64_intrinsic(i64 %n, ptr noalias %y, ptr noalias %x) {
; CHECK-LABEL: @asin_v2f64_intrinsic(
; CHECK: call <2 x double> @_simd_asin_d2(<2 x double>
; CHECK: ret void

entry:
  br label %for.body

for.body:
  %iv = phi i64 [ %iv.next, %for.body ], [ 0, %entry ]
  %gep.y = getelementptr inbounds double, ptr %y, i64 %iv
  %lv = load double, ptr %gep.y, align 4
  %call = tail call double @llvm.asin.f64(double %lv)
  %gep.x = getelementptr inbounds double, ptr %x, i64 %iv
  store double %call, ptr %gep.x, align 4
  %iv.next = add i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

define void @atan_v4f32_intrinsic(i64 %n, ptr noalias %y, ptr noalias %x) {
; CHECK-LABEL: @atan_v4f32_intrinsic(
; CHECK: call <4 x float> @_simd_atan_f4(<4 x float>
; CHECK: ret void

entry:
  br label %for.body

for.body:
  %iv = phi i64 [ %iv.next, %for.body ], [ 0, %entry ]
  %gep.y = getelementptr inbounds float, ptr %y, i64 %iv
  %lv = load float, ptr %gep.y, align 4
  %call = tail call float @llvm.atan.f32(float %lv)
  %gep.x = getelementptr inbounds float, ptr %x, i64 %iv
  store float %call, ptr %gep.x, align 4
  %iv.next = add i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

define void @atan_v2f64_intrinsic(i64 %n, ptr noalias %y, ptr noalias %x) {
; CHECK-LABEL: @atan_v2f64_intrinsic(
; CHECK: call <2 x double> @_simd_atan_d2(<2 x double>
; CHECK: ret void

entry:
  br label %for.body

for.body:
  %iv = phi i64 [ %iv.next, %for.body ], [ 0, %entry ]
  %gep.y = getelementptr inbounds double, ptr %y, i64 %iv
  %lv = load double, ptr %gep.y, align 4
  %call = tail call double @llvm.atan.f64(double %lv)
  %gep.x = getelementptr inbounds double, ptr %x, i64 %iv
  store double %call, ptr %gep.x, align 4
  %iv.next = add i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

define void @atan2_v4f32_intrinsic(i64 %n, ptr noalias %y, ptr noalias %x) {
; CHECK-LABEL: @atan2_v4f32_intrinsic(
; CHECK: call <4 x float> @_simd_atan2_f4(<4 x float>
; CHECK: ret void

entry:
  br label %for.body

for.body:
  %iv = phi i64 [ %iv.next, %for.body ], [ 0, %entry ]
  %gep.y = getelementptr inbounds float, ptr %y, i64 %iv
  %lv = load float, ptr %gep.y, align 4
  %call = tail call float @llvm.atan2.f32(float %lv, float %lv)
  %gep.x = getelementptr inbounds float, ptr %x, i64 %iv
  store float %call, ptr %gep.x, align 4
  %iv.next = add i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

define void @atan2_v2f64_intrinsic(i64 %n, ptr noalias %y, ptr noalias %x) {
; CHECK-LABEL: @atan2_v2f64_intrinsic(
; CHECK: call <2 x double> @_simd_atan2_d2(<2 x double>
; CHECK: ret void

entry:
  br label %for.body

for.body:
  %iv = phi i64 [ %iv.next, %for.body ], [ 0, %entry ]
  %gep.y = getelementptr inbounds double, ptr %y, i64 %iv
  %lv = load double, ptr %gep.y, align 4
  %call = tail call double @llvm.atan2.f64(double %lv, double %lv)
  %gep.x = getelementptr inbounds double, ptr %x, i64 %iv
  store double %call, ptr %gep.x, align 4
  %iv.next = add i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

define void @cosh_v4f32_intrinsic(i64 %n, ptr noalias %y, ptr noalias %x) {
; CHECK-LABEL: @cosh_v4f32_intrinsic(
; CHECK: call <4 x float> @_simd_cosh_f4(<4 x float>
; CHECK: ret void

entry:
  br label %for.body

for.body:
  %iv = phi i64 [ %iv.next, %for.body ], [ 0, %entry ]
  %gep.y = getelementptr inbounds float, ptr %y, i64 %iv
  %lv = load float, ptr %gep.y, align 4
  %call = tail call float @llvm.cosh.f32(float %lv)
  %gep.x = getelementptr inbounds float, ptr %x, i64 %iv
  store float %call, ptr %gep.x, align 4
  %iv.next = add i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

define void @cosh_v2f64_intrinsic(i64 %n, ptr noalias %y, ptr noalias %x) {
; CHECK-LABEL: @cosh_v2f64_intrinsic(
; CHECK: call <2 x double> @_simd_cosh_d2(<2 x double>
; CHECK: ret void

entry:
  br label %for.body

for.body:
  %iv = phi i64 [ %iv.next, %for.body ], [ 0, %entry ]
  %gep.y = getelementptr inbounds double, ptr %y, i64 %iv
  %lv = load double, ptr %gep.y, align 4
  %call = tail call double @llvm.cosh.f64(double %lv)
  %gep.x = getelementptr inbounds double, ptr %x, i64 %iv
  store double %call, ptr %gep.x, align 4
  %iv.next = add i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

define void @sinh_v4f32_intrinsic(i64 %n, ptr noalias %y, ptr noalias %x) {
; CHECK-LABEL: @sinh_v4f32_intrinsic(
; CHECK: call <4 x float> @_simd_sinh_f4(<4 x float>
; CHECK: ret void

entry:
  br label %for.body

for.body:
  %iv = phi i64 [ %iv.next, %for.body ], [ 0, %entry ]
  %gep.y = getelementptr inbounds float, ptr %y, i64 %iv
  %lv = load float, ptr %gep.y, align 4
  %call = tail call float @llvm.sinh.f32(float %lv)
  %gep.x = getelementptr inbounds float, ptr %x, i64 %iv
  store float %call, ptr %gep.x, align 4
  %iv.next = add i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

define void @sinh_v2f64_intrinsic(i64 %n, ptr noalias %y, ptr noalias %x) {
; CHECK-LABEL: @sinh_v2f64_intrinsic(
; CHECK: call <2 x double> @_simd_sinh_d2(<2 x double>
; CHECK: ret void

entry:
  br label %for.body

for.body:
  %iv = phi i64 [ %iv.next, %for.body ], [ 0, %entry ]
  %gep.y = getelementptr inbounds double, ptr %y, i64 %iv
  %lv = load double, ptr %gep.y, align 4
  %call = tail call double @llvm.sinh.f64(double %lv)
  %gep.x = getelementptr inbounds double, ptr %x, i64 %iv
  store double %call, ptr %gep.x, align 4
  %iv.next = add i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

define void @tanh_v4f32_intrinsic(i64 %n, ptr noalias %y, ptr noalias %x) {
; CHECK-LABEL: @tanh_v4f32_intrinsic(
; CHECK: call <4 x float> @_simd_tanh_f4(<4 x float>
; CHECK: ret void

entry:
  br label %for.body

for.body:
  %iv = phi i64 [ %iv.next, %for.body ], [ 0, %entry ]
  %gep.y = getelementptr inbounds float, ptr %y, i64 %iv
  %lv = load float, ptr %gep.y, align 4
  %call = tail call float @llvm.tanh.f32(float %lv)
  %gep.x = getelementptr inbounds float, ptr %x, i64 %iv
  store float %call, ptr %gep.x, align 4
  %iv.next = add i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

define void @tanh_v2f64_intrinsic(i64 %n, ptr noalias %y, ptr noalias %x) {
; CHECK-LABEL: @tanh_v2f64_intrinsic(
; CHECK: call <2 x double> @_simd_tanh_d2(<2 x double>
; CHECK: ret void

entry:
  br label %for.body

for.body:
  %iv = phi i64 [ %iv.next, %for.body ], [ 0, %entry ]
  %gep.y = getelementptr inbounds double, ptr %y, i64 %iv
  %lv = load double, ptr %gep.y, align 4
  %call = tail call double @llvm.tanh.f64(double %lv)
  %gep.x = getelementptr inbounds double, ptr %x, i64 %iv
  store double %call, ptr %gep.x, align 4
  %iv.next = add i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}


declare float @cbrtf(float) nounwind readnone
define void @cbrtf_v4f32(i64 %n, ptr noalias %y, ptr noalias %x) {
; CHECK-LABEL: @cbrtf_v4f32(
; CHECK: call <4 x float> @_simd_cbrt_f4(
; CHECK: ret void

entry:
  br label %for.body

for.body:
  %iv = phi i64 [ %iv.next, %for.body ], [ 0, %entry ]
  %gep.y = getelementptr inbounds float, ptr %y, i64 %iv
  %lv = load float, ptr %gep.y, align 4
  %call = tail call float @cbrtf(float %lv)
  %gep.x = getelementptr inbounds float, ptr %x, i64 %iv
  store float %call, ptr %gep.x, align 4
  %iv.next = add i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

declare double @cbrt(double) nounwind readnone
define void @cbrt_v2f64(i64 %n, ptr noalias %y, ptr noalias %x) {
; CHECK-LABEL: @cbrt_v2f64(
; CHECK: call <2 x double> @_simd_cbrt_d2(
; CHECK: ret void

entry:
  br label %for.body

for.body:
  %iv = phi i64 [ %iv.next, %for.body ], [ 0, %entry ]
  %gep.y = getelementptr inbounds double, ptr %y, i64 %iv
  %lv = load double, ptr %gep.y, align 4
  %call = tail call double @cbrt(double %lv)
  %gep.x = getelementptr inbounds double, ptr %x, i64 %iv
  store double %call, ptr %gep.x, align 4
  %iv.next = add i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

declare float @erff(float) nounwind readnone
define void @erff_v4f32(i64 %n, ptr noalias %y, ptr noalias %x) {
; CHECK-LABEL: @erff_v4f32(
; CHECK: call <4 x float> @_simd_erf_f4(
; CHECK: ret void

entry:
  br label %for.body

for.body:
  %iv = phi i64 [ %iv.next, %for.body ], [ 0, %entry ]
  %gep.y = getelementptr inbounds float, ptr %y, i64 %iv
  %lv = load float, ptr %gep.y, align 4
  %call = tail call float @erff(float %lv)
  %gep.x = getelementptr inbounds float, ptr %x, i64 %iv
  store float %call, ptr %gep.x, align 4
  %iv.next = add i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

declare double @erf(double) nounwind readnone
define void @erf_v2f64(i64 %n, ptr noalias %y, ptr noalias %x) {
; CHECK-LABEL: @erf_v2f64(
; CHECK: call <2 x double> @_simd_erf_d2(
; CHECK: ret void

entry:
  br label %for.body

for.body:
  %iv = phi i64 [ %iv.next, %for.body ], [ 0, %entry ]
  %gep.y = getelementptr inbounds double, ptr %y, i64 %iv
  %lv = load double, ptr %gep.y, align 4
  %call = tail call double @erf(double %lv)
  %gep.x = getelementptr inbounds double, ptr %x, i64 %iv
  store double %call, ptr %gep.x, align 4
  %iv.next = add i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

declare float @powf(float, float) nounwind readnone
define void @powf_v4f32(i64 %n, ptr noalias %y, ptr noalias %x) {
; CHECK-LABEL: @powf_v4f32(
; CHECK: call <4 x float> @_simd_pow_f4(
; CHECK: ret void

entry:
  br label %for.body

for.body:
  %iv = phi i64 [ %iv.next, %for.body ], [ 0, %entry ]
  %gep.y = getelementptr inbounds float, ptr %y, i64 %iv
  %lv = load float, ptr %gep.y, align 4
  %call = tail call float @powf(float %lv, float %lv)
  %gep.x = getelementptr inbounds float, ptr %x, i64 %iv
  store float %call, ptr %gep.x, align 4
  %iv.next = add i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

declare double @pow(double, double) nounwind readnone
define void @pow_v2f64(i64 %n, ptr noalias %y, ptr noalias %x) {
; CHECK-LABEL: @pow_v2f64(
; CHECK: call <2 x double> @_simd_pow_d2(
; CHECK: ret void

entry:
  br label %for.body

for.body:
  %iv = phi i64 [ %iv.next, %for.body ], [ 0, %entry ]
  %gep.y = getelementptr inbounds double, ptr %y, i64 %iv
  %lv = load double, ptr %gep.y, align 4
  %call = tail call double @pow(double %lv, double %lv)
  %gep.x = getelementptr inbounds double, ptr %x, i64 %iv
  store double %call, ptr %gep.x, align 4
  %iv.next = add i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

declare float @sinhf(float) nounwind readnone
define void @sinhf_v4f32(i64 %n, ptr noalias %y, ptr noalias %x) {
; CHECK-LABEL: @sinhf_v4f32(
; CHECK: call <4 x float> @_simd_sinh_f4(
; CHECK: ret void

entry:
  br label %for.body

for.body:
  %iv = phi i64 [ %iv.next, %for.body ], [ 0, %entry ]
  %gep.y = getelementptr inbounds float, ptr %y, i64 %iv
  %lv = load float, ptr %gep.y, align 4
  %call = tail call float @sinhf(float %lv)
  %gep.x = getelementptr inbounds float, ptr %x, i64 %iv
  store float %call, ptr %gep.x, align 4
  %iv.next = add i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

declare double @sinh(double) nounwind readnone
define void @sinh_v2f64(i64 %n, ptr noalias %y, ptr noalias %x) {
; CHECK-LABEL: @sinh_v2f64(
; CHECK: call <2 x double> @_simd_sinh_d2(
; CHECK: ret void

entry:
  br label %for.body

for.body:
  %iv = phi i64 [ %iv.next, %for.body ], [ 0, %entry ]
  %gep.y = getelementptr inbounds double, ptr %y, i64 %iv
  %lv = load double, ptr %gep.y, align 4
  %call = tail call double @sinh(double %lv)
  %gep.x = getelementptr inbounds double, ptr %x, i64 %iv
  store double %call, ptr %gep.x, align 4
  %iv.next = add i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

declare float @coshf(float) nounwind readnone
define void @coshf_v4f32(i64 %n, ptr noalias %y, ptr noalias %x) {
; CHECK-LABEL: @coshf_v4f32(
; CHECK: call <4 x float> @_simd_cosh_f4(
; CHECK: ret void

entry:
  br label %for.body

for.body:
  %iv = phi i64 [ %iv.next, %for.body ], [ 0, %entry ]
  %gep.y = getelementptr inbounds float, ptr %y, i64 %iv
  %lv = load float, ptr %gep.y, align 4
  %call = tail call float @coshf(float %lv)
  %gep.x = getelementptr inbounds float, ptr %x, i64 %iv
  store float %call, ptr %gep.x, align 4
  %iv.next = add i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

declare double @cosh(double) nounwind readnone
define void @cosh_v2f64(i64 %n, ptr noalias %y, ptr noalias %x) {
; CHECK-LABEL: @cosh_v2f64(
; CHECK: call <2 x double> @_simd_cosh_d2(
; CHECK: ret void

entry:
  br label %for.body

for.body:
  %iv = phi i64 [ %iv.next, %for.body ], [ 0, %entry ]
  %gep.y = getelementptr inbounds double, ptr %y, i64 %iv
  %lv = load double, ptr %gep.y, align 4
  %call = tail call double @cosh(double %lv)
  %gep.x = getelementptr inbounds double, ptr %x, i64 %iv
  store double %call, ptr %gep.x, align 4
  %iv.next = add i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

declare float @tanhf(float) nounwind readnone
define void @tanhf_v4f32(i64 %n, ptr noalias %y, ptr noalias %x) {
; CHECK-LABEL: @tanhf_v4f32(
; CHECK: call <4 x float> @_simd_tanh_f4(
; CHECK: ret void

entry:
  br label %for.body

for.body:
  %iv = phi i64 [ %iv.next, %for.body ], [ 0, %entry ]
  %gep.y = getelementptr inbounds float, ptr %y, i64 %iv
  %lv = load float, ptr %gep.y, align 4
  %call = tail call float @tanhf(float %lv)
  %gep.x = getelementptr inbounds float, ptr %x, i64 %iv
  store float %call, ptr %gep.x, align 4
  %iv.next = add i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

declare double @tanh(double) nounwind readnone
define void @tanh_v2f64(i64 %n, ptr noalias %y, ptr noalias %x) {
; CHECK-LABEL: @tanh_v2f64(
; CHECK: call <2 x double> @_simd_tanh_d2(
; CHECK: ret void

entry:
  br label %for.body

for.body:
  %iv = phi i64 [ %iv.next, %for.body ], [ 0, %entry ]
  %gep.y = getelementptr inbounds double, ptr %y, i64 %iv
  %lv = load double, ptr %gep.y, align 4
  %call = tail call double @tanh(double %lv)
  %gep.x = getelementptr inbounds double, ptr %x, i64 %iv
  store double %call, ptr %gep.x, align 4
  %iv.next = add i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

declare float @asinhf(float) nounwind readnone
define void @asinhf_v4f32(i64 %n, ptr noalias %y, ptr noalias %x) {
; CHECK-LABEL: @asinhf_v4f32(
; CHECK: call <4 x float> @_simd_asinh_f4(
; CHECK: ret void

entry:
  br label %for.body

for.body:
  %iv = phi i64 [ %iv.next, %for.body ], [ 0, %entry ]
  %gep.y = getelementptr inbounds float, ptr %y, i64 %iv
  %lv = load float, ptr %gep.y, align 4
  %call = tail call float @asinhf(float %lv)
  %gep.x = getelementptr inbounds float, ptr %x, i64 %iv
  store float %call, ptr %gep.x, align 4
  %iv.next = add i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

declare double @asinh(double) nounwind readnone
define void @asinh_v2f64(i64 %n, ptr noalias %y, ptr noalias %x) {
; CHECK-LABEL: @asinh_v2f64(
; CHECK: call <2 x double> @_simd_asinh_d2(
; CHECK: ret void

entry:
  br label %for.body

for.body:
  %iv = phi i64 [ %iv.next, %for.body ], [ 0, %entry ]
  %gep.y = getelementptr inbounds double, ptr %y, i64 %iv
  %lv = load double, ptr %gep.y, align 4
  %call = tail call double @asinh(double %lv)
  %gep.x = getelementptr inbounds double, ptr %x, i64 %iv
  store double %call, ptr %gep.x, align 4
  %iv.next = add i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

declare float @acoshf(float) nounwind readnone
define void @acoshf_v4f32(i64 %n, ptr noalias %y, ptr noalias %x) {
; CHECK-LABEL: @acoshf_v4f32(
; CHECK: call <4 x float> @_simd_acosh_f4(
; CHECK: ret void

entry:
  br label %for.body

for.body:
  %iv = phi i64 [ %iv.next, %for.body ], [ 0, %entry ]
  %gep.y = getelementptr inbounds float, ptr %y, i64 %iv
  %lv = load float, ptr %gep.y, align 4
  %call = tail call float @acoshf(float %lv)
  %gep.x = getelementptr inbounds float, ptr %x, i64 %iv
  store float %call, ptr %gep.x, align 4
  %iv.next = add i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

declare double @acosh(double) nounwind readnone
define void @acosh_v2f64(i64 %n, ptr noalias %y, ptr noalias %x) {
; CHECK-LABEL: @acosh_v2f64(
; CHECK: call <2 x double> @_simd_acosh_d2(
; CHECK: ret void

entry:
  br label %for.body

for.body:
  %iv = phi i64 [ %iv.next, %for.body ], [ 0, %entry ]
  %gep.y = getelementptr inbounds double, ptr %y, i64 %iv
  %lv = load double, ptr %gep.y, align 4
  %call = tail call double @acosh(double %lv)
  %gep.x = getelementptr inbounds double, ptr %x, i64 %iv
  store double %call, ptr %gep.x, align 4
  %iv.next = add i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

declare float @atanhf(float) nounwind readnone
define void @atanhf_v4f32(i64 %n, ptr noalias %y, ptr noalias %x) {
; CHECK-LABEL: @atanhf_v4f32(
; CHECK: call <4 x float> @_simd_atanh_f4(
; CHECK: ret void

entry:
  br label %for.body

for.body:
  %iv = phi i64 [ %iv.next, %for.body ], [ 0, %entry ]
  %gep.y = getelementptr inbounds float, ptr %y, i64 %iv
  %lv = load float, ptr %gep.y, align 4
  %call = tail call float @atanhf(float %lv)
  %gep.x = getelementptr inbounds float, ptr %x, i64 %iv
  store float %call, ptr %gep.x, align 4
  %iv.next = add i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

declare double @atanh(double) nounwind readnone
define void @atanh_v2f64(i64 %n, ptr noalias %y, ptr noalias %x) {
; CHECK-LABEL: @atanh_v2f64(
; CHECK: call <2 x double> @_simd_atanh_d2(
; CHECK: ret void

entry:
  br label %for.body

for.body:
  %iv = phi i64 [ %iv.next, %for.body ], [ 0, %entry ]
  %gep.y = getelementptr inbounds double, ptr %y, i64 %iv
  %lv = load double, ptr %gep.y, align 4
  %call = tail call double @atanh(double %lv)
  %gep.x = getelementptr inbounds double, ptr %x, i64 %iv
  store double %call, ptr %gep.x, align 4
  %iv.next = add i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

declare float @__cospif(float) nounwind readnone
define void @__cospif_v4f32(i64 %n, ptr noalias %y, ptr noalias %x) {
; CHECK-LABEL: @__cospif_v4f32(
; CHECK: call <4 x float> @_simd_cospi_f4(
; CHECK: ret void

entry:
  br label %for.body

for.body:
  %iv = phi i64 [ %iv.next, %for.body ], [ 0, %entry ]
  %gep.y = getelementptr inbounds float, ptr %y, i64 %iv
  %lv = load float, ptr %gep.y, align 4
  %call = tail call float @__cospif(float %lv)
  %gep.x = getelementptr inbounds float, ptr %x, i64 %iv
  store float %call, ptr %gep.x, align 4
  %iv.next = add i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

declare double @__cospi(double) nounwind readnone
define void @__cospi_v2f64(i64 %n, ptr noalias %y, ptr noalias %x) {
; CHECK-LABEL: @__cospi_v2f64(
; CHECK: call <2 x double> @_simd_cospi_d2(
; CHECK: ret void

entry:
  br label %for.body

for.body:
  %iv = phi i64 [ %iv.next, %for.body ], [ 0, %entry ]
  %gep.y = getelementptr inbounds double, ptr %y, i64 %iv
  %lv = load double, ptr %gep.y, align 4
  %call = tail call double @__cospi(double %lv)
  %gep.x = getelementptr inbounds double, ptr %x, i64 %iv
  store double %call, ptr %gep.x, align 4
  %iv.next = add i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

declare float @__sinpif(float) nounwind readnone
define void @__sinpif_v4f32(i64 %n, ptr noalias %y, ptr noalias %x) {
; CHECK-LABEL: @__sinpif_v4f32(
; CHECK: call <4 x float> @_simd_sinpi_f4(
; CHECK: ret void

entry:
  br label %for.body

for.body:
  %iv = phi i64 [ %iv.next, %for.body ], [ 0, %entry ]
  %gep.y = getelementptr inbounds float, ptr %y, i64 %iv
  %lv = load float, ptr %gep.y, align 4
  %call = tail call float @__sinpif(float %lv)
  %gep.x = getelementptr inbounds float, ptr %x, i64 %iv
  store float %call, ptr %gep.x, align 4
  %iv.next = add i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

declare double @__sinpi(double) nounwind readnone
define void @__sinpi_v2f64(i64 %n, ptr noalias %y, ptr noalias %x) {
; CHECK-LABEL: @__sinpi_v2f64(
; CHECK: call <2 x double> @_simd_sinpi_d2(
; CHECK: ret void

entry:
  br label %for.body

for.body:
  %iv = phi i64 [ %iv.next, %for.body ], [ 0, %entry ]
  %gep.y = getelementptr inbounds double, ptr %y, i64 %iv
  %lv = load double, ptr %gep.y, align 4
  %call = tail call double @__sinpi(double %lv)
  %gep.x = getelementptr inbounds double, ptr %x, i64 %iv
  store double %call, ptr %gep.x, align 4
  %iv.next = add i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

declare float @__tanpif(float) nounwind readnone
define void @__tanpif_v4f32(i64 %n, ptr noalias %y, ptr noalias %x) {
; CHECK-LABEL: @__tanpif_v4f32(
; CHECK: call <4 x float> @_simd_tanpi_f4(
; CHECK: ret void

entry:
  br label %for.body

for.body:
  %iv = phi i64 [ %iv.next, %for.body ], [ 0, %entry ]
  %gep.y = getelementptr inbounds float, ptr %y, i64 %iv
  %lv = load float, ptr %gep.y, align 4
  %call = tail call float @__tanpif(float %lv)
  %gep.x = getelementptr inbounds float, ptr %x, i64 %iv
  store float %call, ptr %gep.x, align 4
  %iv.next = add i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

declare double @__tanpi(double) nounwind readnone
define void @__tanpi_v2f64(i64 %n, ptr noalias %y, ptr noalias %x) {
; CHECK-LABEL: @__tanpi_v2f64(
; CHECK: call <2 x double> @_simd_tanpi_d2(
; CHECK: ret void

entry:
  br label %for.body

for.body:
  %iv = phi i64 [ %iv.next, %for.body ], [ 0, %entry ]
  %gep.y = getelementptr inbounds double, ptr %y, i64 %iv
  %lv = load double, ptr %gep.y, align 4
  %call = tail call double @__tanpi(double %lv)
  %gep.x = getelementptr inbounds double, ptr %x, i64 %iv
  store double %call, ptr %gep.x, align 4
  %iv.next = add i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

declare float @hypotf(float, float) nounwind readnone
define void @hypotf_v4f32(i64 %n, ptr noalias %y, ptr noalias %x) {
; CHECK-LABEL: @hypotf_v4f32(
; CHECK: call <4 x float> @_simd_hypot_f4(
; CHECK: ret void

entry:
  br label %for.body

for.body:
  %iv = phi i64 [ %iv.next, %for.body ], [ 0, %entry ]
  %gep.y = getelementptr inbounds float, ptr %y, i64 %iv
  %lv = load float, ptr %gep.y, align 4
  %call = tail call float @hypotf(float %lv, float %lv)
  %gep.x = getelementptr inbounds float, ptr %x, i64 %iv
  store float %call, ptr %gep.x, align 4
  %iv.next = add i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

declare double @hypot(double, double) nounwind readnone
define void @hypot_v2f64(i64 %n, ptr noalias %y, ptr noalias %x) {
; CHECK-LABEL: @hypot_v2f64(
; CHECK: call <2 x double> @_simd_hypot_d2(
; CHECK: ret void

entry:
  br label %for.body

for.body:
  %iv = phi i64 [ %iv.next, %for.body ], [ 0, %entry ]
  %gep.y = getelementptr inbounds double, ptr %y, i64 %iv
  %lv = load double, ptr %gep.y, align 4
  %call = tail call double @hypot(double %lv, double %lv)
  %gep.x = getelementptr inbounds double, ptr %x, i64 %iv
  store double %call, ptr %gep.x, align 4
  %iv.next = add i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

declare float @erfcf(float) nounwind readnone
define void @erfcf_v4f32(i64 %n, ptr noalias %y, ptr noalias %x) {
; CHECK-LABEL: @erfcf_v4f32(
; CHECK: call <4 x float> @_simd_erfc_f4(
; CHECK: ret void

entry:
  br label %for.body

for.body:
  %iv = phi i64 [ %iv.next, %for.body ], [ 0, %entry ]
  %gep.y = getelementptr inbounds float, ptr %y, i64 %iv
  %lv = load float, ptr %gep.y, align 4
  %call = tail call float @erfcf(float %lv)
  %gep.x = getelementptr inbounds float, ptr %x, i64 %iv
  store float %call, ptr %gep.x, align 4
  %iv.next = add i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

declare double @erfc(double) nounwind readnone
define void @erfc_v2f64(i64 %n, ptr noalias %y, ptr noalias %x) {
; CHECK-LABEL: @erfc_v2f64(
; CHECK: call <2 x double> @_simd_erfc_d2(
; CHECK: ret void

entry:
  br label %for.body

for.body:
  %iv = phi i64 [ %iv.next, %for.body ], [ 0, %entry ]
  %gep.y = getelementptr inbounds double, ptr %y, i64 %iv
  %lv = load double, ptr %gep.y, align 4
  %call = tail call double @erfc(double %lv)
  %gep.x = getelementptr inbounds double, ptr %x, i64 %iv
  store double %call, ptr %gep.x, align 4
  %iv.next = add i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

declare float @lgammaf(float) nounwind readnone
define void @lgammaf_v4f32(i64 %n, ptr noalias %y, ptr noalias %x) {
; CHECK-LABEL: @lgammaf_v4f32(
; CHECK: call <4 x float> @_simd_lgamma_f4(
; CHECK: ret void

entry:
  br label %for.body

for.body:
  %iv = phi i64 [ %iv.next, %for.body ], [ 0, %entry ]
  %gep.y = getelementptr inbounds float, ptr %y, i64 %iv
  %lv = load float, ptr %gep.y, align 4
  %call = tail call float @lgammaf(float %lv)
  %gep.x = getelementptr inbounds float, ptr %x, i64 %iv
  store float %call, ptr %gep.x, align 4
  %iv.next = add i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

declare double @lgamma(double) nounwind readnone
define void @lgamma_v2f64(i64 %n, ptr noalias %y, ptr noalias %x) {
; CHECK-LABEL: @lgamma_v2f64(
; CHECK: call <2 x double> @_simd_lgamma_d2(
; CHECK: ret void

entry:
  br label %for.body

for.body:
  %iv = phi i64 [ %iv.next, %for.body ], [ 0, %entry ]
  %gep.y = getelementptr inbounds double, ptr %y, i64 %iv
  %lv = load double, ptr %gep.y, align 4
  %call = tail call double @lgamma(double %lv)
  %gep.x = getelementptr inbounds double, ptr %x, i64 %iv
  store double %call, ptr %gep.x, align 4
  %iv.next = add i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

declare float @tgammaf(float) nounwind readnone
define void @tgammaf_v4f32(i64 %n, ptr noalias %y, ptr noalias %x) {
; CHECK-LABEL: @tgammaf_v4f32(
; CHECK: call <4 x float> @_simd_tgamma_f4(
; CHECK: ret void

entry:
  br label %for.body

for.body:
  %iv = phi i64 [ %iv.next, %for.body ], [ 0, %entry ]
  %gep.y = getelementptr inbounds float, ptr %y, i64 %iv
  %lv = load float, ptr %gep.y, align 4
  %call = tail call float @tgammaf(float %lv)
  %gep.x = getelementptr inbounds float, ptr %x, i64 %iv
  store float %call, ptr %gep.x, align 4
  %iv.next = add i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

declare double @tgamma(double) nounwind readnone
define void @tgamma_v2f64(i64 %n, ptr noalias %y, ptr noalias %x) {
; CHECK-LABEL: @tgamma_v2f64(
; CHECK: call <2 x double> @_simd_tgamma_d2(
; CHECK: ret void

entry:
  br label %for.body

for.body:
  %iv = phi i64 [ %iv.next, %for.body ], [ 0, %entry ]
  %gep.y = getelementptr inbounds double, ptr %y, i64 %iv
  %lv = load double, ptr %gep.y, align 4
  %call = tail call double @tgamma(double %lv)
  %gep.x = getelementptr inbounds double, ptr %x, i64 %iv
  store double %call, ptr %gep.x, align 4
  %iv.next = add i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

declare float @nextafterf(float, float) nounwind readnone
define void @nextafterf_v4f32(i64 %n, ptr noalias %y, ptr noalias %x) {
; CHECK-LABEL: @nextafterf_v4f32(
; CHECK: call <4 x float> @_simd_nextafter_f4(
; CHECK: ret void

entry:
  br label %for.body

for.body:
  %iv = phi i64 [ %iv.next, %for.body ], [ 0, %entry ]
  %gep.y = getelementptr inbounds float, ptr %y, i64 %iv
  %lv = load float, ptr %gep.y, align 4
  %call = tail call float @nextafterf(float %lv, float %lv)
  %gep.x = getelementptr inbounds float, ptr %x, i64 %iv
  store float %call, ptr %gep.x, align 4
  %iv.next = add i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

declare double @nextafter(double, double) nounwind readnone
define void @nextafter_v2f64(i64 %n, ptr noalias %y, ptr noalias %x) {
; CHECK-LABEL: @nextafter_v2f64(
; CHECK: call <2 x double> @_simd_nextafter_d2(
; CHECK: ret void

entry:
  br label %for.body

for.body:
  %iv = phi i64 [ %iv.next, %for.body ], [ 0, %entry ]
  %gep.y = getelementptr inbounds double, ptr %y, i64 %iv
  %lv = load double, ptr %gep.y, align 4
  %call = tail call double @nextafter(double %lv, double %lv)
  %gep.x = getelementptr inbounds double, ptr %x, i64 %iv
  store double %call, ptr %gep.x, align 4
  %iv.next = add i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

declare float @remainderf(float, float) nounwind readnone
define void @remainderf_v4f32(i64 %n, ptr noalias %y, ptr noalias %x) {
; CHECK-LABEL: @remainderf_v4f32(
; CHECK: call <4 x float> @_simd_remainder_f4(
; CHECK: ret void

entry:
  br label %for.body

for.body:
  %iv = phi i64 [ %iv.next, %for.body ], [ 0, %entry ]
  %gep.y = getelementptr inbounds float, ptr %y, i64 %iv
  %lv = load float, ptr %gep.y, align 4
  %call = tail call float @remainderf(float %lv, float %lv)
  %gep.x = getelementptr inbounds float, ptr %x, i64 %iv
  store float %call, ptr %gep.x, align 4
  %iv.next = add i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

declare double @remainder(double, double) nounwind readnone
define void @remainder_v2f64(i64 %n, ptr noalias %y, ptr noalias %x) {
; CHECK-LABEL: @remainder_v2f64(
; CHECK: call <2 x double> @_simd_remainder_d2(
; CHECK: ret void

entry:
  br label %for.body

for.body:
  %iv = phi i64 [ %iv.next, %for.body ], [ 0, %entry ]
  %gep.y = getelementptr inbounds double, ptr %y, i64 %iv
  %lv = load double, ptr %gep.y, align 4
  %call = tail call double @remainder(double %lv, double %lv)
  %gep.x = getelementptr inbounds double, ptr %x, i64 %iv
  store double %call, ptr %gep.x, align 4
  %iv.next = add i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}
