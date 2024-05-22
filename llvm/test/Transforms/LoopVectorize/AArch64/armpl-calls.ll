; RUN: opt -vector-library=ArmPL -passes=inject-tli-mappings,loop-vectorize -S < %s | FileCheck %s --check-prefixes=CHECK,NEON
; RUN: opt -mattr=+sve -vector-library=ArmPL -passes=inject-tli-mappings,loop-vectorize -S < %s | FileCheck %s --check-prefixes=CHECK,SVE

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64-unknown-linux-gnu"


; Tests are checking if LV can vectorize loops with function calls
; using mappings from TLI for scalable and fixed width vectorization.

declare double @acos(double)
declare float @acosf(float)

define void @acos_f64(ptr nocapture %in.ptr, ptr %out.ptr) {
; CHECK-LABEL: @acos_f64(
; NEON:     [[TMP5:%.*]] = call <2 x double> @armpl_vacosq_f64(<2 x double> [[TMP4:%.*]])
; SVE:      [[TMP5:%.*]] = call <vscale x 2 x double> @armpl_svacos_f64_x(<vscale x 2 x double> [[TMP4:%.*]], <vscale x 2 x i1> {{.*}})
; CHECK:    ret void
;
  entry:
  br label %for.body

  for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %in.gep = getelementptr inbounds double, ptr %in.ptr, i64 %iv
  %in = load double, ptr %in.gep, align 8
  %call = tail call double @acos(double %in)
  %out.gep = getelementptr inbounds double, ptr %out.ptr, i64 %iv
  store double %call, ptr %out.gep, align 8
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

  for.end:
  ret void
}

define void @acos_f32(ptr nocapture %in.ptr, ptr %out.ptr) {
; CHECK-LABEL: @acos_f32(
; NEON: [[TMP5:%.*]] = call <4 x float> @armpl_vacosq_f32(<4 x float> [[TMP4:%.*]])
; SVE: [[TMP5:%.*]] = call <vscale x 4 x float> @armpl_svacos_f32_x(<vscale x 4 x float> [[TMP4:%.*]], <vscale x 4 x i1> {{.*}})
; CHECK: ret void
;
  entry:
  br label %for.body

  for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %in.gep = getelementptr inbounds float, ptr %in.ptr, i64 %iv
  %in = load float, ptr %in.gep, align 8
  %call = tail call float @acosf(float %in)
  %out.gep = getelementptr inbounds float, ptr %out.ptr, i64 %iv
  store float %call, ptr %out.gep, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

  for.end:
  ret void
}

declare double @acosh(double)
declare float @acoshf(float)

define void @acosh_f64(ptr nocapture %in.ptr, ptr %out.ptr) {
; CHECK-LABEL: @acosh_f64(
; NEON:     [[TMP5:%.*]] = call <2 x double> @armpl_vacoshq_f64(<2 x double> [[TMP4:%.*]])
; SVE:      [[TMP5:%.*]] = call <vscale x 2 x double> @armpl_svacosh_f64_x(<vscale x 2 x double> [[TMP4:%.*]], <vscale x 2 x i1> {{.*}})
; CHECK:    ret void
;
  entry:
  br label %for.body

  for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %in.gep = getelementptr inbounds double, ptr %in.ptr, i64 %iv
  %in = load double, ptr %in.gep, align 8
  %call = tail call double @acosh(double %in)
  %out.gep = getelementptr inbounds double, ptr %out.ptr, i64 %iv
  store double %call, ptr %out.gep, align 8
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

  for.end:
  ret void
}

define void @acosh_f32(ptr nocapture %in.ptr, ptr %out.ptr) {
; CHECK-LABEL: @acosh_f32(
; NEON: [[TMP5:%.*]] = call <4 x float> @armpl_vacoshq_f32(<4 x float> [[TMP4:%.*]])
; SVE: [[TMP5:%.*]] = call <vscale x 4 x float> @armpl_svacosh_f32_x(<vscale x 4 x float> [[TMP4:%.*]], <vscale x 4 x i1> {{.*}})
; CHECK: ret void
;
  entry:
  br label %for.body

  for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %in.gep = getelementptr inbounds float, ptr %in.ptr, i64 %iv
  %in = load float, ptr %in.gep, align 8
  %call = tail call float @acoshf(float %in)
  %out.gep = getelementptr inbounds float, ptr %out.ptr, i64 %iv
  store float %call, ptr %out.gep, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

  for.end:
  ret void
}

declare double @asin(double)
declare float @asinf(float)

define void @asin_f64(ptr nocapture %in.ptr, ptr %out.ptr) {
; CHECK-LABEL: @asin_f64(
; NEON:     [[TMP5:%.*]] = call <2 x double> @armpl_vasinq_f64(<2 x double> [[TMP4:%.*]])
; SVE:      [[TMP5:%.*]] = call <vscale x 2 x double> @armpl_svasin_f64_x(<vscale x 2 x double> [[TMP4:%.*]], <vscale x 2 x i1> {{.*}})
; CHECK:    ret void
;
  entry:
  br label %for.body

  for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %in.gep = getelementptr inbounds double, ptr %in.ptr, i64 %iv
  %in = load double, ptr %in.gep, align 8
  %call = tail call double @asin(double %in)
  %out.gep = getelementptr inbounds double, ptr %out.ptr, i64 %iv
  store double %call, ptr %out.gep, align 8
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

  for.end:
  ret void
}

define void @asin_f32(ptr nocapture %in.ptr, ptr %out.ptr) {
; CHECK-LABEL: @asin_f32(
; NEON: [[TMP5:%.*]] = call <4 x float> @armpl_vasinq_f32(<4 x float> [[TMP4:%.*]])
; SVE: [[TMP5:%.*]] = call <vscale x 4 x float> @armpl_svasin_f32_x(<vscale x 4 x float> [[TMP4:%.*]], <vscale x 4 x i1> {{.*}})
; CHECK: ret void
;
  entry:
  br label %for.body

  for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %in.gep = getelementptr inbounds float, ptr %in.ptr, i64 %iv
  %in = load float, ptr %in.gep, align 8
  %call = tail call float @asinf(float %in)
  %out.gep = getelementptr inbounds float, ptr %out.ptr, i64 %iv
  store float %call, ptr %out.gep, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

  for.end:
  ret void
}

declare double @asinh(double)
declare float @asinhf(float)

define void @asinh_f64(ptr nocapture %in.ptr, ptr %out.ptr) {
; CHECK-LABEL: @asinh_f64(
; NEON:     [[TMP5:%.*]] = call <2 x double> @armpl_vasinhq_f64(<2 x double> [[TMP4:%.*]])
; SVE:      [[TMP5:%.*]] = call <vscale x 2 x double> @armpl_svasinh_f64_x(<vscale x 2 x double> [[TMP4:%.*]], <vscale x 2 x i1> {{.*}})
; CHECK:    ret void
;
  entry:
  br label %for.body

  for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %in.gep = getelementptr inbounds double, ptr %in.ptr, i64 %iv
  %in = load double, ptr %in.gep, align 8
  %call = tail call double @asinh(double %in)
  %out.gep = getelementptr inbounds double, ptr %out.ptr, i64 %iv
  store double %call, ptr %out.gep, align 8
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

  for.end:
  ret void
}

define void @asinh_f32(ptr nocapture %in.ptr, ptr %out.ptr) {
; CHECK-LABEL: @asinh_f32(
; NEON: [[TMP5:%.*]] = call <4 x float> @armpl_vasinhq_f32(<4 x float> [[TMP4:%.*]])
; SVE: [[TMP5:%.*]] = call <vscale x 4 x float> @armpl_svasinh_f32_x(<vscale x 4 x float> [[TMP4:%.*]], <vscale x 4 x i1> {{.*}})
; CHECK: ret void
;
  entry:
  br label %for.body

  for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %in.gep = getelementptr inbounds float, ptr %in.ptr, i64 %iv
  %in = load float, ptr %in.gep, align 8
  %call = tail call float @asinhf(float %in)
  %out.gep = getelementptr inbounds float, ptr %out.ptr, i64 %iv
  store float %call, ptr %out.gep, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

  for.end:
  ret void
}

declare double @atan(double)
declare float @atanf(float)

define void @atan_f64(ptr nocapture %in.ptr, ptr %out.ptr) {
; CHECK-LABEL: @atan_f64(
; NEON:     [[TMP5:%.*]] = call <2 x double> @armpl_vatanq_f64(<2 x double> [[TMP4:%.*]])
; SVE:      [[TMP5:%.*]] = call <vscale x 2 x double> @armpl_svatan_f64_x(<vscale x 2 x double> [[TMP4:%.*]], <vscale x 2 x i1> {{.*}})
; CHECK:    ret void
;
  entry:
  br label %for.body

  for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %in.gep = getelementptr inbounds double, ptr %in.ptr, i64 %iv
  %in = load double, ptr %in.gep, align 8
  %call = tail call double @atan(double %in)
  %out.gep = getelementptr inbounds double, ptr %out.ptr, i64 %iv
  store double %call, ptr %out.gep, align 8
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

  for.end:
  ret void
}

define void @atan_f32(ptr nocapture %in.ptr, ptr %out.ptr) {
; CHECK-LABEL: @atan_f32(
; NEON: [[TMP5:%.*]] = call <4 x float> @armpl_vatanq_f32(<4 x float> [[TMP4:%.*]])
; SVE: [[TMP5:%.*]] = call <vscale x 4 x float> @armpl_svatan_f32_x(<vscale x 4 x float> [[TMP4:%.*]], <vscale x 4 x i1> {{.*}})
; CHECK: ret void
;
  entry:
  br label %for.body

  for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %in.gep = getelementptr inbounds float, ptr %in.ptr, i64 %iv
  %in = load float, ptr %in.gep, align 8
  %call = tail call float @atanf(float %in)
  %out.gep = getelementptr inbounds float, ptr %out.ptr, i64 %iv
  store float %call, ptr %out.gep, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

  for.end:
  ret void
}

declare double @atanh(double)
declare float @atanhf(float)

define void @atanh_f64(ptr nocapture %in.ptr, ptr %out.ptr) {
; CHECK-LABEL: @atanh_f64(
; NEON:     [[TMP5:%.*]] = call <2 x double> @armpl_vatanhq_f64(<2 x double> [[TMP4:%.*]])
; SVE:      [[TMP5:%.*]] = call <vscale x 2 x double> @armpl_svatanh_f64_x(<vscale x 2 x double> [[TMP4:%.*]], <vscale x 2 x i1> {{.*}})
; CHECK:    ret void
;
  entry:
  br label %for.body

  for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %in.gep = getelementptr inbounds double, ptr %in.ptr, i64 %iv
  %in = load double, ptr %in.gep, align 8
  %call = tail call double @atanh(double %in)
  %out.gep = getelementptr inbounds double, ptr %out.ptr, i64 %iv
  store double %call, ptr %out.gep, align 8
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

  for.end:
  ret void
}

define void @atanh_f32(ptr nocapture %in.ptr, ptr %out.ptr) {
; CHECK-LABEL: @atanh_f32(
; NEON: [[TMP5:%.*]] = call <4 x float> @armpl_vatanhq_f32(<4 x float> [[TMP4:%.*]])
; SVE: [[TMP5:%.*]] = call <vscale x 4 x float> @armpl_svatanh_f32_x(<vscale x 4 x float> [[TMP4:%.*]], <vscale x 4 x i1> {{.*}})
; CHECK: ret void
;
  entry:
  br label %for.body

  for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %in.gep = getelementptr inbounds float, ptr %in.ptr, i64 %iv
  %in = load float, ptr %in.gep, align 8
  %call = tail call float @atanhf(float %in)
  %out.gep = getelementptr inbounds float, ptr %out.ptr, i64 %iv
  store float %call, ptr %out.gep, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

  for.end:
  ret void
}

declare double @cbrt(double)
declare float @cbrtf(float)

define void @cbrt_f64(ptr nocapture %in.ptr, ptr %out.ptr) {
; CHECK-LABEL: @cbrt_f64(
; NEON:     [[TMP5:%.*]] = call <2 x double> @armpl_vcbrtq_f64(<2 x double> [[TMP4:%.*]])
; SVE:      [[TMP5:%.*]] = call <vscale x 2 x double> @armpl_svcbrt_f64_x(<vscale x 2 x double> [[TMP4:%.*]], <vscale x 2 x i1> {{.*}})
; CHECK:    ret void
;
  entry:
  br label %for.body

  for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %in.gep = getelementptr inbounds double, ptr %in.ptr, i64 %iv
  %in = load double, ptr %in.gep, align 8
  %call = tail call double @cbrt(double %in)
  %out.gep = getelementptr inbounds double, ptr %out.ptr, i64 %iv
  store double %call, ptr %out.gep, align 8
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

  for.end:
  ret void
}

define void @cbrt_f32(ptr nocapture %in.ptr, ptr %out.ptr) {
; CHECK-LABEL: @cbrt_f32(
; NEON: [[TMP5:%.*]] = call <4 x float> @armpl_vcbrtq_f32(<4 x float> [[TMP4:%.*]])
; SVE: [[TMP5:%.*]] = call <vscale x 4 x float> @armpl_svcbrt_f32_x(<vscale x 4 x float> [[TMP4:%.*]], <vscale x 4 x i1> {{.*}})
; CHECK: ret void
;
  entry:
  br label %for.body

  for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %in.gep = getelementptr inbounds float, ptr %in.ptr, i64 %iv
  %in = load float, ptr %in.gep, align 8
  %call = tail call float @cbrtf(float %in)
  %out.gep = getelementptr inbounds float, ptr %out.ptr, i64 %iv
  store float %call, ptr %out.gep, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

  for.end:
  ret void
}

declare double @cos(double)
declare float @cosf(float)

define void @cos_f64(ptr nocapture %in.ptr, ptr %out.ptr) {
; CHECK-LABEL: @cos_f64(
; NEON:     [[TMP5:%.*]] = call <2 x double> @armpl_vcosq_f64(<2 x double> [[TMP4:%.*]])
; SVE:      [[TMP5:%.*]] = call <vscale x 2 x double> @armpl_svcos_f64_x(<vscale x 2 x double> [[TMP4:%.*]], <vscale x 2 x i1> {{.*}})
; CHECK:    ret void
;
  entry:
  br label %for.body

  for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %in.gep = getelementptr inbounds double, ptr %in.ptr, i64 %iv
  %in = load double, ptr %in.gep, align 8
  %call = tail call double @cos(double %in)
  %out.gep = getelementptr inbounds double, ptr %out.ptr, i64 %iv
  store double %call, ptr %out.gep, align 8
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

  for.end:
  ret void
}

define void @cos_f32(ptr nocapture %in.ptr, ptr %out.ptr) {
; CHECK-LABEL: @cos_f32(
; NEON: [[TMP5:%.*]] = call <4 x float> @armpl_vcosq_f32(<4 x float> [[TMP4:%.*]])
; SVE: [[TMP5:%.*]] = call <vscale x 4 x float> @armpl_svcos_f32_x(<vscale x 4 x float> [[TMP4:%.*]], <vscale x 4 x i1> {{.*}})
; CHECK: ret void
;
  entry:
  br label %for.body

  for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %in.gep = getelementptr inbounds float, ptr %in.ptr, i64 %iv
  %in = load float, ptr %in.gep, align 8
  %call = tail call float @cosf(float %in)
  %out.gep = getelementptr inbounds float, ptr %out.ptr, i64 %iv
  store float %call, ptr %out.gep, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

  for.end:
  ret void
}

declare double @cosh(double)
declare float @coshf(float)

define void @cosh_f64(ptr nocapture %in.ptr, ptr %out.ptr) {
; CHECK-LABEL: @cosh_f64(
; NEON:     [[TMP5:%.*]] = call <2 x double> @armpl_vcoshq_f64(<2 x double> [[TMP4:%.*]])
; SVE:      [[TMP5:%.*]] = call <vscale x 2 x double> @armpl_svcosh_f64_x(<vscale x 2 x double> [[TMP4:%.*]], <vscale x 2 x i1> {{.*}})
; CHECK:    ret void
;
  entry:
  br label %for.body

  for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %in.gep = getelementptr inbounds double, ptr %in.ptr, i64 %iv
  %in = load double, ptr %in.gep, align 8
  %call = tail call double @cosh(double %in)
  %out.gep = getelementptr inbounds double, ptr %out.ptr, i64 %iv
  store double %call, ptr %out.gep, align 8
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

  for.end:
  ret void
}

define void @cosh_f32(ptr nocapture %in.ptr, ptr %out.ptr) {
; CHECK-LABEL: @cosh_f32(
; NEON: [[TMP5:%.*]] = call <4 x float> @armpl_vcoshq_f32(<4 x float> [[TMP4:%.*]])
; SVE: [[TMP5:%.*]] = call <vscale x 4 x float> @armpl_svcosh_f32_x(<vscale x 4 x float> [[TMP4:%.*]], <vscale x 4 x i1> {{.*}})
; CHECK: ret void
;
  entry:
  br label %for.body

  for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %in.gep = getelementptr inbounds float, ptr %in.ptr, i64 %iv
  %in = load float, ptr %in.gep, align 8
  %call = tail call float @coshf(float %in)
  %out.gep = getelementptr inbounds float, ptr %out.ptr, i64 %iv
  store float %call, ptr %out.gep, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

  for.end:
  ret void
}

declare double @erf(double)
declare float @erff(float)

define void @erf_f64(ptr nocapture %in.ptr, ptr %out.ptr) {
; CHECK-LABEL: @erf_f64(
; NEON:     [[TMP5:%.*]] = call <2 x double> @armpl_verfq_f64(<2 x double> [[TMP4:%.*]])
; SVE:      [[TMP5:%.*]] = call <vscale x 2 x double> @armpl_sverf_f64_x(<vscale x 2 x double> [[TMP4:%.*]], <vscale x 2 x i1> {{.*}})
; CHECK:    ret void
;
  entry:
  br label %for.body

  for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %in.gep = getelementptr inbounds double, ptr %in.ptr, i64 %iv
  %in = load double, ptr %in.gep, align 8
  %call = tail call double @erf(double %in)
  %out.gep = getelementptr inbounds double, ptr %out.ptr, i64 %iv
  store double %call, ptr %out.gep, align 8
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

  for.end:
  ret void
}

define void @erf_f32(ptr nocapture %in.ptr, ptr %out.ptr) {
; CHECK-LABEL: @erf_f32(
; NEON: [[TMP5:%.*]] = call <4 x float> @armpl_verfq_f32(<4 x float> [[TMP4:%.*]])
; SVE: [[TMP5:%.*]] = call <vscale x 4 x float> @armpl_sverf_f32_x(<vscale x 4 x float> [[TMP4:%.*]], <vscale x 4 x i1> {{.*}})
; CHECK: ret void
;
  entry:
  br label %for.body

  for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %in.gep = getelementptr inbounds float, ptr %in.ptr, i64 %iv
  %in = load float, ptr %in.gep, align 8
  %call = tail call float @erff(float %in)
  %out.gep = getelementptr inbounds float, ptr %out.ptr, i64 %iv
  store float %call, ptr %out.gep, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

  for.end:
  ret void
}

declare double @erfc(double)
declare float @erfcf(float)

define void @erfc_f64(ptr nocapture %in.ptr, ptr %out.ptr) {
; CHECK-LABEL: @erfc_f64(
; NEON:     [[TMP5:%.*]] = call <2 x double> @armpl_verfcq_f64(<2 x double> [[TMP4:%.*]])
; SVE:      [[TMP5:%.*]] = call <vscale x 2 x double> @armpl_sverfc_f64_x(<vscale x 2 x double> [[TMP4:%.*]], <vscale x 2 x i1> {{.*}})
; CHECK:    ret void
;
  entry:
  br label %for.body

  for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %in.gep = getelementptr inbounds double, ptr %in.ptr, i64 %iv
  %in = load double, ptr %in.gep, align 8
  %call = tail call double @erfc(double %in)
  %out.gep = getelementptr inbounds double, ptr %out.ptr, i64 %iv
  store double %call, ptr %out.gep, align 8
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

  for.end:
  ret void
}

define void @erfc_f32(ptr nocapture %in.ptr, ptr %out.ptr) {
; CHECK-LABEL: @erfc_f32(
; NEON: [[TMP5:%.*]] = call <4 x float> @armpl_verfcq_f32(<4 x float> [[TMP4:%.*]])
; SVE: [[TMP5:%.*]] = call <vscale x 4 x float> @armpl_sverfc_f32_x(<vscale x 4 x float> [[TMP4:%.*]], <vscale x 4 x i1> {{.*}})
; CHECK: ret void
;
  entry:
  br label %for.body

  for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %in.gep = getelementptr inbounds float, ptr %in.ptr, i64 %iv
  %in = load float, ptr %in.gep, align 8
  %call = tail call float @erfcf(float %in)
  %out.gep = getelementptr inbounds float, ptr %out.ptr, i64 %iv
  store float %call, ptr %out.gep, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

  for.end:
  ret void
}

declare double @exp(double)
declare float @expf(float)

define void @exp_f64(ptr nocapture %in.ptr, ptr %out.ptr) {
; CHECK-LABEL: @exp_f64(
; NEON:     [[TMP5:%.*]] = call <2 x double> @armpl_vexpq_f64(<2 x double> [[TMP4:%.*]])
; SVE:      [[TMP5:%.*]] = call <vscale x 2 x double> @armpl_svexp_f64_x(<vscale x 2 x double> [[TMP4:%.*]], <vscale x 2 x i1> {{.*}})
; CHECK:    ret void
;
  entry:
  br label %for.body

  for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %in.gep = getelementptr inbounds double, ptr %in.ptr, i64 %iv
  %in = load double, ptr %in.gep, align 8
  %call = tail call double @exp(double %in)
  %out.gep = getelementptr inbounds double, ptr %out.ptr, i64 %iv
  store double %call, ptr %out.gep, align 8
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

  for.end:
  ret void
}

define void @exp_f32(ptr nocapture %in.ptr, ptr %out.ptr) {
; CHECK-LABEL: @exp_f32(
; NEON: [[TMP5:%.*]] = call <4 x float> @armpl_vexpq_f32(<4 x float> [[TMP4:%.*]])
; SVE: [[TMP5:%.*]] = call <vscale x 4 x float> @armpl_svexp_f32_x(<vscale x 4 x float> [[TMP4:%.*]], <vscale x 4 x i1> {{.*}})
; CHECK: ret void
;
  entry:
  br label %for.body

  for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %in.gep = getelementptr inbounds float, ptr %in.ptr, i64 %iv
  %in = load float, ptr %in.gep, align 8
  %call = tail call float @expf(float %in)
  %out.gep = getelementptr inbounds float, ptr %out.ptr, i64 %iv
  store float %call, ptr %out.gep, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

  for.end:
  ret void
}

declare double @exp2(double)
declare float @exp2f(float)

define void @exp2_f64(ptr nocapture %in.ptr, ptr %out.ptr) {
; CHECK-LABEL: @exp2_f64(
; NEON:     [[TMP5:%.*]] = call <2 x double> @armpl_vexp2q_f64(<2 x double> [[TMP4:%.*]])
; SVE:      [[TMP5:%.*]] = call <vscale x 2 x double> @armpl_svexp2_f64_x(<vscale x 2 x double> [[TMP4:%.*]], <vscale x 2 x i1> {{.*}})
; CHECK:    ret void
;
  entry:
  br label %for.body

  for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %in.gep = getelementptr inbounds double, ptr %in.ptr, i64 %iv
  %in = load double, ptr %in.gep, align 8
  %call = tail call double @exp2(double %in)
  %out.gep = getelementptr inbounds double, ptr %out.ptr, i64 %iv
  store double %call, ptr %out.gep, align 8
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

  for.end:
  ret void
}

define void @exp2_f32(ptr nocapture %in.ptr, ptr %out.ptr) {
; CHECK-LABEL: @exp2_f32(
; NEON: [[TMP5:%.*]] = call <4 x float> @armpl_vexp2q_f32(<4 x float> [[TMP4:%.*]])
; SVE: [[TMP5:%.*]] = call <vscale x 4 x float> @armpl_svexp2_f32_x(<vscale x 4 x float> [[TMP4:%.*]], <vscale x 4 x i1> {{.*}})
; CHECK: ret void
;
  entry:
  br label %for.body

  for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %in.gep = getelementptr inbounds float, ptr %in.ptr, i64 %iv
  %in = load float, ptr %in.gep, align 8
  %call = tail call float @exp2f(float %in)
  %out.gep = getelementptr inbounds float, ptr %out.ptr, i64 %iv
  store float %call, ptr %out.gep, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

  for.end:
  ret void
}

declare double @exp10(double)
declare float @exp10f(float)

define void @exp10_f64(ptr nocapture %in.ptr, ptr %out.ptr) {
; CHECK-LABEL: @exp10_f64(
; NEON:     [[TMP5:%.*]] = call <2 x double> @armpl_vexp10q_f64(<2 x double> [[TMP4:%.*]])
; SVE:      [[TMP5:%.*]] = call <vscale x 2 x double> @armpl_svexp10_f64_x(<vscale x 2 x double> [[TMP4:%.*]], <vscale x 2 x i1> {{.*}})
; CHECK:    ret void
;
  entry:
  br label %for.body

  for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %in.gep = getelementptr inbounds double, ptr %in.ptr, i64 %iv
  %in = load double, ptr %in.gep, align 8
  %call = tail call double @exp10(double %in)
  %out.gep = getelementptr inbounds double, ptr %out.ptr, i64 %iv
  store double %call, ptr %out.gep, align 8
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

  for.end:
  ret void
}

define void @exp10_f32(ptr nocapture %in.ptr, ptr %out.ptr) {
; CHECK-LABEL: @exp10_f32(
; NEON: [[TMP5:%.*]] = call <4 x float> @armpl_vexp10q_f32(<4 x float> [[TMP4:%.*]])
; SVE: [[TMP5:%.*]] = call <vscale x 4 x float> @armpl_svexp10_f32_x(<vscale x 4 x float> [[TMP4:%.*]], <vscale x 4 x i1> {{.*}})
; CHECK: ret void
;
  entry:
  br label %for.body

  for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %in.gep = getelementptr inbounds float, ptr %in.ptr, i64 %iv
  %in = load float, ptr %in.gep, align 8
  %call = tail call float @exp10f(float %in)
  %out.gep = getelementptr inbounds float, ptr %out.ptr, i64 %iv
  store float %call, ptr %out.gep, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

  for.end:
  ret void
}

declare double @expm1(double)
declare float @expm1f(float)

define void @expm1_f64(ptr nocapture %in.ptr, ptr %out.ptr) {
; CHECK-LABEL: @expm1_f64(
; NEON:     [[TMP5:%.*]] = call <2 x double> @armpl_vexpm1q_f64(<2 x double> [[TMP4:%.*]])
; SVE:      [[TMP5:%.*]] = call <vscale x 2 x double> @armpl_svexpm1_f64_x(<vscale x 2 x double> [[TMP4:%.*]], <vscale x 2 x i1> {{.*}})
; CHECK:    ret void
;
  entry:
  br label %for.body

  for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %in.gep = getelementptr inbounds double, ptr %in.ptr, i64 %iv
  %in = load double, ptr %in.gep, align 8
  %call = tail call double @expm1(double %in)
  %out.gep = getelementptr inbounds double, ptr %out.ptr, i64 %iv
  store double %call, ptr %out.gep, align 8
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

  for.end:
  ret void
}

define void @expm1_f32(ptr nocapture %in.ptr, ptr %out.ptr) {
; CHECK-LABEL: @expm1_f32(
; NEON: [[TMP5:%.*]] = call <4 x float> @armpl_vexpm1q_f32(<4 x float> [[TMP4:%.*]])
; SVE: [[TMP5:%.*]] = call <vscale x 4 x float> @armpl_svexpm1_f32_x(<vscale x 4 x float> [[TMP4:%.*]], <vscale x 4 x i1> {{.*}})
; CHECK: ret void
;
  entry:
  br label %for.body

  for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %in.gep = getelementptr inbounds float, ptr %in.ptr, i64 %iv
  %in = load float, ptr %in.gep, align 8
  %call = tail call float @expm1f(float %in)
  %out.gep = getelementptr inbounds float, ptr %out.ptr, i64 %iv
  store float %call, ptr %out.gep, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

  for.end:
  ret void
}

declare double @lgamma(double)
declare float @lgammaf(float)

define void @lgamma_f64(ptr nocapture %in.ptr, ptr %out.ptr) {
; CHECK-LABEL: @lgamma_f64(
; NEON:     [[TMP5:%.*]] = call <2 x double> @armpl_vlgammaq_f64(<2 x double> [[TMP4:%.*]])
; SVE:      [[TMP5:%.*]] = call <vscale x 2 x double> @armpl_svlgamma_f64_x(<vscale x 2 x double> [[TMP4:%.*]], <vscale x 2 x i1> {{.*}})
; CHECK:    ret void
;
  entry:
  br label %for.body

  for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %in.gep = getelementptr inbounds double, ptr %in.ptr, i64 %iv
  %in = load double, ptr %in.gep, align 8
  %call = tail call double @lgamma(double %in)
  %out.gep = getelementptr inbounds double, ptr %out.ptr, i64 %iv
  store double %call, ptr %out.gep, align 8
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

  for.end:
  ret void
}

define void @lgamma_f32(ptr nocapture %in.ptr, ptr %out.ptr) {
; CHECK-LABEL: @lgamma_f32(
; NEON: [[TMP5:%.*]] = call <4 x float> @armpl_vlgammaq_f32(<4 x float> [[TMP4:%.*]])
; SVE: [[TMP5:%.*]] = call <vscale x 4 x float> @armpl_svlgamma_f32_x(<vscale x 4 x float> [[TMP4:%.*]], <vscale x 4 x i1> {{.*}})
; CHECK: ret void
;
  entry:
  br label %for.body

  for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %in.gep = getelementptr inbounds float, ptr %in.ptr, i64 %iv
  %in = load float, ptr %in.gep, align 8
  %call = tail call float @lgammaf(float %in)
  %out.gep = getelementptr inbounds float, ptr %out.ptr, i64 %iv
  store float %call, ptr %out.gep, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

  for.end:
  ret void
}

declare double @log(double)
declare float @logf(float)

define void @log_f64(ptr nocapture %in.ptr, ptr %out.ptr) {
; CHECK-LABEL: @log_f64(
; NEON:     [[TMP5:%.*]] = call <2 x double> @armpl_vlogq_f64(<2 x double> [[TMP4:%.*]])
; SVE:      [[TMP5:%.*]] = call <vscale x 2 x double> @armpl_svlog_f64_x(<vscale x 2 x double> [[TMP4:%.*]], <vscale x 2 x i1> {{.*}})
; CHECK:    ret void
;
  entry:
  br label %for.body

  for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %in.gep = getelementptr inbounds double, ptr %in.ptr, i64 %iv
  %in = load double, ptr %in.gep, align 8
  %call = tail call double @log(double %in)
  %out.gep = getelementptr inbounds double, ptr %out.ptr, i64 %iv
  store double %call, ptr %out.gep, align 8
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

  for.end:
  ret void
}

define void @log_f32(ptr nocapture %in.ptr, ptr %out.ptr) {
; CHECK-LABEL: @log_f32(
; NEON: [[TMP5:%.*]] = call <4 x float> @armpl_vlogq_f32(<4 x float> [[TMP4:%.*]])
; SVE: [[TMP5:%.*]] = call <vscale x 4 x float> @armpl_svlog_f32_x(<vscale x 4 x float> [[TMP4:%.*]], <vscale x 4 x i1> {{.*}})
; CHECK: ret void
;
  entry:
  br label %for.body

  for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %in.gep = getelementptr inbounds float, ptr %in.ptr, i64 %iv
  %in = load float, ptr %in.gep, align 8
  %call = tail call float @logf(float %in)
  %out.gep = getelementptr inbounds float, ptr %out.ptr, i64 %iv
  store float %call, ptr %out.gep, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

  for.end:
  ret void
}

declare double @log1p(double)
declare float @log1pf(float)

define void @log1p_f64(ptr nocapture %in.ptr, ptr %out.ptr) {
; CHECK-LABEL: @log1p_f64(
; NEON:     [[TMP5:%.*]] = call <2 x double> @armpl_vlog1pq_f64(<2 x double> [[TMP4:%.*]])
; SVE:      [[TMP5:%.*]] = call <vscale x 2 x double> @armpl_svlog1p_f64_x(<vscale x 2 x double> [[TMP4:%.*]], <vscale x 2 x i1> {{.*}})
; CHECK:    ret void
;
  entry:
  br label %for.body

  for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %in.gep = getelementptr inbounds double, ptr %in.ptr, i64 %iv
  %in = load double, ptr %in.gep, align 8
  %call = tail call double @log1p(double %in)
  %out.gep = getelementptr inbounds double, ptr %out.ptr, i64 %iv
  store double %call, ptr %out.gep, align 8
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

  for.end:
  ret void
}

define void @log1p_f32(ptr nocapture %in.ptr, ptr %out.ptr) {
; CHECK-LABEL: @log1p_f32(
; NEON: [[TMP5:%.*]] = call <4 x float> @armpl_vlog1pq_f32(<4 x float> [[TMP4:%.*]])
; SVE: [[TMP5:%.*]] = call <vscale x 4 x float> @armpl_svlog1p_f32_x(<vscale x 4 x float> [[TMP4:%.*]], <vscale x 4 x i1> {{.*}})
; CHECK: ret void
;
  entry:
  br label %for.body

  for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %in.gep = getelementptr inbounds float, ptr %in.ptr, i64 %iv
  %in = load float, ptr %in.gep, align 8
  %call = tail call float @log1pf(float %in)
  %out.gep = getelementptr inbounds float, ptr %out.ptr, i64 %iv
  store float %call, ptr %out.gep, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

  for.end:
  ret void
}

declare double @log2(double)
declare float @log2f(float)

define void @log2_f64(ptr nocapture %in.ptr, ptr %out.ptr) {
; CHECK-LABEL: @log2_f64(
; NEON:     [[TMP5:%.*]] = call <2 x double> @armpl_vlog2q_f64(<2 x double> [[TMP4:%.*]])
; SVE:      [[TMP5:%.*]] = call <vscale x 2 x double> @armpl_svlog2_f64_x(<vscale x 2 x double> [[TMP4:%.*]], <vscale x 2 x i1> {{.*}})
; CHECK:    ret void
;
  entry:
  br label %for.body

  for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %in.gep = getelementptr inbounds double, ptr %in.ptr, i64 %iv
  %in = load double, ptr %in.gep, align 8
  %call = tail call double @log2(double %in)
  %out.gep = getelementptr inbounds double, ptr %out.ptr, i64 %iv
  store double %call, ptr %out.gep, align 8
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

  for.end:
  ret void
}

define void @log2_f32(ptr nocapture %in.ptr, ptr %out.ptr) {
; CHECK-LABEL: @log2_f32(
; NEON: [[TMP5:%.*]] = call <4 x float> @armpl_vlog2q_f32(<4 x float> [[TMP4:%.*]])
; SVE: [[TMP5:%.*]] = call <vscale x 4 x float> @armpl_svlog2_f32_x(<vscale x 4 x float> [[TMP4:%.*]], <vscale x 4 x i1> {{.*}})
; CHECK: ret void
;
  entry:
  br label %for.body

  for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %in.gep = getelementptr inbounds float, ptr %in.ptr, i64 %iv
  %in = load float, ptr %in.gep, align 8
  %call = tail call float @log2f(float %in)
  %out.gep = getelementptr inbounds float, ptr %out.ptr, i64 %iv
  store float %call, ptr %out.gep, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

  for.end:
  ret void
}

declare double @log10(double)
declare float @log10f(float)

define void @log10_f64(ptr nocapture %in.ptr, ptr %out.ptr) {
; CHECK-LABEL: @log10_f64(
; NEON:     [[TMP5:%.*]] = call <2 x double> @armpl_vlog10q_f64(<2 x double> [[TMP4:%.*]])
; SVE:      [[TMP5:%.*]] = call <vscale x 2 x double> @armpl_svlog10_f64_x(<vscale x 2 x double> [[TMP4:%.*]], <vscale x 2 x i1> {{.*}})
; CHECK:    ret void
;
  entry:
  br label %for.body

  for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %in.gep = getelementptr inbounds double, ptr %in.ptr, i64 %iv
  %in = load double, ptr %in.gep, align 8
  %call = tail call double @log10(double %in)
  %out.gep = getelementptr inbounds double, ptr %out.ptr, i64 %iv
  store double %call, ptr %out.gep, align 8
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

  for.end:
  ret void
}

define void @log10_f32(ptr nocapture %in.ptr, ptr %out.ptr) {
; CHECK-LABEL: @log10_f32(
; NEON: [[TMP5:%.*]] = call <4 x float> @armpl_vlog10q_f32(<4 x float> [[TMP4:%.*]])
; SVE: [[TMP5:%.*]] = call <vscale x 4 x float> @armpl_svlog10_f32_x(<vscale x 4 x float> [[TMP4:%.*]], <vscale x 4 x i1> {{.*}})
; CHECK: ret void
;
  entry:
  br label %for.body

  for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %in.gep = getelementptr inbounds float, ptr %in.ptr, i64 %iv
  %in = load float, ptr %in.gep, align 8
  %call = tail call float @log10f(float %in)
  %out.gep = getelementptr inbounds float, ptr %out.ptr, i64 %iv
  store float %call, ptr %out.gep, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

  for.end:
  ret void
}

declare double @sin(double)
declare float @sinf(float)

define void @sin_f64(ptr nocapture %in.ptr, ptr %out.ptr) {
; CHECK-LABEL: @sin_f64(
; NEON:     [[TMP5:%.*]] = call <2 x double> @armpl_vsinq_f64(<2 x double> [[TMP4:%.*]])
; SVE:      [[TMP5:%.*]] = call <vscale x 2 x double> @armpl_svsin_f64_x(<vscale x 2 x double> [[TMP4:%.*]], <vscale x 2 x i1> {{.*}})
; CHECK:    ret void
;
  entry:
  br label %for.body

  for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %in.gep = getelementptr inbounds double, ptr %in.ptr, i64 %iv
  %in = load double, ptr %in.gep, align 8
  %call = tail call double @sin(double %in)
  %out.gep = getelementptr inbounds double, ptr %out.ptr, i64 %iv
  store double %call, ptr %out.gep, align 8
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

  for.end:
  ret void
}

define void @sin_f32(ptr nocapture %in.ptr, ptr %out.ptr) {
; CHECK-LABEL: @sin_f32(
; NEON: [[TMP5:%.*]] = call <4 x float> @armpl_vsinq_f32(<4 x float> [[TMP4:%.*]])
; SVE: [[TMP5:%.*]] = call <vscale x 4 x float> @armpl_svsin_f32_x(<vscale x 4 x float> [[TMP4:%.*]], <vscale x 4 x i1> {{.*}})
; CHECK: ret void
;
  entry:
  br label %for.body

  for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %in.gep = getelementptr inbounds float, ptr %in.ptr, i64 %iv
  %in = load float, ptr %in.gep, align 8
  %call = tail call float @sinf(float %in)
  %out.gep = getelementptr inbounds float, ptr %out.ptr, i64 %iv
  store float %call, ptr %out.gep, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

  for.end:
  ret void
}

declare double @sinh(double)
declare float @sinhf(float)

define void @sinh_f64(ptr nocapture %in.ptr, ptr %out.ptr) {
; CHECK-LABEL: @sinh_f64(
; NEON:     [[TMP5:%.*]] = call <2 x double> @armpl_vsinhq_f64(<2 x double> [[TMP4:%.*]])
; SVE:      [[TMP5:%.*]] = call <vscale x 2 x double> @armpl_svsinh_f64_x(<vscale x 2 x double> [[TMP4:%.*]], <vscale x 2 x i1> {{.*}})
; CHECK:    ret void
;
  entry:
  br label %for.body

  for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %in.gep = getelementptr inbounds double, ptr %in.ptr, i64 %iv
  %in = load double, ptr %in.gep, align 8
  %call = tail call double @sinh(double %in)
  %out.gep = getelementptr inbounds double, ptr %out.ptr, i64 %iv
  store double %call, ptr %out.gep, align 8
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

  for.end:
  ret void
}

define void @sinh_f32(ptr nocapture %in.ptr, ptr %out.ptr) {
; CHECK-LABEL: @sinh_f32(
; NEON: [[TMP5:%.*]] = call <4 x float> @armpl_vsinhq_f32(<4 x float> [[TMP4:%.*]])
; SVE: [[TMP5:%.*]] = call <vscale x 4 x float> @armpl_svsinh_f32_x(<vscale x 4 x float> [[TMP4:%.*]], <vscale x 4 x i1> {{.*}})
; CHECK: ret void
;
  entry:
  br label %for.body

  for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %in.gep = getelementptr inbounds float, ptr %in.ptr, i64 %iv
  %in = load float, ptr %in.gep, align 8
  %call = tail call float @sinhf(float %in)
  %out.gep = getelementptr inbounds float, ptr %out.ptr, i64 %iv
  store float %call, ptr %out.gep, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

  for.end:
  ret void
}

declare double @sinpi(double)
declare float @sinpif(float)

define void @sinpi_f64(ptr nocapture %in.ptr, ptr %out.ptr) {
; CHECK-LABEL: @sinpi_f64(
; NEON:     [[TMP5:%.*]] = call <2 x double> @armpl_vsinpiq_f64(<2 x double> [[TMP4:%.*]])
; SVE:      [[TMP5:%.*]] = call <vscale x 2 x double> @armpl_svsinpi_f64_x(<vscale x 2 x double> [[TMP4:%.*]], <vscale x 2 x i1> {{.*}})
; CHECK:    ret void
;
  entry:
  br label %for.body

  for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %in.gep = getelementptr inbounds double, ptr %in.ptr, i64 %iv
  %in = load double, ptr %in.gep, align 8
  %call = tail call double @sinpi(double %in)
  %out.gep = getelementptr inbounds double, ptr %out.ptr, i64 %iv
  store double %call, ptr %out.gep, align 8
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

  for.end:
  ret void
}

define void @sinpi_f32(ptr nocapture %in.ptr, ptr %out.ptr) {
; CHECK-LABEL: @sinpi_f32(
; NEON: [[TMP5:%.*]] = call <4 x float> @armpl_vsinpiq_f32(<4 x float> [[TMP4:%.*]])
; SVE: [[TMP5:%.*]] = call <vscale x 4 x float> @armpl_svsinpi_f32_x(<vscale x 4 x float> [[TMP4:%.*]], <vscale x 4 x i1> {{.*}})
; CHECK: ret void
;
  entry:
  br label %for.body

  for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %in.gep = getelementptr inbounds float, ptr %in.ptr, i64 %iv
  %in = load float, ptr %in.gep, align 8
  %call = tail call float @sinpif(float %in)
  %out.gep = getelementptr inbounds float, ptr %out.ptr, i64 %iv
  store float %call, ptr %out.gep, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

  for.end:
  ret void
}

declare double @sqrt(double)
declare float @sqrtf(float)

define void @sqrt_f64(ptr nocapture %in.ptr, ptr %out.ptr) {
; CHECK-LABEL: @sqrt_f64(
; NEON:     [[TMP5:%.*]] = call <2 x double> @armpl_vsqrtq_f64(<2 x double> [[TMP4:%.*]])
; SVE:      [[TMP5:%.*]] = call <vscale x 2 x double> @armpl_svsqrt_f64_x(<vscale x 2 x double> [[TMP4:%.*]], <vscale x 2 x i1> {{.*}})
; CHECK:    ret void
;
  entry:
  br label %for.body

  for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %in.gep = getelementptr inbounds double, ptr %in.ptr, i64 %iv
  %in = load double, ptr %in.gep, align 8
  %call = tail call double @sqrt(double %in)
  %out.gep = getelementptr inbounds double, ptr %out.ptr, i64 %iv
  store double %call, ptr %out.gep, align 8
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

  for.end:
  ret void
}

define void @sqrt_f32(ptr nocapture %in.ptr, ptr %out.ptr) {
; CHECK-LABEL: @sqrt_f32(
; NEON: [[TMP5:%.*]] = call <4 x float> @armpl_vsqrtq_f32(<4 x float> [[TMP4:%.*]])
; SVE: [[TMP5:%.*]] = call <vscale x 4 x float> @armpl_svsqrt_f32_x(<vscale x 4 x float> [[TMP4:%.*]], <vscale x 4 x i1> {{.*}})
; CHECK: ret void
;
  entry:
  br label %for.body

  for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %in.gep = getelementptr inbounds float, ptr %in.ptr, i64 %iv
  %in = load float, ptr %in.gep, align 8
  %call = tail call float @sqrtf(float %in)
  %out.gep = getelementptr inbounds float, ptr %out.ptr, i64 %iv
  store float %call, ptr %out.gep, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

  for.end:
  ret void
}

declare double @tan(double)
declare float @tanf(float)

define void @tan_f64(ptr nocapture %in.ptr, ptr %out.ptr) {
; CHECK-LABEL: @tan_f64(
; NEON:     [[TMP5:%.*]] = call <2 x double> @armpl_vtanq_f64(<2 x double> [[TMP4:%.*]])
; SVE:      [[TMP5:%.*]] = call <vscale x 2 x double> @armpl_svtan_f64_x(<vscale x 2 x double> [[TMP4:%.*]], <vscale x 2 x i1> {{.*}})
; CHECK:    ret void
;
  entry:
  br label %for.body

  for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %in.gep = getelementptr inbounds double, ptr %in.ptr, i64 %iv
  %in = load double, ptr %in.gep, align 8
  %call = tail call double @tan(double %in)
  %out.gep = getelementptr inbounds double, ptr %out.ptr, i64 %iv
  store double %call, ptr %out.gep, align 8
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

  for.end:
  ret void
}

define void @tan_f32(ptr nocapture %in.ptr, ptr %out.ptr) {
; CHECK-LABEL: @tan_f32(
; NEON: [[TMP5:%.*]] = call <4 x float> @armpl_vtanq_f32(<4 x float> [[TMP4:%.*]])
; SVE: [[TMP5:%.*]] = call <vscale x 4 x float> @armpl_svtan_f32_x(<vscale x 4 x float> [[TMP4:%.*]], <vscale x 4 x i1> {{.*}})
; CHECK: ret void
;
  entry:
  br label %for.body

  for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %in.gep = getelementptr inbounds float, ptr %in.ptr, i64 %iv
  %in = load float, ptr %in.gep, align 8
  %call = tail call float @tanf(float %in)
  %out.gep = getelementptr inbounds float, ptr %out.ptr, i64 %iv
  store float %call, ptr %out.gep, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

  for.end:
  ret void
}

declare double @tanh(double)
declare float @tanhf(float)

define void @tanh_f64(ptr nocapture %in.ptr, ptr %out.ptr) {
; CHECK-LABEL: @tanh_f64(
; NEON:     [[TMP5:%.*]] = call <2 x double> @armpl_vtanhq_f64(<2 x double> [[TMP4:%.*]])
; SVE:      [[TMP5:%.*]] = call <vscale x 2 x double> @armpl_svtanh_f64_x(<vscale x 2 x double> [[TMP4:%.*]], <vscale x 2 x i1> {{.*}})
; CHECK:    ret void
;
  entry:
  br label %for.body

  for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %in.gep = getelementptr inbounds double, ptr %in.ptr, i64 %iv
  %in = load double, ptr %in.gep, align 8
  %call = tail call double @tanh(double %in)
  %out.gep = getelementptr inbounds double, ptr %out.ptr, i64 %iv
  store double %call, ptr %out.gep, align 8
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

  for.end:
  ret void
}

define void @tanh_f32(ptr nocapture %in.ptr, ptr %out.ptr) {
; CHECK-LABEL: @tanh_f32(
; NEON: [[TMP5:%.*]] = call <4 x float> @armpl_vtanhq_f32(<4 x float> [[TMP4:%.*]])
; SVE: [[TMP5:%.*]] = call <vscale x 4 x float> @armpl_svtanh_f32_x(<vscale x 4 x float> [[TMP4:%.*]], <vscale x 4 x i1> {{.*}})
; CHECK: ret void
;
  entry:
  br label %for.body

  for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %in.gep = getelementptr inbounds float, ptr %in.ptr, i64 %iv
  %in = load float, ptr %in.gep, align 8
  %call = tail call float @tanhf(float %in)
  %out.gep = getelementptr inbounds float, ptr %out.ptr, i64 %iv
  store float %call, ptr %out.gep, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

  for.end:
  ret void
}

declare double @tgamma(double)
declare float @tgammaf(float)

define void @tgamma_f64(ptr nocapture %in.ptr, ptr %out.ptr) {
; CHECK-LABEL: @tgamma_f64(
; NEON:     [[TMP5:%.*]] = call <2 x double> @armpl_vtgammaq_f64(<2 x double> [[TMP4:%.*]])
; SVE:      [[TMP5:%.*]] = call <vscale x 2 x double> @armpl_svtgamma_f64_x(<vscale x 2 x double> [[TMP4:%.*]], <vscale x 2 x i1> {{.*}})
; CHECK:    ret void
;
  entry:
  br label %for.body

  for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %in.gep = getelementptr inbounds double, ptr %in.ptr, i64 %iv
  %in = load double, ptr %in.gep, align 8
  %call = tail call double @tgamma(double %in)
  %out.gep = getelementptr inbounds double, ptr %out.ptr, i64 %iv
  store double %call, ptr %out.gep, align 8
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

  for.end:
  ret void
}

define void @tgamma_f32(ptr nocapture %in.ptr, ptr %out.ptr) {
; CHECK-LABEL: @tgamma_f32(
; NEON: [[TMP5:%.*]] = call <4 x float> @armpl_vtgammaq_f32(<4 x float> [[TMP4:%.*]])
; SVE: [[TMP5:%.*]] = call <vscale x 4 x float> @armpl_svtgamma_f32_x(<vscale x 4 x float> [[TMP4:%.*]], <vscale x 4 x i1> {{.*}})
; CHECK: ret void
;
  entry:
  br label %for.body

  for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %in.gep = getelementptr inbounds float, ptr %in.ptr, i64 %iv
  %in = load float, ptr %in.gep, align 8
  %call = tail call float @tgammaf(float %in)
  %out.gep = getelementptr inbounds float, ptr %out.ptr, i64 %iv
  store float %call, ptr %out.gep, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

  for.end:
  ret void
}

declare double @atan2(double, double)
declare float @atan2f(float, float)

define void @atan2_f64(ptr nocapture %in.ptr, ptr %out.ptr) {
; CHECK-LABEL: @atan2_f64(
; NEON:     [[TMP5:%.*]] = call <2 x double> @armpl_vatan2q_f64(<2 x double> [[TMP4:%.*]], <2 x double> [[TMP4:%.*]])
; SVE:      [[TMP5:%.*]] = call <vscale x 2 x double> @armpl_svatan2_f64_x(<vscale x 2 x double> [[TMP4:%.*]], <vscale x 2 x double> [[TMP4:%.*]], <vscale x 2 x i1> {{.*}})
; CHECK:    ret void
;
  entry:
  br label %for.body

  for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %in.gep = getelementptr inbounds double, ptr %in.ptr, i64 %iv
  %in = load double, ptr %in.gep, align 8
  %call = tail call double @atan2(double %in, double %in)
  %out.gep = getelementptr inbounds double, ptr %out.ptr, i64 %iv
  store double %call, ptr %out.gep, align 8
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

  for.end:
  ret void
}

define void @atan2_f32(ptr nocapture %in.ptr, ptr %out.ptr) {
; CHECK-LABEL: @atan2_f32(
; NEON: [[TMP5:%.*]] = call <4 x float> @armpl_vatan2q_f32(<4 x float> [[TMP4:%.*]], <4 x float> [[TMP4:%.*]])
; SVE: [[TMP5:%.*]] = call <vscale x 4 x float> @armpl_svatan2_f32_x(<vscale x 4 x float> [[TMP4:%.*]], <vscale x 4 x float> [[TMP4:%.*]], <vscale x 4 x i1> {{.*}})
; CHECK: ret void
;
  entry:
  br label %for.body

  for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %in.gep = getelementptr inbounds float, ptr %in.ptr, i64 %iv
  %in = load float, ptr %in.gep, align 8
  %call = tail call float @atan2f(float %in, float %in)
  %out.gep = getelementptr inbounds float, ptr %out.ptr, i64 %iv
  store float %call, ptr %out.gep, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

  for.end:
  ret void
}

declare double @copysign(double, double)
declare float @copysignf(float, float)

define void @copysign_f64(ptr nocapture %in.ptr, ptr %out.ptr) {
; CHECK-LABEL: @copysign_f64(
; NEON:     [[TMP5:%.*]] = call <2 x double> @armpl_vcopysignq_f64(<2 x double> [[TMP4:%.*]], <2 x double> [[TMP4:%.*]])
; SVE:      [[TMP5:%.*]] = call <vscale x 2 x double> @armpl_svcopysign_f64_x(<vscale x 2 x double> [[TMP4:%.*]], <vscale x 2 x double> [[TMP4:%.*]], <vscale x 2 x i1> {{.*}})
; CHECK:    ret void
;
  entry:
  br label %for.body

  for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %in.gep = getelementptr inbounds double, ptr %in.ptr, i64 %iv
  %in = load double, ptr %in.gep, align 8
  %call = tail call double @copysign(double %in, double %in)
  %out.gep = getelementptr inbounds double, ptr %out.ptr, i64 %iv
  store double %call, ptr %out.gep, align 8
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

  for.end:
  ret void
}

define void @copysign_f32(ptr nocapture %in.ptr, ptr %out.ptr) {
; CHECK-LABEL: @copysign_f32(
; NEON: [[TMP5:%.*]] = call <4 x float> @armpl_vcopysignq_f32(<4 x float> [[TMP4:%.*]], <4 x float> [[TMP4:%.*]])
; SVE: [[TMP5:%.*]] = call <vscale x 4 x float> @armpl_svcopysign_f32_x(<vscale x 4 x float> [[TMP4:%.*]], <vscale x 4 x float> [[TMP4:%.*]], <vscale x 4 x i1> {{.*}})
; CHECK: ret void
;
  entry:
  br label %for.body

  for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %in.gep = getelementptr inbounds float, ptr %in.ptr, i64 %iv
  %in = load float, ptr %in.gep, align 8
  %call = tail call float @copysignf(float %in, float %in)
  %out.gep = getelementptr inbounds float, ptr %out.ptr, i64 %iv
  store float %call, ptr %out.gep, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

  for.end:
  ret void
}

declare double @fdim(double, double)
declare float @fdimf(float, float)

define void @fdim_f64(ptr nocapture %in.ptr, ptr %out.ptr) {
; CHECK-LABEL: @fdim_f64(
; NEON:     [[TMP5:%.*]] = call <2 x double> @armpl_vfdimq_f64(<2 x double> [[TMP4:%.*]], <2 x double> [[TMP4:%.*]])
; SVE:      [[TMP5:%.*]] = call <vscale x 2 x double> @armpl_svfdim_f64_x(<vscale x 2 x double> [[TMP4:%.*]], <vscale x 2 x double> [[TMP4:%.*]], <vscale x 2 x i1> {{.*}})
; CHECK:    ret void
;
  entry:
  br label %for.body

  for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %in.gep = getelementptr inbounds double, ptr %in.ptr, i64 %iv
  %in = load double, ptr %in.gep, align 8
  %call = tail call double @fdim(double %in, double %in)
  %out.gep = getelementptr inbounds double, ptr %out.ptr, i64 %iv
  store double %call, ptr %out.gep, align 8
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

  for.end:
  ret void
}

define void @fdim_f32(ptr nocapture %in.ptr, ptr %out.ptr) {
; CHECK-LABEL: @fdim_f32(
; NEON: [[TMP5:%.*]] = call <4 x float> @armpl_vfdimq_f32(<4 x float> [[TMP4:%.*]], <4 x float> [[TMP4:%.*]])
; SVE: [[TMP5:%.*]] = call <vscale x 4 x float> @armpl_svfdim_f32_x(<vscale x 4 x float> [[TMP4:%.*]], <vscale x 4 x float> [[TMP4:%.*]], <vscale x 4 x i1> {{.*}})
; CHECK: ret void
;
  entry:
  br label %for.body

  for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %in.gep = getelementptr inbounds float, ptr %in.ptr, i64 %iv
  %in = load float, ptr %in.gep, align 8
  %call = tail call float @fdimf(float %in, float %in)
  %out.gep = getelementptr inbounds float, ptr %out.ptr, i64 %iv
  store float %call, ptr %out.gep, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

  for.end:
  ret void
}

declare double @fmin(double, double)
declare float @fminf(float, float)

define void @fmin_f64(ptr nocapture %in.ptr, ptr %out.ptr) {
; CHECK-LABEL: @fmin_f64(
; NEON:     [[TMP5:%.*]] = call <2 x double> @armpl_vfminq_f64(<2 x double> [[TMP4:%.*]], <2 x double> [[TMP4:%.*]])
; SVE:      [[TMP5:%.*]] = call <vscale x 2 x double> @armpl_svfmin_f64_x(<vscale x 2 x double> [[TMP4:%.*]], <vscale x 2 x double> [[TMP4:%.*]], <vscale x 2 x i1> {{.*}})
; CHECK:    ret void
;
  entry:
  br label %for.body

  for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %in.gep = getelementptr inbounds double, ptr %in.ptr, i64 %iv
  %in = load double, ptr %in.gep, align 8
  %call = tail call double @fmin(double %in, double %in)
  %out.gep = getelementptr inbounds double, ptr %out.ptr, i64 %iv
  store double %call, ptr %out.gep, align 8
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

  for.end:
  ret void
}

define void @fmin_f32(ptr nocapture %in.ptr, ptr %out.ptr) {
; CHECK-LABEL: @fmin_f32(
; NEON: [[TMP5:%.*]] = call <4 x float> @armpl_vfminq_f32(<4 x float> [[TMP4:%.*]], <4 x float> [[TMP4:%.*]])
; SVE: [[TMP5:%.*]] = call <vscale x 4 x float> @armpl_svfmin_f32_x(<vscale x 4 x float> [[TMP4:%.*]], <vscale x 4 x float> [[TMP4:%.*]], <vscale x 4 x i1> {{.*}})
; CHECK: ret void
;
  entry:
  br label %for.body

  for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %in.gep = getelementptr inbounds float, ptr %in.ptr, i64 %iv
  %in = load float, ptr %in.gep, align 8
  %call = tail call float @fminf(float %in, float %in)
  %out.gep = getelementptr inbounds float, ptr %out.ptr, i64 %iv
  store float %call, ptr %out.gep, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

  for.end:
  ret void
}

declare double @fmod(double, double)
declare float @fmodf(float, float)

define void @fmod_f64(ptr nocapture %in.ptr, ptr %out.ptr) {
; CHECK-LABEL: @fmod_f64(
; NEON:     [[TMP5:%.*]] = call <2 x double> @armpl_vfmodq_f64(<2 x double> [[TMP4:%.*]], <2 x double> [[TMP4:%.*]])
; SVE:      [[TMP5:%.*]] = call <vscale x 2 x double> @armpl_svfmod_f64_x(<vscale x 2 x double> [[TMP4:%.*]], <vscale x 2 x double> [[TMP4:%.*]], <vscale x 2 x i1> {{.*}})
; CHECK:    ret void
;
  entry:
  br label %for.body

  for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %in.gep = getelementptr inbounds double, ptr %in.ptr, i64 %iv
  %in = load double, ptr %in.gep, align 8
  %call = tail call double @fmod(double %in, double %in)
  %out.gep = getelementptr inbounds double, ptr %out.ptr, i64 %iv
  store double %call, ptr %out.gep, align 8
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

  for.end:
  ret void
}

define void @fmod_f32(ptr nocapture %in.ptr, ptr %out.ptr) {
; CHECK-LABEL: @fmod_f32(
; NEON: [[TMP5:%.*]] = call <4 x float> @armpl_vfmodq_f32(<4 x float> [[TMP4:%.*]], <4 x float> [[TMP4:%.*]])
; SVE: [[TMP5:%.*]] = call <vscale x 4 x float> @armpl_svfmod_f32_x(<vscale x 4 x float> [[TMP4:%.*]], <vscale x 4 x float> [[TMP4:%.*]], <vscale x 4 x i1> {{.*}})
; CHECK: ret void
;
  entry:
  br label %for.body

  for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %in.gep = getelementptr inbounds float, ptr %in.ptr, i64 %iv
  %in = load float, ptr %in.gep, align 8
  %call = tail call float @fmodf(float %in, float %in)
  %out.gep = getelementptr inbounds float, ptr %out.ptr, i64 %iv
  store float %call, ptr %out.gep, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

  for.end:
  ret void
}

declare double @hypot(double, double)
declare float @hypotf(float, float)

define void @hypot_f64(ptr nocapture %in.ptr, ptr %out.ptr) {
; CHECK-LABEL: @hypot_f64(
; NEON:     [[TMP5:%.*]] = call <2 x double> @armpl_vhypotq_f64(<2 x double> [[TMP4:%.*]], <2 x double> [[TMP4:%.*]])
; SVE:      [[TMP5:%.*]] = call <vscale x 2 x double> @armpl_svhypot_f64_x(<vscale x 2 x double> [[TMP4:%.*]], <vscale x 2 x double> [[TMP4:%.*]], <vscale x 2 x i1> {{.*}})
; CHECK:    ret void
;
  entry:
  br label %for.body

  for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %in.gep = getelementptr inbounds double, ptr %in.ptr, i64 %iv
  %in = load double, ptr %in.gep, align 8
  %call = tail call double @hypot(double %in, double %in)
  %out.gep = getelementptr inbounds double, ptr %out.ptr, i64 %iv
  store double %call, ptr %out.gep, align 8
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

  for.end:
  ret void
}

define void @hypot_f32(ptr nocapture %in.ptr, ptr %out.ptr) {
; CHECK-LABEL: @hypot_f32(
; NEON: [[TMP5:%.*]] = call <4 x float> @armpl_vhypotq_f32(<4 x float> [[TMP4:%.*]], <4 x float> [[TMP4:%.*]])
; SVE: [[TMP5:%.*]] = call <vscale x 4 x float> @armpl_svhypot_f32_x(<vscale x 4 x float> [[TMP4:%.*]], <vscale x 4 x float> [[TMP4:%.*]], <vscale x 4 x i1> {{.*}})
; CHECK: ret void
;
  entry:
  br label %for.body

  for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %in.gep = getelementptr inbounds float, ptr %in.ptr, i64 %iv
  %in = load float, ptr %in.gep, align 8
  %call = tail call float @hypotf(float %in, float %in)
  %out.gep = getelementptr inbounds float, ptr %out.ptr, i64 %iv
  store float %call, ptr %out.gep, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

  for.end:
  ret void
}

declare double @nextafter(double, double)
declare float @nextafterf(float, float)

define void @nextafter_f64(ptr nocapture %in.ptr, ptr %out.ptr) {
; CHECK-LABEL: @nextafter_f64(
; NEON:     [[TMP5:%.*]] = call <2 x double> @armpl_vnextafterq_f64(<2 x double> [[TMP4:%.*]], <2 x double> [[TMP4:%.*]])
; SVE:      [[TMP5:%.*]] = call <vscale x 2 x double> @armpl_svnextafter_f64_x(<vscale x 2 x double> [[TMP4:%.*]], <vscale x 2 x double> [[TMP4:%.*]], <vscale x 2 x i1> {{.*}})
; CHECK:    ret void
;
  entry:
  br label %for.body

  for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %in.gep = getelementptr inbounds double, ptr %in.ptr, i64 %iv
  %in = load double, ptr %in.gep, align 8
  %call = tail call double @nextafter(double %in, double %in)
  %out.gep = getelementptr inbounds double, ptr %out.ptr, i64 %iv
  store double %call, ptr %out.gep, align 8
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

  for.end:
  ret void
}

define void @nextafter_f32(ptr nocapture %in.ptr, ptr %out.ptr) {
; CHECK-LABEL: @nextafter_f32(
; NEON: [[TMP5:%.*]] = call <4 x float> @armpl_vnextafterq_f32(<4 x float> [[TMP4:%.*]], <4 x float> [[TMP4:%.*]])
; SVE: [[TMP5:%.*]] = call <vscale x 4 x float> @armpl_svnextafter_f32_x(<vscale x 4 x float> [[TMP4:%.*]], <vscale x 4 x float> [[TMP4:%.*]], <vscale x 4 x i1> {{.*}})
; CHECK: ret void
;
  entry:
  br label %for.body

  for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %in.gep = getelementptr inbounds float, ptr %in.ptr, i64 %iv
  %in = load float, ptr %in.gep, align 8
  %call = tail call float @nextafterf(float %in, float %in)
  %out.gep = getelementptr inbounds float, ptr %out.ptr, i64 %iv
  store float %call, ptr %out.gep, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

  for.end:
  ret void
}

declare double @pow(double, double)
declare float @powf(float, float)

define void @pow_f64(ptr nocapture %in.ptr, ptr %out.ptr) {
; CHECK-LABEL: @pow_f64(
; NEON:     [[TMP5:%.*]] = call <2 x double> @armpl_vpowq_f64(<2 x double> [[TMP4:%.*]], <2 x double> [[TMP4:%.*]])
; SVE:      [[TMP5:%.*]] = call <vscale x 2 x double> @armpl_svpow_f64_x(<vscale x 2 x double> [[TMP4:%.*]], <vscale x 2 x double> [[TMP4:%.*]], <vscale x 2 x i1> {{.*}})
; CHECK:    ret void
;
  entry:
  br label %for.body

  for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %in.gep = getelementptr inbounds double, ptr %in.ptr, i64 %iv
  %in = load double, ptr %in.gep, align 8
  %call = tail call double @pow(double %in, double %in)
  %out.gep = getelementptr inbounds double, ptr %out.ptr, i64 %iv
  store double %call, ptr %out.gep, align 8
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

  for.end:
  ret void
}

define void @pow_f32(ptr nocapture %in.ptr, ptr %out.ptr) {
; CHECK-LABEL: @pow_f32(
; NEON: [[TMP5:%.*]] = call <4 x float> @armpl_vpowq_f32(<4 x float> [[TMP4:%.*]], <4 x float> [[TMP4:%.*]])
; SVE: [[TMP5:%.*]] = call <vscale x 4 x float> @armpl_svpow_f32_x(<vscale x 4 x float> [[TMP4:%.*]], <vscale x 4 x float> [[TMP4:%.*]], <vscale x 4 x i1> {{.*}})
; CHECK: ret void
;
  entry:
  br label %for.body

  for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %in.gep = getelementptr inbounds float, ptr %in.ptr, i64 %iv
  %in = load float, ptr %in.gep, align 8
  %call = tail call float @powf(float %in, float %in)
  %out.gep = getelementptr inbounds float, ptr %out.ptr, i64 %iv
  store float %call, ptr %out.gep, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

  for.end:
  ret void
}

declare double @fma(double, double, double)
declare float @fmaf(float, float, float)

define void @fma_f64(ptr nocapture %in.ptr, ptr %out.ptr) {
; CHECK-LABEL: @fma_f64(
; NEON: [[TMP5:%.*]] = call <2 x double> @armpl_vfmaq_f64(<2 x double> [[TMP4:%.*]], <2 x double> [[TMP4:%.*]], <2 x double> [[TMP4:%.*]])
; SVE: [[TMP5:%.*]] = call <vscale x 2 x double> @armpl_svfma_f64_x(<vscale x 2 x double> [[TMP4:%.*]], <vscale x 2 x double> [[TMP4:%.*]], <vscale x 2 x double> [[TMP4:%.*]], <vscale x 2 x i1> {{.*}})
; CHECK: ret void
;
  entry:
  br label %for.body

  for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %in.gep = getelementptr inbounds double, ptr %in.ptr, i64 %iv
  %in = load double, ptr %in.gep, align 8
  %call = tail call double @fma(double %in, double %in, double %in)
  %out.gep = getelementptr inbounds double, ptr %out.ptr, i64 %iv
  store double %call, ptr %out.gep, align 8
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

  for.end:
  ret void
}

define void @fma_f32(ptr nocapture %in.ptr, ptr %out.ptr) {
; CHECK-LABEL: @fma_f32(
; NEON: [[TMP5:%.*]] = call <4 x float> @armpl_vfmaq_f32(<4 x float> [[TMP4:%.*]], <4 x float> [[TMP4:%.*]])
; SVE: [[TMP5:%.*]] = call <vscale x 4 x float> @armpl_svfma_f32_x(<vscale x 4 x float> [[TMP4:%.*]], <vscale x 4 x float> [[TMP4:%.*]], <vscale x 4 x i1> {{.*}})
; CHECK: ret void
;
  entry:
  br label %for.body

  for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %in.gep = getelementptr inbounds float, ptr %in.ptr, i64 %iv
  %in = load float, ptr %in.gep, align 8
  %call = tail call float @fmaf(float %in, float %in, float %in)
  %out.gep = getelementptr inbounds float, ptr %out.ptr, i64 %iv
  store float %call, ptr %out.gep, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

  for.end:
  ret void
}

