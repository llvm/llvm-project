; RUN: opt -vector-library=ArmPL -passes=inject-tli-mappings,loop-vectorize -S < %s | FileCheck %s --check-prefixes=CHECK,NEON
; RUN: opt -mattr=+sve -vector-library=ArmPL -passes=inject-tli-mappings,loop-vectorize -S < %s | FileCheck %s --check-prefixes=CHECK,SVE

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64-unknown-linux-gnu"


; Tests are checking if LV can vectorize loops with llvm math intrinsics
; using mappings from TLI for scalable and fixed width vectorization.

declare double @llvm.cos.f64(double)
declare float @llvm.cos.f32(float)

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
  %call = tail call double @llvm.cos.f64(double %in)
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
  %call = tail call float @llvm.cos.f32(float %in)
  %out.gep = getelementptr inbounds float, ptr %out.ptr, i64 %iv
  store float %call, ptr %out.gep, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

  for.end:
  ret void
}

declare double @llvm.exp.f64(double)
declare float @llvm.exp.f32(float)

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
  %call = tail call double @llvm.exp.f64(double %in)
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
  %call = tail call float @llvm.exp.f32(float %in)
  %out.gep = getelementptr inbounds float, ptr %out.ptr, i64 %iv
  store float %call, ptr %out.gep, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

  for.end:
  ret void
}

declare double @llvm.exp2.f64(double)
declare float @llvm.exp2.f32(float)

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
  %call = tail call double @llvm.exp2.f64(double %in)
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
  %call = tail call float @llvm.exp2.f32(float %in)
  %out.gep = getelementptr inbounds float, ptr %out.ptr, i64 %iv
  store float %call, ptr %out.gep, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

  for.end:
  ret void
}

declare double @llvm.log.f64(double)
declare float @llvm.log.f32(float)

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
  %call = tail call double @llvm.log.f64(double %in)
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
  %call = tail call float @llvm.log.f32(float %in)
  %out.gep = getelementptr inbounds float, ptr %out.ptr, i64 %iv
  store float %call, ptr %out.gep, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

  for.end:
  ret void
}

declare double @llvm.log2.f64(double)
declare float @llvm.log2.f32(float)

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
  %call = tail call double @llvm.log2.f64(double %in)
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
  %call = tail call float @llvm.log2.f32(float %in)
  %out.gep = getelementptr inbounds float, ptr %out.ptr, i64 %iv
  store float %call, ptr %out.gep, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

  for.end:
  ret void
}

declare double @llvm.log10.f64(double)
declare float @llvm.log10.f32(float)

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
  %call = tail call double @llvm.log10.f64(double %in)
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
  %call = tail call float @llvm.log10.f32(float %in)
  %out.gep = getelementptr inbounds float, ptr %out.ptr, i64 %iv
  store float %call, ptr %out.gep, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

  for.end:
  ret void
}

declare double @llvm.sin.f64(double)
declare float @llvm.sin.f32(float)

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
  %call = tail call double @llvm.sin.f64(double %in)
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
  %call = tail call float @llvm.sin.f32(float %in)
  %out.gep = getelementptr inbounds float, ptr %out.ptr, i64 %iv
  store float %call, ptr %out.gep, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

  for.end:
  ret void
}

declare double @llvm.pow.f64(double, double)
declare float @llvm.pow.f32(float, float)

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
  %call = tail call double @llvm.pow.f64(double %in, double %in)
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
  %call = tail call float @llvm.pow.f32(float %in, float %in)
  %out.gep = getelementptr inbounds float, ptr %out.ptr, i64 %iv
  store float %call, ptr %out.gep, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

  for.end:
  ret void
}

