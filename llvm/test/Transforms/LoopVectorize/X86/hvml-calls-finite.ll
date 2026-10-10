; RUN: opt -vector-library=HVML -passes=inject-tli-mappings,loop-vectorize -S < %s | FileCheck %s

; Test to verify that when math headers are built with
; __FINITE_MATH_ONLY__ enabled, causing use of __<func>_finite
; function versions, vectorization can map these to vector versions.

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @asin_f64(ptr nocapture %varray) {
; CHECK-LABEL: @asin_f64
; CHECK: call fast <4 x double> @hvml_vd4_asin(<4 x double> {{.*}})
; CHECK: ret
entry:
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %tmp = trunc i64 %indvars.iv to i32
  %conv = sitofp i32 %tmp to double
  %call = tail call fast double @__asin_finite(double %conv)
  %arrayidx = getelementptr inbounds double, ptr %varray, i64 %indvars.iv
  store double %call, ptr %arrayidx, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body, !llvm.loop !4

for.end:
  ret void
}


define void @asin_f32(ptr nocapture %varray) {
; CHECK-LABEL: @asin_f32
; CHECK: call fast <4 x float> @hvml_vs4_asin(<4 x float> {{.*}})
; CHECK: ret
entry:
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %tmp = trunc i64 %indvars.iv to i32
  %conv = sitofp i32 %tmp to float
  %call = tail call fast float @__asinf_finite(float %conv)
  %arrayidx = getelementptr inbounds float, ptr %varray, i64 %indvars.iv
  store float %call, ptr %arrayidx, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body, !llvm.loop !4

for.end:
  ret void
}


define void @acos_f64(ptr nocapture %varray) {
; CHECK-LABEL: @acos_f64
; CHECK: call fast <4 x double> @hvml_vd4_acos(<4 x double> {{.*}})
; CHECK: ret
entry:
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %tmp = trunc i64 %indvars.iv to i32
  %conv = sitofp i32 %tmp to double
  %call = tail call fast double @__acos_finite(double %conv)
  %arrayidx = getelementptr inbounds double, ptr %varray, i64 %indvars.iv
  store double %call, ptr %arrayidx, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body, !llvm.loop !4

for.end:
  ret void
}


define void @acos_f32(ptr nocapture %varray) {
; CHECK-LABEL: @acos_f32
; CHECK: call fast <4 x float> @hvml_vs4_acos(<4 x float> {{.*}})
; CHECK: ret
entry:
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %tmp = trunc i64 %indvars.iv to i32
  %conv = sitofp i32 %tmp to float
  %call = tail call fast float @__acosf_finite(float %conv)
  %arrayidx = getelementptr inbounds float, ptr %varray, i64 %indvars.iv
  store float %call, ptr %arrayidx, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body, !llvm.loop !4

for.end:
  ret void
}


define void @sinh_f64(ptr nocapture %varray) {
; CHECK-LABEL: @sinh_f64
; CHECK: call fast <4 x double> @hvml_vd4_sinh(<4 x double> {{.*}})
; CHECK: ret
entry:
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %tmp = trunc i64 %indvars.iv to i32
  %conv = sitofp i32 %tmp to double
  %call = tail call fast double @__sinh_finite(double %conv)
  %arrayidx = getelementptr inbounds double, ptr %varray, i64 %indvars.iv
  store double %call, ptr %arrayidx, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body, !llvm.loop !4

for.end:
  ret void
}


define void @sinh_f32(ptr nocapture %varray) {
; CHECK-LABEL: @sinh_f32
; CHECK: call fast <4 x float> @hvml_vs4_sinh(<4 x float> {{.*}})
; CHECK: ret
entry:
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %tmp = trunc i64 %indvars.iv to i32
  %conv = sitofp i32 %tmp to float
  %call = tail call fast float @__sinhf_finite(float %conv)
  %arrayidx = getelementptr inbounds float, ptr %varray, i64 %indvars.iv
  store float %call, ptr %arrayidx, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body, !llvm.loop !4

for.end:
  ret void
}


define void @cosh_f64(ptr nocapture %varray) {
; CHECK-LABEL: @cosh_f64
; CHECK: call fast <4 x double> @hvml_vd4_cosh(<4 x double> {{.*}})
; CHECK: ret
entry:
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %tmp = trunc i64 %indvars.iv to i32
  %conv = sitofp i32 %tmp to double
  %call = tail call fast double @__cosh_finite(double %conv)
  %arrayidx = getelementptr inbounds double, ptr %varray, i64 %indvars.iv
  store double %call, ptr %arrayidx, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body, !llvm.loop !4

for.end:
  ret void
}


define void @cosh_f32(ptr nocapture %varray) {
; CHECK-LABEL: @cosh_f32
; CHECK: call fast <4 x float> @hvml_vs4_cosh(<4 x float> {{.*}})
; CHECK: ret
entry:
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %tmp = trunc i64 %indvars.iv to i32
  %conv = sitofp i32 %tmp to float
  %call = tail call fast float @__coshf_finite(float %conv)
  %arrayidx = getelementptr inbounds float, ptr %varray, i64 %indvars.iv
  store float %call, ptr %arrayidx, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body, !llvm.loop !4

for.end:
  ret void
}


define void @exp_f64(ptr nocapture %varray) {
; CHECK-LABEL: @exp_f64
; CHECK: call fast <4 x double> @hvml_vd4_exp(<4 x double> {{.*}})
; CHECK: ret
entry:
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %tmp = trunc i64 %indvars.iv to i32
  %conv = sitofp i32 %tmp to double
  %call = tail call fast double @__exp_finite(double %conv)
  %arrayidx = getelementptr inbounds double, ptr %varray, i64 %indvars.iv
  store double %call, ptr %arrayidx, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body, !llvm.loop !4

for.end:
  ret void
}


define void @exp_f32(ptr nocapture %varray) {
; CHECK-LABEL: @exp_f32
; CHECK: call fast <4 x float> @hvml_vs4_exp(<4 x float> {{.*}})
; CHECK: ret
entry:
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %tmp = trunc i64 %indvars.iv to i32
  %conv = sitofp i32 %tmp to float
  %call = tail call fast float @__expf_finite(float %conv)
  %arrayidx = getelementptr inbounds float, ptr %varray, i64 %indvars.iv
  store float %call, ptr %arrayidx, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body, !llvm.loop !4

for.end:
  ret void
}


define void @exp2_f64(ptr nocapture %varray) {
; CHECK-LABEL: @exp2_f64
; CHECK: call fast <4 x double> @hvml_vd4_exp2(<4 x double> {{.*}})
; CHECK: ret
entry:
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %tmp = trunc i64 %indvars.iv to i32
  %conv = sitofp i32 %tmp to double
  %call = tail call fast double @__exp2_finite(double %conv)
  %arrayidx = getelementptr inbounds double, ptr %varray, i64 %indvars.iv
  store double %call, ptr %arrayidx, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body, !llvm.loop !4

for.end:
  ret void
}


define void @exp2_f32(ptr nocapture %varray) {
; CHECK-LABEL: @exp2_f32
; CHECK: call fast <4 x float> @hvml_vs4_exp2(<4 x float> {{.*}})
; CHECK: ret
entry:
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %tmp = trunc i64 %indvars.iv to i32
  %conv = sitofp i32 %tmp to float
  %call = tail call fast float @__exp2f_finite(float %conv)
  %arrayidx = getelementptr inbounds float, ptr %varray, i64 %indvars.iv
  store float %call, ptr %arrayidx, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body, !llvm.loop !4

for.end:
  ret void
}


define void @log_f64(ptr nocapture %varray) {
; CHECK-LABEL: @log_f64
; CHECK: call fast <4 x double> @hvml_vd4_ln(<4 x double> {{.*}})
; CHECK: ret
entry:
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %tmp = trunc i64 %indvars.iv to i32
  %conv = sitofp i32 %tmp to double
  %call = tail call fast double @__log_finite(double %conv)
  %arrayidx = getelementptr inbounds double, ptr %varray, i64 %indvars.iv
  store double %call, ptr %arrayidx, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body, !llvm.loop !4

for.end:
  ret void
}


define void @log_f32(ptr nocapture %varray) {
; CHECK-LABEL: @log_f32
; CHECK: call fast <4 x float> @hvml_vs4_ln(<4 x float> {{.*}})
; CHECK: ret
entry:
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %tmp = trunc i64 %indvars.iv to i32
  %conv = sitofp i32 %tmp to float
  %call = tail call fast float @__logf_finite(float %conv)
  %arrayidx = getelementptr inbounds float, ptr %varray, i64 %indvars.iv
  store float %call, ptr %arrayidx, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body, !llvm.loop !4

for.end:
  ret void
}


define void @log2_f64(ptr nocapture %varray) {
; CHECK-LABEL: @log2_f64
; CHECK: call fast <4 x double> @hvml_vd4_log2(<4 x double> {{.*}})
; CHECK: ret
entry:
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %tmp = trunc i64 %indvars.iv to i32
  %conv = sitofp i32 %tmp to double
  %call = tail call fast double @__log2_finite(double %conv)
  %arrayidx = getelementptr inbounds double, ptr %varray, i64 %indvars.iv
  store double %call, ptr %arrayidx, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body, !llvm.loop !4

for.end:
  ret void
}


define void @log2_f32(ptr nocapture %varray) {
; CHECK-LABEL: @log2_f32
; CHECK: call fast <4 x float> @hvml_vs4_log2(<4 x float> {{.*}})
; CHECK: ret
entry:
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %tmp = trunc i64 %indvars.iv to i32
  %conv = sitofp i32 %tmp to float
  %call = tail call fast float @__log2f_finite(float %conv)
  %arrayidx = getelementptr inbounds float, ptr %varray, i64 %indvars.iv
  store float %call, ptr %arrayidx, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body, !llvm.loop !4

for.end:
  ret void
}


define void @log10_f64(ptr nocapture %varray) {
; CHECK-LABEL: @log10_f64
; CHECK: call fast <4 x double> @hvml_vd4_log10(<4 x double> {{.*}})
; CHECK: ret
entry:
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %tmp = trunc i64 %indvars.iv to i32
  %conv = sitofp i32 %tmp to double
  %call = tail call fast double @__log10_finite(double %conv)
  %arrayidx = getelementptr inbounds double, ptr %varray, i64 %indvars.iv
  store double %call, ptr %arrayidx, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body, !llvm.loop !4

for.end:
  ret void
}


define void @log10_f32(ptr nocapture %varray) {
; CHECK-LABEL: @log10_f32
; CHECK: call fast <4 x float> @hvml_vs4_log10(<4 x float> {{.*}})
; CHECK: ret
entry:
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %tmp = trunc i64 %indvars.iv to i32
  %conv = sitofp i32 %tmp to float
  %call = tail call fast float @__log10f_finite(float %conv)
  %arrayidx = getelementptr inbounds float, ptr %varray, i64 %indvars.iv
  store float %call, ptr %arrayidx, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body, !llvm.loop !4

for.end:
  ret void
}


define void @pow_f64(ptr nocapture %varray, ptr nocapture readonly %exp) {
; CHECK-LABEL: @pow_f64
; CHECK: call fast <4 x double> @hvml_vd4_pow(<4 x double> {{.*}}, <4 x double> {{.*}})
; CHECK: ret
entry:
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %tmp = trunc i64 %indvars.iv to i32
  %conv = sitofp i32 %tmp to double
  %arrayidx = getelementptr inbounds double, ptr %exp, i64 %indvars.iv
  %tmp1 = load double, ptr %arrayidx, align 4
  %tmp2 = tail call fast double @__pow_finite(double %conv, double %tmp1)
  %arrayidx2 = getelementptr inbounds double, ptr %varray, i64 %indvars.iv
  store double %tmp2, ptr %arrayidx2, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body, !llvm.loop !4

for.end:
  ret void
}


define void @pow_f32(ptr nocapture %varray, ptr nocapture readonly %exp) {
; CHECK-LABEL: @pow_f32
; CHECK: call fast <4 x float> @hvml_vs4_pow(<4 x float> {{.*}}, <4 x float> {{.*}})
; CHECK: ret
entry:
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %tmp = trunc i64 %indvars.iv to i32
  %conv = sitofp i32 %tmp to float
  %arrayidx = getelementptr inbounds float, ptr %exp, i64 %indvars.iv
  %tmp1 = load float, ptr %arrayidx, align 4
  %tmp2 = tail call fast float @__powf_finite(float %conv, float %tmp1)
  %arrayidx2 = getelementptr inbounds float, ptr %varray, i64 %indvars.iv
  store float %tmp2, ptr %arrayidx2, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body, !llvm.loop !4

for.end:
  ret void
}


define void @sqrt_f64(ptr nocapture %varray) {
; CHECK-LABEL: @sqrt_f64
; CHECK: call fast <4 x double> @hvml_vd4_sqrt(<4 x double> {{.*}})
; CHECK: ret
entry:
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %tmp = trunc i64 %indvars.iv to i32
  %conv = sitofp i32 %tmp to double
  %call = tail call fast double @__sqrt_finite(double %conv)
  %arrayidx = getelementptr inbounds double, ptr %varray, i64 %indvars.iv
  store double %call, ptr %arrayidx, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body, !llvm.loop !4

for.end:
  ret void
}


define void @sqrt_f32(ptr nocapture %varray) {
; CHECK-LABEL: @sqrt_f32
; CHECK: call fast <4 x float> @hvml_vs4_sqrt(<4 x float> {{.*}})
; CHECK: ret
entry:
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %tmp = trunc i64 %indvars.iv to i32
  %conv = sitofp i32 %tmp to float
  %call = tail call fast float @__sqrtf_finite(float %conv)
  %arrayidx = getelementptr inbounds float, ptr %varray, i64 %indvars.iv
  store float %call, ptr %arrayidx, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body, !llvm.loop !4

for.end:
  ret void
}


define void @asin_vf2_f64(ptr nocapture %varray) {
; CHECK-LABEL: @asin_vf2_f64
; CHECK: call fast <2 x double> @hvml_vd2_asin(<2 x double> {{.*}})
; CHECK: ret
entry:
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %tmp = trunc i64 %indvars.iv to i32
  %conv = sitofp i32 %tmp to double
  %call = tail call fast double @__asin_finite(double %conv)
  %arrayidx = getelementptr inbounds double, ptr %varray, i64 %indvars.iv
  store double %call, ptr %arrayidx, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body, !llvm.loop !1

for.end:
  ret void
}


define void @acos_vf2_f64(ptr nocapture %varray) {
; CHECK-LABEL: @acos_vf2_f64
; CHECK: call fast <2 x double> @hvml_vd2_acos(<2 x double> {{.*}})
; CHECK: ret
entry:
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %tmp = trunc i64 %indvars.iv to i32
  %conv = sitofp i32 %tmp to double
  %call = tail call fast double @__acos_finite(double %conv)
  %arrayidx = getelementptr inbounds double, ptr %varray, i64 %indvars.iv
  store double %call, ptr %arrayidx, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body, !llvm.loop !1

for.end:
  ret void
}


define void @exp_vf8_f32(ptr nocapture %varray) {
; CHECK-LABEL: @exp_vf8_f32
; CHECK: call fast <8 x float> @hvml_vs8_exp(<8 x float> {{.*}})
; CHECK: ret
entry:
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %tmp = trunc i64 %indvars.iv to i32
  %conv = sitofp i32 %tmp to float
  %call = tail call fast float @__expf_finite(float %conv)
  %arrayidx = getelementptr inbounds float, ptr %varray, i64 %indvars.iv
  store float %call, ptr %arrayidx, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body, !llvm.loop !7

for.end:
  ret void
}


define void @log2_vf8_f32(ptr nocapture %varray) {
; CHECK-LABEL: @log2_vf8_f32
; CHECK: call fast <8 x float> @hvml_vs8_log2(<8 x float> {{.*}})
; CHECK: ret
entry:
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %tmp = trunc i64 %indvars.iv to i32
  %conv = sitofp i32 %tmp to float
  %call = tail call fast float @__log2f_finite(float %conv)
  %arrayidx = getelementptr inbounds float, ptr %varray, i64 %indvars.iv
  store float %call, ptr %arrayidx, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body, !llvm.loop !7

for.end:
  ret void
}



!1 = distinct !{!1, !2, !3}
!4 = distinct !{!4, !5, !6}
!7 = distinct !{!7, !8, !9}
!2 = !{!"llvm.loop.vectorize.width", i32 2}
!3 = !{!"llvm.loop.vectorize.enable"}
!5 = !{!"llvm.loop.vectorize.width", i32 4}
!6 = !{!"llvm.loop.vectorize.enable"}
!8 = !{!"llvm.loop.vectorize.width", i32 8}
!9 = !{!"llvm.loop.vectorize.enable"}

attributes #0 = { nounwind readnone }

declare double @__asin_finite(double) #0
declare float @__asinf_finite(float) #0
declare double @__acos_finite(double) #0
declare float @__acosf_finite(float) #0
declare double @__sinh_finite(double) #0
declare float @__sinhf_finite(float) #0
declare double @__cosh_finite(double) #0
declare float @__coshf_finite(float) #0
declare double @__exp_finite(double) #0
declare float @__expf_finite(float) #0
declare double @__exp2_finite(double) #0
declare float @__exp2f_finite(float) #0
declare double @__log_finite(double) #0
declare float @__logf_finite(float) #0
declare double @__log2_finite(double) #0
declare float @__log2f_finite(float) #0
declare double @__log10_finite(double) #0
declare float @__log10f_finite(float) #0
declare double @__pow_finite(double, double) #0
declare float @__powf_finite(float, float) #0
declare double @__sqrt_finite(double) #0
declare float @__sqrtf_finite(float) #0
