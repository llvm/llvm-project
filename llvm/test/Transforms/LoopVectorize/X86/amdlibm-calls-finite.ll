; RUN: opt -vector-library=AMDLIBM -passes=inject-tli-mappings,loop-vectorize -S < %s | FileCheck %s

; Test to verify that when math headers are built with
; __FINITE_MATH_ONLY__ enabled, causing use of __<func>_finite
; function versions, vectorization can map these to vector versions.

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @exp_f32(ptr nocapture %varray) {
; CHECK-LABEL: @exp_f32
; CHECK: <4 x float> @amd_vrs4_expf
; CHECK: ret
entry:
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %tmp = trunc i64 %indvars.iv to i32
  %conv = sitofp i32 %tmp to float
  %call = tail call fast float @__expf_finite(float %conv)
  %arrayidx = getelementptr inbounds float, ptr %varray, i64 %indvars.iv
  store float %call, ptr %arrayidx, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body, !llvm.loop !4

for.end:                                          ; preds = %for.body
  ret void
}

define void @exp_f64(ptr nocapture %varray) {
; CHECK-LABEL: @exp_f64
; CHECK: <4 x double> @amd_vrd4_exp
; CHECK: ret
entry:
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %tmp = trunc i64 %indvars.iv to i32
  %conv = sitofp i32 %tmp to double
  %call = tail call fast double @__exp_finite(double %conv)
  %arrayidx = getelementptr inbounds double, ptr %varray, i64 %indvars.iv
  store double %call, ptr %arrayidx, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body, !llvm.loop !4

for.end:                                          ; preds = %for.body
  ret void
}

define void @log_f32(ptr nocapture %varray) {
; CHECK-LABEL: @log_f32
; CHECK: <4 x float> @amd_vrs4_logf
; CHECK: ret
entry:
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %tmp = trunc i64 %indvars.iv to i32
  %conv = sitofp i32 %tmp to float
  %call = tail call fast float @__logf_finite(float %conv)
  %arrayidx = getelementptr inbounds float, ptr %varray, i64 %indvars.iv
  store float %call, ptr %arrayidx, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body, !llvm.loop !4

for.end:                                          ; preds = %for.body
  ret void
}

define void @log_f64(ptr nocapture %varray) {
; CHECK-LABEL: @log_f64
; CHECK: <4 x double> @amd_vrd4_log
; CHECK: ret
entry:
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %tmp = trunc i64 %indvars.iv to i32
  %conv = sitofp i32 %tmp to double
  %call = tail call fast double @__log_finite(double %conv)
  %arrayidx = getelementptr inbounds double, ptr %varray, i64 %indvars.iv
  store double %call, ptr %arrayidx, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body, !llvm.loop !4

for.end:                                          ; preds = %for.body
  ret void
}

define void @pow_f32(ptr nocapture %varray, ptr nocapture readonly %exp) {
; CHECK-LABEL: @pow_f32
; CHECK: <4 x float> @amd_vrs4_powf
; CHECK: ret
entry:
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
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

for.end:                                          ; preds = %for.body
  ret void
}

define void @pow_f64(ptr nocapture %varray, ptr nocapture readonly %exp) {
; CHECK-LABEL: @pow_f64
; CHECK: <4 x double> @amd_vrd4_pow
; CHECK: ret
entry:
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
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

for.end:                                          ; preds = %for.body
  ret void
}

define void @exp2f_finite(ptr nocapture %varray) {
; CHECK-LABEL: @exp2f_finite(
; CHECK:    call <4 x float> @amd_vrs4_exp2f(<4 x float> %{{.*}})
; CHECK:    ret void
;
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %tmp = trunc i64 %iv to i32
  %conv = sitofp i32 %tmp to float
  %call = tail call float @__exp2f_finite(float %conv)
  %arrayidx = getelementptr inbounds float, ptr %varray, i64 %iv
  store float %call, ptr %arrayidx, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body, !llvm.loop !4

for.end:
  ret void
}

define void @exp2_finite(ptr nocapture %varray) {
; CHECK-LABEL: @exp2_finite(
; CHECK:    call <4 x double> @amd_vrd4_exp2(<4 x double> {{.*}})
; CHECK:    ret void
;
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %tmp = trunc i64 %iv to i32
  %conv = sitofp i32 %tmp to double
  %call = tail call double @__exp2_finite(double %conv)
  %arrayidx = getelementptr inbounds double, ptr %varray, i64 %iv
  store double %call, ptr %arrayidx, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body, !llvm.loop !4

for.end:
  ret void
}

define void @log2_f32(ptr nocapture %varray) {
; CHECK-LABEL: @log2_f32
; CHECK: <4 x float> @amd_vrs4_log2f
; CHECK: ret
entry:
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %tmp = trunc i64 %indvars.iv to i32
  %conv = sitofp i32 %tmp to float
  %call = tail call fast float @__log2f_finite(float %conv)
  %arrayidx = getelementptr inbounds float, ptr %varray, i64 %indvars.iv
  store float %call, ptr %arrayidx, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body, !llvm.loop !4

for.end:                                          ; preds = %for.body
  ret void
}

define void @log2_f64(ptr nocapture %varray) {
; CHECK-LABEL: @log2_f64
; CHECK: <4 x double> @amd_vrd4_log2
; CHECK: ret
entry:
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %tmp = trunc i64 %indvars.iv to i32
  %conv = sitofp i32 %tmp to double
  %call = tail call fast double @__log2_finite(double %conv)
  %arrayidx = getelementptr inbounds double, ptr %varray, i64 %indvars.iv
  store double %call, ptr %arrayidx, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body, !llvm.loop !4

for.end:                                          ; preds = %for.body
  ret void
}

define void @log10_f32(ptr nocapture %varray) {
; CHECK-LABEL: @log10_f32
; CHECK: <4 x float> @amd_vrs4_log10f
; CHECK: ret
entry:
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %tmp = trunc i64 %indvars.iv to i32
  %conv = sitofp i32 %tmp to float
  %call = tail call fast float @__log10f_finite(float %conv)
  %arrayidx = getelementptr inbounds float, ptr %varray, i64 %indvars.iv
  store float %call, ptr %arrayidx, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body, !llvm.loop !4

for.end:                                          ; preds = %for.body
  ret void
}

define void @log10_finite(ptr nocapture %varray) {
; CHECK-LABEL: @log10_finite(
; CHECK:    call <2 x double> @amd_vrd2_log10(<2 x double> {{.*}})
; CHECK:    ret void
;
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %tmp = trunc i64 %iv to i32
  %conv = sitofp i32 %tmp to double
  %call = tail call double @__log10_finite(double %conv)
  %arrayidx = getelementptr inbounds double, ptr %varray, i64 %iv
  store double %call, ptr %arrayidx, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body, !llvm.loop !1

for.end:
  ret void
}

define void @exp10_finite(ptr nocapture %varray) {
; CHECK-LABEL: @exp10_finite(
; CHECK:    call <2 x double> @amd_vrd2_exp10(<2 x double> {{.*}})
; CHECK:    ret void
;
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %tmp = trunc i64 %iv to i32
  %conv = sitofp i32 %tmp to double
  %call = tail call double @__exp10_finite(double %conv)
  %arrayidx = getelementptr inbounds double, ptr %varray, i64 %iv
  store double %call, ptr %arrayidx, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body, !llvm.loop !1

for.end:
  ret void
}

define void @exp10_f32(ptr nocapture %varray) {
; CHECK-LABEL: @exp10_f32
; CHECK: <4 x float> @amd_vrs4_exp10f
; CHECK: ret
entry:
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %tmp = trunc i64 %indvars.iv to i32
  %conv = sitofp i32 %tmp to float
  %call = tail call fast float @__exp10f_finite(float %conv)
  %arrayidx = getelementptr inbounds float, ptr %varray, i64 %indvars.iv
  store float %call, ptr %arrayidx, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body, !llvm.loop !4

for.end:                                          ; preds = %for.body
  ret void
}

define void @asin_finite(ptr nocapture %varray) {
; CHECK-LABEL: @asin_finite(
; CHECK:    call <8 x double> @amd_vrd8_asin(<8 x double> {{.*}})
; CHECK:    ret void
;
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %tmp = trunc i64 %iv to i32
  %conv = sitofp i32 %tmp to double
  %call = tail call double @__asin_finite(double %conv)
  %arrayidx = getelementptr inbounds double, ptr %varray, i64 %iv
  store double %call, ptr %arrayidx, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body, !llvm.loop !7

for.end:
  ret void
}

define void @asinf_finite(ptr nocapture %varray) {
; CHECK-LABEL: @asinf_finite
; CHECK: <4 x float> @amd_vrs4_asinf
; CHECK: ret
entry:
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %tmp = trunc i64 %indvars.iv to i32
  %conv = sitofp i32 %tmp to float
  %call = tail call fast float @__asinf_finite(float %conv)
  %arrayidx = getelementptr inbounds float, ptr %varray, i64 %indvars.iv
  store float %call, ptr %arrayidx, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body, !llvm.loop !4

for.end:                                          ; preds = %for.body
  ret void
}

define void @acosf_finite(ptr nocapture %varray) {
; CHECK-LABEL: @acosf_finite
; CHECK: <4 x float> @amd_vrs4_acosf
; CHECK: ret
entry:
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %tmp = trunc i64 %indvars.iv to i32
  %conv = sitofp i32 %tmp to float
  %call = tail call fast float @__acosf_finite(float %conv)
  %arrayidx = getelementptr inbounds float, ptr %varray, i64 %indvars.iv
  store float %call, ptr %arrayidx, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body, !llvm.loop !4

for.end:                                          ; preds = %for.body
  ret void
}

!1 = distinct !{!1, !2, !3}
!2 = !{!"llvm.loop.vectorize.width", i32 2}
!3 = !{!"llvm.loop.vectorize.enable", i1 true}

!4 = distinct !{!4, !5, !6}
!5 = !{!"llvm.loop.vectorize.width", i32 4}
!6 = !{!"llvm.loop.vectorize.enable", i1 true}

!7 = distinct !{!7, !8, !9}
!8 = !{!"llvm.loop.vectorize.width", i32 8}
!9 = !{!"llvm.loop.vectorize.enable", i1 true}

declare float @__expf_finite(float) #0
declare double @__exp_finite(double) #0
declare double @__log_finite(double) #0
declare float @__logf_finite(float) #0
declare float @__powf_finite(float, float) #0
declare double @__pow_finite(double, double) #0
declare float @__exp2f_finite(float) #0
declare double @__exp2_finite(double) #0
declare float @__log2f_finite(float) #0
declare double @__log2_finite(double) #0
declare float @__log10f_finite(float) #0
declare double @__log10_finite(double) #0
declare double @__exp10_finite(double) #0
declare float @__exp10f_finite(float) #0
declare double @__asin_finite(double) #0
declare float @__asinf_finite(float) #0
declare float @__acosf_finite(float) #0