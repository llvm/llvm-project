; RUN: opt -vector-library=SVML -inject-tli-mappings -loop-vectorize -S < %s | FileCheck %s

; Test to verify that when math headers are built with
; __FINITE_MATH_ONLY__ enabled, causing use of __<func>_finite
; function versions, vectorization can map these to vector versions.

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare float @__expf_finite(float) #0

; CHECK-LABEL: @exp_f32
; CHECK: <4 x float> @__svml_expf4
; CHECK: ret
define void @exp_f32(float* nocapture %varray) {
entry:
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %tmp = trunc i64 %indvars.iv to i32
  %conv = sitofp i32 %tmp to float
  %call = tail call fast float @__expf_finite(float %conv)
  %arrayidx = getelementptr inbounds float, float* %varray, i64 %indvars.iv
  store float %call, float* %arrayidx, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body, !llvm.loop !1

for.end:                                          ; preds = %for.body
  ret void
}

!1 = distinct !{!1, !2, !3}
!2 = !{!"llvm.loop.vectorize.width", i32 4}
!3 = !{!"llvm.loop.vectorize.enable", i1 true}


declare double @__exp_finite(double) #0

; CHECK-LABEL: @exp_f64
; CHECK: <4 x double> @__svml_exp4
; CHECK: ret
define void @exp_f64(double* nocapture %varray) {
entry:
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %tmp = trunc i64 %indvars.iv to i32
  %conv = sitofp i32 %tmp to double
  %call = tail call fast double @__exp_finite(double %conv)
  %arrayidx = getelementptr inbounds double, double* %varray, i64 %indvars.iv
  store double %call, double* %arrayidx, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body, !llvm.loop !11

for.end:                                          ; preds = %for.body
  ret void
}

!11 = distinct !{!11, !12, !13}
!12 = !{!"llvm.loop.vectorize.width", i32 4}
!13 = !{!"llvm.loop.vectorize.enable", i1 true}




declare float @__logf_finite(float) #0

; CHECK-LABEL: @log_f32
; CHECK: <4 x float> @__svml_logf4
; CHECK: ret
define void @log_f32(float* nocapture %varray) {
entry:
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %tmp = trunc i64 %indvars.iv to i32
  %conv = sitofp i32 %tmp to float
  %call = tail call fast float @__logf_finite(float %conv)
  %arrayidx = getelementptr inbounds float, float* %varray, i64 %indvars.iv
  store float %call, float* %arrayidx, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body, !llvm.loop !21

for.end:                                          ; preds = %for.body
  ret void
}

!21 = distinct !{!21, !22, !23}
!22 = !{!"llvm.loop.vectorize.width", i32 4}
!23 = !{!"llvm.loop.vectorize.enable", i1 true}


declare double @__log_finite(double) #0

; CHECK-LABEL: @log_f64
; CHECK: <4 x double> @__svml_log4
; CHECK: ret
define void @log_f64(double* nocapture %varray) {
entry:
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %tmp = trunc i64 %indvars.iv to i32
  %conv = sitofp i32 %tmp to double
  %call = tail call fast double @__log_finite(double %conv)
  %arrayidx = getelementptr inbounds double, double* %varray, i64 %indvars.iv
  store double %call, double* %arrayidx, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body, !llvm.loop !31

for.end:                                          ; preds = %for.body
  ret void
}

!31 = distinct !{!31, !32, !33}
!32 = !{!"llvm.loop.vectorize.width", i32 4}
!33 = !{!"llvm.loop.vectorize.enable", i1 true}


declare float @__powf_finite(float, float) #0

; CHECK-LABEL: @pow_f32
; CHECK: <4 x float> @__svml_powf4
; CHECK: ret
define void @pow_f32(float* nocapture %varray, float* nocapture readonly %exp) {
entry:
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %tmp = trunc i64 %indvars.iv to i32
  %conv = sitofp i32 %tmp to float
  %arrayidx = getelementptr inbounds float, float* %exp, i64 %indvars.iv
  %tmp1 = load float, float* %arrayidx, align 4
  %tmp2 = tail call fast float @__powf_finite(float %conv, float %tmp1)
  %arrayidx2 = getelementptr inbounds float, float* %varray, i64 %indvars.iv
  store float %tmp2, float* %arrayidx2, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body, !llvm.loop !41

for.end:                                          ; preds = %for.body
  ret void
}

!41 = distinct !{!41, !42, !43}
!42 = !{!"llvm.loop.vectorize.width", i32 4}
!43 = !{!"llvm.loop.vectorize.enable", i1 true}


declare double @__pow_finite(double, double) #0

; CHECK-LABEL: @pow_f64
; CHECK: <4 x double> @__svml_pow4
; CHECK: ret
define void @pow_f64(double* nocapture %varray, double* nocapture readonly %exp) {
entry:
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %tmp = trunc i64 %indvars.iv to i32
  %conv = sitofp i32 %tmp to double
  %arrayidx = getelementptr inbounds double, double* %exp, i64 %indvars.iv
  %tmp1 = load double, double* %arrayidx, align 4
  %tmp2 = tail call fast double @__pow_finite(double %conv, double %tmp1)
  %arrayidx2 = getelementptr inbounds double, double* %varray, i64 %indvars.iv
  store double %tmp2, double* %arrayidx2, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body, !llvm.loop !51

for.end:                                          ; preds = %for.body
  ret void
}

!51 = distinct !{!51, !52, !53}
!52 = !{!"llvm.loop.vectorize.width", i32 4}
!53 = !{!"llvm.loop.vectorize.enable", i1 true}

declare float @__exp2f_finite(float) #0

define void @exp2f_finite(float* nocapture %varray) {
; CHECK-LABEL: @exp2f_finite(
; CHECK:    call <4 x float> @__svml_exp2f4(<4 x float> %{{.*}})
; CHECK:    ret void
;
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %tmp = trunc i64 %iv to i32
  %conv = sitofp i32 %tmp to float
  %call = tail call float @__exp2f_finite(float %conv)
  %arrayidx = getelementptr inbounds float, float* %varray, i64 %iv
  store float %call, float* %arrayidx, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body, !llvm.loop !61

for.end:
  ret void
}

!61 = distinct !{!61, !62, !63}
!62 = !{!"llvm.loop.vectorize.width", i32 4}
!63 = !{!"llvm.loop.vectorize.enable", i1 true}

declare double @__exp2_finite(double) #0

define void @exp2_finite(double* nocapture %varray) {
; CHECK-LABEL: @exp2_finite(
; CHECK:    call <4 x double> @__svml_exp24(<4 x double> {{.*}})
; CHECK:    ret void
;
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %tmp = trunc i64 %iv to i32
  %conv = sitofp i32 %tmp to double
  %call = tail call double @__exp2_finite(double %conv)
  %arrayidx = getelementptr inbounds double, double* %varray, i64 %iv
  store double %call, double* %arrayidx, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body, !llvm.loop !71

for.end:
  ret void
}

!71 = distinct !{!71, !72, !73}
!72 = !{!"llvm.loop.vectorize.width", i32 4}
!73 = !{!"llvm.loop.vectorize.enable", i1 true}
