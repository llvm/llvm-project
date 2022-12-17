; RUN: opt -vector-library=LIBMVEC-X86 -passes=inject-tli-mappings,loop-vectorize -S < %s | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @sin_f64(double* nocapture %varray) {
; CHECK-LABEL: @sin_f64(
; CHECK-LABEL:    vector.body
; CHECK:    [[TMP5:%.*]] = call <4 x double> @_ZGVdN4v_sin(<4 x double> [[TMP4:%.*]])
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
  br i1 %exitcond, label %for.end, label %for.body, !llvm.loop !1

for.end:
  ret void
}

!1 = distinct !{!1, !2, !3}
!2 = !{!"llvm.loop.vectorize.width", i32 4}
!3 = !{!"llvm.loop.vectorize.enable", i1 true}


define void @sin_f32(float* nocapture %varray) {
; CHECK-LABEL: @sin_f32(
; CHECK-LABEL:    vector.body
; CHECK:    [[TMP5:%.*]] = call <4 x float> @_ZGVbN4v_sinf(<4 x float> [[TMP4:%.*]])
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
  br i1 %exitcond, label %for.end, label %for.body, !llvm.loop !21

for.end:
  ret void
}

!21 = distinct !{!21, !22, !23}
!22 = !{!"llvm.loop.vectorize.width", i32 4}
!23 = !{!"llvm.loop.vectorize.enable", i1 true}

define void @sin_f64_intrinsic(double* nocapture %varray) {
; CHECK-LABEL: @sin_f64_intrinsic(
; CHECK-LABEL:    vector.body
; CHECK:    [[TMP5:%.*]] = call <4 x double> @_ZGVdN4v_sin(<4 x double> [[TMP4:%.*]])
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
  br i1 %exitcond, label %for.end, label %for.body, !llvm.loop !31

for.end:
  ret void
}

!31 = distinct !{!31, !32, !33}
!32 = !{!"llvm.loop.vectorize.width", i32 4}
!33 = !{!"llvm.loop.vectorize.enable", i1 true}

define void @sin_f32_intrinsic(float* nocapture %varray) {
; CHECK-LABEL: @sin_f32_intrinsic(
; CHECK-LABEL:    vector.body
; CHECK:    [[TMP5:%.*]] = call <4 x float> @_ZGVbN4v_sinf(<4 x float> [[TMP4:%.*]])
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
  br i1 %exitcond, label %for.end, label %for.body, !llvm.loop !41

for.end:
  ret void
}

!41 = distinct !{!41, !42, !43}
!42 = !{!"llvm.loop.vectorize.width", i32 4}
!43 = !{!"llvm.loop.vectorize.enable", i1 true}

define void @cos_f64(double* nocapture %varray) {
; CHECK-LABEL: @cos_f64(
; CHECK-LABEL:    vector.body
; CHECK:    [[TMP5:%.*]] = call <4 x double> @_ZGVdN4v_cos(<4 x double> [[TMP4:%.*]])
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
  br i1 %exitcond, label %for.end, label %for.body, !llvm.loop !51

for.end:
  ret void
}

!51 = distinct !{!51, !52, !53}
!52 = !{!"llvm.loop.vectorize.width", i32 4}
!53 = !{!"llvm.loop.vectorize.enable", i1 true}

define void @cos_f32(float* nocapture %varray) {
; CHECK-LABEL: @cos_f32(
; CHECK-LABEL:    vector.body
; CHECK:    [[TMP5:%.*]] = call <4 x float> @_ZGVbN4v_cosf(<4 x float> [[TMP4:%.*]])
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
  br i1 %exitcond, label %for.end, label %for.body, !llvm.loop !61

for.end:
  ret void
}

!61 = distinct !{!61, !62, !63}
!62 = !{!"llvm.loop.vectorize.width", i32 4}
!63 = !{!"llvm.loop.vectorize.enable", i1 true}

define void @cos_f64_intrinsic(double* nocapture %varray) {
; CHECK-LABEL: @cos_f64_intrinsic(
; CHECK-LABEL:    vector.body
; CHECK:    [[TMP5:%.*]] = call <4 x double> @_ZGVdN4v_cos(<4 x double> [[TMP4:%.*]])
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
  br i1 %exitcond, label %for.end, label %for.body, !llvm.loop !71

for.end:
  ret void
}

!71 = distinct !{!71, !72, !73}
!72 = !{!"llvm.loop.vectorize.width", i32 4}
!73 = !{!"llvm.loop.vectorize.enable", i1 true}

define void @cos_f32_intrinsic(float* nocapture %varray) {
; CHECK-LABEL: @cos_f32_intrinsic(
; CHECK-LABEL:    vector.body
; CHECK:    [[TMP5:%.*]] = call <4 x float> @_ZGVbN4v_cosf(<4 x float> [[TMP4:%.*]])
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
  br i1 %exitcond, label %for.end, label %for.body, !llvm.loop !81

for.end:
  ret void
}

!81 = distinct !{!81, !82, !83}
!82 = !{!"llvm.loop.vectorize.width", i32 4}
!83 = !{!"llvm.loop.vectorize.enable", i1 true}


define void @exp_f32(float* nocapture %varray) {
; CHECK-LABEL: @exp_f32
; CHECK-LABEL:    vector.body
; CHECK: <4 x float> @_ZGVbN4v_expf
entry:
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %tmp = trunc i64 %indvars.iv to i32
  %conv = sitofp i32 %tmp to float
  %call = tail call fast float @expf(float %conv)
  %arrayidx = getelementptr inbounds float, float* %varray, i64 %indvars.iv
  store float %call, float* %arrayidx, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body, !llvm.loop !91

for.end:                                          ; preds = %for.body
  ret void
}

!91 = distinct !{!91, !92, !93}
!92 = !{!"llvm.loop.vectorize.width", i32 4}
!93 = !{!"llvm.loop.vectorize.enable", i1 true}

define void @exp_f32_intrin(float* nocapture %varray) {
; CHECK-LABEL: @exp_f32_intrin
; CHECK-LABEL: vector.body
; CHECK: <4 x float> @_ZGVbN4v_expf
entry:
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %tmp = trunc i64 %indvars.iv to i32
  %conv = sitofp i32 %tmp to float
  %call = tail call fast float @llvm.exp.f32(float %conv)
  %arrayidx = getelementptr inbounds float, float* %varray, i64 %indvars.iv
  store float %call, float* %arrayidx, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body, !llvm.loop !101

for.end:                                          ; preds = %for.body
  ret void
}

!101 = distinct !{!101, !102, !103}
!102 = !{!"llvm.loop.vectorize.width", i32 4}
!103 = !{!"llvm.loop.vectorize.enable", i1 true}


define void @log_f32(float* nocapture %varray) {
; CHECK-LABEL: @log_f32
; CHECK-LABEL: vector.body
; CHECK: <4 x float> @_ZGVbN4v_logf
entry:
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %tmp = trunc i64 %indvars.iv to i32
  %conv = sitofp i32 %tmp to float
  %call = tail call fast float @logf(float %conv)
  %arrayidx = getelementptr inbounds float, float* %varray, i64 %indvars.iv
  store float %call, float* %arrayidx, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body, !llvm.loop !111

for.end:                                          ; preds = %for.body
  ret void
}

!111 = distinct !{!111, !112, !113}
!112 = !{!"llvm.loop.vectorize.width", i32 4}
!113 = !{!"llvm.loop.vectorize.enable", i1 true}

define void @pow_f32(float* nocapture %varray, float* nocapture readonly %exp) {
; CHECK-LABEL: @pow_f32
; CHECK-LABEL:    vector.body
; CHECK: <4 x float> @_ZGVbN4vv_powf
entry:
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %tmp = trunc i64 %indvars.iv to i32
  %conv = sitofp i32 %tmp to float
  %arrayidx = getelementptr inbounds float, float* %exp, i64 %indvars.iv
  %tmp1 = load float, float* %arrayidx, align 4
  %tmp2 = tail call fast float @powf(float %conv, float %tmp1)
  %arrayidx2 = getelementptr inbounds float, float* %varray, i64 %indvars.iv
  store float %tmp2, float* %arrayidx2, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body, !llvm.loop !121

for.end:                                          ; preds = %for.body
  ret void
}

!121 = distinct !{!121, !122, !123}
!122 = !{!"llvm.loop.vectorize.width", i32 4}
!123 = !{!"llvm.loop.vectorize.enable", i1 true}

define void @pow_f32_intrin(float* nocapture %varray, float* nocapture readonly %exp) {
; CHECK-LABEL: @pow_f32_intrin
; CHECK-LABEL:    vector.body
; CHECK: <4 x float> @_ZGVbN4vv_powf
entry:
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %tmp = trunc i64 %indvars.iv to i32
  %conv = sitofp i32 %tmp to float
  %arrayidx = getelementptr inbounds float, float* %exp, i64 %indvars.iv
  %tmp1 = load float, float* %arrayidx, align 4
  %tmp2 = tail call fast float @llvm.pow.f32(float %conv, float %tmp1)
  %arrayidx2 = getelementptr inbounds float, float* %varray, i64 %indvars.iv
  store float %tmp2, float* %arrayidx2, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body, !llvm.loop !131

for.end:                                          ; preds = %for.body
  ret void
}

!131 = distinct !{!131, !132, !133}
!132 = !{!"llvm.loop.vectorize.width", i32 4}
!133 = !{!"llvm.loop.vectorize.enable", i1 true}

attributes #0 = { nounwind readnone }

declare double @sin(double) #0
declare float @sinf(float) #0
declare double @llvm.sin.f64(double) #0
declare float @llvm.sin.f32(float) #0
declare double @cos(double) #0
declare float @cosf(float) #0
declare double @llvm.cos.f64(double) #0
declare float @llvm.cos.f32(float) #0
declare float @expf(float) #0
declare float @powf(float, float) #0
declare float @llvm.exp.f32(float) #0
declare float @logf(float) #0
declare float @llvm.pow.f32(float, float) #0
