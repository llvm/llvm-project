; RUN: opt -vector-library=LIBMVEC -passes=inject-tli-mappings,loop-vectorize -S < %s | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @sin_f64(ptr nocapture %varray) {
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
  %arrayidx = getelementptr inbounds double, ptr %varray, i64 %iv
  store double %call, ptr %arrayidx, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body, !llvm.loop !1

for.end:
  ret void
}

!1 = distinct !{!1, !2, !3}
!2 = !{!"llvm.loop.vectorize.width", i32 4}
!3 = !{!"llvm.loop.vectorize.enable", i1 true}


define void @sin_f32(ptr nocapture %varray) {
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
  %arrayidx = getelementptr inbounds float, ptr %varray, i64 %iv
  store float %call, ptr %arrayidx, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body, !llvm.loop !21

for.end:
  ret void
}

!21 = distinct !{!21, !22, !23}
!22 = !{!"llvm.loop.vectorize.width", i32 4}
!23 = !{!"llvm.loop.vectorize.enable", i1 true}

define void @sin_f64_intrinsic(ptr nocapture %varray) {
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
  %arrayidx = getelementptr inbounds double, ptr %varray, i64 %iv
  store double %call, ptr %arrayidx, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body, !llvm.loop !31

for.end:
  ret void
}

!31 = distinct !{!31, !32, !33}
!32 = !{!"llvm.loop.vectorize.width", i32 4}
!33 = !{!"llvm.loop.vectorize.enable", i1 true}

define void @sin_f32_intrinsic(ptr nocapture %varray) {
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
  %arrayidx = getelementptr inbounds float, ptr %varray, i64 %iv
  store float %call, ptr %arrayidx, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body, !llvm.loop !41

for.end:
  ret void
}

!41 = distinct !{!41, !42, !43}
!42 = !{!"llvm.loop.vectorize.width", i32 4}
!43 = !{!"llvm.loop.vectorize.enable", i1 true}

define void @cos_f64(ptr nocapture %varray) {
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
  %arrayidx = getelementptr inbounds double, ptr %varray, i64 %iv
  store double %call, ptr %arrayidx, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body, !llvm.loop !51

for.end:
  ret void
}

!51 = distinct !{!51, !52, !53}
!52 = !{!"llvm.loop.vectorize.width", i32 4}
!53 = !{!"llvm.loop.vectorize.enable", i1 true}

define void @cos_f32(ptr nocapture %varray) {
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
  %arrayidx = getelementptr inbounds float, ptr %varray, i64 %iv
  store float %call, ptr %arrayidx, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body, !llvm.loop !61

for.end:
  ret void
}

!61 = distinct !{!61, !62, !63}
!62 = !{!"llvm.loop.vectorize.width", i32 4}
!63 = !{!"llvm.loop.vectorize.enable", i1 true}

define void @cos_f64_intrinsic(ptr nocapture %varray) {
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
  %arrayidx = getelementptr inbounds double, ptr %varray, i64 %iv
  store double %call, ptr %arrayidx, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body, !llvm.loop !71

for.end:
  ret void
}

!71 = distinct !{!71, !72, !73}
!72 = !{!"llvm.loop.vectorize.width", i32 4}
!73 = !{!"llvm.loop.vectorize.enable", i1 true}

define void @cos_f32_intrinsic(ptr nocapture %varray) {
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
  %arrayidx = getelementptr inbounds float, ptr %varray, i64 %iv
  store float %call, ptr %arrayidx, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body, !llvm.loop !81

for.end:
  ret void
}

!81 = distinct !{!81, !82, !83}
!82 = !{!"llvm.loop.vectorize.width", i32 4}
!83 = !{!"llvm.loop.vectorize.enable", i1 true}


define void @exp_f32(ptr nocapture %varray) {
; CHECK-LABEL: @exp_f32
; CHECK-LABEL:    vector.body
; CHECK: <4 x float> @_ZGVbN4v_expf
entry:
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %tmp = trunc i64 %indvars.iv to i32
  %conv = sitofp i32 %tmp to float
  %call = tail call fast float @expf(float %conv)
  %arrayidx = getelementptr inbounds float, ptr %varray, i64 %indvars.iv
  store float %call, ptr %arrayidx, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body, !llvm.loop !91

for.end:
  ret void
}

!91 = distinct !{!91, !92, !93}
!92 = !{!"llvm.loop.vectorize.width", i32 4}
!93 = !{!"llvm.loop.vectorize.enable", i1 true}

define void @exp_f32_intrin(ptr nocapture %varray) {
; CHECK-LABEL: @exp_f32_intrin
; CHECK-LABEL: vector.body
; CHECK: <4 x float> @_ZGVbN4v_expf
entry:
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %tmp = trunc i64 %indvars.iv to i32
  %conv = sitofp i32 %tmp to float
  %call = tail call fast float @llvm.exp.f32(float %conv)
  %arrayidx = getelementptr inbounds float, ptr %varray, i64 %indvars.iv
  store float %call, ptr %arrayidx, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body, !llvm.loop !101

for.end:
  ret void
}

!101 = distinct !{!101, !102, !103}
!102 = !{!"llvm.loop.vectorize.width", i32 4}
!103 = !{!"llvm.loop.vectorize.enable", i1 true}


define void @log_f32(ptr nocapture %varray) {
; CHECK-LABEL: @log_f32
; CHECK-LABEL: vector.body
; CHECK: <4 x float> @_ZGVbN4v_logf
entry:
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %tmp = trunc i64 %indvars.iv to i32
  %conv = sitofp i32 %tmp to float
  %call = tail call fast float @logf(float %conv)
  %arrayidx = getelementptr inbounds float, ptr %varray, i64 %indvars.iv
  store float %call, ptr %arrayidx, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body, !llvm.loop !111

for.end:
  ret void
}

!111 = distinct !{!111, !112, !113}
!112 = !{!"llvm.loop.vectorize.width", i32 4}
!113 = !{!"llvm.loop.vectorize.enable", i1 true}

define void @pow_f32(ptr nocapture %varray, ptr nocapture readonly %exp) {
; CHECK-LABEL: @pow_f32
; CHECK-LABEL:    vector.body
; CHECK: <4 x float> @_ZGVbN4vv_powf
entry:
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %tmp = trunc i64 %indvars.iv to i32
  %conv = sitofp i32 %tmp to float
  %arrayidx = getelementptr inbounds float, ptr %exp, i64 %indvars.iv
  %tmp1 = load float, ptr %arrayidx, align 4
  %tmp2 = tail call fast float @powf(float %conv, float %tmp1)
  %arrayidx2 = getelementptr inbounds float, ptr %varray, i64 %indvars.iv
  store float %tmp2, ptr %arrayidx2, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body, !llvm.loop !121

for.end:
  ret void
}

!121 = distinct !{!121, !122, !123}
!122 = !{!"llvm.loop.vectorize.width", i32 4}
!123 = !{!"llvm.loop.vectorize.enable", i1 true}

define void @pow_f32_intrin(ptr nocapture %varray, ptr nocapture readonly %exp) {
; CHECK-LABEL: @pow_f32_intrin
; CHECK-LABEL:    vector.body
; CHECK: <4 x float> @_ZGVbN4vv_powf
entry:
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %tmp = trunc i64 %indvars.iv to i32
  %conv = sitofp i32 %tmp to float
  %arrayidx = getelementptr inbounds float, ptr %exp, i64 %indvars.iv
  %tmp1 = load float, ptr %arrayidx, align 4
  %tmp2 = tail call fast float @llvm.pow.f32(float %conv, float %tmp1)
  %arrayidx2 = getelementptr inbounds float, ptr %varray, i64 %indvars.iv
  store float %tmp2, ptr %arrayidx2, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body, !llvm.loop !131

for.end:
  ret void
}

!131 = distinct !{!131, !132, !133}
!132 = !{!"llvm.loop.vectorize.width", i32 4}
!133 = !{!"llvm.loop.vectorize.enable", i1 true}

attributes #0 = { nounwind readnone }

declare double @sin(double) #0
declare float @sinf(float) #0
declare double @cos(double) #0
declare float @cosf(float) #0
declare float @expf(float) #0
declare float @powf(float, float) #0
declare float @logf(float) #0


; GLIBC 2.35 libmvec functions (no corresponding LLVM intrinsic)
declare float @erff(float) #0
declare float @erfcf(float) #0
declare float @cbrtf(float) #0
declare float @expm1f(float) #0
declare float @log1pf(float) #0
declare float @asinhf(float) #0
declare float @acoshf(float) #0
declare float @atanhf(float) #0

define void @erf_f32(ptr nocapture %varray) {
; CHECK-LABEL: @erf_f32
; CHECK-LABEL:    vector.body
; CHECK: <4 x float> @_ZGVbN4v_erff
entry:
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %tmp = trunc i64 %indvars.iv to i32
  %conv = sitofp i32 %tmp to float
  %call = tail call fast float @erff(float %conv)
  %arrayidx = getelementptr inbounds float, ptr %varray, i64 %indvars.iv
  store float %call, ptr %arrayidx, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body, !llvm.loop !134

for.end:
  ret void
}

!134 = distinct !{!134, !135, !136}
!135 = !{!"llvm.loop.vectorize.width", i32 4}
!136 = !{!"llvm.loop.vectorize.enable", i1 true}

define void @erfc_f32(ptr nocapture %varray) {
; CHECK-LABEL: @erfc_f32
; CHECK-LABEL:    vector.body
; CHECK: <4 x float> @_ZGVbN4v_erfcf
entry:
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %tmp = trunc i64 %indvars.iv to i32
  %conv = sitofp i32 %tmp to float
  %call = tail call fast float @erfcf(float %conv)
  %arrayidx = getelementptr inbounds float, ptr %varray, i64 %indvars.iv
  store float %call, ptr %arrayidx, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body, !llvm.loop !137

for.end:
  ret void
}

!137 = distinct !{!137, !138, !139}
!138 = !{!"llvm.loop.vectorize.width", i32 4}
!139 = !{!"llvm.loop.vectorize.enable", i1 true}

define void @cbrt_f32(ptr nocapture %varray) {
; CHECK-LABEL: @cbrt_f32
; CHECK-LABEL:    vector.body
; CHECK: <4 x float> @_ZGVbN4v_cbrtf
entry:
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %tmp = trunc i64 %indvars.iv to i32
  %conv = sitofp i32 %tmp to float
  %call = tail call fast float @cbrtf(float %conv)
  %arrayidx = getelementptr inbounds float, ptr %varray, i64 %indvars.iv
  store float %call, ptr %arrayidx, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body, !llvm.loop !140

for.end:
  ret void
}

!140 = distinct !{!140, !141, !142}
!141 = !{!"llvm.loop.vectorize.width", i32 4}
!142 = !{!"llvm.loop.vectorize.enable", i1 true}

define void @expm1_f32(ptr nocapture %varray) {
; CHECK-LABEL: @expm1_f32
; CHECK-LABEL:    vector.body
; CHECK: <4 x float> @_ZGVbN4v_expm1f
entry:
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %tmp = trunc i64 %indvars.iv to i32
  %conv = sitofp i32 %tmp to float
  %call = tail call fast float @expm1f(float %conv)
  %arrayidx = getelementptr inbounds float, ptr %varray, i64 %indvars.iv
  store float %call, ptr %arrayidx, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body, !llvm.loop !143

for.end:
  ret void
}

!143 = distinct !{!143, !144, !145}
!144 = !{!"llvm.loop.vectorize.width", i32 4}
!145 = !{!"llvm.loop.vectorize.enable", i1 true}

define void @log1p_f32(ptr nocapture %varray) {
; CHECK-LABEL: @log1p_f32
; CHECK-LABEL:    vector.body
; CHECK: <4 x float> @_ZGVbN4v_log1pf
entry:
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %tmp = trunc i64 %indvars.iv to i32
  %conv = sitofp i32 %tmp to float
  %call = tail call fast float @log1pf(float %conv)
  %arrayidx = getelementptr inbounds float, ptr %varray, i64 %indvars.iv
  store float %call, ptr %arrayidx, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body, !llvm.loop !146

for.end:
  ret void
}

!146 = distinct !{!146, !147, !148}
!147 = !{!"llvm.loop.vectorize.width", i32 4}
!148 = !{!"llvm.loop.vectorize.enable", i1 true}

define void @asinh_f32(ptr nocapture %varray) {
; CHECK-LABEL: @asinh_f32
; CHECK-LABEL:    vector.body
; CHECK: <4 x float> @_ZGVbN4v_asinhf
entry:
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %tmp = trunc i64 %indvars.iv to i32
  %conv = sitofp i32 %tmp to float
  %call = tail call fast float @asinhf(float %conv)
  %arrayidx = getelementptr inbounds float, ptr %varray, i64 %indvars.iv
  store float %call, ptr %arrayidx, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body, !llvm.loop !149

for.end:
  ret void
}

!149 = distinct !{!149, !150, !151}
!150 = !{!"llvm.loop.vectorize.width", i32 4}
!151 = !{!"llvm.loop.vectorize.enable", i1 true}

define void @acosh_f32(ptr nocapture %varray) {
; CHECK-LABEL: @acosh_f32
; CHECK-LABEL:    vector.body
; CHECK: <4 x float> @_ZGVbN4v_acoshf
entry:
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %tmp = trunc i64 %indvars.iv to i32
  %conv = sitofp i32 %tmp to float
  %call = tail call fast float @acoshf(float %conv)
  %arrayidx = getelementptr inbounds float, ptr %varray, i64 %indvars.iv
  store float %call, ptr %arrayidx, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body, !llvm.loop !152

for.end:
  ret void
}

!152 = distinct !{!152, !153, !154}
!153 = !{!"llvm.loop.vectorize.width", i32 4}
!154 = !{!"llvm.loop.vectorize.enable", i1 true}

define void @atanh_f32(ptr nocapture %varray) {
; CHECK-LABEL: @atanh_f32
; CHECK-LABEL:    vector.body
; CHECK: <4 x float> @_ZGVbN4v_atanhf
entry:
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %tmp = trunc i64 %indvars.iv to i32
  %conv = sitofp i32 %tmp to float
  %call = tail call fast float @atanhf(float %conv)
  %arrayidx = getelementptr inbounds float, ptr %varray, i64 %indvars.iv
  store float %call, ptr %arrayidx, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body, !llvm.loop !155

for.end:
  ret void
}

!155 = distinct !{!155, !156, !157}
!156 = !{!"llvm.loop.vectorize.width", i32 4}
!157 = !{!"llvm.loop.vectorize.enable", i1 true}



; GLIBC 2.35 libmvec functions, f64 (no corresponding LLVM intrinsic)
declare double @erf(double) #0
declare double @erfc(double) #0
declare double @cbrt(double) #0
declare double @expm1(double) #0
declare double @log1p(double) #0
declare double @asinh(double) #0
declare double @acosh(double) #0
declare double @atanh(double) #0

define void @erf_f64(ptr nocapture %varray) {
; CHECK-LABEL: @erf_f64
; CHECK-LABEL:    vector.body
; CHECK: <2 x double> @_ZGVbN2v_erf
entry:
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %tmp = trunc i64 %indvars.iv to i32
  %conv = sitofp i32 %tmp to double
  %call = tail call fast double @erf(double %conv)
  %arrayidx = getelementptr inbounds double, ptr %varray, i64 %indvars.iv
  store double %call, ptr %arrayidx, align 8
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body, !llvm.loop !158

for.end:
  ret void
}

!158 = distinct !{!158, !159, !160}
!159 = !{!"llvm.loop.vectorize.width", i32 2}
!160 = !{!"llvm.loop.vectorize.enable", i1 true}

define void @erfc_f64(ptr nocapture %varray) {
; CHECK-LABEL: @erfc_f64
; CHECK-LABEL:    vector.body
; CHECK: <2 x double> @_ZGVbN2v_erfc
entry:
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %tmp = trunc i64 %indvars.iv to i32
  %conv = sitofp i32 %tmp to double
  %call = tail call fast double @erfc(double %conv)
  %arrayidx = getelementptr inbounds double, ptr %varray, i64 %indvars.iv
  store double %call, ptr %arrayidx, align 8
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body, !llvm.loop !161

for.end:
  ret void
}

!161 = distinct !{!161, !162, !163}
!162 = !{!"llvm.loop.vectorize.width", i32 2}
!163 = !{!"llvm.loop.vectorize.enable", i1 true}

define void @cbrt_f64(ptr nocapture %varray) {
; CHECK-LABEL: @cbrt_f64
; CHECK-LABEL:    vector.body
; CHECK: <2 x double> @_ZGVbN2v_cbrt
entry:
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %tmp = trunc i64 %indvars.iv to i32
  %conv = sitofp i32 %tmp to double
  %call = tail call fast double @cbrt(double %conv)
  %arrayidx = getelementptr inbounds double, ptr %varray, i64 %indvars.iv
  store double %call, ptr %arrayidx, align 8
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body, !llvm.loop !164

for.end:
  ret void
}

!164 = distinct !{!164, !165, !166}
!165 = !{!"llvm.loop.vectorize.width", i32 2}
!166 = !{!"llvm.loop.vectorize.enable", i1 true}

define void @expm1_f64(ptr nocapture %varray) {
; CHECK-LABEL: @expm1_f64
; CHECK-LABEL:    vector.body
; CHECK: <2 x double> @_ZGVbN2v_expm1
entry:
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %tmp = trunc i64 %indvars.iv to i32
  %conv = sitofp i32 %tmp to double
  %call = tail call fast double @expm1(double %conv)
  %arrayidx = getelementptr inbounds double, ptr %varray, i64 %indvars.iv
  store double %call, ptr %arrayidx, align 8
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body, !llvm.loop !167

for.end:
  ret void
}

!167 = distinct !{!167, !168, !169}
!168 = !{!"llvm.loop.vectorize.width", i32 2}
!169 = !{!"llvm.loop.vectorize.enable", i1 true}

define void @log1p_f64(ptr nocapture %varray) {
; CHECK-LABEL: @log1p_f64
; CHECK-LABEL:    vector.body
; CHECK: <2 x double> @_ZGVbN2v_log1p
entry:
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %tmp = trunc i64 %indvars.iv to i32
  %conv = sitofp i32 %tmp to double
  %call = tail call fast double @log1p(double %conv)
  %arrayidx = getelementptr inbounds double, ptr %varray, i64 %indvars.iv
  store double %call, ptr %arrayidx, align 8
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body, !llvm.loop !170

for.end:
  ret void
}

!170 = distinct !{!170, !171, !172}
!171 = !{!"llvm.loop.vectorize.width", i32 2}
!172 = !{!"llvm.loop.vectorize.enable", i1 true}

define void @asinh_f64(ptr nocapture %varray) {
; CHECK-LABEL: @asinh_f64
; CHECK-LABEL:    vector.body
; CHECK: <2 x double> @_ZGVbN2v_asinh
entry:
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %tmp = trunc i64 %indvars.iv to i32
  %conv = sitofp i32 %tmp to double
  %call = tail call fast double @asinh(double %conv)
  %arrayidx = getelementptr inbounds double, ptr %varray, i64 %indvars.iv
  store double %call, ptr %arrayidx, align 8
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body, !llvm.loop !173

for.end:
  ret void
}

!173 = distinct !{!173, !174, !175}
!174 = !{!"llvm.loop.vectorize.width", i32 2}
!175 = !{!"llvm.loop.vectorize.enable", i1 true}

define void @acosh_f64(ptr nocapture %varray) {
; CHECK-LABEL: @acosh_f64
; CHECK-LABEL:    vector.body
; CHECK: <2 x double> @_ZGVbN2v_acosh
entry:
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %tmp = trunc i64 %indvars.iv to i32
  %conv = sitofp i32 %tmp to double
  %call = tail call fast double @acosh(double %conv)
  %arrayidx = getelementptr inbounds double, ptr %varray, i64 %indvars.iv
  store double %call, ptr %arrayidx, align 8
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body, !llvm.loop !176

for.end:
  ret void
}

!176 = distinct !{!176, !177, !178}
!177 = !{!"llvm.loop.vectorize.width", i32 2}
!178 = !{!"llvm.loop.vectorize.enable", i1 true}

define void @atanh_f64(ptr nocapture %varray) {
; CHECK-LABEL: @atanh_f64
; CHECK-LABEL:    vector.body
; CHECK: <2 x double> @_ZGVbN2v_atanh
entry:
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %tmp = trunc i64 %indvars.iv to i32
  %conv = sitofp i32 %tmp to double
  %call = tail call fast double @atanh(double %conv)
  %arrayidx = getelementptr inbounds double, ptr %varray, i64 %indvars.iv
  store double %call, ptr %arrayidx, align 8
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body, !llvm.loop !179

for.end:
  ret void
}

!179 = distinct !{!179, !180, !181}
!180 = !{!"llvm.loop.vectorize.width", i32 2}
!181 = !{!"llvm.loop.vectorize.enable", i1 true}

