; Test that the Hygon vector math library (HVML) vector functions are
; selected when vectorizing the corresponding scalar math calls, at the
; fixed vector widths registered in VecFuncs.def. HVML registers 2/4-wide
; double (hvml_vd2_*/hvml_vd4_*) and 4/8-wide float (hvml_vs4_*/hvml_vs8_*)
; variants; widths outside that set fall back to the scalar libm intrinsic.
; RUN: opt -vector-library=HVML -passes=inject-tli-mappings,loop-vectorize,replace-with-veclib -force-vector-width=4 -force-vector-interleave=1 -mattr=avx -S < %s | FileCheck %s --check-prefixes=CHECK,CHECK-VF4
; RUN: opt -vector-library=HVML -passes=inject-tli-mappings,loop-vectorize,replace-with-veclib -force-vector-width=2 -force-vector-interleave=1 -mattr=avx -S < %s | FileCheck %s --check-prefixes=CHECK,CHECK-VF2
; RUN: opt -vector-library=HVML -passes=inject-tli-mappings,loop-vectorize,replace-with-veclib -force-vector-width=8 -force-vector-interleave=1 -mattr=avx -S < %s | FileCheck %s --check-prefixes=CHECK,CHECK-VF8

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @sin_f64(ptr nocapture %varray) {
; CHECK-LABEL: @sin_f64(
; CHECK-VF2:    [[TMP5:%.*]] = call <2 x double> @hvml_vd2_sin(<2 x double> [[TMP4:%.*]])
; CHECK-VF4:    [[TMP5:%.*]] = call <4 x double> @hvml_vd4_sin(<4 x double> [[TMP4:%.*]])
; CHECK-VF8:    [[TMP5:%.*]] = call <8 x double> @llvm.sin.v8f64(<8 x double> [[TMP4:%.*]])
; CHECK:        ret void
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


define void @sin_f64_intrinsic(ptr nocapture %varray) {
; CHECK-LABEL: @sin_f64_intrinsic(
; CHECK-VF2:    [[TMP5:%.*]] = call <2 x double> @hvml_vd2_sin(<2 x double> [[TMP4:%.*]])
; CHECK-VF4:    [[TMP5:%.*]] = call <4 x double> @hvml_vd4_sin(<4 x double> [[TMP4:%.*]])
; CHECK-VF8:    [[TMP5:%.*]] = call <8 x double> @llvm.sin.v8f64(<8 x double> [[TMP4:%.*]])
; CHECK:        ret void
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


define void @sin_f32(ptr nocapture %varray) {
; CHECK-LABEL: @sin_f32(
; CHECK-VF2:    [[TMP5:%.*]] = call <2 x float> @llvm.sin.v2f32(<2 x float> [[TMP4:%.*]])
; CHECK-VF4:    [[TMP5:%.*]] = call <4 x float> @hvml_vs4_sin(<4 x float> [[TMP4:%.*]])
; CHECK-VF8:    [[TMP5:%.*]] = call <8 x float> @hvml_vs8_sin(<8 x float> [[TMP4:%.*]])
; CHECK:        ret void
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


define void @sin_f32_intrinsic(ptr nocapture %varray) {
; CHECK-LABEL: @sin_f32_intrinsic(
; CHECK-VF2:    [[TMP5:%.*]] = call <2 x float> @llvm.sin.v2f32(<2 x float> [[TMP4:%.*]])
; CHECK-VF4:    [[TMP5:%.*]] = call <4 x float> @hvml_vs4_sin(<4 x float> [[TMP4:%.*]])
; CHECK-VF8:    [[TMP5:%.*]] = call <8 x float> @hvml_vs8_sin(<8 x float> [[TMP4:%.*]])
; CHECK:        ret void
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
; CHECK-VF2:    [[TMP5:%.*]] = call <2 x double> @hvml_vd2_cos(<2 x double> [[TMP4:%.*]])
; CHECK-VF4:    [[TMP5:%.*]] = call <4 x double> @hvml_vd4_cos(<4 x double> [[TMP4:%.*]])
; CHECK-VF8:    [[TMP5:%.*]] = call <8 x double> @llvm.cos.v8f64(<8 x double> [[TMP4:%.*]])
; CHECK:        ret void
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


define void @cos_f64_intrinsic(ptr nocapture %varray) {
; CHECK-LABEL: @cos_f64_intrinsic(
; CHECK-VF2:    [[TMP5:%.*]] = call <2 x double> @hvml_vd2_cos(<2 x double> [[TMP4:%.*]])
; CHECK-VF4:    [[TMP5:%.*]] = call <4 x double> @hvml_vd4_cos(<4 x double> [[TMP4:%.*]])
; CHECK-VF8:    [[TMP5:%.*]] = call <8 x double> @llvm.cos.v8f64(<8 x double> [[TMP4:%.*]])
; CHECK:        ret void
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


define void @cos_f32(ptr nocapture %varray) {
; CHECK-LABEL: @cos_f32(
; CHECK-VF2:    [[TMP5:%.*]] = call <2 x float> @llvm.cos.v2f32(<2 x float> [[TMP4:%.*]])
; CHECK-VF4:    [[TMP5:%.*]] = call <4 x float> @hvml_vs4_cos(<4 x float> [[TMP4:%.*]])
; CHECK-VF8:    [[TMP5:%.*]] = call <8 x float> @hvml_vs8_cos(<8 x float> [[TMP4:%.*]])
; CHECK:        ret void
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


define void @cos_f32_intrinsic(ptr nocapture %varray) {
; CHECK-LABEL: @cos_f32_intrinsic(
; CHECK-VF2:    [[TMP5:%.*]] = call <2 x float> @llvm.cos.v2f32(<2 x float> [[TMP4:%.*]])
; CHECK-VF4:    [[TMP5:%.*]] = call <4 x float> @hvml_vs4_cos(<4 x float> [[TMP4:%.*]])
; CHECK-VF8:    [[TMP5:%.*]] = call <8 x float> @hvml_vs8_cos(<8 x float> [[TMP4:%.*]])
; CHECK:        ret void
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



define void @tan_f64(ptr nocapture %varray) {
; CHECK-LABEL: @tan_f64(
; CHECK-VF2:    [[TMP5:%.*]] = call <2 x double> @hvml_vd2_tan(<2 x double> [[TMP4:%.*]])
; CHECK-VF4:    [[TMP5:%.*]] = call <4 x double> @hvml_vd4_tan(<4 x double> [[TMP4:%.*]])
; CHECK-VF8:    [[TMP5:%.*]] = call <8 x double> @llvm.tan.v8f64(<8 x double> [[TMP4:%.*]])
; CHECK:        ret void
;
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %tmp = trunc i64 %iv to i32
  %conv = sitofp i32 %tmp to double
  %call = tail call double @tan(double %conv)
  %arrayidx = getelementptr inbounds double, ptr %varray, i64 %iv
  store double %call, ptr %arrayidx, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}


define void @tan_f64_intrinsic(ptr nocapture %varray) {
; CHECK-LABEL: @tan_f64_intrinsic(
; CHECK-VF2:    [[TMP5:%.*]] = call <2 x double> @hvml_vd2_tan(<2 x double> [[TMP4:%.*]])
; CHECK-VF4:    [[TMP5:%.*]] = call <4 x double> @hvml_vd4_tan(<4 x double> [[TMP4:%.*]])
; CHECK-VF8:    [[TMP5:%.*]] = call <8 x double> @llvm.tan.v8f64(<8 x double> [[TMP4:%.*]])
; CHECK:        ret void
;
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %tmp = trunc i64 %iv to i32
  %conv = sitofp i32 %tmp to double
  %call = tail call double @llvm.tan.f64(double %conv)
  %arrayidx = getelementptr inbounds double, ptr %varray, i64 %iv
  store double %call, ptr %arrayidx, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}


define void @tan_f32(ptr nocapture %varray) {
; CHECK-LABEL: @tan_f32(
; CHECK-VF2:    [[TMP5:%.*]] = call <2 x float> @llvm.tan.v2f32(<2 x float> [[TMP4:%.*]])
; CHECK-VF4:    [[TMP5:%.*]] = call <4 x float> @hvml_vs4_tan(<4 x float> [[TMP4:%.*]])
; CHECK-VF8:    [[TMP5:%.*]] = call <8 x float> @hvml_vs8_tan(<8 x float> [[TMP4:%.*]])
; CHECK:        ret void
;
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %tmp = trunc i64 %iv to i32
  %conv = sitofp i32 %tmp to float
  %call = tail call float @tanf(float %conv)
  %arrayidx = getelementptr inbounds float, ptr %varray, i64 %iv
  store float %call, ptr %arrayidx, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}


define void @tan_f32_intrinsic(ptr nocapture %varray) {
; CHECK-LABEL: @tan_f32_intrinsic(
; CHECK-VF2:    [[TMP5:%.*]] = call <2 x float> @llvm.tan.v2f32(<2 x float> [[TMP4:%.*]])
; CHECK-VF4:    [[TMP5:%.*]] = call <4 x float> @hvml_vs4_tan(<4 x float> [[TMP4:%.*]])
; CHECK-VF8:    [[TMP5:%.*]] = call <8 x float> @hvml_vs8_tan(<8 x float> [[TMP4:%.*]])
; CHECK:        ret void
;
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %tmp = trunc i64 %iv to i32
  %conv = sitofp i32 %tmp to float
  %call = tail call float @llvm.tan.f32(float %conv)
  %arrayidx = getelementptr inbounds float, ptr %varray, i64 %iv
  store float %call, ptr %arrayidx, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}



define void @asin_f64(ptr nocapture %varray) {
; CHECK-LABEL: @asin_f64(
; CHECK-VF2:    [[TMP5:%.*]] = call <2 x double> @hvml_vd2_asin(<2 x double> [[TMP4:%.*]])
; CHECK-VF4:    [[TMP5:%.*]] = call <4 x double> @hvml_vd4_asin(<4 x double> [[TMP4:%.*]])
; CHECK-VF8:    [[TMP5:%.*]] = call <8 x double> @llvm.asin.v8f64(<8 x double> [[TMP4:%.*]])
; CHECK:        ret void
;
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %tmp = trunc i64 %iv to i32
  %conv = sitofp i32 %tmp to double
  %call = tail call double @asin(double %conv)
  %arrayidx = getelementptr inbounds double, ptr %varray, i64 %iv
  store double %call, ptr %arrayidx, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}


define void @asin_f64_intrinsic(ptr nocapture %varray) {
; CHECK-LABEL: @asin_f64_intrinsic(
; CHECK-VF2:    [[TMP5:%.*]] = call <2 x double> @hvml_vd2_asin(<2 x double> [[TMP4:%.*]])
; CHECK-VF4:    [[TMP5:%.*]] = call <4 x double> @hvml_vd4_asin(<4 x double> [[TMP4:%.*]])
; CHECK-VF8:    [[TMP5:%.*]] = call <8 x double> @llvm.asin.v8f64(<8 x double> [[TMP4:%.*]])
; CHECK:        ret void
;
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %tmp = trunc i64 %iv to i32
  %conv = sitofp i32 %tmp to double
  %call = tail call double @llvm.asin.f64(double %conv)
  %arrayidx = getelementptr inbounds double, ptr %varray, i64 %iv
  store double %call, ptr %arrayidx, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}


define void @asin_f32(ptr nocapture %varray) {
; CHECK-LABEL: @asin_f32(
; CHECK-VF2:    [[TMP5:%.*]] = call <2 x float> @llvm.asin.v2f32(<2 x float> [[TMP4:%.*]])
; CHECK-VF4:    [[TMP5:%.*]] = call <4 x float> @hvml_vs4_asin(<4 x float> [[TMP4:%.*]])
; CHECK-VF8:    [[TMP5:%.*]] = call <8 x float> @hvml_vs8_asin(<8 x float> [[TMP4:%.*]])
; CHECK:        ret void
;
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %tmp = trunc i64 %iv to i32
  %conv = sitofp i32 %tmp to float
  %call = tail call float @asinf(float %conv)
  %arrayidx = getelementptr inbounds float, ptr %varray, i64 %iv
  store float %call, ptr %arrayidx, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}


define void @asin_f32_intrinsic(ptr nocapture %varray) {
; CHECK-LABEL: @asin_f32_intrinsic(
; CHECK-VF2:    [[TMP5:%.*]] = call <2 x float> @llvm.asin.v2f32(<2 x float> [[TMP4:%.*]])
; CHECK-VF4:    [[TMP5:%.*]] = call <4 x float> @hvml_vs4_asin(<4 x float> [[TMP4:%.*]])
; CHECK-VF8:    [[TMP5:%.*]] = call <8 x float> @hvml_vs8_asin(<8 x float> [[TMP4:%.*]])
; CHECK:        ret void
;
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %tmp = trunc i64 %iv to i32
  %conv = sitofp i32 %tmp to float
  %call = tail call float @llvm.asin.f32(float %conv)
  %arrayidx = getelementptr inbounds float, ptr %varray, i64 %iv
  store float %call, ptr %arrayidx, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}



define void @acos_f64(ptr nocapture %varray) {
; CHECK-LABEL: @acos_f64(
; CHECK-VF2:    [[TMP5:%.*]] = call <2 x double> @hvml_vd2_acos(<2 x double> [[TMP4:%.*]])
; CHECK-VF4:    [[TMP5:%.*]] = call <4 x double> @hvml_vd4_acos(<4 x double> [[TMP4:%.*]])
; CHECK-VF8:    [[TMP5:%.*]] = call <8 x double> @llvm.acos.v8f64(<8 x double> [[TMP4:%.*]])
; CHECK:        ret void
;
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %tmp = trunc i64 %iv to i32
  %conv = sitofp i32 %tmp to double
  %call = tail call double @acos(double %conv)
  %arrayidx = getelementptr inbounds double, ptr %varray, i64 %iv
  store double %call, ptr %arrayidx, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}


define void @acos_f64_intrinsic(ptr nocapture %varray) {
; CHECK-LABEL: @acos_f64_intrinsic(
; CHECK-VF2:    [[TMP5:%.*]] = call <2 x double> @hvml_vd2_acos(<2 x double> [[TMP4:%.*]])
; CHECK-VF4:    [[TMP5:%.*]] = call <4 x double> @hvml_vd4_acos(<4 x double> [[TMP4:%.*]])
; CHECK-VF8:    [[TMP5:%.*]] = call <8 x double> @llvm.acos.v8f64(<8 x double> [[TMP4:%.*]])
; CHECK:        ret void
;
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %tmp = trunc i64 %iv to i32
  %conv = sitofp i32 %tmp to double
  %call = tail call double @llvm.acos.f64(double %conv)
  %arrayidx = getelementptr inbounds double, ptr %varray, i64 %iv
  store double %call, ptr %arrayidx, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}


define void @acos_f32(ptr nocapture %varray) {
; CHECK-LABEL: @acos_f32(
; CHECK-VF2:    [[TMP5:%.*]] = call <2 x float> @llvm.acos.v2f32(<2 x float> [[TMP4:%.*]])
; CHECK-VF4:    [[TMP5:%.*]] = call <4 x float> @hvml_vs4_acos(<4 x float> [[TMP4:%.*]])
; CHECK-VF8:    [[TMP5:%.*]] = call <8 x float> @hvml_vs8_acos(<8 x float> [[TMP4:%.*]])
; CHECK:        ret void
;
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %tmp = trunc i64 %iv to i32
  %conv = sitofp i32 %tmp to float
  %call = tail call float @acosf(float %conv)
  %arrayidx = getelementptr inbounds float, ptr %varray, i64 %iv
  store float %call, ptr %arrayidx, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}


define void @acos_f32_intrinsic(ptr nocapture %varray) {
; CHECK-LABEL: @acos_f32_intrinsic(
; CHECK-VF2:    [[TMP5:%.*]] = call <2 x float> @llvm.acos.v2f32(<2 x float> [[TMP4:%.*]])
; CHECK-VF4:    [[TMP5:%.*]] = call <4 x float> @hvml_vs4_acos(<4 x float> [[TMP4:%.*]])
; CHECK-VF8:    [[TMP5:%.*]] = call <8 x float> @hvml_vs8_acos(<8 x float> [[TMP4:%.*]])
; CHECK:        ret void
;
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %tmp = trunc i64 %iv to i32
  %conv = sitofp i32 %tmp to float
  %call = tail call float @llvm.acos.f32(float %conv)
  %arrayidx = getelementptr inbounds float, ptr %varray, i64 %iv
  store float %call, ptr %arrayidx, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}



define void @atan_f64(ptr nocapture %varray) {
; CHECK-LABEL: @atan_f64(
; CHECK-VF2:    [[TMP5:%.*]] = call <2 x double> @hvml_vd2_atan(<2 x double> [[TMP4:%.*]])
; CHECK-VF4:    [[TMP5:%.*]] = call <4 x double> @hvml_vd4_atan(<4 x double> [[TMP4:%.*]])
; CHECK-VF8:    [[TMP5:%.*]] = call <8 x double> @llvm.atan.v8f64(<8 x double> [[TMP4:%.*]])
; CHECK:        ret void
;
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %tmp = trunc i64 %iv to i32
  %conv = sitofp i32 %tmp to double
  %call = tail call double @atan(double %conv)
  %arrayidx = getelementptr inbounds double, ptr %varray, i64 %iv
  store double %call, ptr %arrayidx, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}


define void @atan_f64_intrinsic(ptr nocapture %varray) {
; CHECK-LABEL: @atan_f64_intrinsic(
; CHECK-VF2:    [[TMP5:%.*]] = call <2 x double> @hvml_vd2_atan(<2 x double> [[TMP4:%.*]])
; CHECK-VF4:    [[TMP5:%.*]] = call <4 x double> @hvml_vd4_atan(<4 x double> [[TMP4:%.*]])
; CHECK-VF8:    [[TMP5:%.*]] = call <8 x double> @llvm.atan.v8f64(<8 x double> [[TMP4:%.*]])
; CHECK:        ret void
;
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %tmp = trunc i64 %iv to i32
  %conv = sitofp i32 %tmp to double
  %call = tail call double @llvm.atan.f64(double %conv)
  %arrayidx = getelementptr inbounds double, ptr %varray, i64 %iv
  store double %call, ptr %arrayidx, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}


define void @atan_f32(ptr nocapture %varray) {
; CHECK-LABEL: @atan_f32(
; CHECK-VF2:    [[TMP5:%.*]] = call <2 x float> @llvm.atan.v2f32(<2 x float> [[TMP4:%.*]])
; CHECK-VF4:    [[TMP5:%.*]] = call <4 x float> @hvml_vs4_atan(<4 x float> [[TMP4:%.*]])
; CHECK-VF8:    [[TMP5:%.*]] = call <8 x float> @hvml_vs8_atan(<8 x float> [[TMP4:%.*]])
; CHECK:        ret void
;
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %tmp = trunc i64 %iv to i32
  %conv = sitofp i32 %tmp to float
  %call = tail call float @atanf(float %conv)
  %arrayidx = getelementptr inbounds float, ptr %varray, i64 %iv
  store float %call, ptr %arrayidx, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}


define void @atan_f32_intrinsic(ptr nocapture %varray) {
; CHECK-LABEL: @atan_f32_intrinsic(
; CHECK-VF2:    [[TMP5:%.*]] = call <2 x float> @llvm.atan.v2f32(<2 x float> [[TMP4:%.*]])
; CHECK-VF4:    [[TMP5:%.*]] = call <4 x float> @hvml_vs4_atan(<4 x float> [[TMP4:%.*]])
; CHECK-VF8:    [[TMP5:%.*]] = call <8 x float> @hvml_vs8_atan(<8 x float> [[TMP4:%.*]])
; CHECK:        ret void
;
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %tmp = trunc i64 %iv to i32
  %conv = sitofp i32 %tmp to float
  %call = tail call float @llvm.atan.f32(float %conv)
  %arrayidx = getelementptr inbounds float, ptr %varray, i64 %iv
  store float %call, ptr %arrayidx, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}



define void @sinh_f64(ptr nocapture %varray) {
; CHECK-LABEL: @sinh_f64(
; CHECK-VF2:    [[TMP5:%.*]] = call <2 x double> @hvml_vd2_sinh(<2 x double> [[TMP4:%.*]])
; CHECK-VF4:    [[TMP5:%.*]] = call <4 x double> @hvml_vd4_sinh(<4 x double> [[TMP4:%.*]])
; CHECK-VF8:    [[TMP5:%.*]] = call <8 x double> @llvm.sinh.v8f64(<8 x double> [[TMP4:%.*]])
; CHECK:        ret void
;
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %tmp = trunc i64 %iv to i32
  %conv = sitofp i32 %tmp to double
  %call = tail call double @sinh(double %conv)
  %arrayidx = getelementptr inbounds double, ptr %varray, i64 %iv
  store double %call, ptr %arrayidx, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}


define void @sinh_f64_intrinsic(ptr nocapture %varray) {
; CHECK-LABEL: @sinh_f64_intrinsic(
; CHECK-VF2:    [[TMP5:%.*]] = call <2 x double> @hvml_vd2_sinh(<2 x double> [[TMP4:%.*]])
; CHECK-VF4:    [[TMP5:%.*]] = call <4 x double> @hvml_vd4_sinh(<4 x double> [[TMP4:%.*]])
; CHECK-VF8:    [[TMP5:%.*]] = call <8 x double> @llvm.sinh.v8f64(<8 x double> [[TMP4:%.*]])
; CHECK:        ret void
;
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %tmp = trunc i64 %iv to i32
  %conv = sitofp i32 %tmp to double
  %call = tail call double @llvm.sinh.f64(double %conv)
  %arrayidx = getelementptr inbounds double, ptr %varray, i64 %iv
  store double %call, ptr %arrayidx, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}


define void @sinh_f32(ptr nocapture %varray) {
; CHECK-LABEL: @sinh_f32(
; CHECK-VF2:    [[TMP5:%.*]] = call <2 x float> @llvm.sinh.v2f32(<2 x float> [[TMP4:%.*]])
; CHECK-VF4:    [[TMP5:%.*]] = call <4 x float> @hvml_vs4_sinh(<4 x float> [[TMP4:%.*]])
; CHECK-VF8:    [[TMP5:%.*]] = call <8 x float> @hvml_vs8_sinh(<8 x float> [[TMP4:%.*]])
; CHECK:        ret void
;
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %tmp = trunc i64 %iv to i32
  %conv = sitofp i32 %tmp to float
  %call = tail call float @sinhf(float %conv)
  %arrayidx = getelementptr inbounds float, ptr %varray, i64 %iv
  store float %call, ptr %arrayidx, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}


define void @sinh_f32_intrinsic(ptr nocapture %varray) {
; CHECK-LABEL: @sinh_f32_intrinsic(
; CHECK-VF2:    [[TMP5:%.*]] = call <2 x float> @llvm.sinh.v2f32(<2 x float> [[TMP4:%.*]])
; CHECK-VF4:    [[TMP5:%.*]] = call <4 x float> @hvml_vs4_sinh(<4 x float> [[TMP4:%.*]])
; CHECK-VF8:    [[TMP5:%.*]] = call <8 x float> @hvml_vs8_sinh(<8 x float> [[TMP4:%.*]])
; CHECK:        ret void
;
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %tmp = trunc i64 %iv to i32
  %conv = sitofp i32 %tmp to float
  %call = tail call float @llvm.sinh.f32(float %conv)
  %arrayidx = getelementptr inbounds float, ptr %varray, i64 %iv
  store float %call, ptr %arrayidx, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}



define void @cosh_f64(ptr nocapture %varray) {
; CHECK-LABEL: @cosh_f64(
; CHECK-VF2:    [[TMP5:%.*]] = call <2 x double> @hvml_vd2_cosh(<2 x double> [[TMP4:%.*]])
; CHECK-VF4:    [[TMP5:%.*]] = call <4 x double> @hvml_vd4_cosh(<4 x double> [[TMP4:%.*]])
; CHECK-VF8:    [[TMP5:%.*]] = call <8 x double> @llvm.cosh.v8f64(<8 x double> [[TMP4:%.*]])
; CHECK:        ret void
;
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %tmp = trunc i64 %iv to i32
  %conv = sitofp i32 %tmp to double
  %call = tail call double @cosh(double %conv)
  %arrayidx = getelementptr inbounds double, ptr %varray, i64 %iv
  store double %call, ptr %arrayidx, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}


define void @cosh_f64_intrinsic(ptr nocapture %varray) {
; CHECK-LABEL: @cosh_f64_intrinsic(
; CHECK-VF2:    [[TMP5:%.*]] = call <2 x double> @hvml_vd2_cosh(<2 x double> [[TMP4:%.*]])
; CHECK-VF4:    [[TMP5:%.*]] = call <4 x double> @hvml_vd4_cosh(<4 x double> [[TMP4:%.*]])
; CHECK-VF8:    [[TMP5:%.*]] = call <8 x double> @llvm.cosh.v8f64(<8 x double> [[TMP4:%.*]])
; CHECK:        ret void
;
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %tmp = trunc i64 %iv to i32
  %conv = sitofp i32 %tmp to double
  %call = tail call double @llvm.cosh.f64(double %conv)
  %arrayidx = getelementptr inbounds double, ptr %varray, i64 %iv
  store double %call, ptr %arrayidx, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}


define void @cosh_f32(ptr nocapture %varray) {
; CHECK-LABEL: @cosh_f32(
; CHECK-VF2:    [[TMP5:%.*]] = call <2 x float> @llvm.cosh.v2f32(<2 x float> [[TMP4:%.*]])
; CHECK-VF4:    [[TMP5:%.*]] = call <4 x float> @hvml_vs4_cosh(<4 x float> [[TMP4:%.*]])
; CHECK-VF8:    [[TMP5:%.*]] = call <8 x float> @hvml_vs8_cosh(<8 x float> [[TMP4:%.*]])
; CHECK:        ret void
;
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %tmp = trunc i64 %iv to i32
  %conv = sitofp i32 %tmp to float
  %call = tail call float @coshf(float %conv)
  %arrayidx = getelementptr inbounds float, ptr %varray, i64 %iv
  store float %call, ptr %arrayidx, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}


define void @cosh_f32_intrinsic(ptr nocapture %varray) {
; CHECK-LABEL: @cosh_f32_intrinsic(
; CHECK-VF2:    [[TMP5:%.*]] = call <2 x float> @llvm.cosh.v2f32(<2 x float> [[TMP4:%.*]])
; CHECK-VF4:    [[TMP5:%.*]] = call <4 x float> @hvml_vs4_cosh(<4 x float> [[TMP4:%.*]])
; CHECK-VF8:    [[TMP5:%.*]] = call <8 x float> @hvml_vs8_cosh(<8 x float> [[TMP4:%.*]])
; CHECK:        ret void
;
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %tmp = trunc i64 %iv to i32
  %conv = sitofp i32 %tmp to float
  %call = tail call float @llvm.cosh.f32(float %conv)
  %arrayidx = getelementptr inbounds float, ptr %varray, i64 %iv
  store float %call, ptr %arrayidx, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}



define void @tanh_f64(ptr nocapture %varray) {
; CHECK-LABEL: @tanh_f64(
; CHECK-VF2:    [[TMP5:%.*]] = call <2 x double> @hvml_vd2_tanh(<2 x double> [[TMP4:%.*]])
; CHECK-VF4:    [[TMP5:%.*]] = call <4 x double> @hvml_vd4_tanh(<4 x double> [[TMP4:%.*]])
; CHECK-VF8:    [[TMP5:%.*]] = call <8 x double> @llvm.tanh.v8f64(<8 x double> [[TMP4:%.*]])
; CHECK:        ret void
;
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %tmp = trunc i64 %iv to i32
  %conv = sitofp i32 %tmp to double
  %call = tail call double @tanh(double %conv)
  %arrayidx = getelementptr inbounds double, ptr %varray, i64 %iv
  store double %call, ptr %arrayidx, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}


define void @tanh_f64_intrinsic(ptr nocapture %varray) {
; CHECK-LABEL: @tanh_f64_intrinsic(
; CHECK-VF2:    [[TMP5:%.*]] = call <2 x double> @hvml_vd2_tanh(<2 x double> [[TMP4:%.*]])
; CHECK-VF4:    [[TMP5:%.*]] = call <4 x double> @hvml_vd4_tanh(<4 x double> [[TMP4:%.*]])
; CHECK-VF8:    [[TMP5:%.*]] = call <8 x double> @llvm.tanh.v8f64(<8 x double> [[TMP4:%.*]])
; CHECK:        ret void
;
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %tmp = trunc i64 %iv to i32
  %conv = sitofp i32 %tmp to double
  %call = tail call double @llvm.tanh.f64(double %conv)
  %arrayidx = getelementptr inbounds double, ptr %varray, i64 %iv
  store double %call, ptr %arrayidx, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}


define void @tanh_f32(ptr nocapture %varray) {
; CHECK-LABEL: @tanh_f32(
; CHECK-VF2:    [[TMP5:%.*]] = call <2 x float> @llvm.tanh.v2f32(<2 x float> [[TMP4:%.*]])
; CHECK-VF4:    [[TMP5:%.*]] = call <4 x float> @hvml_vs4_tanh(<4 x float> [[TMP4:%.*]])
; CHECK-VF8:    [[TMP5:%.*]] = call <8 x float> @hvml_vs8_tanh(<8 x float> [[TMP4:%.*]])
; CHECK:        ret void
;
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %tmp = trunc i64 %iv to i32
  %conv = sitofp i32 %tmp to float
  %call = tail call float @tanhf(float %conv)
  %arrayidx = getelementptr inbounds float, ptr %varray, i64 %iv
  store float %call, ptr %arrayidx, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}


define void @tanh_f32_intrinsic(ptr nocapture %varray) {
; CHECK-LABEL: @tanh_f32_intrinsic(
; CHECK-VF2:    [[TMP5:%.*]] = call <2 x float> @llvm.tanh.v2f32(<2 x float> [[TMP4:%.*]])
; CHECK-VF4:    [[TMP5:%.*]] = call <4 x float> @hvml_vs4_tanh(<4 x float> [[TMP4:%.*]])
; CHECK-VF8:    [[TMP5:%.*]] = call <8 x float> @hvml_vs8_tanh(<8 x float> [[TMP4:%.*]])
; CHECK:        ret void
;
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %tmp = trunc i64 %iv to i32
  %conv = sitofp i32 %tmp to float
  %call = tail call float @llvm.tanh.f32(float %conv)
  %arrayidx = getelementptr inbounds float, ptr %varray, i64 %iv
  store float %call, ptr %arrayidx, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}



define void @exp_f64(ptr nocapture %varray) {
; CHECK-LABEL: @exp_f64(
; CHECK-VF2:    [[TMP5:%.*]] = call <2 x double> @hvml_vd2_exp(<2 x double> [[TMP4:%.*]])
; CHECK-VF4:    [[TMP5:%.*]] = call <4 x double> @hvml_vd4_exp(<4 x double> [[TMP4:%.*]])
; CHECK-VF8:    [[TMP5:%.*]] = call <8 x double> @llvm.exp.v8f64(<8 x double> [[TMP4:%.*]])
; CHECK:        ret void
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


define void @exp_f64_intrinsic(ptr nocapture %varray) {
; CHECK-LABEL: @exp_f64_intrinsic(
; CHECK-VF2:    [[TMP5:%.*]] = call <2 x double> @hvml_vd2_exp(<2 x double> [[TMP4:%.*]])
; CHECK-VF4:    [[TMP5:%.*]] = call <4 x double> @hvml_vd4_exp(<4 x double> [[TMP4:%.*]])
; CHECK-VF8:    [[TMP5:%.*]] = call <8 x double> @llvm.exp.v8f64(<8 x double> [[TMP4:%.*]])
; CHECK:        ret void
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


define void @exp_f32(ptr nocapture %varray) {
; CHECK-LABEL: @exp_f32(
; CHECK-VF2:    [[TMP5:%.*]] = call <2 x float> @llvm.exp.v2f32(<2 x float> [[TMP4:%.*]])
; CHECK-VF4:    [[TMP5:%.*]] = call <4 x float> @hvml_vs4_exp(<4 x float> [[TMP4:%.*]])
; CHECK-VF8:    [[TMP5:%.*]] = call <8 x float> @hvml_vs8_exp(<8 x float> [[TMP4:%.*]])
; CHECK:        ret void
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


define void @exp_f32_intrinsic(ptr nocapture %varray) {
; CHECK-LABEL: @exp_f32_intrinsic(
; CHECK-VF2:    [[TMP5:%.*]] = call <2 x float> @llvm.exp.v2f32(<2 x float> [[TMP4:%.*]])
; CHECK-VF4:    [[TMP5:%.*]] = call <4 x float> @hvml_vs4_exp(<4 x float> [[TMP4:%.*]])
; CHECK-VF8:    [[TMP5:%.*]] = call <8 x float> @hvml_vs8_exp(<8 x float> [[TMP4:%.*]])
; CHECK:        ret void
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



define void @exp2_f64(ptr nocapture %varray) {
; CHECK-LABEL: @exp2_f64(
; CHECK-VF2:    [[TMP5:%.*]] = call <2 x double> @hvml_vd2_exp2(<2 x double> [[TMP4:%.*]])
; CHECK-VF4:    [[TMP5:%.*]] = call <4 x double> @hvml_vd4_exp2(<4 x double> [[TMP4:%.*]])
; CHECK-VF8:    [[TMP5:%.*]] = call <8 x double> @llvm.exp2.v8f64(<8 x double> [[TMP4:%.*]])
; CHECK:        ret void
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


define void @exp2_f64_intrinsic(ptr nocapture %varray) {
; CHECK-LABEL: @exp2_f64_intrinsic(
; CHECK-VF2:    [[TMP5:%.*]] = call <2 x double> @hvml_vd2_exp2(<2 x double> [[TMP4:%.*]])
; CHECK-VF4:    [[TMP5:%.*]] = call <4 x double> @hvml_vd4_exp2(<4 x double> [[TMP4:%.*]])
; CHECK-VF8:    [[TMP5:%.*]] = call <8 x double> @llvm.exp2.v8f64(<8 x double> [[TMP4:%.*]])
; CHECK:        ret void
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


define void @exp2_f32(ptr nocapture %varray) {
; CHECK-LABEL: @exp2_f32(
; CHECK-VF2:    [[TMP5:%.*]] = call <2 x float> @llvm.exp2.v2f32(<2 x float> [[TMP4:%.*]])
; CHECK-VF4:    [[TMP5:%.*]] = call <4 x float> @hvml_vs4_exp2(<4 x float> [[TMP4:%.*]])
; CHECK-VF8:    [[TMP5:%.*]] = call <8 x float> @hvml_vs8_exp2(<8 x float> [[TMP4:%.*]])
; CHECK:        ret void
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


define void @exp2_f32_intrinsic(ptr nocapture %varray) {
; CHECK-LABEL: @exp2_f32_intrinsic(
; CHECK-VF2:    [[TMP5:%.*]] = call <2 x float> @llvm.exp2.v2f32(<2 x float> [[TMP4:%.*]])
; CHECK-VF4:    [[TMP5:%.*]] = call <4 x float> @hvml_vs4_exp2(<4 x float> [[TMP4:%.*]])
; CHECK-VF8:    [[TMP5:%.*]] = call <8 x float> @hvml_vs8_exp2(<8 x float> [[TMP4:%.*]])
; CHECK:        ret void
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



define void @log_f64(ptr nocapture %varray) {
; CHECK-LABEL: @log_f64(
; CHECK-VF2:    [[TMP5:%.*]] = call <2 x double> @hvml_vd2_ln(<2 x double> [[TMP4:%.*]])
; CHECK-VF4:    [[TMP5:%.*]] = call <4 x double> @hvml_vd4_ln(<4 x double> [[TMP4:%.*]])
; CHECK-VF8:    [[TMP5:%.*]] = call <8 x double> @llvm.log.v8f64(<8 x double> [[TMP4:%.*]])
; CHECK:        ret void
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


define void @log_f64_intrinsic(ptr nocapture %varray) {
; CHECK-LABEL: @log_f64_intrinsic(
; CHECK-VF2:    [[TMP5:%.*]] = call <2 x double> @hvml_vd2_ln(<2 x double> [[TMP4:%.*]])
; CHECK-VF4:    [[TMP5:%.*]] = call <4 x double> @hvml_vd4_ln(<4 x double> [[TMP4:%.*]])
; CHECK-VF8:    [[TMP5:%.*]] = call <8 x double> @llvm.log.v8f64(<8 x double> [[TMP4:%.*]])
; CHECK:        ret void
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


define void @log_f32(ptr nocapture %varray) {
; CHECK-LABEL: @log_f32(
; CHECK-VF2:    [[TMP5:%.*]] = call <2 x float> @llvm.log.v2f32(<2 x float> [[TMP4:%.*]])
; CHECK-VF4:    [[TMP5:%.*]] = call <4 x float> @hvml_vs4_ln(<4 x float> [[TMP4:%.*]])
; CHECK-VF8:    [[TMP5:%.*]] = call <8 x float> @hvml_vs8_ln(<8 x float> [[TMP4:%.*]])
; CHECK:        ret void
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


define void @log_f32_intrinsic(ptr nocapture %varray) {
; CHECK-LABEL: @log_f32_intrinsic(
; CHECK-VF2:    [[TMP5:%.*]] = call <2 x float> @llvm.log.v2f32(<2 x float> [[TMP4:%.*]])
; CHECK-VF4:    [[TMP5:%.*]] = call <4 x float> @hvml_vs4_ln(<4 x float> [[TMP4:%.*]])
; CHECK-VF8:    [[TMP5:%.*]] = call <8 x float> @hvml_vs8_ln(<8 x float> [[TMP4:%.*]])
; CHECK:        ret void
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
; CHECK-VF2:    [[TMP5:%.*]] = call <2 x double> @hvml_vd2_log2(<2 x double> [[TMP4:%.*]])
; CHECK-VF4:    [[TMP5:%.*]] = call <4 x double> @hvml_vd4_log2(<4 x double> [[TMP4:%.*]])
; CHECK-VF8:    [[TMP5:%.*]] = call <8 x double> @llvm.log2.v8f64(<8 x double> [[TMP4:%.*]])
; CHECK:        ret void
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


define void @log2_f64_intrinsic(ptr nocapture %varray) {
; CHECK-LABEL: @log2_f64_intrinsic(
; CHECK-VF2:    [[TMP5:%.*]] = call <2 x double> @hvml_vd2_log2(<2 x double> [[TMP4:%.*]])
; CHECK-VF4:    [[TMP5:%.*]] = call <4 x double> @hvml_vd4_log2(<4 x double> [[TMP4:%.*]])
; CHECK-VF8:    [[TMP5:%.*]] = call <8 x double> @llvm.log2.v8f64(<8 x double> [[TMP4:%.*]])
; CHECK:        ret void
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


define void @log2_f32(ptr nocapture %varray) {
; CHECK-LABEL: @log2_f32(
; CHECK-VF2:    [[TMP5:%.*]] = call <2 x float> @llvm.log2.v2f32(<2 x float> [[TMP4:%.*]])
; CHECK-VF4:    [[TMP5:%.*]] = call <4 x float> @hvml_vs4_log2(<4 x float> [[TMP4:%.*]])
; CHECK-VF8:    [[TMP5:%.*]] = call <8 x float> @hvml_vs8_log2(<8 x float> [[TMP4:%.*]])
; CHECK:        ret void
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


define void @log2_f32_intrinsic(ptr nocapture %varray) {
; CHECK-LABEL: @log2_f32_intrinsic(
; CHECK-VF2:    [[TMP5:%.*]] = call <2 x float> @llvm.log2.v2f32(<2 x float> [[TMP4:%.*]])
; CHECK-VF4:    [[TMP5:%.*]] = call <4 x float> @hvml_vs4_log2(<4 x float> [[TMP4:%.*]])
; CHECK-VF8:    [[TMP5:%.*]] = call <8 x float> @hvml_vs8_log2(<8 x float> [[TMP4:%.*]])
; CHECK:        ret void
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



define void @log10_f64(ptr nocapture %varray) {
; CHECK-LABEL: @log10_f64(
; CHECK-VF2:    [[TMP5:%.*]] = call <2 x double> @hvml_vd2_log10(<2 x double> [[TMP4:%.*]])
; CHECK-VF4:    [[TMP5:%.*]] = call <4 x double> @hvml_vd4_log10(<4 x double> [[TMP4:%.*]])
; CHECK-VF8:    [[TMP5:%.*]] = call <8 x double> @llvm.log10.v8f64(<8 x double> [[TMP4:%.*]])
; CHECK:        ret void
;
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %tmp = trunc i64 %iv to i32
  %conv = sitofp i32 %tmp to double
  %call = tail call double @log10(double %conv)
  %arrayidx = getelementptr inbounds double, ptr %varray, i64 %iv
  store double %call, ptr %arrayidx, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}


define void @log10_f64_intrinsic(ptr nocapture %varray) {
; CHECK-LABEL: @log10_f64_intrinsic(
; CHECK-VF2:    [[TMP5:%.*]] = call <2 x double> @hvml_vd2_log10(<2 x double> [[TMP4:%.*]])
; CHECK-VF4:    [[TMP5:%.*]] = call <4 x double> @hvml_vd4_log10(<4 x double> [[TMP4:%.*]])
; CHECK-VF8:    [[TMP5:%.*]] = call <8 x double> @llvm.log10.v8f64(<8 x double> [[TMP4:%.*]])
; CHECK:        ret void
;
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %tmp = trunc i64 %iv to i32
  %conv = sitofp i32 %tmp to double
  %call = tail call double @llvm.log10.f64(double %conv)
  %arrayidx = getelementptr inbounds double, ptr %varray, i64 %iv
  store double %call, ptr %arrayidx, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}


define void @log10_f32(ptr nocapture %varray) {
; CHECK-LABEL: @log10_f32(
; CHECK-VF2:    [[TMP5:%.*]] = call <2 x float> @llvm.log10.v2f32(<2 x float> [[TMP4:%.*]])
; CHECK-VF4:    [[TMP5:%.*]] = call <4 x float> @hvml_vs4_log10(<4 x float> [[TMP4:%.*]])
; CHECK-VF8:    [[TMP5:%.*]] = call <8 x float> @hvml_vs8_log10(<8 x float> [[TMP4:%.*]])
; CHECK:        ret void
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
; CHECK-VF2:    [[TMP5:%.*]] = call <2 x float> @llvm.log10.v2f32(<2 x float> [[TMP4:%.*]])
; CHECK-VF4:    [[TMP5:%.*]] = call <4 x float> @hvml_vs4_log10(<4 x float> [[TMP4:%.*]])
; CHECK-VF8:    [[TMP5:%.*]] = call <8 x float> @hvml_vs8_log10(<8 x float> [[TMP4:%.*]])
; CHECK:        ret void
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



define void @sqrt_f64(ptr nocapture %varray) {
; CHECK-LABEL: @sqrt_f64(
; CHECK-VF2:    [[TMP5:%.*]] = call <2 x double> @hvml_vd2_sqrt(<2 x double> [[TMP4:%.*]])
; CHECK-VF4:    [[TMP5:%.*]] = call <4 x double> @hvml_vd4_sqrt(<4 x double> [[TMP4:%.*]])
; CHECK-VF8:    [[TMP5:%.*]] = tail call double @sqrt(double [[TMP4:%.*]])
; CHECK:        ret void
;
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %tmp = trunc i64 %iv to i32
  %conv = sitofp i32 %tmp to double
  %call = tail call double @sqrt(double %conv)
  %arrayidx = getelementptr inbounds double, ptr %varray, i64 %iv
  store double %call, ptr %arrayidx, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}


define void @sqrt_f32(ptr nocapture %varray) {
; CHECK-LABEL: @sqrt_f32(
; CHECK-VF2:    [[TMP5:%.*]] = call <2 x float> @llvm.sqrt.v2f32(<2 x float> [[TMP4:%.*]])
; CHECK-VF4:    [[TMP5:%.*]] = call <4 x float> @hvml_vs4_sqrt(<4 x float> [[TMP4:%.*]])
; CHECK-VF8:    [[TMP5:%.*]] = call <8 x float> @hvml_vs8_sqrt(<8 x float> [[TMP4:%.*]])
; CHECK:        ret void
;
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %tmp = trunc i64 %iv to i32
  %conv = sitofp i32 %tmp to float
  %call = tail call float @sqrtf(float %conv)
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
; CHECK-VF2:    [[TMP8:%.*]] = call <2 x double> @hvml_vd2_pow(<2 x double> [[TMP4:%.*]], <2 x double> [[WIDE_LOAD:%.*]])
; CHECK-VF4:    [[TMP8:%.*]] = call <4 x double> @hvml_vd4_pow(<4 x double> [[TMP4:%.*]], <4 x double> [[WIDE_LOAD:%.*]])
; CHECK-VF8:    [[TMP8:%.*]] = call <8 x double> @llvm.pow.v8f64(<8 x double> [[TMP4:%.*]], <8 x double> [[WIDE_LOAD:%.*]])
; CHECK:        ret void
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
; CHECK-VF2:    [[TMP8:%.*]] = call <2 x double> @hvml_vd2_pow(<2 x double> [[TMP4:%.*]], <2 x double> [[WIDE_LOAD:%.*]])
; CHECK-VF4:    [[TMP8:%.*]] = call <4 x double> @hvml_vd4_pow(<4 x double> [[TMP4:%.*]], <4 x double> [[WIDE_LOAD:%.*]])
; CHECK-VF8:    [[TMP8:%.*]] = call <8 x double> @llvm.pow.v8f64(<8 x double> [[TMP4:%.*]], <8 x double> [[WIDE_LOAD:%.*]])
; CHECK:        ret void
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
; CHECK-VF2:    [[TMP8:%.*]] = call <2 x float> @llvm.pow.v2f32(<2 x float> [[TMP4:%.*]], <2 x float> [[WIDE_LOAD:%.*]])
; CHECK-VF4:    [[TMP8:%.*]] = call <4 x float> @hvml_vs4_pow(<4 x float> [[TMP4:%.*]], <4 x float> [[WIDE_LOAD:%.*]])
; CHECK-VF8:    [[TMP8:%.*]] = call <8 x float> @hvml_vs8_pow(<8 x float> [[TMP4:%.*]], <8 x float> [[WIDE_LOAD:%.*]])
; CHECK:        ret void
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
; CHECK-VF2:    [[TMP8:%.*]] = call <2 x float> @llvm.pow.v2f32(<2 x float> [[TMP4:%.*]], <2 x float> [[WIDE_LOAD:%.*]])
; CHECK-VF4:    [[TMP8:%.*]] = call <4 x float> @hvml_vs4_pow(<4 x float> [[TMP4:%.*]], <4 x float> [[WIDE_LOAD:%.*]])
; CHECK-VF8:    [[TMP8:%.*]] = call <8 x float> @hvml_vs8_pow(<8 x float> [[TMP4:%.*]], <8 x float> [[WIDE_LOAD:%.*]])
; CHECK:        ret void
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



attributes #0 = { nounwind readnone }

declare double @sin(double) #0
declare float @sinf(float) #0
declare double @cos(double) #0
declare float @cosf(float) #0
declare double @tan(double) #0
declare float @tanf(float) #0
declare double @asin(double) #0
declare float @asinf(float) #0
declare double @acos(double) #0
declare float @acosf(float) #0
declare double @atan(double) #0
declare float @atanf(float) #0
declare double @sinh(double) #0
declare float @sinhf(float) #0
declare double @cosh(double) #0
declare float @coshf(float) #0
declare double @tanh(double) #0
declare float @tanhf(float) #0
declare double @exp(double) #0
declare float @expf(float) #0
declare double @exp2(double) #0
declare float @exp2f(float) #0
declare double @log(double) #0
declare float @logf(float) #0
declare double @log2(double) #0
declare float @log2f(float) #0
declare double @log10(double) #0
declare float @log10f(float) #0
declare double @sqrt(double) #0
declare float @sqrtf(float) #0
declare double @pow(double, double) #0
declare float @powf(float, float) #0
