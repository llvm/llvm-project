; RUN: opt < %s -basicaa -slp-vectorizer -slp-threshold=-999 -dce -S -mtriple=x86_64-apple-macosx10.8.0 -mcpu=corei7-avx | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.8.0"

declare double @sin(double)
declare double @cos(double)
declare double @pow(double, double)
declare double @exp2(double)
declare double @sqrt(double)
declare i64 @round(i64)


; CHECK: sin_libm
; CHECK: call <2 x double> @llvm.sin.v2f64
; CHECK: ret void
define void @sin_libm(double* %a, double* %b, double* %c) {
entry:
  %i0 = load double, double* %a, align 8
  %i1 = load double, double* %b, align 8
  %mul = fmul double %i0, %i1
  %call = tail call double @sin(double %mul) nounwind readnone
  %arrayidx3 = getelementptr inbounds double, double* %a, i64 1
  %i3 = load double, double* %arrayidx3, align 8
  %arrayidx4 = getelementptr inbounds double, double* %b, i64 1
  %i4 = load double, double* %arrayidx4, align 8
  %mul5 = fmul double %i3, %i4
  %call5 = tail call double @sin(double %mul5) nounwind readnone
  store double %call, double* %c, align 8
  %arrayidx5 = getelementptr inbounds double, double* %c, i64 1
  store double %call5, double* %arrayidx5, align 8
  ret void
}

; CHECK: cos_libm
; CHECK: call <2 x double> @llvm.cos.v2f64
; CHECK: ret void
define void @cos_libm(double* %a, double* %b, double* %c) {
entry:
  %i0 = load double, double* %a, align 8
  %i1 = load double, double* %b, align 8
  %mul = fmul double %i0, %i1
  %call = tail call double @cos(double %mul) nounwind readnone
  %arrayidx3 = getelementptr inbounds double, double* %a, i64 1
  %i3 = load double, double* %arrayidx3, align 8
  %arrayidx4 = getelementptr inbounds double, double* %b, i64 1
  %i4 = load double, double* %arrayidx4, align 8
  %mul5 = fmul double %i3, %i4
  %call5 = tail call double @cos(double %mul5) nounwind readnone
  store double %call, double* %c, align 8
  %arrayidx5 = getelementptr inbounds double, double* %c, i64 1
  store double %call5, double* %arrayidx5, align 8
  ret void
}

; CHECK: pow_libm
; CHECK: call <2 x double> @llvm.pow.v2f64
; CHECK: ret void
define void @pow_libm(double* %a, double* %b, double* %c) {
entry:
  %i0 = load double, double* %a, align 8
  %i1 = load double, double* %b, align 8
  %mul = fmul double %i0, %i1
  %call = tail call double @pow(double %mul,double %mul) nounwind readnone
  %arrayidx3 = getelementptr inbounds double, double* %a, i64 1
  %i3 = load double, double* %arrayidx3, align 8
  %arrayidx4 = getelementptr inbounds double, double* %b, i64 1
  %i4 = load double, double* %arrayidx4, align 8
  %mul5 = fmul double %i3, %i4
  %call5 = tail call double @pow(double %mul5,double %mul5) nounwind readnone
  store double %call, double* %c, align 8
  %arrayidx5 = getelementptr inbounds double, double* %c, i64 1
  store double %call5, double* %arrayidx5, align 8
  ret void
}


; CHECK: exp2_libm
; CHECK: call <2 x double> @llvm.exp2.v2f64
; CHECK: ret void
define void @exp2_libm(double* %a, double* %b, double* %c) {
entry:
  %i0 = load double, double* %a, align 8
  %i1 = load double, double* %b, align 8
  %mul = fmul double %i0, %i1
  %call = tail call double @exp2(double %mul) nounwind readnone
  %arrayidx3 = getelementptr inbounds double, double* %a, i64 1
  %i3 = load double, double* %arrayidx3, align 8
  %arrayidx4 = getelementptr inbounds double, double* %b, i64 1
  %i4 = load double, double* %arrayidx4, align 8
  %mul5 = fmul double %i3, %i4
  %call5 = tail call double @exp2(double %mul5) nounwind readnone
  store double %call, double* %c, align 8
  %arrayidx5 = getelementptr inbounds double, double* %c, i64 1
  store double %call5, double* %arrayidx5, align 8
  ret void
}


; CHECK: sqrt_libm
; CHECK: call nnan <2 x double> @llvm.sqrt.v2f64
; CHECK: ret void
define void @sqrt_libm(double* %a, double* %b, double* %c) {
entry:
  %i0 = load double, double* %a, align 8
  %i1 = load double, double* %b, align 8
  %mul = fmul double %i0, %i1
  %call = tail call nnan double @sqrt(double %mul) nounwind readnone
  %arrayidx3 = getelementptr inbounds double, double* %a, i64 1
  %i3 = load double, double* %arrayidx3, align 8
  %arrayidx4 = getelementptr inbounds double, double* %b, i64 1
  %i4 = load double, double* %arrayidx4, align 8
  %mul5 = fmul double %i3, %i4
  %call5 = tail call nnan double @sqrt(double %mul5) nounwind readnone
  store double %call, double* %c, align 8
  %arrayidx5 = getelementptr inbounds double, double* %c, i64 1
  store double %call5, double* %arrayidx5, align 8
  ret void
}


; Negative test case
; CHECK: round_custom
; CHECK-NOT: load <4 x i64>
; CHECK: ret void
define void @round_custom(i64* %a, i64* %b, i64* %c) {
entry:
  %i0 = load i64, i64* %a, align 8
  %i1 = load i64, i64* %b, align 8
  %mul = mul i64 %i0, %i1
  %call = tail call i64 @round(i64 %mul) nounwind readnone
  %arrayidx3 = getelementptr inbounds i64, i64* %a, i64 1
  %i3 = load i64, i64* %arrayidx3, align 8
  %arrayidx4 = getelementptr inbounds i64, i64* %b, i64 1
  %i4 = load i64, i64* %arrayidx4, align 8
  %mul5 = mul i64 %i3, %i4
  %call5 = tail call i64 @round(i64 %mul5) nounwind readnone
  store i64 %call, i64* %c, align 8
  %arrayidx5 = getelementptr inbounds i64, i64* %c, i64 1
  store i64 %call5, i64* %arrayidx5, align 8
  ret void
}


; CHECK: declare <2 x double> @llvm.sin.v2f64(<2 x double>) [[ATTR0:#[0-9]+]]
; CHECK: declare <2 x double> @llvm.cos.v2f64(<2 x double>) [[ATTR0]]
; CHECK: declare <2 x double> @llvm.pow.v2f64(<2 x double>, <2 x double>) [[ATTR0]]
; CHECK: declare <2 x double> @llvm.exp2.v2f64(<2 x double>) [[ATTR0]]

; CHECK: attributes [[ATTR0]] = { nounwind readnone speculatable }

