; RUN: opt -S -passes=instcombine %s | FileCheck %s

define float @fabs_fneg_f32(float %x) {
; CHECK-LABEL: define float @fabs_fneg_f32(
; CHECK-NEXT:    [[FABS:%.*]] = call float @llvm.fabs.f32(float [[X:%.*]])
; CHECK-NEXT:    ret float [[FABS]]
;
  %neg = fneg float %x
  %fabs = call float @llvm.fabs.f32(float %neg)
  ret float %fabs
}

define <2 x float> @fabs_fneg_v2f32(<2 x float> %x) {
; CHECK-LABEL: define <2 x float> @fabs_fneg_v2f32(
; CHECK-NEXT:    [[FABS:%.*]] = call <2 x float> @llvm.fabs.v2f32(<2 x float> [[X:%.*]])
; CHECK-NEXT:    ret <2 x float> [[FABS]]
;
  %neg = fneg <2 x float> %x
  %fabs = call <2 x float> @llvm.fabs.v2f32(<2 x float> %neg)
  ret <2 x float> %fabs
}

define double @fabs_fneg_f64(double %x) {
; CHECK-LABEL: define double @fabs_fneg_f64(
; CHECK-NEXT:    [[FABS:%.*]] = call double @llvm.fabs.f64(double [[X:%.*]])
; CHECK-NEXT:    ret double [[FABS]]
;
  %neg = fneg double %x
  %fabs = call double @llvm.fabs.f64(double %neg)
  ret double %fabs
}

define <4 x double> @fabs_fneg_v4f64(<4 x double> %x) {
; CHECK-LABEL: define <4 x double> @fabs_fneg_v4f64(
; CHECK-NEXT:    [[FABS:%.*]] = call <4 x double> @llvm.fabs.v4f64(<4 x double> [[X:%.*]])
; CHECK-NEXT:    ret <4 x double> [[FABS]]
;
  %neg = fneg <4 x double> %x
  %fabs = call <4 x double> @llvm.fabs.v4f64(<4 x double> %neg)
  ret <4 x double> %fabs
}

define half @fabs_fneg_f16(half %x) {
; CHECK-LABEL: define half @fabs_fneg_f16(
; CHECK-NEXT:    [[FABS:%.*]] = call half @llvm.fabs.f16(half [[X:%.*]])
; CHECK-NEXT:    ret half [[FABS]]
;
  %neg = fneg half %x
  %fabs = call half @llvm.fabs.f16(half %neg)
  ret half %fabs
}

declare float @llvm.fabs.f32(float)
declare <2 x float> @llvm.fabs.v2f32(<2 x float>)
declare double @llvm.fabs.f64(double)
declare <4 x double> @llvm.fabs.v4f64(<4 x double>)
declare half @llvm.fabs.f16(half)
