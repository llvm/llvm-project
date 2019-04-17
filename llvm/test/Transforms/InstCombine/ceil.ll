; RUN: opt -S -instcombine < %s | FileCheck %s

declare float @llvm.ceil.f32(float) #0
declare double @llvm.ceil.f64(double) #0
declare <4 x float> @llvm.ceil.v4f32(<4 x float>) #0

; CHECK-LABEL: @constant_fold_ceil_f32_01
; CHECK-NEXT: ret float 1.000000e+00
define float @constant_fold_ceil_f32_01() #0 {
  %x = call float @llvm.ceil.f32(float 1.00) #0
  ret float %x
}

; CHECK-LABEL: @constant_fold_ceil_f32_02
; CHECK-NEXT: ret float 2.000000e+00
define float @constant_fold_ceil_f32_02() #0 {
  %x = call float @llvm.ceil.f32(float 1.25) #0
  ret float %x
}

; CHECK-LABEL: @constant_fold_ceil_f32_03
; CHECK-NEXT: ret float -1.000000e+00
define float @constant_fold_ceil_f32_03() #0 {
  %x = call float @llvm.ceil.f32(float -1.25) #0
  ret float %x
}

; CHECK-LABEL: @constant_fold_ceil_v4f32_01
; CHECK-NEXT: ret <4 x float> <float 1.000000e+00, float 2.000000e+00, float -1.000000e+00, float -1.000000e+00>
define <4 x float> @constant_fold_ceil_v4f32_01() #0 {
  %x = call <4 x float> @llvm.ceil.v4f32(<4 x float> <float 1.00, float 1.25, float -1.25, float -1.00>)
  ret <4 x float> %x
}

; CHECK-LABEL: @constant_fold_ceil_f64_01
; CHECK-NEXT: ret double 1.000000e+00
define double @constant_fold_ceil_f64_01() #0 {
  %x = call double @llvm.ceil.f64(double 1.0) #0
  ret double %x
}

; CHECK-LABEL: @constant_fold_ceil_f64_02
; CHECK-NEXT: ret double 2.000000e+00
define double @constant_fold_ceil_f64_02() #0 {
  %x = call double @llvm.ceil.f64(double 1.3) #0
  ret double %x
}

; CHECK-LABEL: @constant_fold_ceil_f64_03
; CHECK-NEXT: ret double -1.000000e+00
define double @constant_fold_ceil_f64_03() #0 {
  %x = call double @llvm.ceil.f64(double -1.75) #0
  ret double %x
}

attributes #0 = { nounwind readnone }
