; RUN: opt -S -instcombine < %s | FileCheck %s

declare float @llvm.fma.f32(float, float, float) #0
declare float @llvm.fmuladd.f32(float, float, float) #0
declare <4 x float> @llvm.fma.v4f32(<4 x float>, <4 x float>, <4 x float>) #0

declare double @llvm.fma.f64(double, double, double) #0
declare double @llvm.fmuladd.f64(double, double, double) #0

declare double @llvm.sqrt.f64(double) #0


; CHECK-LABEL: @constant_fold_fma_f32
; CHECK-NEXT: ret float 6.000000e+00
define float @constant_fold_fma_f32() #0 {
  %x = call float @llvm.fma.f32(float 1.0, float 2.0, float 4.0) #0
  ret float %x
}

; CHECK-LABEL: @constant_fold_fma_v4f32
; CHECK-NEXT: ret <4 x float> <float 1.200000e+01, float 1.400000e+01, float 1.600000e+01, float 1.800000e+01>
define <4 x float> @constant_fold_fma_v4f32() #0 {
  %x = call <4 x float> @llvm.fma.v4f32(<4 x float> <float 1.0, float 2.0, float 3.0, float 4.0>, <4 x float> <float 2.0, float 2.0, float 2.0, float 2.0>, <4 x float> <float 10.0, float 10.0, float 10.0, float 10.0>)
  ret <4 x float> %x
}

; CHECK-LABEL: @constant_fold_fmuladd_f32
; CHECK-NEXT: ret float 6.000000e+00
define float @constant_fold_fmuladd_f32() #0 {
  %x = call float @llvm.fmuladd.f32(float 1.0, float 2.0, float 4.0) #0
  ret float %x
}

; CHECK-LABEL: @constant_fold_fma_f64
; CHECK-NEXT: ret double 6.000000e+00
define double @constant_fold_fma_f64() #0 {
  %x = call double @llvm.fma.f64(double 1.0, double 2.0, double 4.0) #0
  ret double %x
}

; CHECK-LABEL: @constant_fold_fmuladd_f64
; CHECK-NEXT: ret double 6.000000e+00
define double @constant_fold_fmuladd_f64() #0 {
  %x = call double @llvm.fmuladd.f64(double 1.0, double 2.0, double 4.0) #0
  ret double %x
}

; PR32177

; CHECK-LABEL: @constant_fold_frem_f32
; CHECK-NEXT: ret float 0x41A61B2000000000
define float @constant_fold_frem_f32() #0 {
  %x = frem float 0x43cbfcd960000000, 0xc1e2b34a00000000
  ret float %x
}

; PR3316

; CHECK-LABEL: @constant_fold_frem_f64
; CHECK-NEXT: ret double 0.000000e+00
define double @constant_fold_frem_f64() {
  %x = frem double 0x43E0000000000000, 1.000000e+00
  ret double %x
}

attributes #0 = { nounwind readnone }
