; RUN: opt -S -instcombine < %s | FileCheck %s

declare float @llvm.maxnum.f32(float, float) #0
declare float @llvm.maxnum.v2f32(<2 x float>, <2 x float>) #0
declare <4 x float> @llvm.maxnum.v4f32(<4 x float>, <4 x float>) #0

declare double @llvm.maxnum.f64(double, double) #0
declare <2 x double> @llvm.maxnum.v2f64(<2 x double>, <2 x double>) #0

; CHECK-LABEL: @constant_fold_maxnum_f32
; CHECK-NEXT: ret float 2.000000e+00
define float @constant_fold_maxnum_f32() #0 {
  %x = call float @llvm.maxnum.f32(float 1.0, float 2.0) #0
  ret float %x
}

; CHECK-LABEL: @constant_fold_maxnum_f32_inv
; CHECK-NEXT: ret float 2.000000e+00
define float @constant_fold_maxnum_f32_inv() #0 {
  %x = call float @llvm.maxnum.f32(float 2.0, float 1.0) #0
  ret float %x
}

; CHECK-LABEL: @constant_fold_maxnum_f32_nan0
; CHECK-NEXT: ret float 2.000000e+00
define float @constant_fold_maxnum_f32_nan0() #0 {
  %x = call float @llvm.maxnum.f32(float 0x7FF8000000000000, float 2.0) #0
  ret float %x
}

; CHECK-LABEL: @constant_fold_maxnum_f32_nan1
; CHECK-NEXT: ret float 2.000000e+00
define float @constant_fold_maxnum_f32_nan1() #0 {
  %x = call float @llvm.maxnum.f32(float 2.0, float 0x7FF8000000000000) #0
  ret float %x
}

; CHECK-LABEL: @constant_fold_maxnum_f32_nan_nan
; CHECK-NEXT: ret float 0x7FF8000000000000
define float @constant_fold_maxnum_f32_nan_nan() #0 {
  %x = call float @llvm.maxnum.f32(float 0x7FF8000000000000, float 0x7FF8000000000000) #0
  ret float %x
}

; CHECK-LABEL: @constant_fold_maxnum_f32_p0_p0
; CHECK-NEXT: ret float 0.000000e+00
define float @constant_fold_maxnum_f32_p0_p0() #0 {
  %x = call float @llvm.maxnum.f32(float 0.0, float 0.0) #0
  ret float %x
}

; CHECK-LABEL: @constant_fold_maxnum_f32_p0_n0
; CHECK-NEXT: ret float 0.000000e+00
define float @constant_fold_maxnum_f32_p0_n0() #0 {
  %x = call float @llvm.maxnum.f32(float 0.0, float -0.0) #0
  ret float %x
}

; CHECK-LABEL: @constant_fold_maxnum_f32_n0_p0
; CHECK-NEXT: ret float -0.000000e+00
define float @constant_fold_maxnum_f32_n0_p0() #0 {
  %x = call float @llvm.maxnum.f32(float -0.0, float 0.0) #0
  ret float %x
}

; CHECK-LABEL: @constant_fold_maxnum_f32_n0_n0
; CHECK-NEXT: ret float -0.000000e+00
define float @constant_fold_maxnum_f32_n0_n0() #0 {
  %x = call float @llvm.maxnum.f32(float -0.0, float -0.0) #0
  ret float %x
}

; CHECK-LABEL: @constant_fold_maxnum_v4f32
; CHECK-NEXT: ret <4 x float> <float 2.000000e+00, float 8.000000e+00, float 1.000000e+01, float 9.000000e+00>
define <4 x float> @constant_fold_maxnum_v4f32() #0 {
  %x = call <4 x float> @llvm.maxnum.v4f32(<4 x float> <float 1.0, float 8.0, float 3.0, float 9.0>, <4 x float> <float 2.0, float 2.0, float 10.0, float 5.0>)
  ret <4 x float> %x
}

; CHECK-LABEL: @constant_fold_maxnum_f64
; CHECK-NEXT: ret double 2.000000e+00
define double @constant_fold_maxnum_f64() #0 {
  %x = call double @llvm.maxnum.f64(double 1.0, double 2.0) #0
  ret double %x
}

; CHECK-LABEL: @constant_fold_maxnum_f64_nan0
; CHECK-NEXT: ret double 2.000000e+00
define double @constant_fold_maxnum_f64_nan0() #0 {
  %x = call double @llvm.maxnum.f64(double 0x7FF8000000000000, double 2.0) #0
  ret double %x
}

; CHECK-LABEL: @constant_fold_maxnum_f64_nan1
; CHECK-NEXT: ret double 2.000000e+00
define double @constant_fold_maxnum_f64_nan1() #0 {
  %x = call double @llvm.maxnum.f64(double 2.0, double 0x7FF8000000000000) #0
  ret double %x
}

; CHECK-LABEL: @constant_fold_maxnum_f64_nan_nan
; CHECK-NEXT: ret double 0x7FF8000000000000
define double @constant_fold_maxnum_f64_nan_nan() #0 {
  %x = call double @llvm.maxnum.f64(double 0x7FF8000000000000, double 0x7FF8000000000000) #0
  ret double %x
}

; CHECK-LABEL: @canonicalize_constant_maxnum_f32
; CHECK: call float @llvm.maxnum.f32(float %x, float 1.000000e+00)
define float @canonicalize_constant_maxnum_f32(float %x) #0 {
  %y = call float @llvm.maxnum.f32(float 1.0, float %x) #0
  ret float %y
}

; CHECK-LABEL: @noop_maxnum_f32
; CHECK-NEXT: ret float %x
define float @noop_maxnum_f32(float %x) #0 {
  %y = call float @llvm.maxnum.f32(float %x, float %x) #0
  ret float %y
}

; CHECK-LABEL: @maxnum_f32_nan_val
; CHECK-NEXT: ret float %x
define float @maxnum_f32_nan_val(float %x) #0 {
  %y = call float @llvm.maxnum.f32(float 0x7FF8000000000000, float %x) #0
  ret float %y
}

; CHECK-LABEL: @maxnum_f32_val_nan
; CHECK-NEXT: ret float %x
define float @maxnum_f32_val_nan(float %x) #0 {
  %y = call float @llvm.maxnum.f32(float %x, float 0x7FF8000000000000) #0
  ret float %y
}

; CHECK-LABEL: @fold_maxnum_f32_undef_undef
; CHECK-NEXT: ret float undef
define float @fold_maxnum_f32_undef_undef(float %x) nounwind {
  %val = call float @llvm.maxnum.f32(float undef, float undef) #0
  ret float %val
}

; CHECK-LABEL: @fold_maxnum_f32_val_undef
; CHECK-NEXT: ret float %x
define float @fold_maxnum_f32_val_undef(float %x) nounwind {
  %val = call float @llvm.maxnum.f32(float %x, float undef) #0
  ret float %val
}

; CHECK-LABEL: @fold_maxnum_f32_undef_val
; CHECK-NEXT: ret float %x
define float @fold_maxnum_f32_undef_val(float %x) nounwind {
  %val = call float @llvm.maxnum.f32(float undef, float %x) #0
  ret float %val
}

; CHECK-LABEL: @maxnum_x_maxnum_x_y
; CHECK-NEXT: call float @llvm.maxnum.f32(float %x, float %y)
; CHECK-NEXT: ret float
define float @maxnum_x_maxnum_x_y(float %x, float %y) #0 {
  %a = call float @llvm.maxnum.f32(float %x, float %y) #0
  %b = call float @llvm.maxnum.f32(float %x, float %a) #0
  ret float %b
}

; CHECK-LABEL: @maxnum_y_maxnum_x_y
; CHECK-NEXT: call float @llvm.maxnum.f32(float %x, float %y)
; CHECK-NEXT: ret float
define float @maxnum_y_maxnum_x_y(float %x, float %y) #0 {
  %a = call float @llvm.maxnum.f32(float %x, float %y) #0
  %b = call float @llvm.maxnum.f32(float %y, float %a) #0
  ret float %b
}

; CHECK-LABEL: @maxnum_z_maxnum_x_y
; CHECK-NEXT: call float @llvm.maxnum.f32(float %x, float %y)
; CHECK-NEXT: call float @llvm.maxnum.f32(float %z, float %a)
; CHECK-NEXT: ret float
define float @maxnum_z_maxnum_x_y(float %x, float %y, float %z) #0 {
  %a = call float @llvm.maxnum.f32(float %x, float %y) #0
  %b = call float @llvm.maxnum.f32(float %z, float %a) #0
  ret float %b
}

; CHECK-LABEL: @maxnum_maxnum_x_y_z
; CHECK-NEXT: call float @llvm.maxnum.f32(float %x, float %y)
; CHECK-NEXT: call float @llvm.maxnum.f32(float %a, float %z)
; CHECK-NEXT: ret float
define float @maxnum_maxnum_x_y_z(float %x, float %y, float %z) #0 {
  %a = call float @llvm.maxnum.f32(float %x, float %y) #0
  %b = call float @llvm.maxnum.f32(float %a, float %z) #0
  ret float %b
}

; CHECK-LABEL: @maxnum4
; CHECK-NEXT: call float @llvm.maxnum.f32(float %x, float %y)
; CHECK-NEXT: call float @llvm.maxnum.f32(float %z, float %w)
; CHECK-NEXT: call float @llvm.maxnum.f32(float %a, float %b)
; CHECK-NEXT: ret float
define float @maxnum4(float %x, float %y, float %z, float %w) #0 {
  %a = call float @llvm.maxnum.f32(float %x, float %y) #0
  %b = call float @llvm.maxnum.f32(float %z, float %w) #0
  %c = call float @llvm.maxnum.f32(float %a, float %b) #0
  ret float %c
}

; CHECK-LABEL: @fold_maxnum_f32_inf_val
; CHECK-NEXT: ret float 0x7FF0000000000000
define float @fold_maxnum_f32_inf_val(float %x) nounwind {
  %val = call float @llvm.maxnum.f32(float 0x7FF0000000000000, float %x) #0
  ret float %val
}

; CHECK-LABEL: @fold_maxnum_f32_neginf_val
; CHECK-NEXT: call float @llvm.maxnum.f32(float %x, float 0xFFF0000000000000)
; CHECK-NEXT: ret float
define float @fold_maxnum_f32_neginf_val(float %x) nounwind {
  %val = call float @llvm.maxnum.f32(float 0xFFF0000000000000, float %x) #0
  ret float %val
}

attributes #0 = { nounwind readnone }
