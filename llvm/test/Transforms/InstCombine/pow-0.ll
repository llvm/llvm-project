; RUN: opt < %s -instcombine -S | FileCheck %s

; CHECK-LABEL: @zero(
; CHECK-NEXT:  ret double 1.000000e+00
define double @zero(double %value) {
  %res = call double @llvm.pow.f64(double %value, double 0.000000e+00)
  ret double %res
}

; CHECK-LABEL: @minus_zero(
; CHECK-NEXT:  ret double 1.000000e+00
define double @minus_zero(double %value) {
  %res = call double @llvm.pow.f64(double %value, double -0.000000e+00)
  ret double %res
}

; CHECK-LABEL: @fast_zero(
; CHECK-NEXT:  ret double 1.000000e+00
define double @fast_zero(double %value) {
  %res = call fast double @llvm.pow.f64(double %value, double 0.000000e+00)
  ret double %res
}

; CHECK-LABEL: @fast_minus_zero(
; CHECK-NEXT:  ret double 1.000000e+00
define double @fast_minus_zero(double %value) {
  %res = call fast double @llvm.pow.f64(double %value, double -0.000000e+00)
  ret double %res
}

; CHECK-LABEL: @vec_zero(
; CHECK-NEXT:  ret <2 x double> <double 1.000000e+00, double 1.000000e+00>
define <2 x double> @vec_zero(<2 x double> %value) {
  %res = call <2 x double> @llvm.pow.v2f64(<2 x double> %value, <2 x double> <double 0.000000e+00, double 0.000000e+00>)
  ret <2 x double> %res
}

; CHECK-LABEL: @vec_minus_zero(
; CHECK-NEXT:  ret <2 x double> <double 1.000000e+00, double 1.000000e+00>
define <2 x double> @vec_minus_zero(<2 x double> %value) {
  %res = call <2 x double> @llvm.pow.v2f64(<2 x double> %value, <2 x double> <double -0.000000e+00, double -0.000000e+00>)
  ret <2 x double> %res
}

; CHECK-LABEL: @vec_fast_zero(
; CHECK-NEXT:  ret <2 x double> <double 1.000000e+00, double 1.000000e+00>
define <2 x double> @vec_fast_zero(<2 x double> %value) {
  %res = call fast <2 x double> @llvm.pow.v2f64(<2 x double> %value, <2 x double> <double 0.000000e+00, double 0.000000e+00>)
  ret <2 x double> %res
}

; CHECK-LABEL: @vec_fast_minus_zero(
; CHECK-NEXT:  ret <2 x double> <double 1.000000e+00, double 1.000000e+00>
define <2 x double> @vec_fast_minus_zero(<2 x double> %value) {
  %res = call fast <2 x double> @llvm.pow.v2f64(<2 x double> %value, <2 x double> <double -0.000000e+00, double -0.000000e+00>)
  ret <2 x double> %res
}

declare double @llvm.pow.f64(double, double)
declare <2 x double> @llvm.pow.v2f64(<2 x double>, <2 x double>)
