; RUN: not opt -S < %s 2>&1 | FileCheck %s

; CHECK: Intrinsic has incorrect return type!
; CHECK-NEXT: float (double, <2 x double>)* @llvm.vector.reduce.fadd.f32.f64.v2f64
define float @fadd_invalid_scalar_res(double %acc, <2 x double> %in) {
  %res = call float @llvm.vector.reduce.fadd.f32.f64.v2f64(double %acc, <2 x double> %in)
  ret float %res
}

; CHECK: Intrinsic has incorrect argument type!
; CHECK-NEXT: double (float, <2 x double>)* @llvm.vector.reduce.fadd.f64.f32.v2f64
define double @fadd_invalid_scalar_start(float %acc, <2 x double> %in) {
  %res = call double @llvm.vector.reduce.fadd.f64.f32.v2f64(float %acc, <2 x double> %in)
  ret double %res
}

; CHECK: Intrinsic has incorrect return type!
; CHECK-NEXT: <2 x double> (double, <2 x double>)* @llvm.vector.reduce.fadd.v2f64.f64.v2f64
define <2 x double> @fadd_invalid_vector_res(double %acc, <2 x double> %in) {
  %res = call <2 x double> @llvm.vector.reduce.fadd.v2f64.f64.v2f64(double %acc, <2 x double> %in)
  ret <2 x double> %res
}

; CHECK: Intrinsic has incorrect argument type!
; CHECK-NEXT: double (<2 x double>, <2 x double>)* @llvm.vector.reduce.fadd.f64.v2f64.v2f64
define double @fadd_invalid_vector_start(<2 x double> %in, <2 x double> %acc) {
  %res = call double @llvm.vector.reduce.fadd.f64.v2f64.v2f64(<2 x double> %acc, <2 x double> %in)
  ret double %res
}

declare float @llvm.vector.reduce.fadd.f32.f64.v2f64(double %acc, <2 x double> %in)
declare double @llvm.vector.reduce.fadd.f64.f32.v2f64(float %acc, <2 x double> %in)
declare double @llvm.vector.reduce.fadd.f64.v2f64.v2f64(<2 x double> %acc, <2 x double> %in)
declare <2 x double> @llvm.vector.reduce.fadd.v2f64.f64.v2f64(double %acc, <2 x double> %in)
