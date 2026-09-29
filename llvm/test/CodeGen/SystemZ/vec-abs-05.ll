; Test f64 and v2f64 absolute.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z13 | FileCheck %s

declare double @llvm.fabs.f64(double)
declare <2 x double> @llvm.fabs.v2f64(<2 x double>)

; Test a plain absolute.
define <2 x double> @f1(<2 x double> %val) {
; CHECK-LABEL: f1:
; CHECK: vflpdb %v24, %v24
; CHECK: br %r14
  %ret = call <2 x double> @llvm.fabs.v2f64(<2 x double> %val)
  ret <2 x double> %ret
}

; Test a negative absolute.
define <2 x double> @f2(<2 x double> %val) {
; CHECK-LABEL: f2:
; CHECK: vflndb %v24, %v24
; CHECK: br %r14
  %abs = call <2 x double> @llvm.fabs.v2f64(<2 x double> %val)
  %ret = fneg <2 x double> %abs
  ret <2 x double> %ret
}

; Test an f64 absolute that uses vector registers.
define double @f3(<2 x double> %val) {
; CHECK-LABEL: f3:
; CHECK: wflpdb %f0, %v24
; CHECK: br %r14
  %scalar = extractelement <2 x double> %val, i32 0
  %ret = call double @llvm.fabs.f64(double %scalar)
  ret double %ret
}

; Test an f64 negative absolute that uses vector registers.
define double @f4(<2 x double> %val) {
; CHECK-LABEL: f4:
; CHECK: wflndb %f0, %v24
; CHECK: br %r14
  %scalar = extractelement <2 x double> %val, i32 0
  %abs = call double @llvm.fabs.f64(double %scalar)
  %ret = fneg double %abs
  ret double %ret
}
