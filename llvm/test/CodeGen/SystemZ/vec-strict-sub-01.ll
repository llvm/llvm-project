; Test strict vector subtraction.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z13 | FileCheck %s

declare double @llvm.experimental.constrained.fsub.f64(double, double, metadata, metadata)
declare <2 x double> @llvm.experimental.constrained.fsub.v2f64(<2 x double>, <2 x double>, metadata, metadata)

; Test a v2f64 subtraction.
define <2 x double> @f6(<2 x double> %dummy, <2 x double> %val1,
                        <2 x double> %val2) {
; CHECK-LABEL: f6:
; CHECK: vfsdb %v24, %v26, %v28
; CHECK: br %r14
  %ret = call <2 x double> @llvm.experimental.constrained.fsub.v2f64(
                        <2 x double> %val1, <2 x double> %val2,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict")
  ret <2 x double> %ret
}

; Test an f64 subtraction that uses vector registers.
define double @f7(<2 x double> %val1, <2 x double> %val2) {
; CHECK-LABEL: f7:
; CHECK: wfsdb %f0, %v24, %v26
; CHECK: br %r14
  %scalar1 = extractelement <2 x double> %val1, i32 0
  %scalar2 = extractelement <2 x double> %val2, i32 0
  %ret = call double @llvm.experimental.constrained.fsub.f64(
                        double %scalar1, double %scalar2,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict")
  ret double %ret
}

