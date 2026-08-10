; Test strict vector addition.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z13 | FileCheck %s

declare double @llvm.experimental.constrained.fadd.f64(double, double, metadata, metadata)
declare <2 x double> @llvm.experimental.constrained.fadd.v2f64(<2 x double>, <2 x double>, metadata, metadata)

; Test a v2f64 addition.
define <2 x double> @f5(<2 x double> %dummy, <2 x double> %val1,
                        <2 x double> %val2) strictfp {
; CHECK-LABEL: f5:
; CHECK: vfadb %v24, %v26, %v28
; CHECK: br %r14
  %ret = call <2 x double> @llvm.experimental.constrained.fadd.v2f64(
                        <2 x double> %val1, <2 x double> %val2,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict") strictfp
  ret <2 x double> %ret
}

; Test an f64 addition that uses vector registers.
define double @f6(<2 x double> %val1, <2 x double> %val2) strictfp {
; CHECK-LABEL: f6:
; CHECK: wfadb %f0, %v24, %v26
; CHECK: br %r14
  %scalar1 = extractelement <2 x double> %val1, i32 0
  %scalar2 = extractelement <2 x double> %val2, i32 0
  %ret = call double @llvm.experimental.constrained.fadd.f64(
                        double %scalar1, double %scalar2,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict") strictfp
  ret double %ret
}
