; Test strict vector division.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z13 | FileCheck %s

declare double @llvm.experimental.constrained.fdiv.f64(double, double, metadata, metadata)
declare <2 x double> @llvm.experimental.constrained.fdiv.v2f64(<2 x double>, <2 x double>, metadata, metadata)

; Test a v2f64 division.
define <2 x double> @f5(<2 x double> %dummy, <2 x double> %val1,
                        <2 x double> %val2) #0 {
; CHECK-LABEL: f5:
; CHECK: vfddb %v24, %v26, %v28
; CHECK: br %r14
  %ret = call <2 x double> @llvm.experimental.constrained.fdiv.v2f64(
                        <2 x double> %val1, <2 x double> %val2,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict") #0
  ret <2 x double> %ret
}

; Test an f64 division that uses vector registers.
define double @f6(<2 x double> %val1, <2 x double> %val2) #0 {
; CHECK-LABEL: f6:
; CHECK: wfddb %f0, %v24, %v26
; CHECK: br %r14
  %scalar1 = extractelement <2 x double> %val1, i32 0
  %scalar2 = extractelement <2 x double> %val2, i32 0
  %ret = call double @llvm.experimental.constrained.fdiv.f64(
                        double %scalar1, double %scalar2,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict") #0
  ret double %ret
}

attributes #0 = { strictfp }
