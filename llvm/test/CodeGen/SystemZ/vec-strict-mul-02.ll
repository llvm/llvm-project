; Test strict vector multiply-and-add.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z13 | FileCheck %s

declare <2 x double> @llvm.experimental.constrained.fma.v2f64(<2 x double>, <2 x double>, <2 x double>, metadata, metadata)

; Test a v2f64 multiply-and-add.
define <2 x double> @f4(<2 x double> %dummy, <2 x double> %val1,
                        <2 x double> %val2, <2 x double> %val3) #0 {
; CHECK-LABEL: f4:
; CHECK: vfmadb %v24, %v26, %v28, %v30
; CHECK: br %r14
  %ret = call <2 x double> @llvm.experimental.constrained.fma.v2f64 (
                        <2 x double> %val1,
                        <2 x double> %val2,
                        <2 x double> %val3,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict") #0
  ret <2 x double> %ret
}

; Test a v2f64 multiply-and-subtract.
define <2 x double> @f5(<2 x double> %dummy, <2 x double> %val1,
                        <2 x double> %val2, <2 x double> %val3) #0 {
; CHECK-LABEL: f5:
; CHECK: vfmsdb %v24, %v26, %v28, %v30
; CHECK: br %r14
  %negval3 = fneg <2 x double> %val3
  %ret = call <2 x double> @llvm.experimental.constrained.fma.v2f64 (
                        <2 x double> %val1,
                        <2 x double> %val2,
                        <2 x double> %negval3,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict") #0
  ret <2 x double> %ret
}

attributes #0 = { strictfp }
