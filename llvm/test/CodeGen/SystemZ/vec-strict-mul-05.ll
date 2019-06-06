; Test vector negative multiply-and-add on z14.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z14 | FileCheck %s

declare <2 x double> @llvm.experimental.constrained.fma.v2f64(<2 x double>, <2 x double>, <2 x double>, metadata, metadata)
declare <4 x float> @llvm.experimental.constrained.fma.v4f32(<4 x float>, <4 x float>, <4 x float>, metadata, metadata)

; Test a v2f64 negative multiply-and-add.
define <2 x double> @f1(<2 x double> %dummy, <2 x double> %val1,
                        <2 x double> %val2, <2 x double> %val3) {
; CHECK-LABEL: f1:
; CHECK: vfnmadb %v24, %v26, %v28, %v30
; CHECK: br %r14
  %ret = call <2 x double> @llvm.experimental.constrained.fma.v2f64 (
                        <2 x double> %val1,
                        <2 x double> %val2,
                        <2 x double> %val3,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict")
  %negret = fsub <2 x double> <double -0.0, double -0.0>, %ret
  ret <2 x double> %negret
}

; Test a v2f64 negative multiply-and-subtract.
define <2 x double> @f2(<2 x double> %dummy, <2 x double> %val1,
                        <2 x double> %val2, <2 x double> %val3) {
; CHECK-LABEL: f2:
; CHECK: vfnmsdb %v24, %v26, %v28, %v30
; CHECK: br %r14
  %negval3 = fsub <2 x double> <double -0.0, double -0.0>, %val3
  %ret = call <2 x double> @llvm.experimental.constrained.fma.v2f64 (
                        <2 x double> %val1,
                        <2 x double> %val2,
                        <2 x double> %negval3,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict")
  %negret = fsub <2 x double> <double -0.0, double -0.0>, %ret
  ret <2 x double> %negret
}

; Test a v4f32 negative multiply-and-add.
define <4 x float> @f3(<4 x float> %dummy, <4 x float> %val1,
                       <4 x float> %val2, <4 x float> %val3) {
; CHECK-LABEL: f3:
; CHECK: vfnmasb %v24, %v26, %v28, %v30
; CHECK: br %r14
  %ret = call <4 x float> @llvm.experimental.constrained.fma.v4f32 (
                        <4 x float> %val1,
                        <4 x float> %val2,
                        <4 x float> %val3,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict")
  %negret = fsub <4 x float> <float -0.0, float -0.0,
                              float -0.0, float -0.0>, %ret
  ret <4 x float> %negret
}

; Test a v4f32 negative multiply-and-subtract.
define <4 x float> @f4(<4 x float> %dummy, <4 x float> %val1,
                       <4 x float> %val2, <4 x float> %val3) {
; CHECK-LABEL: f4:
; CHECK: vfnmssb %v24, %v26, %v28, %v30
; CHECK: br %r14
  %negval3 = fsub <4 x float> <float -0.0, float -0.0,
                               float -0.0, float -0.0>, %val3
  %ret = call <4 x float> @llvm.experimental.constrained.fma.v4f32 (
                        <4 x float> %val1,
                        <4 x float> %val2,
                        <4 x float> %negval3,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict")
  %negret = fsub <4 x float> <float -0.0, float -0.0,
                               float -0.0, float -0.0>, %ret
  ret <4 x float> %negret
}
