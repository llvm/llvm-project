; Test strict v2f64 rounding.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z13 | FileCheck %s

declare double @llvm.experimental.constrained.rint.f64(double, metadata, metadata)
declare double @llvm.experimental.constrained.nearbyint.f64(double, metadata, metadata)
declare double @llvm.experimental.constrained.floor.f64(double, metadata)
declare double @llvm.experimental.constrained.ceil.f64(double, metadata)
declare double @llvm.experimental.constrained.trunc.f64(double, metadata)
declare double @llvm.experimental.constrained.round.f64(double, metadata)
declare <2 x double> @llvm.experimental.constrained.rint.v2f64(<2 x double>, metadata, metadata)
declare <2 x double> @llvm.experimental.constrained.nearbyint.v2f64(<2 x double>, metadata, metadata)
declare <2 x double> @llvm.experimental.constrained.floor.v2f64(<2 x double>, metadata)
declare <2 x double> @llvm.experimental.constrained.ceil.v2f64(<2 x double>, metadata)
declare <2 x double> @llvm.experimental.constrained.trunc.v2f64(<2 x double>, metadata)
declare <2 x double> @llvm.experimental.constrained.round.v2f64(<2 x double>, metadata)

define <2 x double> @f1(<2 x double> %val) #0 {
; CHECK-LABEL: f1:
; CHECK: vfidb %v24, %v24, 0, 0
; CHECK: br %r14
  %res = call <2 x double> @llvm.experimental.constrained.rint.v2f64(
                        <2 x double> %val,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict") #0
  ret <2 x double> %res
}

define <2 x double> @f2(<2 x double> %val) #0 {
; CHECK-LABEL: f2:
; CHECK: vfidb %v24, %v24, 4, 0
; CHECK: br %r14
  %res = call <2 x double> @llvm.experimental.constrained.nearbyint.v2f64(
                        <2 x double> %val,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict") #0
  ret <2 x double> %res
}

define <2 x double> @f3(<2 x double> %val) #0 {
; CHECK-LABEL: f3:
; CHECK: vfidb %v24, %v24, 4, 7
; CHECK: br %r14
  %res = call <2 x double> @llvm.experimental.constrained.floor.v2f64(
                        <2 x double> %val,
                        metadata !"fpexcept.strict") #0
  ret <2 x double> %res
}

define <2 x double> @f4(<2 x double> %val) #0 {
; CHECK-LABEL: f4:
; CHECK: vfidb %v24, %v24, 4, 6
; CHECK: br %r14
  %res = call <2 x double> @llvm.experimental.constrained.ceil.v2f64(
                        <2 x double> %val,
                        metadata !"fpexcept.strict") #0
  ret <2 x double> %res
}

define <2 x double> @f5(<2 x double> %val) #0 {
; CHECK-LABEL: f5:
; CHECK: vfidb %v24, %v24, 4, 5
; CHECK: br %r14
  %res = call <2 x double> @llvm.experimental.constrained.trunc.v2f64(
                        <2 x double> %val,
                        metadata !"fpexcept.strict") #0
  ret <2 x double> %res
}

define <2 x double> @f6(<2 x double> %val) #0 {
; CHECK-LABEL: f6:
; CHECK: vfidb %v24, %v24, 4, 1
; CHECK: br %r14
  %res = call <2 x double> @llvm.experimental.constrained.round.v2f64(
                        <2 x double> %val,
                        metadata !"fpexcept.strict") #0
  ret <2 x double> %res
}

define double @f7(<2 x double> %val) #0 {
; CHECK-LABEL: f7:
; CHECK: wfidb %f0, %v24, 0, 0
; CHECK: br %r14
  %scalar = extractelement <2 x double> %val, i32 0
  %res = call double @llvm.experimental.constrained.rint.f64(
                        double %scalar,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict") #0
  ret double %res
}

define double @f8(<2 x double> %val) #0 {
; CHECK-LABEL: f8:
; CHECK: wfidb %f0, %v24, 4, 0
; CHECK: br %r14
  %scalar = extractelement <2 x double> %val, i32 0
  %res = call double @llvm.experimental.constrained.nearbyint.f64(
                        double %scalar,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict") #0
  ret double %res
}

define double @f9(<2 x double> %val) #0 {
; CHECK-LABEL: f9:
; CHECK: wfidb %f0, %v24, 4, 7
; CHECK: br %r14
  %scalar = extractelement <2 x double> %val, i32 0
  %res = call double @llvm.experimental.constrained.floor.f64(
                        double %scalar,
                        metadata !"fpexcept.strict") #0
  ret double %res
}


define double @f10(<2 x double> %val) #0 {
; CHECK-LABEL: f10:
; CHECK: wfidb %f0, %v24, 4, 6
; CHECK: br %r14
  %scalar = extractelement <2 x double> %val, i32 0
  %res = call double @llvm.experimental.constrained.ceil.f64(
                        double %scalar,
                        metadata !"fpexcept.strict") #0
  ret double %res
}

define double @f11(<2 x double> %val) #0 {
; CHECK-LABEL: f11:
; CHECK: wfidb %f0, %v24, 4, 5
; CHECK: br %r14
  %scalar = extractelement <2 x double> %val, i32 0
  %res = call double @llvm.experimental.constrained.trunc.f64(
                        double %scalar,
                        metadata !"fpexcept.strict") #0
  ret double %res
}

define double @f12(<2 x double> %val) #0 {
; CHECK-LABEL: f12:
; CHECK: wfidb %f0, %v24, 4, 1
; CHECK: br %r14
  %scalar = extractelement <2 x double> %val, i32 0
  %res = call double @llvm.experimental.constrained.round.f64(
                        double %scalar,
                        metadata !"fpexcept.strict") #0
  ret double %res
}

attributes #0 = { strictfp }
