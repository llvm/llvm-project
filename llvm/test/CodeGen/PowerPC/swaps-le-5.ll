; RUN: llc -verify-machineinstrs -mcpu=pwr8 -mtriple=powerpc64le-unknown-linux-gnu -O3 < %s | FileCheck %s

; These tests verify that VSX swap optimization works for various
; manipulations of <2 x double> vectors.

@x = global <2 x double> <double 9.970000e+01, double -1.032220e+02>, align 16
@z = global <2 x double> <double 2.332000e+01, double 3.111111e+01>, align 16

define void @bar0(double %y) {
entry:
  %0 = load <2 x double>, ptr @x, align 16
  %vecins = insertelement <2 x double> %0, double %y, i32 0
  store <2 x double> %vecins, ptr @z, align 16
  ret void
}

; CHECK-LABEL: @bar0
; CHECK-DAG: xxswapd 1, 1
; CHECK-DAG: lxvd2x [[REG1:[0-9]+]]
; CHECK: xxmrgld [[REG2:[0-9]+]], 1, [[REG1]]
; CHECK: stxvd2x [[REG2]]
; CHECK-NOT: xxswapd

define void @bar1(double %y) {
entry:
  %0 = load <2 x double>, ptr @x, align 16
  %vecins = insertelement <2 x double> %0, double %y, i32 1
  store <2 x double> %vecins, ptr @z, align 16
  ret void
}

; CHECK-LABEL: @bar1
; CHECK-DAG: xxswapd 1, 1
; CHECK-DAG: lxvd2x [[REG1:[0-9]+]]
; CHECK: xxpermdi [[REG2:[0-9]+]], [[REG1]], 1, 1
; CHECK: stxvd2x [[REG2]]
; CHECK-NOT: xxswapd

define void @baz0() {
entry:
  %0 = load <2 x double>, ptr @z, align 16
  %1 = load <2 x double>, ptr @x, align 16
  %vecins = shufflevector <2 x double> %0, <2 x double> %1, <2 x i32> <i32 0, i32 2>
  store <2 x double> %vecins, ptr @z, align 16
  ret void
}

; CHECK-LABEL: @baz0
; CHECK: lxvd2x
; CHECK: lxvd2x
; CHECK: xxmrghd
; CHECK: stxvd2x
; CHECK-NOT: xxswapd

define void @baz1() {
entry:
  %0 = load <2 x double>, ptr @z, align 16
  %1 = load <2 x double>, ptr @x, align 16
  %vecins = shufflevector <2 x double> %0, <2 x double> %1, <2 x i32> <i32 3, i32 1>
  store <2 x double> %vecins, ptr @z, align 16
  ret void
}

; CHECK-LABEL: @baz1
; CHECK: lxvd2x
; CHECK: lxvd2x
; CHECK: xxmrgld
; CHECK: stxvd2x
; CHECK-NOT: xxswapd

