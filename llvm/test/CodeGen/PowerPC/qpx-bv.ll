; RUN: llc -verify-machineinstrs < %s -mcpu=a2q | FileCheck %s

target datalayout = "E-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-f128:128:128-v128:128:128-n32:64"
target triple = "powerpc64-bgq-linux"

define <4 x double> @foo(double %f1, double %f2, double %f3, double %f4) {
  %v1 = insertelement <4 x double> undef, double %f1, i32 0
  %v2 = insertelement <4 x double> %v1,   double %f2, i32 1
  %v3 = insertelement <4 x double> %v2,   double %f3, i32 2
  %v4 = insertelement <4 x double> %v3,   double %f4, i32 3
  ret <4 x double> %v4

; CHECK-LABEL: @foo
; CHECK: qvgpci [[REG1:[0-9]+]], 275
; CHECK-DAG: qvgpci [[REG2:[0-9]+]], 101
; CHECK-DAG: qvfperm [[REG3:[0-9]+]], 3, 4, [[REG1]]
; CHECK-DAG: qvfperm [[REG4:[0-9]+]], 1, 2, [[REG1]]
; CHECK-DAG: qvfperm 1, [[REG4]], [[REG3]], [[REG2]]
; CHECK: blr
}

define <4 x float> @goo(float %f1, float %f2, float %f3, float %f4) {
  %v1 = insertelement <4 x float> undef, float %f1, i32 0
  %v2 = insertelement <4 x float> %v1,   float %f2, i32 1
  %v3 = insertelement <4 x float> %v2,   float %f3, i32 2
  %v4 = insertelement <4 x float> %v3,   float %f4, i32 3
  ret <4 x float> %v4

; CHECK-LABEL: @goo
; CHECK: qvgpci [[REG1:[0-9]+]], 275
; CHECK-DAG: qvgpci [[REG2:[0-9]+]], 101
; CHECK-DAG: qvfperm [[REG3:[0-9]+]], 3, 4, [[REG1]]
; CHECK-DAG: qvfperm [[REG4:[0-9]+]], 1, 2, [[REG1]]
; CHECK-DAG: qvfperm 1, [[REG4]], [[REG3]], [[REG2]]
; CHECK: blr
}

