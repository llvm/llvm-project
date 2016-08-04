; RUN: llc -verify-machineinstrs < %s -mcpu=pwr7 -mattr=+vsx | FileCheck %s
; RUN: llc -verify-machineinstrs < %s -mcpu=pwr7 -mattr=+vsx -fast-isel -O0 | FileCheck %s
target datalayout = "E-m:e-i64:64-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

declare <2 x double> @sv(<2 x double>, <2 x i64>, <4 x float>) #0

define <2 x double> @main(<4 x float> %a, <2 x double> %b, <2 x i64> %c) #1 {
entry:
  %ca = tail call <2 x double> @sv(<2 x double> %b, <2 x i64> %c,  <4 x float> %a)
  %v = fadd <2 x double> %ca, <double 1.0, double 1.0>
  ret <2 x double> %v

; CHECK-LABEL: @main
; CHECK-DAG: vor [[V:[0-9]+]], 2, 2
; CHECK-DAG: xxlor 34, 35, 35
; CHECK-DAG: xxlor 35, 36, 36
; CHECK-DAG: vor 4, [[V]], [[V]]
; CHECK-DAG: bl sv
; CHECK-DAG: lxvd2x [[VC:[0-9]+]],
; CHECK: xvadddp 34, 34, [[VC]]
; CHECK: blr
}

attributes #0 = { noinline nounwind readnone }
attributes #1 = { nounwind }

