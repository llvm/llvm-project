; RUN: opt -passes=aggressive-instcombine -S < %s | FileCheck %s
; REQUIRES: x86-registered-target

target triple = "x86_64-unknown-linux-gnu"

declare double @sqrt(double)

; CHECK-LABEL: @strict_sqrt(
; CHECK: call double @sqrt(double 2.000000e+00)
; CHECK: ret double %result
define double @strict_sqrt() strictfp {
  %result = call double @sqrt(double 2.0) strictfp
  ret double %result
}

; CHECK-LABEL: @ordinary_sqrt(
; CHECK: ret double f0x3FF6A09E667F3BCD
define double @ordinary_sqrt() {
  %result = call double @sqrt(double 2.0)
  ret double %result
}
