; RUN: llc -mcpu=cortex-m4 < %s | FileCheck %s

target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-a:0:32-n32-S64"
target triple = "thumbv7em-none--eabi"

; CHECK: vcmp.f32
define double @f(double %a, double %b, double %c, float %d) {
  %1 = fcmp oeq float %d, 0.0
  %2 = select i1 %1, double %a, double %c
  ret double %2
}
