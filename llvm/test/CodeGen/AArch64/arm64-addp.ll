; RUN: llc < %s -mtriple=arm64-eabi -aarch64-neon-syntax=apple -mcpu=cyclone | FileCheck %s

define double @foo(<2 x double> %a) nounwind {
; CHECK-LABEL: foo:
; CHECK: faddp.2d d0, v0
; CHECK-NEXT: ret
  %lane0.i = extractelement <2 x double> %a, i32 0
  %lane1.i = extractelement <2 x double> %a, i32 1
  %vpaddd.i = fadd double %lane0.i, %lane1.i
  ret double %vpaddd.i
}

define i64 @foo0(<2 x i64> %a) nounwind {
; CHECK-LABEL: foo0:
; CHECK: addp.2d d0, v0
; CHECK-NEXT: fmov x0, d0
; CHECK-NEXT: ret
  %lane0.i = extractelement <2 x i64> %a, i32 0
  %lane1.i = extractelement <2 x i64> %a, i32 1
  %vpaddd.i = add i64 %lane0.i, %lane1.i
  ret i64 %vpaddd.i
}

define float @foo1(<2 x float> %a) nounwind {
; CHECK-LABEL: foo1:
; CHECK: faddp.2s
; CHECK-NEXT: ret
  %lane0.i = extractelement <2 x float> %a, i32 0
  %lane1.i = extractelement <2 x float> %a, i32 1
  %vpaddd.i = fadd float %lane0.i, %lane1.i
  ret float %vpaddd.i
}
