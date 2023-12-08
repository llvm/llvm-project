; RUN: llc < %s -mtriple=i386-apple-darwin10  -mattr=+fma,-fma4  | FileCheck %s
; RUN: llc < %s -mtriple=x86_64-apple-darwin10 -mattr=+fma,-fma4 | FileCheck %s
; RUN: llc < %s -mtriple=i686-- -mcpu=bdver2 -mattr=-fma4  | FileCheck %s

; Test FMA3 variant selection

; CHECK-LABEL: fma3_select231ssX:
; CHECK: vfmadd231ss %xmm
define float @fma3_select231ssX(float %x, float %y) {
entry:
  br label %while.body
while.body:
  %acc.01 = phi float [ 0.000000e+00, %entry ], [ %acc, %while.body ]
  %acc = call float @llvm.fma.f32(float %x, float %y, float %acc.01)
  %b = fcmp ueq float %acc, 0.0
  br i1 %b, label %while.body, label %while.end
while.end:
  ret float %acc
}

; CHECK-LABEL: fma3_select231pdY:
; CHECK: vfmadd231pd %ymm
define <4 x double> @fma3_select231pdY(<4 x double> %x, <4 x double> %y) {
entry:
  br label %while.body
while.body:
  %acc.04 = phi <4 x double> [ zeroinitializer, %entry ], [ %add, %while.body ]
  %add = call <4 x double> @llvm.fma.v4f64(<4 x double> %x, <4 x double> %y, <4 x double> %acc.04)
  %vecext = extractelement <4 x double> %add, i32 0
  %cmp = fcmp oeq double %vecext, 0.000000e+00
  br i1 %cmp, label %while.body, label %while.end
while.end:
  ret <4 x double> %add
}

declare float @llvm.fma.f32(float, float, float)
declare <4 x double> @llvm.fma.v4f64(<4 x double>, <4 x double>, <4 x double>)
