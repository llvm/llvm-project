; RUN: llc -mtriple=arm64-eabi < %s | FileCheck %s

define float @fma32(float %a, float %b, float %c) nounwind readnone ssp {
entry:
; CHECK-LABEL: fma32:
; CHECK: fmadd s0, s0, s1, s2
  %0 = tail call float @llvm.fma.f32(float %a, float %b, float %c)
  ret float %0
}

define float @fnma32(float %a, float %b, float %c) nounwind readnone ssp {
entry:
; CHECK-LABEL: fnma32:
; CHECK: fnmadd s0, s0, s1, s2
  %0 = tail call float @llvm.fma.f32(float %a, float %b, float %c)
  %mul = fmul float %0, -1.000000e+00
  ret float %mul
}

define float @fms32(float %a, float %b, float %c) nounwind readnone ssp {
entry:
; CHECK-LABEL: fms32:
; CHECK: fmsub s0, s0, s1, s2
  %mul = fmul float %b, -1.000000e+00
  %0 = tail call float @llvm.fma.f32(float %a, float %mul, float %c)
  ret float %0
}

define float @fms32_com(float %a, float %b, float %c) nounwind readnone ssp {
entry:
; CHECK-LABEL: fms32_com:
; CHECK: fmsub s0, s1, s0, s2
  %mul = fmul float %b, -1.000000e+00
  %0 = tail call float @llvm.fma.f32(float %mul, float %a, float %c)
  ret float %0
}

define float @fnms32(float %a, float %b, float %c) nounwind readnone ssp {
entry:
; CHECK-LABEL: fnms32:
; CHECK: fnmsub s0, s0, s1, s2
  %mul = fmul float %c, -1.000000e+00
  %0 = tail call float @llvm.fma.f32(float %a, float %b, float %mul)
  ret float %0
}

define double @fma64(double %a, double %b, double %c) nounwind readnone ssp {
; CHECK-LABEL: fma64:
; CHECK: fmadd d0, d0, d1, d2
entry:
  %0 = tail call double @llvm.fma.f64(double %a, double %b, double %c)
  ret double %0
}

define double @fnma64(double %a, double %b, double %c) nounwind readnone ssp {
; CHECK-LABEL: fnma64:
; CHECK: fnmadd d0, d0, d1, d2
entry:
  %0 = tail call double @llvm.fma.f64(double %a, double %b, double %c)
  %mul = fmul double %0, -1.000000e+00
  ret double %mul
}

define double @fms64(double %a, double %b, double %c) nounwind readnone ssp {
; CHECK-LABEL: fms64:
; CHECK: fmsub d0, d0, d1, d2
entry:
  %mul = fmul double %b, -1.000000e+00
  %0 = tail call double @llvm.fma.f64(double %a, double %mul, double %c)
  ret double %0
}

define double @fms64_com(double %a, double %b, double %c) nounwind readnone ssp {
; CHECK-LABEL: fms64_com:
; CHECK: fmsub d0, d1, d0, d2
entry:
  %mul = fmul double %b, -1.000000e+00
  %0 = tail call double @llvm.fma.f64(double %mul, double %a, double %c)
  ret double %0
}

define double @fnms64(double %a, double %b, double %c) nounwind readnone ssp {
; CHECK-LABEL: fnms64:
; CHECK: fnmsub d0, d0, d1, d2
entry:
  %mul = fmul double %c, -1.000000e+00
  %0 = tail call double @llvm.fma.f64(double %a, double %b, double %mul)
  ret double %0
}

; This would crash while trying getNegatedExpression().

define float @negated_constant(float %x) {
; CHECK-LABEL: negated_constant:
; CHECK:       // %bb.0:
; CHECK-NEXT:    mov w8, #-1037565952
; CHECK-NEXT:    mov w9, #1109917696
; CHECK-NEXT:    fmov s1, w8
; CHECK-NEXT:    fmul s1, s0, s1
; CHECK-NEXT:    fmov s2, w9
; CHECK-NEXT:    fmadd s0, s0, s2, s1
; CHECK-NEXT:    ret
  %m = fmul float %x, 42.0
  %fma = call nsz float @llvm.fma.f32(float %x, float -42.0, float %m)
  %nfma = fneg float %fma
  ret float %nfma
}

declare float @llvm.fma.f32(float, float, float) nounwind readnone
declare double @llvm.fma.f64(double, double, double) nounwind readnone
