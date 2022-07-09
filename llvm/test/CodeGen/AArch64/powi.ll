; RUN: llc < %s -mtriple=aarch64-- | FileCheck %s

declare double @llvm.powi.f64.i32(double, i32)
declare float  @llvm.powi.f32.i32(float,  i32)
declare float  @pow(double noundef, double noundef)

define float @powi_f32(float %x) nounwind {
; CHECK-LABEL: powi_f32:
; CHECK:       // %bb.0:
; CHECK-NEXT:    fmul s0, s0, s0
; CHECK-NEXT:    fmul s0, s0, s0
; CHECK-NEXT:    ret
  %1 = tail call float @llvm.powi.f32.i32(float %x, i32 4)
  ret float %1
}

define double @powi_f64(double %x) nounwind {
; CHECK-LABEL: powi_f64:
; CHECK:       // %bb.0:
; CHECK-NEXT:    fmul d1, d0, d0
; CHECK-NEXT:    fmul d0, d0, d1
; CHECK-NEXT:    ret
  %1 = tail call double @llvm.powi.f64.i32(double %x, i32 3)
  ret double %1
}
