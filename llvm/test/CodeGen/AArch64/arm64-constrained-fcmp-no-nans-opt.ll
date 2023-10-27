; RUN: llc < %s -mtriple=arm64-eabi -mattr=+fullfp16 -enable-no-nans-fp-math | FileCheck %s

declare i1 @llvm.experimental.constrained.fcmp.f32(float, float, metadata, metadata)
declare i1 @llvm.experimental.constrained.fcmp.f64(double, double, metadata, metadata)

; CHECK-LABEL: @f32_constrained_fcmp_ueq
; CHECK: fcmp s0, s1
; CHECK-NEXT: cset w0, eq
; CHECK-NEXT: ret
define i1 @f32_constrained_fcmp_ueq(float %a, float %b) nounwind ssp strictfp {
  %cmp = tail call i1 @llvm.experimental.constrained.fcmp.f32(float %a, float %b, metadata !"ueq", metadata !"fpexcept.strict")
  ret i1 %cmp
}

; CHECK-LABEL: @f32_constrained_fcmp_une
; CHECK: fcmp s0, s1
; CHECK-NEXT: cset w0, ne
; CHECK-NEXT: ret
define i1 @f32_constrained_fcmp_une(float %a, float %b) nounwind ssp strictfp {
  %cmp = tail call i1 @llvm.experimental.constrained.fcmp.f32(float %a, float %b, metadata !"une", metadata !"fpexcept.strict")
  ret i1 %cmp
}

; CHECK-LABEL: @f32_constrained_fcmp_ugt
; CHECK: fcmp s0, s1
; CHECK-NEXT: cset w0, gt
; CHECK-NEXT: ret
define i1 @f32_constrained_fcmp_ugt(float %a, float %b) nounwind ssp strictfp {
  %cmp = tail call i1 @llvm.experimental.constrained.fcmp.f32(float %a, float %b, metadata !"ugt", metadata !"fpexcept.strict")
  ret i1 %cmp
}

; CHECK-LABEL: @f32_constrained_fcmp_uge
; CHECK: fcmp s0, s1
; CHECK-NEXT: cset w0, ge
; CHECK-NEXT: ret
define i1 @f32_constrained_fcmp_uge(float %a, float %b) nounwind ssp strictfp {
  %cmp = tail call i1 @llvm.experimental.constrained.fcmp.f32(float %a, float %b, metadata !"uge", metadata !"fpexcept.strict")
  ret i1 %cmp
}

; CHECK-LABEL: @f32_constrained_fcmp_ult
; CHECK: fcmp s0, s1
; CHECK-NEXT: cset w0, lt
; CHECK-NEXT: ret
define i1 @f32_constrained_fcmp_ult(float %a, float %b) nounwind ssp strictfp {
  %cmp = tail call i1 @llvm.experimental.constrained.fcmp.f32(float %a, float %b, metadata !"ult", metadata !"fpexcept.strict")
  ret i1 %cmp
}

; CHECK-LABEL: @f32_constrained_fcmp_ule
; CHECK: fcmp s0, s1
; CHECK-NEXT: cset w0, le
; CHECK-NEXT: ret
define i1 @f32_constrained_fcmp_ule(float %a, float %b) nounwind ssp strictfp {
  %cmp = tail call i1 @llvm.experimental.constrained.fcmp.f32(float %a, float %b, metadata !"ule", metadata !"fpexcept.strict")
  ret i1 %cmp
}

; CHECK-LABEL: @f64_constrained_fcmp_ueq
; CHECK: fcmp d0, d1
; CHECK-NEXT: cset w0, eq
; CHECK-NEXT: ret
define i1 @f64_constrained_fcmp_ueq(double %a, double %b) nounwind ssp strictfp {
  %cmp = tail call i1 @llvm.experimental.constrained.fcmp.f64(double %a, double %b, metadata !"ueq", metadata !"fpexcept.strict")
  ret i1 %cmp
}

; CHECK-LABEL: @f64_constrained_fcmp_une
; CHECK: fcmp d0, d1
; CHECK-NEXT: cset w0, ne
; CHECK-NEXT: ret
define i1 @f64_constrained_fcmp_une(double %a, double %b) nounwind ssp strictfp {
  %cmp = tail call i1 @llvm.experimental.constrained.fcmp.f64(double %a, double %b, metadata !"une", metadata !"fpexcept.strict")
  ret i1 %cmp
}

; CHECK-LABEL: @f64_constrained_fcmp_ugt
; CHECK: fcmp d0, d1
; CHECK-NEXT: cset w0, gt
; CHECK-NEXT: ret
define i1 @f64_constrained_fcmp_ugt(double %a, double %b) nounwind ssp strictfp {
  %cmp = tail call i1 @llvm.experimental.constrained.fcmp.f64(double %a, double %b, metadata !"ugt", metadata !"fpexcept.strict")
  ret i1 %cmp
}

; CHECK-LABEL: @f64_constrained_fcmp_uge
; CHECK: fcmp d0, d1
; CHECK-NEXT: cset w0, ge
; CHECK-NEXT: ret
define i1 @f64_constrained_fcmp_uge(double %a, double %b) nounwind ssp strictfp {
  %cmp = tail call i1 @llvm.experimental.constrained.fcmp.f64(double %a, double %b, metadata !"uge", metadata !"fpexcept.strict")
  ret i1 %cmp
}

; CHECK-LABEL: @f64_constrained_fcmp_ult
; CHECK: fcmp d0, d1
; CHECK-NEXT: cset w0, lt
; CHECK-NEXT: ret
define i1 @f64_constrained_fcmp_ult(double %a, double %b) nounwind ssp strictfp {
  %cmp = tail call i1 @llvm.experimental.constrained.fcmp.f64(double %a, double %b, metadata !"ult", metadata !"fpexcept.strict")
  ret i1 %cmp
}

; CHECK-LABEL: @f64_constrained_fcmp_ule
; CHECK: fcmp d0, d1
; CHECK-NEXT: cset w0, le
; CHECK-NEXT: ret
define i1 @f64_constrained_fcmp_ule(double %a, double %b) nounwind ssp strictfp {
  %cmp = tail call i1 @llvm.experimental.constrained.fcmp.f64(double %a, double %b, metadata !"ule", metadata !"fpexcept.strict")
  ret i1 %cmp
}
