; Test that floating-point instructions that set cc are *not* used to
; eliminate *strict* compares for load complement, load negative and load
; positive
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z10 | FileCheck %s
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z13 | FileCheck %s

; Load complement (sign-bit flipped).
; Test f32
define float @f1(float %a, float %b, float %f) #0 {
; CHECK-LABEL: f1:
; CHECK: ltebr
; CHECK-NEXT: ber %r14
  %neg = fneg float %f
  %cond = call i1 @llvm.experimental.constrained.fcmp.f32(
                                               float %neg, float 0.0,
                                               metadata !"oeq",
                                               metadata !"fpexcept.strict") #0
  %res = select i1 %cond, float %a, float %b
  ret float %res
}

; Test f64
define double @f2(double %a, double %b, double %f) #0 {
; CHECK-LABEL: f2:
; CHECK: ltdbr
; CHECK-NEXT: ber %r14
  %neg = fneg double %f
  %cond = call i1 @llvm.experimental.constrained.fcmp.f64(
                                               double %neg, double 0.0,
                                               metadata !"oeq",
                                               metadata !"fpexcept.strict") #0
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

; Negation of floating-point absolute.
; Test f32
declare float @llvm.fabs.f32(float %f)
define float @f3(float %a, float %b, float %f) #0 {
; CHECK-LABEL: f3:
; CHECK: ltebr
; CHECK-NEXT: ber %r14
  %abs = call float @llvm.fabs.f32(float %f)
  %neg = fneg float %abs
  %cond = call i1 @llvm.experimental.constrained.fcmp.f32(
                                               float %neg, float 0.0,
                                               metadata !"oeq",
                                               metadata !"fpexcept.strict") #0
  %res = select i1 %cond, float %a, float %b
  ret float %res
}

; Test f64
declare double @llvm.fabs.f64(double %f)
define double @f4(double %a, double %b, double %f) #0 {
; CHECK-LABEL: f4:
; CHECK: ltdbr
; CHECK-NEXT: ber %r14
  %abs = call double @llvm.fabs.f64(double %f)
  %neg = fneg double %abs
  %cond = call i1 @llvm.experimental.constrained.fcmp.f64(
                                               double %neg, double 0.0,
                                               metadata !"oeq",
                                               metadata !"fpexcept.strict") #0
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

; Absolute floating-point value.
; Test f32
define float @f5(float %a, float %b, float %f) #0 {
; CHECK-LABEL: f5:
; CHECK: ltebr
; CHECK-NEXT: ber %r14
  %abs = call float @llvm.fabs.f32(float %f)
  %cond = call i1 @llvm.experimental.constrained.fcmp.f32(
                                               float %abs, float 0.0,
                                               metadata !"oeq",
                                               metadata !"fpexcept.strict") #0
  %res = select i1 %cond, float %a, float %b
  ret float %res
}

; Test f64
define double @f6(double %a, double %b, double %f) #0 {
; CHECK-LABEL: f6:
; CHECK: ltdbr
; CHECK-NEXT: ber %r14
  %abs = call double @llvm.fabs.f64(double %f)
  %cond = call i1 @llvm.experimental.constrained.fcmp.f64(
                                               double %abs, double 0.0,
                                               metadata !"oeq",
                                               metadata !"fpexcept.strict") #0
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

attributes #0 = { strictfp }

declare i1 @llvm.experimental.constrained.fcmp.f32(float, float, metadata, metadata)
declare i1 @llvm.experimental.constrained.fcmp.f64(double, double, metadata, metadata)

