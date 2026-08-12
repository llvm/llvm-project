; RUN: llc -mtriple=armv7a-none-eabi -mattr=vfp4 %s -o - | FileCheck %s
; RUN: llc -mtriple=thumbv7m-none-eabi -mattr=vfp4 %s -o - | FileCheck %s

; Test that STRICT_F* nodes (produced by constrained FP intrinsics with
; fpexcept.maytrap) lower to native VFP instructions via the
; mutateStrictFPToFP path in SelectionDAGISel.  The STRICT_F* operations
; have ISD action = Expand in ARMISelLowering, which triggers the mutation
; to non-strict FP nodes that then match normal VFP patterns.
;
; If a future upstream change breaks the mutation path (e.g. by adding
; explicit STRICT_F* patterns or changing the Expand fall-through logic),
; these CHECK lines will catch it.

; CHECK-LABEL: test_fadd:
; CHECK:       vadd.f32
define float @test_fadd(float %a, float %b) #0 {
  %r = call float @llvm.experimental.constrained.fadd.f32(
    float %a, float %b,
    metadata !"round.tonearest", metadata !"fpexcept.maytrap") #0
  ret float %r
}

; CHECK-LABEL: test_fsub:
; CHECK:       vsub.f32
define float @test_fsub(float %a, float %b) #0 {
  %r = call float @llvm.experimental.constrained.fsub.f32(
    float %a, float %b,
    metadata !"round.tonearest", metadata !"fpexcept.maytrap") #0
  ret float %r
}

; CHECK-LABEL: test_fmul:
; CHECK:       vmul.f32
define float @test_fmul(float %a, float %b) #0 {
  %r = call float @llvm.experimental.constrained.fmul.f32(
    float %a, float %b,
    metadata !"round.tonearest", metadata !"fpexcept.maytrap") #0
  ret float %r
}

; CHECK-LABEL: test_fdiv:
; CHECK:       vdiv.f32
define float @test_fdiv(float %a, float %b) #0 {
  %r = call float @llvm.experimental.constrained.fdiv.f32(
    float %a, float %b,
    metadata !"round.tonearest", metadata !"fpexcept.maytrap") #0
  ret float %r
}

; CHECK-LABEL: test_fsqrt:
; CHECK:       vsqrt.f32
define float @test_fsqrt(float %a) #0 {
  %r = call float @llvm.experimental.constrained.sqrt.f32(
    float %a,
    metadata !"round.tonearest", metadata !"fpexcept.maytrap") #0
  ret float %r
}

; CHECK-LABEL: test_fma:
; CHECK:       vfma.f32
define float @test_fma(float %a, float %b, float %c) #0 {
  %r = call float @llvm.experimental.constrained.fma.f32(
    float %a, float %b, float %c,
    metadata !"round.tonearest", metadata !"fpexcept.maytrap") #0
  ret float %r
}

; Double-precision (f64) — requires VFP with double support
; CHECK-LABEL: test_fadd_f64:
; CHECK:       vadd.f64
define double @test_fadd_f64(double %a, double %b) #0 {
  %r = call double @llvm.experimental.constrained.fadd.f64(
    double %a, double %b,
    metadata !"round.tonearest", metadata !"fpexcept.maytrap") #0
  ret double %r
}

; CHECK-LABEL: test_fdiv_f64:
; CHECK:       vdiv.f64
define double @test_fdiv_f64(double %a, double %b) #0 {
  %r = call double @llvm.experimental.constrained.fdiv.f64(
    double %a, double %b,
    metadata !"round.tonearest", metadata !"fpexcept.maytrap") #0
  ret double %r
}

declare float @llvm.experimental.constrained.fadd.f32(float, float, metadata, metadata)
declare float @llvm.experimental.constrained.fsub.f32(float, float, metadata, metadata)
declare float @llvm.experimental.constrained.fmul.f32(float, float, metadata, metadata)
declare float @llvm.experimental.constrained.fdiv.f32(float, float, metadata, metadata)
declare float @llvm.experimental.constrained.sqrt.f32(float, metadata, metadata)
declare float @llvm.experimental.constrained.fma.f32(float, float, float, metadata, metadata)
declare double @llvm.experimental.constrained.fadd.f64(double, double, metadata, metadata)
declare double @llvm.experimental.constrained.fdiv.f64(double, double, metadata, metadata)

attributes #0 = { strictfp }
