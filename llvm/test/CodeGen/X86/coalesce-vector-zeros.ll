; RUN: llc < %s -mtriple=x86_64-unknown-linux-gnu -mattr=+avx512f | FileCheck %s

; Verify that a scalar double 0.0 and a <4 x double> zeroinitializer in the same
; function share the same zero register rather than generating separate vxorpd instructions.
define double @test_coalesce_zeros(ptr %p, double %d) {
; CHECK-LABEL: test_coalesce_zeros:
; CHECK:       # %bb.0:
; CHECK:         vxorp{{s|d}} %xmm{{[0-9]+}}, %xmm{{[0-9]+}}, %xmm{{[0-9]+}}
; CHECK-NOT:     vxorpd
; CHECK-NOT:     vxorps
; CHECK:         vmovap{{s|d}} %ymm
; CHECK:         vmaxsd
; CHECK:         retq

entry:
  store <4 x double> zeroinitializer, ptr %p
  %z_scalar = call double @llvm.maxnum.f64(double 0.0, double %d)
  ret double %z_scalar
}

declare double @llvm.maxnum.f64(double, double)
