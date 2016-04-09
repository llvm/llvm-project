; RUN: opt < %s -instcombine -S | FileCheck %s
; This test makes sure that the undef is propagated for the cos instrinsic

declare double    @llvm.cos.f64(double %Val)
declare float     @llvm.cos.f32(float %Val)

; Function Attrs: nounwind readnone
define double @test1() {
; CHECK-LABEL: define double @test1(
; CHECK-NEXT: ret double 0.000000e+00
  %1 = call double @llvm.cos.f64(double undef)
  ret double %1
}


; Function Attrs: nounwind readnone
define float @test2(float %d) {
; CHECK-LABEL: define float @test2(
; CHECK-NEXT: %cosval = call float @llvm.cos.f32(float %d)
   %cosval   = call float @llvm.cos.f32(float %d)
   %cosval2  = call float @llvm.cos.f32(float undef)
   %fsum   = fadd float %cosval2, %cosval
   ret float %fsum
; CHECK-NEXT: %fsum
; CHECK: ret float %fsum
}
