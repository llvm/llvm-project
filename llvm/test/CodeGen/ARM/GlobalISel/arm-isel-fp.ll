; RUN: llc -mtriple arm-gnueabihf -mattr=+vfp2 -float-abi=hard -global-isel %s -o - | FileCheck %s
; RUN: llc -mtriple arm-- -mattr=+vfp2 -float-abi=soft -global-isel %s -o - | FileCheck %s
; RUN: llc -mtriple arm-- -float-abi=soft -global-isel %s -o - | FileCheck %s

define arm_aapcscc float @test_frem_float(float %x, float %y) {
; CHECK-LABEL: test_frem_float:
; CHECK: blx fmodf
  %r = frem float %x, %y
  ret float %r
}

define arm_aapcscc double @test_frem_double(double %x, double %y) {
; CHECK-LABEL: test_frem_double:
; CHECK: blx fmod
  %r = frem double %x, %y
  ret double %r
}

declare float @llvm.pow.f32(float %x, float %y)
define arm_aapcscc float @test_fpow_float(float %x, float %y) {
; CHECK-LABEL: test_fpow_float:
; CHECK: blx powf
  %r = call float @llvm.pow.f32(float %x, float %y)
  ret float %r
}

declare double @llvm.pow.f64(double %x, double %y)
define arm_aapcscc double @test_fpow_double(double %x, double %y) {
; CHECK-LABEL: test_fpow_double:
; CHECK: blx pow
  %r = call double @llvm.pow.f64(double %x, double %y)
  ret double %r
}
