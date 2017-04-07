; RUN: llc -mtriple arm-unknown -mattr=+vfp2 -float-abi=hard -global-isel %s -o - | FileCheck %s
; RUN: llc -mtriple arm-unknown -mattr=+vfp2 -float-abi=soft -global-isel %s -o - | FileCheck %s
; RUN: llc -mtriple arm-unknwon -float-abi=soft -global-isel %s -o - | FileCheck %s

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
