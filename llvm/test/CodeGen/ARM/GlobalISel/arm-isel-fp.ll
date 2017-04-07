; RUN: llc -mtriple arm-unknown -mattr=+vfp2 -float-abi=hard -global-isel %s -o - | FileCheck %s
; RUN: llc -mtriple arm-unknown -mattr=+vfp2 -float-abi=soft -global-isel %s -o - | FileCheck %s
; RUN: llc -mtriple arm-unknwon -float-abi=soft -global-isel %s -o - | FileCheck %s

define arm_aapcscc float @test_frem_float(float %x, float %y) {
; CHECK-LABEL: test_frem_float:
; CHECK: blx fmodf
  %r = frem float %x, %y
  ret float %r
}

