; RUN: llc %s -O0 -march=sparc -mattr=replacefmuls -o - | FileCheck %s

; CHECK-LABEL: test_replace_fmuls
; CHECK:       fsmuld %f1, %f0, %f2
; CHECK:       fdtos %f2, %f0
; NOFIX-LABEL: test_replace_fmuls
; NOFIX:       fmuls %f1, %f0, %f0
define float @test_replace_fmuls(float %a, float %b) {
entry:
  %mul = fmul float %a, %b

  ret float %mul
}
