; Test vector negation on z14.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z14 | FileCheck %s

; Test a v4f32 negation.
define <4 x float> @f1(<4 x float> %dummy, <4 x float> %val) {
; CHECK-LABEL: f1:
; CHECK: vflcsb %v24, %v26
; CHECK: br %r14
  %ret = fneg <4 x float> %val
  ret <4 x float> %ret
}

; Test an f32 negation that uses vector registers.
define float @f2(<4 x float> %val) {
; CHECK-LABEL: f2:
; CHECK: wflcsb %f0, %v24
; CHECK: br %r14
  %scalar = extractelement <4 x float> %val, i32 0
  %ret = fneg float %scalar
  ret float %ret
}
