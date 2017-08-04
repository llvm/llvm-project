; Test vector division on z14.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z14 | FileCheck %s

; Test a v4f32 division.
define <4 x float> @f1(<4 x float> %dummy, <4 x float> %val1,
                       <4 x float> %val2) {
; CHECK-LABEL: f1:
; CHECK: vfdsb %v24, %v26, %v28
; CHECK: br %r14
  %ret = fdiv <4 x float> %val1, %val2
  ret <4 x float> %ret
}

; Test an f32 division that uses vector registers.
define float @f2(<4 x float> %val1, <4 x float> %val2) {
; CHECK-LABEL: f2:
; CHECK: wfdsb %f0, %v24, %v26
; CHECK: br %r14
  %scalar1 = extractelement <4 x float> %val1, i32 0
  %scalar2 = extractelement <4 x float> %val2, i32 0
  %ret = fdiv float %scalar1, %scalar2
  ret float %ret
}
