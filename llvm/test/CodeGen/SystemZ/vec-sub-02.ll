; Test vector subtraction on z14.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z14 | FileCheck %s

; Test a v4f32 subtraction.
define <4 x float> @f6(<4 x float> %dummy, <4 x float> %val1,
                       <4 x float> %val2) {
; CHECK-LABEL: f6:
; CHECK: vfssb %v24, %v26, %v28
; CHECK: br %r14
  %ret = fsub <4 x float> %val1, %val2
  ret <4 x float> %ret
}

; Test an f32 subtraction that uses vector registers.
define float @f7(<4 x float> %val1, <4 x float> %val2) {
; CHECK-LABEL: f7:
; CHECK: wfssb %f0, %v24, %v26
; CHECK: br %r14
  %scalar1 = extractelement <4 x float> %val1, i32 0
  %scalar2 = extractelement <4 x float> %val2, i32 0
  %ret = fsub float %scalar1, %scalar2
  ret float %ret
}

; Test a v2f32 subtraction, which gets promoted to v4f32.
define <2 x float> @f14(<2 x float> %val1, <2 x float> %val2) {
; No particular output expected, but must compile.
  %ret = fsub <2 x float> %val1, %val2
  ret <2 x float> %ret
}

; Test a v1f128 subtraction.
define <1 x fp128> @f15(<1 x fp128> %dummy, <1 x fp128> %val1,
                        <1 x fp128> %val2) {
; CHECK-LABEL: f15:
; CHECK: wfsxb %v24, %v26, %v28
; CHECK: br %r14
  %ret = fsub <1 x fp128> %val1, %val2
  ret <1 x fp128> %ret
}
