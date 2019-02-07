; Test vector byte masks, v4f32 version. Only all-zero vectors are handled.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z13 | FileCheck %s

; Test an all-zeros vector.
define <4 x float> @f0() {
; CHECK-LABEL: f0:
; CHECK: vgbm %v24, 0
; CHECK: br %r14
  ret <4 x float> zeroinitializer
}

; Test that undefs are treated as zero.
define <4 x float> @f1() {
; CHECK-LABEL: f1:
; CHECK: vgbm %v24, 0
; CHECK: br %r14
  ret <4 x float> <float zeroinitializer, float undef,
                   float zeroinitializer, float undef>
}

; Test an all-zeros v2f32 that gets promoted to v4f32.
define <2 x float> @f2() {
; CHECK-LABEL: f2:
; CHECK: vgbm %v24, 0
; CHECK: br %r14
  ret <2 x float> zeroinitializer
}
