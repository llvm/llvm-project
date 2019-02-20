; Test vector byte masks, v2f64 version. Only all-zero vectors are handled.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z13 | FileCheck %s

; Test an all-zeros vector.
define <2 x double> @f0() {
; CHECK-LABEL: f0:
; CHECK: vgbm %v24, 0
; CHECK: br %r14
  ret <2 x double> zeroinitializer
}

; Test that undefs are treated as zero.
define <2 x double> @f1() {
; CHECK-LABEL: f1:
; CHECK: vgbm %v24, 0
; CHECK: br %r14
  ret <2 x double> <double zeroinitializer, double undef>
}
