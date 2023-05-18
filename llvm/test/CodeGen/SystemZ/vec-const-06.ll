; Test vector byte masks, v2f64 version.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z13 | FileCheck %s

; Test an all-zeros vector.
define <2 x double> @f1() {
; CHECK-LABEL: f1:
; CHECK: vgbm %v24, 0
; CHECK: br %r14
  ret <2 x double> zeroinitializer
}

; Test an all-ones vector.
define <2 x double> @f2() {
; CHECK-LABEL: f2:
; CHECK: vgbm %v24, 65535
; CHECK: br %r14
  ret <2 x double> <double 0xffffffffffffffff, double 0xffffffffffffffff>
}

; Test a mixed vector (mask 0x8c76).
define <2 x double> @f3() {
; CHECK-LABEL: f3:
; CHECK: vgbm %v24, 35958
; CHECK: br %r14
  ret <2 x double> <double 0xff000000ffff0000, double 0x00ffffff00ffff00>
}

; Test that undefs are treated as zero (mask 0x8c00).
define <2 x double> @f4() {
; CHECK-LABEL: f4:
; CHECK: vgbm %v24, 35840
; CHECK: br %r14
  ret <2 x double> <double 0xff000000ffff0000, double undef>
}

; Test that we don't use VGBM if one of the bytes is not 0 or 0xff.
define <2 x double> @f5() {
; CHECK-LABEL: f5:
; CHECK-NOT: vgbm
; CHECK: br %r14
  ret <2 x double> <double 0xfe000000ffff0000, double 0x00ffffff00ffff00>
}
