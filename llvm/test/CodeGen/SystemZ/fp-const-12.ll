; Test loads of FP constants with VGM and VGBM.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z13 | FileCheck %s

define double @f1() {
; CHECK-LABEL: f1:
; CHECK: vgmg %v0, 2, 11
  ret double 1.0
}

define double @f2() {
; CHECK-LABEL: f2:
; CHECK: vgmg %v0, 1, 1
  ret double 2.0
}

define double @f3() {
; CHECK-LABEL: f3:
; CHECK: vgmg %v0, 0, 1
  ret double -2.0
}

define double @f4() {
; CHECK-LABEL: f4:
; CHECK: vgmg %v0, 2, 10
  ret double 0.5
}

define double @f5() {
; CHECK-LABEL: f5:
; CHECK: vgmg %v0, 2, 9
  ret double 0.125
}

define float @f6() {
; CHECK-LABEL: f6:
; CHECK: vgmf %v0, 2, 8
  ret float 1.0
}

define float @f7() {
; CHECK-LABEL: f7:
; CHECK: vgmf %v0, 1, 1
  ret float 2.0
}

define float @f8() {
; CHECK-LABEL: f8:
; CHECK: vgmf %v0, 0, 1
  ret float -2.0
}

define float @f9() {
; CHECK-LABEL: f9:
; CHECK: vgmf %v0, 2, 7
  ret float 0.5
}

define float @f10() {
; CHECK-LABEL: f10:
; CHECK: vgmf %v0, 2, 6
  ret float 0.125
}

define float @f11() {
; CHECK-LABEL: f11:
; CHECK: vgbm %v0, 61440
  ret float 0xFFFFFFFFE0000000
}

define double @f12() {
; CHECK-LABEL: f12:
; CHECK: vgbm %v0, 61440
  ret double 0xFFFFFFFF00000000
}
