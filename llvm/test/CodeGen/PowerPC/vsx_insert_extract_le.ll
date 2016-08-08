; RUN: llc -verify-machineinstrs -mcpu=pwr8 -mattr=+vsx -mtriple=powerpc64le-unknown-linux-gnu < %s | FileCheck %s

define <2 x double> @testi0(<2 x double>* %p1, double* %p2) {
  %v = load <2 x double>, <2 x double>* %p1
  %s = load double, double* %p2
  %r = insertelement <2 x double> %v, double %s, i32 0
  ret <2 x double> %r

; CHECK-LABEL: testi0
; CHECK: lxvd2x 0, 0, 3
; CHECK: lxsdx 1, 0, 4
; CHECK: xxswapd 0, 0
; CHECK: xxspltd 1, 1, 0
; CHECK: xxpermdi 34, 0, 1, 1
}

define <2 x double> @testi1(<2 x double>* %p1, double* %p2) {
  %v = load <2 x double>, <2 x double>* %p1
  %s = load double, double* %p2
  %r = insertelement <2 x double> %v, double %s, i32 1
  ret <2 x double> %r

; CHECK-LABEL: testi1
; CHECK: lxvd2x 0, 0, 3
; CHECK: lxsdx 1, 0, 4
; CHECK: xxswapd 0, 0
; CHECK: xxspltd 1, 1, 0
; CHECK: xxmrgld 34, 1, 0
}

define double @teste0(<2 x double>* %p1) {
  %v = load <2 x double>, <2 x double>* %p1
  %r = extractelement <2 x double> %v, i32 0
  ret double %r

; CHECK-LABEL: teste0
; CHECK: lxvd2x 1, 0, 3
}

define double @teste1(<2 x double>* %p1) {
  %v = load <2 x double>, <2 x double>* %p1
  %r = extractelement <2 x double> %v, i32 1
  ret double %r

; CHECK-LABEL: teste1
; CHECK: lxvd2x 0, 0, 3
; CHECK: xxswapd 1, 0
}
