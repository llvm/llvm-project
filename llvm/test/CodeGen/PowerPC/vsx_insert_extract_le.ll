; RUN: llc -verify-machineinstrs -mcpu=pwr8 -mattr=+vsx \
; RUN:   -mtriple=powerpc64le-unknown-linux-gnu < %s | FileCheck %s

; RUN: llc -verify-machineinstrs -mcpu=pwr9 -mattr=-power9-vector \
; RUN:   -mtriple=powerpc64le-unknown-linux-gnu < %s | FileCheck %s

; RUN: llc -verify-machineinstrs -mcpu=pwr9 \
; RUN:   -mtriple=powerpc64le-unknown-linux-gnu < %s | FileCheck %s \
; RUN:   --check-prefix=CHECK-P9 --implicit-check-not xxswapd

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

; CHECK-P9-LABEL: testi0
; CHECK-P9: lfd [[REG1:[0-9]+]], 0(4)
; CHECK-P9: lxv [[REG2:[0-9]+]], 0(3)
; CHECK-P9: xxspltd [[REG3:[0-9]+]], [[REG1]], 0
; CHECK-P9: xxpermdi 34, [[REG2]], [[REG3]], 1
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

; CHECK-P9-LABEL: testi1
; CHECK-P9: lfd [[REG1:[0-9]+]], 0(4)
; CHECK-P9: lxv [[REG2:[0-9]+]], 0(3)
; CHECK-P9: xxspltd [[REG3:[0-9]+]], [[REG1]], 0
; CHECK-P9: xxmrgld 34, [[REG3]], [[REG2]]
}

define double @teste0(<2 x double>* %p1) {
  %v = load <2 x double>, <2 x double>* %p1
  %r = extractelement <2 x double> %v, i32 0
  ret double %r

; CHECK-LABEL: teste0
; CHECK: lxvd2x 1, 0, 3

; CHECK-P9-LABEL: teste0
; CHECK-P9: lfd 1, 0(3)
}

define double @teste1(<2 x double>* %p1) {
  %v = load <2 x double>, <2 x double>* %p1
  %r = extractelement <2 x double> %v, i32 1
  ret double %r

; CHECK-LABEL: teste1
; CHECK: lxvd2x 0, 0, 3
; CHECK: xxswapd 1, 0

; CHECK-P9-LABEL: teste1
; CHECK-P9: lfd 1, 8(3)
}
