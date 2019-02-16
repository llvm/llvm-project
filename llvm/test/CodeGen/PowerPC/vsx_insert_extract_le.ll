; RUN: llc -verify-machineinstrs -mcpu=pwr8 -mattr=+vsx -ppc-vsr-nums-as-vr \
; RUN:   -ppc-asm-full-reg-names -mtriple=powerpc64le-unknown-linux-gnu < %s \
; RUN:   | FileCheck %s

; RUN: llc -verify-machineinstrs -mcpu=pwr9 -mattr=-power9-vector -ppc-vsr-nums-as-vr \
; RUN:   -ppc-asm-full-reg-names -mtriple=powerpc64le-unknown-linux-gnu < %s \
; RUN:   | FileCheck --check-prefix=CHECK-P9-VECTOR %s

; RUN: llc -verify-machineinstrs -mcpu=pwr9 -ppc-vsr-nums-as-vr -ppc-asm-full-reg-names \
; RUN:   -mtriple=powerpc64le-unknown-linux-gnu < %s | FileCheck %s \
; RUN:   --check-prefix=CHECK-P9 --implicit-check-not xxswapd

define <2 x double> @testi0(<2 x double>* %p1, double* %p2) {
; CHECK-LABEL: testi0:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lxvd2x vs0, 0, r3
; CHECK-NEXT:    lfdx f1, 0, r4
; CHECK-NEXT:    xxswapd vs0, vs0
; CHECK-NEXT:    xxspltd vs1, vs1, 0
; CHECK-NEXT:    xxpermdi v2, vs0, vs1, 1
; CHECK-NEXT:    blr
;
; CHECK-P9-VECTOR-LABEL: testi0:
; CHECK-P9-VECTOR:       # %bb.0:
; CHECK-P9-VECTOR-NEXT:    lxvd2x vs0, 0, r3
; CHECK-P9-VECTOR-NEXT:    lfdx f1, 0, r4
; CHECK-P9-VECTOR-NEXT:    xxspltd vs1, vs1, 0
; CHECK-P9-VECTOR-NEXT:    xxswapd vs0, vs0
; CHECK-P9-VECTOR-NEXT:    xxpermdi v2, vs0, vs1, 1
; CHECK-P9-VECTOR-NEXT:    blr
;
; CHECK-P9-LABEL: testi0:
; CHECK-P9:       # %bb.0:
; CHECK-P9-NEXT:    lfd [[REG:f[0-9]+]], 0(r4)
; CHECK-P9-NEXT:    lxv [[REG1:vs[0-9]+]], 0(r3)
; CHECK-P9-NEXT:    xxpermdi [[REG2:vs[0-9]+]], [[REG]], [[REG]], 2
; CHECK-P9-NEXT:    xxpermdi v2, [[REG1]], [[REG2]], 1
; CHECK-P9-NEXT:    blr
  %v = load <2 x double>, <2 x double>* %p1
  %s = load double, double* %p2
  %r = insertelement <2 x double> %v, double %s, i32 0
  ret <2 x double> %r


}

define <2 x double> @testi1(<2 x double>* %p1, double* %p2) {
; CHECK-LABEL: testi1:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lxvd2x vs0, 0, r3
; CHECK-NEXT:    lfdx f1, 0, r4
; CHECK-NEXT:    xxswapd vs0, vs0
; CHECK-NEXT:    xxspltd vs1, vs1, 0
; CHECK-NEXT:    xxmrgld v2, vs1, vs0
; CHECK-NEXT:    blr
;
; CHECK-P9-VECTOR-LABEL: testi1:
; CHECK-P9-VECTOR:       # %bb.0:
; CHECK-P9-VECTOR-NEXT:    lxvd2x vs0, 0, r3
; CHECK-P9-VECTOR-NEXT:    lfdx f1, 0, r4
; CHECK-P9-VECTOR-NEXT:    xxspltd vs1, vs1, 0
; CHECK-P9-VECTOR-NEXT:    xxswapd vs0, vs0
; CHECK-P9-VECTOR-NEXT:    xxmrgld v2, vs1, vs0
; CHECK-P9-VECTOR-NEXT:    blr
;
; CHECK-P9-LABEL: testi1:
; CHECK-P9:       # %bb.0:
; CHECK-P9-NEXT:    lfd [[REG:f[0-9]+]], 0(r4)
; CHECK-P9-NEXT:    lxv [[REG1:vs[0-9]+]], 0(r3)
; CHECK-P9-NEXT:    xxpermdi [[REG2:vs[0-9]+]], [[REG]], [[REG]], 2
; CHECK-P9-NEXT:    xxmrgld v2, [[REG2]], [[REG1]] 
; CHECK-P9-NEXT:    blr
  %v = load <2 x double>, <2 x double>* %p1
  %s = load double, double* %p2
  %r = insertelement <2 x double> %v, double %s, i32 1
  ret <2 x double> %r


}

define double @teste0(<2 x double>* %p1) {
; CHECK-LABEL: teste0:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lxvd2x vs1, 0, r3
; CHECK:         blr
;
; CHECK-P9-VECTOR-LABEL: teste0:
; CHECK-P9-VECTOR:       # %bb.0:
; CHECK-P9-VECTOR-NEXT:    lxvd2x vs1, 0, r3
; CHECK-P9-VECTOR:         blr
;
; CHECK-P9-LABEL: teste0:
; CHECK-P9:       # %bb.0:
; CHECK-P9-NEXT:    lfd f1, 0(r3)
; CHECK-P9-NEXT:    blr
  %v = load <2 x double>, <2 x double>* %p1
  %r = extractelement <2 x double> %v, i32 0
  ret double %r


}

define double @teste1(<2 x double>* %p1) {
; CHECK-LABEL: teste1:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lxvd2x vs0, 0, r3
; CHECK-NEXT:    xxswapd vs1, vs0
; CHECK:         blr
;
; CHECK-P9-VECTOR-LABEL: teste1:
; CHECK-P9-VECTOR:       # %bb.0:
; CHECK-P9-VECTOR-NEXT:    lxvd2x vs0, 0, r3
; CHECK-P9-VECTOR-NEXT:    xxswapd vs1, vs0
; CHECK-P9-VECTOR:         blr
;
; CHECK-P9-LABEL: teste1:
; CHECK-P9:       # %bb.0:
; CHECK-P9-NEXT:    lfd f1, 8(r3)
; CHECK-P9-NEXT:    blr
  %v = load <2 x double>, <2 x double>* %p1
  %r = extractelement <2 x double> %v, i32 1
  ret double %r


}
