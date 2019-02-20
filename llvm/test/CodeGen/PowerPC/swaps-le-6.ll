; RUN: llc -relocation-model=pic -verify-machineinstrs -mcpu=pwr8 -ppc-vsr-nums-as-vr \
; RUN:   -ppc-asm-full-reg-names -mtriple=powerpc64le-unknown-linux-gnu \
; RUN:   -O3 < %s | FileCheck %s

; RUN: llc -relocation-model=pic -mcpu=pwr9 -mtriple=powerpc64le-unknown-linux-gnu -O3 \
; RUN:   -ppc-vsr-nums-as-vr -ppc-asm-full-reg-names -verify-machineinstrs \
; RUN:   < %s | FileCheck %s --check-prefix=CHECK-P9 \
; RUN:   --implicit-check-not xxswapd

; RUN: llc -relocation-model=pic -mcpu=pwr9 -mtriple=powerpc64le-unknown-linux-gnu -O3 \
; RUN:   -ppc-vsr-nums-as-vr -ppc-asm-full-reg-names -verify-machineinstrs \
; RUN:   -mattr=-power9-vector < %s | FileCheck %s --check-prefix=CHECK-P9-NOVECTOR

; These tests verify that VSX swap optimization works when loading a scalar
; into a vector register.


@x = global <2 x double> <double 9.970000e+01, double -1.032220e+02>, align 16
@z = global <2 x double> <double 2.332000e+01, double 3.111111e+01>, align 16
@y = global double 1.780000e+00, align 8

define void @bar0() {
; CHECK-LABEL: bar0:
; CHECK:   # %bb.0: # %entry
; CHECK:     addis r3, r2, .LC0@toc@ha
; CHECK:     addis r4, r2, .LC1@toc@ha
; CHECK:     ld r3, .LC0@toc@l(r3)
; CHECK:     addis r3, r2, .LC2@toc@ha
; CHECK:     ld r3, .LC2@toc@l(r3)
; CHECK:     xxpermdi vs0, vs0, vs1, 1
; CHECK:     stxvd2x vs0, 0, r3
; CHECK:     blr
;
; CHECK-P9-NOVECTOR-LABEL: bar0:
; CHECK-P9-NOVECTOR:   # %bb.0: # %entry
; CHECK-P9-NOVECTOR:     addis r3, r2, .LC0@toc@ha
; CHECK-P9-NOVECTOR:     ld r3, .LC0@toc@l(r3)
; CHECK-P9-NOVECTOR:     addis r3, r2, .LC1@toc@ha
; CHECK-P9-NOVECTOR:     addis r3, r2, .LC2@toc@ha
; CHECK-P9-NOVECTOR:     ld r3, .LC2@toc@l(r3)
; CHECK-P9-NOVECTOR:     xxpermdi vs0, vs1, vs0, 1
; CHECK-P9-NOVECTOR:     stxvd2x vs0, 0, r3
; CHECK-P9-NOVECTOR:     blr
;
; CHECK-P9-LABEL: bar0:
; CHECK-P9:   # %bb.0: # %entry
; CHECK-P9:     addis r3, r2, .LC0@toc@ha
; CHECK-P9:     ld r3, .LC0@toc@l(r3)
; CHECK-P9:     lxvx vs0, 0, r3
; CHECK-P9:     addis r3, r2, .LC1@toc@ha
; CHECK-P9:     ld r3, .LC1@toc@l(r3)
; CHECK-P9:     lfd f1, 0(r3)
; CHECK-P9:     addis r3, r2, .LC2@toc@ha
; CHECK-P9:     ld r3, .LC2@toc@l(r3)
; CHECK-P9:     xxpermdi vs1, f1, f1, 2
; CHECK-P9:     xxpermdi vs0, vs0, vs1, 1
; CHECK-P9:     stxvx vs0, 0, r3
; CHECK-P9:     blr
entry:
  %0 = load <2 x double>, <2 x double>* @x, align 16
  %1 = load double, double* @y, align 8
  %vecins = insertelement <2 x double> %0, double %1, i32 0
  store <2 x double> %vecins, <2 x double>* @z, align 16
  ret void
}

define void @bar1() {
; CHECK-LABEL: bar1:
; CHECK:   # %bb.0: # %entry
; CHECK:     addis r3, r2, .LC0@toc@ha
; CHECK:     addis r4, r2, .LC1@toc@ha
; CHECK:     ld r3, .LC0@toc@l(r3)
; CHECK:     addis r3, r2, .LC2@toc@ha
; CHECK:     ld r3, .LC2@toc@l(r3)
; CHECK:     xxmrghd vs0, vs1, vs0
; CHECK:     stxvd2x vs0, 0, r3
; CHECK:     blr
;
; CHECK-P9-NOVECTOR-LABEL: bar1:
; CHECK-P9-NOVECTOR:   # %bb.0: # %entry
; CHECK-P9-NOVECTOR:     addis r3, r2, .LC0@toc@ha
; CHECK-P9-NOVECTOR:     ld r3, .LC0@toc@l(r3)
; CHECK-P9-NOVECTOR:     addis r3, r2, .LC1@toc@ha
; CHECK-P9-NOVECTOR:     addis r3, r2, .LC2@toc@ha
; CHECK-P9-NOVECTOR:     ld r3, .LC2@toc@l(r3)
; CHECK-P9-NOVECTOR:     xxmrghd vs0, vs0, vs1
; CHECK-P9-NOVECTOR:     stxvd2x vs0, 0, r3
; CHECK-P9-NOVECTOR:     blr
;
; CHECK-P9-LABEL: bar1:
; CHECK-P9:   # %bb.0: # %entry
; CHECK-P9:     addis r3, r2, .LC0@toc@ha
; CHECK-P9:     ld r3, .LC0@toc@l(r3)
; CHECK-P9:     lxvx vs0, 0, r3
; CHECK-P9:     addis r3, r2, .LC1@toc@ha
; CHECK-P9:     ld r3, .LC1@toc@l(r3)
; CHECK-P9:     lfd f1, 0(r3)
; CHECK-P9:     addis r3, r2, .LC2@toc@ha
; CHECK-P9:     ld r3, .LC2@toc@l(r3)
; CHECK-P9:     xxpermdi vs1, f1, f1, 2
; CHECK-P9:     xxmrgld vs0, vs1, vs0
; CHECK-P9:     stxvx vs0, 0, r3
; CHECK-P9:     blr
entry:
  %0 = load <2 x double>, <2 x double>* @x, align 16
  %1 = load double, double* @y, align 8
  %vecins = insertelement <2 x double> %0, double %1, i32 1
  store <2 x double> %vecins, <2 x double>* @z, align 16
  ret void
}

