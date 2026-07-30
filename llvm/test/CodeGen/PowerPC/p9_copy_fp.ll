; RUN: llc -verify-machineinstrs -mcpu=pwr9 -mattr=+vsx -ppc-vsr-nums-as-vr \
; RUN:     -mtriple=powerpc64le-unknown-linux-gnu -ppc-asm-full-reg-names < %s \
; RUN:     | FileCheck %s
; RUN: llc -verify-machineinstrs -mcpu=pwr9 -mattr=+vsx -ppc-vsr-nums-as-vr \
; RUN:     -mtriple=powerpc64-unknown-linux-gnu -ppc-asm-full-reg-names < %s \
; RUN:     | FileCheck -check-prefix=CHECK-BE %s

; Function Attrs: norecurse nounwind readnone
define double @cp_fp1(<2 x double> %v) {
  ; CHECK-LABEL: cp_fp1:
  ; CHECK: xscpsgndp f1, v2, v2
  ; CHECK: blr

  ; CHECK-BE-LABEL: cp_fp1:
  ; CHECK-BE: xxswapd vs1, v2
  ; CHECK-BE: blr
  entry:
    %vecext = extractelement <2 x double> %v, i32 1
      ret double %vecext
}

; Function Attrs: norecurse nounwind readnone
define double @cp_fp2(<2 x double> %v) {
  ; CHECK-LABEL: cp_fp2:
  ; CHECK:    xxswapd vs1, v2
  ; CHECK:    blr

  ; CHECK-BE-LABEL: cp_fp2:
  ; CHECK-BE: xscpsgndp f1, v2, v2
  ; CHECK-BE: blr
  entry:
    %vecext = extractelement <2 x double> %v, i32 0
      ret double %vecext
}

; Function Attrs: norecurse nounwind readnone
define <2 x double> @cp_fp3(double %v) {
  ; CHECK-LABEL: cp_fp3:
  ; CHECK:    xxspltd v2, vs1, 0
  ; CHECK:    blr

  ; CHECK-BE-LABEL: cp_fp3:
  ; CHECK-BE: xscpsgndp v2, f1, f1
  ; CHECK-BE: blr
  entry:
    %vecins = insertelement <2 x double> undef, double %v, i32 0
      ret <2 x double> %vecins
}
