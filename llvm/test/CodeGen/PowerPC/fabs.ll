; RUN: llc -verify-machineinstrs -ppc-asm-full-reg-names < %s -mattr=-vsx -mtriple=powerpc-unknown-linux-gnu | FileCheck %s

define double @fabs(double %f) {
; CHECK-LABEL: fabs:
; CHECK:       # %bb.0:
; CHECK-NEXT:    fabs f1, f1
; CHECK-NEXT:    blr
;
  %t = tail call double @fabs( double %f ) readnone
  ret double %t
}

define float @bitcast_fabs(float %x) {
; CHECK-LABEL: bitcast_fabs:
; CHECK:       # %bb.0:
; CHECK:         stfs f1, 8(r1)
; CHECK:         lwz r3, 8(r1)
; CHECK-NEXT:    clrlwi r3, r3, 1
; CHECK-NEXT:    stw r3, 12(r1)
; CHECK-NEXT:    lfs f1, 12(r1)
; CHECK-NEXT:    addi r1, r1, 16
; CHECK-NEXT:    blr
;
  %bc1 = bitcast float %x to i32
  %and = and i32 %bc1, 2147483647
  %bc2 = bitcast i32 %and to float
  ret float %bc2
}

