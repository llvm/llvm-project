; RUN: llc < %s -mattr=-vsx -march=ppc32 -mtriple=powerpc-apple-darwin | FileCheck %s

define double @fabs(double %f) {
; CHECK-LABEL: fabs:
; CHECK:       ; BB#0:
; CHECK-NEXT:    fabs f1, f1
; CHECK-NEXT:    blr
;
  %t = tail call double @fabs( double %f ) readnone
  ret double %t
}

define float @bitcast_fabs(float %x) {
; CHECK-LABEL: bitcast_fabs:
; CHECK:       ; BB#0:
; CHECK-NEXT:    stfs f1, -8(r1)
; CHECK-NEXT:    nop
; CHECK-NEXT:    nop
; CHECK-NEXT:    lwz r2, -8(r1)
; CHECK-NEXT:    clrlwi r2, r2, 1
; CHECK-NEXT:    stw r2, -4(r1)
; CHECK-NEXT:    lfs f1, -4(r1)
; CHECK-NEXT:    blr
;
  %bc1 = bitcast float %x to i32
  %and = and i32 %bc1, 2147483647
  %bc2 = bitcast i32 %and to float
  ret float %bc2
}

