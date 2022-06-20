; RUN: llc --mtriple=loongarch32 --mattr=+d < %s | FileCheck %s --check-prefix=LA32
; RUN: llc --mtriple=loongarch64 --mattr=+d < %s | FileCheck %s --check-prefix=LA64

;; Exercise the 'fdiv' LLVM IR: https://llvm.org/docs/LangRef.html#fdiv-instruction

define float @fdiv_s(float %x, float %y) {
; LA32-LABEL: fdiv_s:
; LA32:       # %bb.0:
; LA32-NEXT:    fdiv.s $fa0, $fa0, $fa1
; LA32-NEXT:    jirl $zero, $ra, 0
;
; LA64-LABEL: fdiv_s:
; LA64:       # %bb.0:
; LA64-NEXT:    fdiv.s $fa0, $fa0, $fa1
; LA64-NEXT:    jirl $zero, $ra, 0
  %div = fdiv float %x, %y
  ret float %div
}

define double @fdiv_d(double %x, double %y) {
; LA32-LABEL: fdiv_d:
; LA32:       # %bb.0:
; LA32-NEXT:    fdiv.d $fa0, $fa0, $fa1
; LA32-NEXT:    jirl $zero, $ra, 0
;
; LA64-LABEL: fdiv_d:
; LA64:       # %bb.0:
; LA64-NEXT:    fdiv.d $fa0, $fa0, $fa1
; LA64-NEXT:    jirl $zero, $ra, 0
  %div = fdiv double %x, %y
  ret double %div
}
