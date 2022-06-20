; RUN: llc --mtriple=loongarch32 --mattr=+d < %s | FileCheck %s --check-prefix=LA32
; RUN: llc --mtriple=loongarch64 --mattr=+d < %s | FileCheck %s --check-prefix=LA64

;; Exercise the 'fneg' LLVM IR: https://llvm.org/docs/LangRef.html#fneg-instruction

define float @fneg_s(float %x) {
; LA32-LABEL: fneg_s:
; LA32:       # %bb.0:
; LA32-NEXT:    fneg.s $fa0, $fa0
; LA32-NEXT:    jirl $zero, $ra, 0
;
; LA64-LABEL: fneg_s:
; LA64:       # %bb.0:
; LA64-NEXT:    fneg.s $fa0, $fa0
; LA64-NEXT:    jirl $zero, $ra, 0
  %neg = fneg float %x
  ret float %neg
}

define double @fneg_d(double %x) {
; LA32-LABEL: fneg_d:
; LA32:       # %bb.0:
; LA32-NEXT:    fneg.d $fa0, $fa0
; LA32-NEXT:    jirl $zero, $ra, 0
;
; LA64-LABEL: fneg_d:
; LA64:       # %bb.0:
; LA64-NEXT:    fneg.d $fa0, $fa0
; LA64-NEXT:    jirl $zero, $ra, 0
  %neg = fneg double %x
  ret double %neg
}
