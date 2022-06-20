; RUN: llc --mtriple=loongarch32 --mattr=+d < %s | FileCheck %s --check-prefix=LA32
; RUN: llc --mtriple=loongarch64 --mattr=+d < %s | FileCheck %s --check-prefix=LA64

;; Exercise the 'fsub' LLVM IR: https://llvm.org/docs/LangRef.html#fsub-instruction

define float @fsub_s(float %x, float %y) {
; LA32-LABEL: fsub_s:
; LA32:       # %bb.0:
; LA32-NEXT:    fsub.s $fa0, $fa0, $fa1
; LA32-NEXT:    jirl $zero, $ra, 0
;
; LA64-LABEL: fsub_s:
; LA64:       # %bb.0:
; LA64-NEXT:    fsub.s $fa0, $fa0, $fa1
; LA64-NEXT:    jirl $zero, $ra, 0
  %sub = fsub float %x, %y
  ret float %sub
}

define double @fsub_d(double %x, double %y) {
; LA32-LABEL: fsub_d:
; LA32:       # %bb.0:
; LA32-NEXT:    fsub.d $fa0, $fa0, $fa1
; LA32-NEXT:    jirl $zero, $ra, 0
;
; LA64-LABEL: fsub_d:
; LA64:       # %bb.0:
; LA64-NEXT:    fsub.d $fa0, $fa0, $fa1
; LA64-NEXT:    jirl $zero, $ra, 0
  %sub = fsub double %x, %y
  ret double %sub
}

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
    %res = fsub float -0.0, %x
    ret float %res
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
    %res = fsub double -0.0, %x
    ret double %res
}
