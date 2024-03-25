; RUN: llc --mtriple=loongarch32 --mattr=+f,+frecipe < %s | FileCheck %s
; RUN: llc --mtriple=loongarch64 --mattr=+f,+frecipe < %s | FileCheck %s

declare float @llvm.loongarch.frecipe.s(float)

define float @frecipe_s(float %a) {
; CHECK-LABEL: frecipe_s:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    frecipe.s $fa0, $fa0
; CHECK-NEXT:    ret
entry:
  %res = call float @llvm.loongarch.frecipe.s(float %a)
  ret float %res
}

declare float @llvm.loongarch.frsqrte.s(float)

define float @frsqrte_s(float %a) {
; CHECK-LABEL: frsqrte_s:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    frsqrte.s $fa0, $fa0
; CHECK-NEXT:    ret
entry:
  %res = call float @llvm.loongarch.frsqrte.s(float %a)
  ret float %res
}
