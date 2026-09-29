; RUN: llc --mtriple=loongarch32 --mattr=+d,+frecipe < %s | FileCheck %s
; RUN: llc --mtriple=loongarch64 --mattr=+d,+frecipe < %s | FileCheck %s

declare double @llvm.loongarch.frecipe.d(double)

define double @frecipe_d(double %a) {
; CHECK-LABEL: frecipe_d:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    frecipe.d $fa0, $fa0
; CHECK-NEXT:    ret
entry:
  %res = call double @llvm.loongarch.frecipe.d(double %a)
  ret double %res
}

declare double @llvm.loongarch.frsqrte.d(double)

define double @frsqrte_d(double %a) {
; CHECK-LABEL: frsqrte_d:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    frsqrte.d $fa0, $fa0
; CHECK-NEXT:    ret
entry:
  %res = call double @llvm.loongarch.frsqrte.d(double %a)
  ret double %res
}
