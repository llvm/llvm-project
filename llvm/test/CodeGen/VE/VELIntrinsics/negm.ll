; RUN: llc < %s -mtriple=ve -mattr=+vpu | FileCheck %s

;;; Test negate vm intrinsic instructions
;;;
;;; Note:
;;;   We test NEGM*m and NEGM*y instructions.

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @negm_mm(<256 x i1> %0) {
; CHECK-LABEL: negm_mm:
; CHECK:       # %bb.0:
; CHECK-NEXT:    negm %vm1, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call <256 x i1> @llvm.ve.vl.negm.mm(<256 x i1> %0)
  ret <256 x i1> %2
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.negm.mm(<256 x i1>)

; Function Attrs: nounwind readnone
define fastcc <512 x i1> @negm_MM(<512 x i1> %0) {
; CHECK-LABEL: negm_MM:
; CHECK:       # %bb.0:
; CHECK-NEXT:    negm %vm2, %vm2
; CHECK-NEXT:    negm %vm3, %vm3
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call <512 x i1> @llvm.ve.vl.negm.MM(<512 x i1> %0)
  ret <512 x i1> %2
}

; Function Attrs: nounwind readnone
declare <512 x i1> @llvm.ve.vl.negm.MM(<512 x i1>)
