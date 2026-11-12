; RUN: llc < %s -mtriple=ve -mattr=+vpu | FileCheck %s

;;; Test exclusive or vm intrinsic instructions
;;;
;;; Note:
;;;   We test XORM*mm and XORM*yy instructions.

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @xorm_mmm(<256 x i1> %0, <256 x i1> %1) {
; CHECK-LABEL: xorm_mmm:
; CHECK:       # %bb.0:
; CHECK-NEXT:    xorm %vm1, %vm1, %vm2
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call <256 x i1> @llvm.ve.vl.xorm.mmm(<256 x i1> %0, <256 x i1> %1)
  ret <256 x i1> %3
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.xorm.mmm(<256 x i1>, <256 x i1>)

; Function Attrs: nounwind readnone
define fastcc <512 x i1> @xorm_MMM(<512 x i1> %0, <512 x i1> %1) {
; CHECK-LABEL: xorm_MMM:
; CHECK:       # %bb.0:
; CHECK-NEXT:    xorm %vm2, %vm2, %vm4
; CHECK-NEXT:    xorm %vm3, %vm3, %vm5
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call <512 x i1> @llvm.ve.vl.xorm.MMM(<512 x i1> %0, <512 x i1> %1)
  ret <512 x i1> %3
}

; Function Attrs: nounwind readnone
declare <512 x i1> @llvm.ve.vl.xorm.MMM(<512 x i1>, <512 x i1>)
