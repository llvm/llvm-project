; RUN: llc < %s -mtriple=ve -mattr=+vpu | FileCheck %s

;;; Test or vm intrinsic instructions
;;;
;;; Note:
;;;   We test ORM*mm and ORM*yy instructions.

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @orm_mmm(<256 x i1> %0, <256 x i1> %1) {
; CHECK-LABEL: orm_mmm:
; CHECK:       # %bb.0:
; CHECK-NEXT:    orm %vm1, %vm1, %vm2
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call <256 x i1> @llvm.ve.vl.orm.mmm(<256 x i1> %0, <256 x i1> %1)
  ret <256 x i1> %3
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.orm.mmm(<256 x i1>, <256 x i1>)

; Function Attrs: nounwind readnone
define fastcc <512 x i1> @orm_MMM(<512 x i1> %0, <512 x i1> %1) {
; CHECK-LABEL: orm_MMM:
; CHECK:       # %bb.0:
; CHECK-NEXT:    orm %vm2, %vm2, %vm4
; CHECK-NEXT:    orm %vm3, %vm3, %vm5
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call <512 x i1> @llvm.ve.vl.orm.MMM(<512 x i1> %0, <512 x i1> %1)
  ret <512 x i1> %3
}

; Function Attrs: nounwind readnone
declare <512 x i1> @llvm.ve.vl.orm.MMM(<512 x i1>, <512 x i1>)
