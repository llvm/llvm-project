; RUN: llc < %s -mtriple=ve -mattr=+vpu | FileCheck %s

;;; Test insert intrinsic instructions
;;;
;;; Note:
;;;   We test insert_vm512u and insert_vm512l pseudo instructions.

; Function Attrs: nounwind readnone
define fastcc <512 x i1> @insert_vm512u(<512 x i1> %0, <256 x i1> %1) {
; CHECK-LABEL: insert_vm512u:
; CHECK:       # %bb.0:
; CHECK-NEXT:    andm %vm2, %vm0, %vm4
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call <512 x i1> @llvm.ve.vl.insert.vm512u(<512 x i1> %0, <256 x i1> %1)
  ret <512 x i1> %3
}

; Function Attrs: nounwind readnone
declare <512 x i1> @llvm.ve.vl.insert.vm512u(<512 x i1>, <256 x i1>)

; Function Attrs: nounwind readnone
define fastcc <512 x i1> @insert_vm512l(<512 x i1> %0, <256 x i1> %1) {
; CHECK-LABEL: insert_vm512l:
; CHECK:       # %bb.0:
; CHECK-NEXT:    andm %vm3, %vm0, %vm4
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call <512 x i1> @llvm.ve.vl.insert.vm512l(<512 x i1> %0, <256 x i1> %1)
  ret <512 x i1> %3
}

; Function Attrs: nounwind readnone
declare <512 x i1> @llvm.ve.vl.insert.vm512l(<512 x i1>, <256 x i1>)
