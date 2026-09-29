; RUN: llc < %s -mtriple=ve -mattr=+vpu | FileCheck %s

;;; Test set vector out-of-order memory access boundary intrinsic instructions
;;;
;;; Note:
;;;   We test SVOB instruction.

; Function Attrs: nounwind
define fastcc void @svob_svob() {
; CHECK-LABEL: svob_svob:
; CHECK:       # %bb.0:
; CHECK-NEXT:    svob
; CHECK-NEXT:    b.l.t (, %s10)
  tail call void @llvm.ve.vl.svob()
  ret void
}

; Function Attrs: nounwind
declare void @llvm.ve.vl.svob()
