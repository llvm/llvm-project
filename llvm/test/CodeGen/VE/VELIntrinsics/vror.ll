; RUN: llc < %s -mtriple=ve -mattr=+vpu | FileCheck %s

;;; Test vector reduction or intrinsic instructions
;;;
;;; Note:
;;;   We test VROR*vl and VROR*vml instructions.

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vror_vvl(<256 x double> %0) {
; CHECK-LABEL: vror_vvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vror %v0, %v0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call fast <256 x double> @llvm.ve.vl.vror.vvl(<256 x double> %0, i32 256)
  ret <256 x double> %2
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vror.vvl(<256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vror_vvml(<256 x double> %0, <256 x i1> %1) {
; CHECK-LABEL: vror_vvml:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vror %v0, %v0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.vror.vvml(<256 x double> %0, <256 x i1> %1, i32 256)
  ret <256 x double> %3
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vror.vvml(<256 x double>, <256 x i1>, i32)
