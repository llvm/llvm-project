; RUN: llc < %s -mtriple=ve -mattr=+vpu | FileCheck %s

;;; Test vector expand intrinsic instructions
;;;
;;; Note:
;;;   We test VEX*vml_v instruction.

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vex_vvmvl(<256 x double> %0, <256 x i1> %1, <256 x double> %2) {
; CHECK-LABEL: vex_vvmvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 128
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vex %v1, %v0, %vm1
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v1
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = tail call fast <256 x double> @llvm.ve.vl.vex.vvmvl(<256 x double> %0, <256 x i1> %1, <256 x double> %2, i32 128)
  ret <256 x double> %4
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vex.vvmvl(<256 x double>, <256 x i1>, <256 x double>, i32)
