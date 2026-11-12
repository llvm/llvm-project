; RUN: llc < %s -mtriple=ve -mattr=+vpu | FileCheck %s

;;; Test vector shuffle intrinsic instructions
;;;
;;; Note:
;;;   We test VSHF*vvrl, VSHF*vvrl_v, VSHF*vvil, and VSHF*vvil_v instructions.

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vshf_vvvsl(<256 x double> %0, <256 x double> %1, i64 %2) {
; CHECK-LABEL: vshf_vvvsl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vshf %v0, %v0, %v1, %s0
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = tail call fast <256 x double> @llvm.ve.vl.vshf.vvvsl(<256 x double> %0, <256 x double> %1, i64 %2, i32 256)
  ret <256 x double> %4
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vshf.vvvsl(<256 x double>, <256 x double>, i64, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vshf_vvvsvl(<256 x double> %0, <256 x double> %1, i64 %2, <256 x double> %3) {
; CHECK-LABEL: vshf_vvvsvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 128
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vshf %v2, %v0, %v1, %s0
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v2
; CHECK-NEXT:    b.l.t (, %s10)
  %5 = tail call fast <256 x double> @llvm.ve.vl.vshf.vvvsvl(<256 x double> %0, <256 x double> %1, i64 %2, <256 x double> %3, i32 128)
  ret <256 x double> %5
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vshf.vvvsvl(<256 x double>, <256 x double>, i64, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vshf_vvvsl_imm(<256 x double> %0, <256 x double> %1) {
; CHECK-LABEL: vshf_vvvsl_imm:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vshf %v0, %v0, %v1, 8
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.vshf.vvvsl(<256 x double> %0, <256 x double> %1, i64 8, i32 256)
  ret <256 x double> %3
}

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vshf_vvvsvl_imm(<256 x double> %0, <256 x double> %1, <256 x double> %2) {
; CHECK-LABEL: vshf_vvvsvl_imm:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 128
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vshf %v2, %v0, %v1, 8
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v2
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = tail call fast <256 x double> @llvm.ve.vl.vshf.vvvsvl(<256 x double> %0, <256 x double> %1, i64 8, <256 x double> %2, i32 128)
  ret <256 x double> %4
}
