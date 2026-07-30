; RUN: llc < %s -mtriple=ve -mattr=+vpu | FileCheck %s

;;; Test vector shift left and add intrinsic instructions
;;;
;;; Note:
;;;   We test VSFA*vrrl, VSFA*vrrl_v, VSFA*virl, VSFA*virl_v, VSFA*vrrml_v, and
;;;   VSFA*virml_v instructions.

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vsfa_vvssl(<256 x double> %0, i64 %1, i64 %2) {
; CHECK-LABEL: vsfa_vvssl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s2, 256
; CHECK-NEXT:    lvl %s2
; CHECK-NEXT:    vsfa %v0, %v0, %s0, %s1
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = tail call fast <256 x double> @llvm.ve.vl.vsfa.vvssl(<256 x double> %0, i64 %1, i64 %2, i32 256)
  ret <256 x double> %4
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vsfa.vvssl(<256 x double>, i64, i64, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vsfa_vvssvl(<256 x double> %0, i64 %1, i64 %2, <256 x double> %3) {
; CHECK-LABEL: vsfa_vvssvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s2, 128
; CHECK-NEXT:    lvl %s2
; CHECK-NEXT:    vsfa %v1, %v0, %s0, %s1
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v1
; CHECK-NEXT:    b.l.t (, %s10)
  %5 = tail call fast <256 x double> @llvm.ve.vl.vsfa.vvssvl(<256 x double> %0, i64 %1, i64 %2, <256 x double> %3, i32 128)
  ret <256 x double> %5
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vsfa.vvssvl(<256 x double>, i64, i64, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vsfa_vvssl_imm(<256 x double> %0, i64 %1) {
; CHECK-LABEL: vsfa_vvssl_imm:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vsfa %v0, %v0, 8, %s0
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.vsfa.vvssl(<256 x double> %0, i64 8, i64 %1, i32 256)
  ret <256 x double> %3
}

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vsfa_vvssvl_imm(<256 x double> %0, i64 %1, <256 x double> %2) {
; CHECK-LABEL: vsfa_vvssvl_imm:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 128
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vsfa %v1, %v0, 8, %s0
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v1
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = tail call fast <256 x double> @llvm.ve.vl.vsfa.vvssvl(<256 x double> %0, i64 8, i64 %1, <256 x double> %2, i32 128)
  ret <256 x double> %4
}

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vsfa_vvssmvl(<256 x double> %0, i64 %1, i64 %2, <256 x i1> %3, <256 x double> %4) {
; CHECK-LABEL: vsfa_vvssmvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s2, 128
; CHECK-NEXT:    lvl %s2
; CHECK-NEXT:    vsfa %v1, %v0, %s0, %s1, %vm1
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v1
; CHECK-NEXT:    b.l.t (, %s10)
  %6 = tail call fast <256 x double> @llvm.ve.vl.vsfa.vvssmvl(<256 x double> %0, i64 %1, i64 %2, <256 x i1> %3, <256 x double> %4, i32 128)
  ret <256 x double> %6
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vsfa.vvssmvl(<256 x double>, i64, i64, <256 x i1>, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vsfa_vvssmvl_imm(<256 x double> %0, i64 %1, <256 x i1> %2, <256 x double> %3) {
; CHECK-LABEL: vsfa_vvssmvl_imm:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 128
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vsfa %v1, %v0, 8, %s0, %vm1
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v1
; CHECK-NEXT:    b.l.t (, %s10)
  %5 = tail call fast <256 x double> @llvm.ve.vl.vsfa.vvssmvl(<256 x double> %0, i64 8, i64 %1, <256 x i1> %2, <256 x double> %3, i32 128)
  ret <256 x double> %5
}
