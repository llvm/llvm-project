; RUN: llc < %s -mtriple=ve -mattr=+vpu | FileCheck %s

;;; Test vector shift right logical intrinsic instructions
;;;
;;; Note:
;;;   We test VSRL*vvl, VSRL*vvl_v, VSRL*vrl, VSRL*vrl_v, VSRL*vil, VSRL*vil_v,
;;;   VSRL*vvml_v, VSRL*vrml_v, VSRL*viml_v, PVSRL*vvl, PVSRL*vvl_v, PVSRL*vrl,
;;;   PVSRL*vrl_v, PVSRL*vvml_v, and PVSRL*vrml_v instructions.

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vsrl_vvvl(<256 x double> %0, <256 x double> %1) {
; CHECK-LABEL: vsrl_vvvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vsrl %v0, %v0, %v1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.vsrl.vvvl(<256 x double> %0, <256 x double> %1, i32 256)
  ret <256 x double> %3
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vsrl.vvvl(<256 x double>, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vsrl_vvvvl(<256 x double> %0, <256 x double> %1, <256 x double> %2) {
; CHECK-LABEL: vsrl_vvvvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 128
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vsrl %v2, %v0, %v1
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v2
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = tail call fast <256 x double> @llvm.ve.vl.vsrl.vvvvl(<256 x double> %0, <256 x double> %1, <256 x double> %2, i32 128)
  ret <256 x double> %4
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vsrl.vvvvl(<256 x double>, <256 x double>, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vsrl_vvsl(<256 x double> %0, i64 %1) {
; CHECK-LABEL: vsrl_vvsl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vsrl %v0, %v0, %s0
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.vsrl.vvsl(<256 x double> %0, i64 %1, i32 256)
  ret <256 x double> %3
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vsrl.vvsl(<256 x double>, i64, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vsrl_vvsvl(<256 x double> %0, i64 %1, <256 x double> %2) {
; CHECK-LABEL: vsrl_vvsvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 128
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vsrl %v1, %v0, %s0
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v1
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = tail call fast <256 x double> @llvm.ve.vl.vsrl.vvsvl(<256 x double> %0, i64 %1, <256 x double> %2, i32 128)
  ret <256 x double> %4
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vsrl.vvsvl(<256 x double>, i64, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vsrl_vvsl_imm(<256 x double> %0) {
; CHECK-LABEL: vsrl_vvsl_imm:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vsrl %v0, %v0, 8
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call fast <256 x double> @llvm.ve.vl.vsrl.vvsl(<256 x double> %0, i64 8, i32 256)
  ret <256 x double> %2
}

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vsrl_vvsvl_imm(<256 x double> %0, <256 x double> %1) {
; CHECK-LABEL: vsrl_vvsvl_imm:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 128
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vsrl %v1, %v0, 8
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.vsrl.vvsvl(<256 x double> %0, i64 8, <256 x double> %1, i32 128)
  ret <256 x double> %3
}

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vsrl_vvvmvl(<256 x double> %0, <256 x double> %1, <256 x i1> %2, <256 x double> %3) {
; CHECK-LABEL: vsrl_vvvmvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 128
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vsrl %v2, %v0, %v1, %vm1
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v2
; CHECK-NEXT:    b.l.t (, %s10)
  %5 = tail call fast <256 x double> @llvm.ve.vl.vsrl.vvvmvl(<256 x double> %0, <256 x double> %1, <256 x i1> %2, <256 x double> %3, i32 128)
  ret <256 x double> %5
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vsrl.vvvmvl(<256 x double>, <256 x double>, <256 x i1>, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vsrl_vvsmvl(<256 x double> %0, i64 %1, <256 x i1> %2, <256 x double> %3) {
; CHECK-LABEL: vsrl_vvsmvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 128
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vsrl %v1, %v0, %s0, %vm1
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v1
; CHECK-NEXT:    b.l.t (, %s10)
  %5 = tail call fast <256 x double> @llvm.ve.vl.vsrl.vvsmvl(<256 x double> %0, i64 %1, <256 x i1> %2, <256 x double> %3, i32 128)
  ret <256 x double> %5
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vsrl.vvsmvl(<256 x double>, i64, <256 x i1>, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vsrl_vvsmvl_imm(<256 x double> %0, <256 x i1> %1, <256 x double> %2) {
; CHECK-LABEL: vsrl_vvsmvl_imm:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 128
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vsrl %v1, %v0, 8, %vm1
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v1
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = tail call fast <256 x double> @llvm.ve.vl.vsrl.vvsmvl(<256 x double> %0, i64 8, <256 x i1> %1, <256 x double> %2, i32 128)
  ret <256 x double> %4
}

; Function Attrs: nounwind readnone
define fastcc <256 x double> @pvsrl_vvvl(<256 x double> %0, <256 x double> %1) {
; CHECK-LABEL: pvsrl_vvvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    pvsrl %v0, %v0, %v1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.pvsrl.vvvl(<256 x double> %0, <256 x double> %1, i32 256)
  ret <256 x double> %3
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.pvsrl.vvvl(<256 x double>, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @pvsrl_vvvvl(<256 x double> %0, <256 x double> %1, <256 x double> %2) {
; CHECK-LABEL: pvsrl_vvvvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 128
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    pvsrl %v2, %v0, %v1
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v2
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = tail call fast <256 x double> @llvm.ve.vl.pvsrl.vvvvl(<256 x double> %0, <256 x double> %1, <256 x double> %2, i32 128)
  ret <256 x double> %4
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.pvsrl.vvvvl(<256 x double>, <256 x double>, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @pvsrl_vvsl(<256 x double> %0, i64 %1) {
; CHECK-LABEL: pvsrl_vvsl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    pvsrl %v0, %v0, %s0
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.pvsrl.vvsl(<256 x double> %0, i64 %1, i32 256)
  ret <256 x double> %3
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.pvsrl.vvsl(<256 x double>, i64, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @pvsrl_vvsvl(<256 x double> %0, i64 %1, <256 x double> %2) {
; CHECK-LABEL: pvsrl_vvsvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 128
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    pvsrl %v1, %v0, %s0
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v1
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = tail call fast <256 x double> @llvm.ve.vl.pvsrl.vvsvl(<256 x double> %0, i64 %1, <256 x double> %2, i32 128)
  ret <256 x double> %4
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.pvsrl.vvsvl(<256 x double>, i64, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @pvsrl_vvvMvl(<256 x double> %0, <256 x double> %1, <512 x i1> %2, <256 x double> %3) {
; CHECK-LABEL: pvsrl_vvvMvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 128
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    pvsrl %v2, %v0, %v1, %vm2
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v2
; CHECK-NEXT:    b.l.t (, %s10)
  %5 = tail call fast <256 x double> @llvm.ve.vl.pvsrl.vvvMvl(<256 x double> %0, <256 x double> %1, <512 x i1> %2, <256 x double> %3, i32 128)
  ret <256 x double> %5
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.pvsrl.vvvMvl(<256 x double>, <256 x double>, <512 x i1>, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @pvsrl_vvsMvl(<256 x double> %0, i64 %1, <512 x i1> %2, <256 x double> %3) {
; CHECK-LABEL: pvsrl_vvsMvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 128
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    pvsrl %v1, %v0, %s0, %vm2
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v1
; CHECK-NEXT:    b.l.t (, %s10)
  %5 = tail call fast <256 x double> @llvm.ve.vl.pvsrl.vvsMvl(<256 x double> %0, i64 %1, <512 x i1> %2, <256 x double> %3, i32 128)
  ret <256 x double> %5
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.pvsrl.vvsMvl(<256 x double>, i64, <512 x i1>, <256 x double>, i32)
