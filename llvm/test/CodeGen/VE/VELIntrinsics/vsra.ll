; RUN: llc < %s -mtriple=ve -mattr=+vpu | FileCheck %s

;;; Test vector shift right arithmetic intrinsic instructions
;;;
;;; Note:
;;;   We test VSRA*vvl, VSRA*vvl_v, VSRA*vrl, VSRA*vrl_v, VSRA*vil, VSRA*vil_v,
;;;   VSRA*vvml_v, VSRA*vrml_v, VSRA*viml_v, PVSRA*vvl, PVSRA*vvl_v, PVSRA*vrl,
;;;   PVSRA*vrl_v, PVSRA*vvml_v, and PVSRA*vrml_v instructions.

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vsrawsx_vvvl(<256 x double> %0, <256 x double> %1) {
; CHECK-LABEL: vsrawsx_vvvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vsra.w.sx %v0, %v0, %v1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.vsrawsx.vvvl(<256 x double> %0, <256 x double> %1, i32 256)
  ret <256 x double> %3
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vsrawsx.vvvl(<256 x double>, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vsrawsx_vvvvl(<256 x double> %0, <256 x double> %1, <256 x double> %2) {
; CHECK-LABEL: vsrawsx_vvvvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 128
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vsra.w.sx %v2, %v0, %v1
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v2
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = tail call fast <256 x double> @llvm.ve.vl.vsrawsx.vvvvl(<256 x double> %0, <256 x double> %1, <256 x double> %2, i32 128)
  ret <256 x double> %4
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vsrawsx.vvvvl(<256 x double>, <256 x double>, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vsrawsx_vvsl(<256 x double> %0, i32 signext %1) {
; CHECK-LABEL: vsrawsx_vvsl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vsra.w.sx %v0, %v0, %s0
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.vsrawsx.vvsl(<256 x double> %0, i32 %1, i32 256)
  ret <256 x double> %3
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vsrawsx.vvsl(<256 x double>, i32, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vsrawsx_vvsvl(<256 x double> %0, i32 signext %1, <256 x double> %2) {
; CHECK-LABEL: vsrawsx_vvsvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    lea %s1, 128
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vsra.w.sx %v1, %v0, %s0
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v1
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = tail call fast <256 x double> @llvm.ve.vl.vsrawsx.vvsvl(<256 x double> %0, i32 %1, <256 x double> %2, i32 128)
  ret <256 x double> %4
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vsrawsx.vvsvl(<256 x double>, i32, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vsrawsx_vvsl_imm(<256 x double> %0) {
; CHECK-LABEL: vsrawsx_vvsl_imm:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vsra.w.sx %v0, %v0, 8
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call fast <256 x double> @llvm.ve.vl.vsrawsx.vvsl(<256 x double> %0, i32 8, i32 256)
  ret <256 x double> %2
}

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vsrawsx_vvsvl_imm(<256 x double> %0, <256 x double> %1) {
; CHECK-LABEL: vsrawsx_vvsvl_imm:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 128
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vsra.w.sx %v1, %v0, 8
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.vsrawsx.vvsvl(<256 x double> %0, i32 8, <256 x double> %1, i32 128)
  ret <256 x double> %3
}

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vsrawsx_vvvmvl(<256 x double> %0, <256 x double> %1, <256 x i1> %2, <256 x double> %3) {
; CHECK-LABEL: vsrawsx_vvvmvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 128
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vsra.w.sx %v2, %v0, %v1, %vm1
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v2
; CHECK-NEXT:    b.l.t (, %s10)
  %5 = tail call fast <256 x double> @llvm.ve.vl.vsrawsx.vvvmvl(<256 x double> %0, <256 x double> %1, <256 x i1> %2, <256 x double> %3, i32 128)
  ret <256 x double> %5
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vsrawsx.vvvmvl(<256 x double>, <256 x double>, <256 x i1>, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vsrawsx_vvsmvl(<256 x double> %0, i32 signext %1, <256 x i1> %2, <256 x double> %3) {
; CHECK-LABEL: vsrawsx_vvsmvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    lea %s1, 128
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vsra.w.sx %v1, %v0, %s0, %vm1
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v1
; CHECK-NEXT:    b.l.t (, %s10)
  %5 = tail call fast <256 x double> @llvm.ve.vl.vsrawsx.vvsmvl(<256 x double> %0, i32 %1, <256 x i1> %2, <256 x double> %3, i32 128)
  ret <256 x double> %5
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vsrawsx.vvsmvl(<256 x double>, i32, <256 x i1>, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vsrawsx_vvsmvl_imm(<256 x double> %0, <256 x i1> %1, <256 x double> %2) {
; CHECK-LABEL: vsrawsx_vvsmvl_imm:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 128
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vsra.w.sx %v1, %v0, 8, %vm1
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v1
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = tail call fast <256 x double> @llvm.ve.vl.vsrawsx.vvsmvl(<256 x double> %0, i32 8, <256 x i1> %1, <256 x double> %2, i32 128)
  ret <256 x double> %4
}

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vsrawzx_vvvl(<256 x double> %0, <256 x double> %1) {
; CHECK-LABEL: vsrawzx_vvvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vsra.w.zx %v0, %v0, %v1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.vsrawzx.vvvl(<256 x double> %0, <256 x double> %1, i32 256)
  ret <256 x double> %3
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vsrawzx.vvvl(<256 x double>, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vsrawzx_vvvvl(<256 x double> %0, <256 x double> %1, <256 x double> %2) {
; CHECK-LABEL: vsrawzx_vvvvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 128
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vsra.w.zx %v2, %v0, %v1
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v2
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = tail call fast <256 x double> @llvm.ve.vl.vsrawzx.vvvvl(<256 x double> %0, <256 x double> %1, <256 x double> %2, i32 128)
  ret <256 x double> %4
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vsrawzx.vvvvl(<256 x double>, <256 x double>, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vsrawzx_vvsl(<256 x double> %0, i32 signext %1) {
; CHECK-LABEL: vsrawzx_vvsl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vsra.w.zx %v0, %v0, %s0
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.vsrawzx.vvsl(<256 x double> %0, i32 %1, i32 256)
  ret <256 x double> %3
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vsrawzx.vvsl(<256 x double>, i32, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vsrawzx_vvsvl(<256 x double> %0, i32 signext %1, <256 x double> %2) {
; CHECK-LABEL: vsrawzx_vvsvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    lea %s1, 128
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vsra.w.zx %v1, %v0, %s0
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v1
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = tail call fast <256 x double> @llvm.ve.vl.vsrawzx.vvsvl(<256 x double> %0, i32 %1, <256 x double> %2, i32 128)
  ret <256 x double> %4
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vsrawzx.vvsvl(<256 x double>, i32, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vsrawzx_vvsl_imm(<256 x double> %0) {
; CHECK-LABEL: vsrawzx_vvsl_imm:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vsra.w.zx %v0, %v0, 8
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call fast <256 x double> @llvm.ve.vl.vsrawzx.vvsl(<256 x double> %0, i32 8, i32 256)
  ret <256 x double> %2
}

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vsrawzx_vvsvl_imm(<256 x double> %0, <256 x double> %1) {
; CHECK-LABEL: vsrawzx_vvsvl_imm:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 128
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vsra.w.zx %v1, %v0, 8
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.vsrawzx.vvsvl(<256 x double> %0, i32 8, <256 x double> %1, i32 128)
  ret <256 x double> %3
}

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vsrawzx_vvvmvl(<256 x double> %0, <256 x double> %1, <256 x i1> %2, <256 x double> %3) {
; CHECK-LABEL: vsrawzx_vvvmvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 128
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vsra.w.zx %v2, %v0, %v1, %vm1
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v2
; CHECK-NEXT:    b.l.t (, %s10)
  %5 = tail call fast <256 x double> @llvm.ve.vl.vsrawzx.vvvmvl(<256 x double> %0, <256 x double> %1, <256 x i1> %2, <256 x double> %3, i32 128)
  ret <256 x double> %5
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vsrawzx.vvvmvl(<256 x double>, <256 x double>, <256 x i1>, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vsrawzx_vvsmvl(<256 x double> %0, i32 signext %1, <256 x i1> %2, <256 x double> %3) {
; CHECK-LABEL: vsrawzx_vvsmvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    lea %s1, 128
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vsra.w.zx %v1, %v0, %s0, %vm1
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v1
; CHECK-NEXT:    b.l.t (, %s10)
  %5 = tail call fast <256 x double> @llvm.ve.vl.vsrawzx.vvsmvl(<256 x double> %0, i32 %1, <256 x i1> %2, <256 x double> %3, i32 128)
  ret <256 x double> %5
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vsrawzx.vvsmvl(<256 x double>, i32, <256 x i1>, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vsrawzx_vvsmvl_imm(<256 x double> %0, <256 x i1> %1, <256 x double> %2) {
; CHECK-LABEL: vsrawzx_vvsmvl_imm:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 128
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vsra.w.zx %v1, %v0, 8, %vm1
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v1
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = tail call fast <256 x double> @llvm.ve.vl.vsrawzx.vvsmvl(<256 x double> %0, i32 8, <256 x i1> %1, <256 x double> %2, i32 128)
  ret <256 x double> %4
}

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vsral_vvvl(<256 x double> %0, <256 x double> %1) {
; CHECK-LABEL: vsral_vvvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vsra.l %v0, %v0, %v1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.vsral.vvvl(<256 x double> %0, <256 x double> %1, i32 256)
  ret <256 x double> %3
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vsral.vvvl(<256 x double>, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vsral_vvvvl(<256 x double> %0, <256 x double> %1, <256 x double> %2) {
; CHECK-LABEL: vsral_vvvvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 128
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vsra.l %v2, %v0, %v1
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v2
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = tail call fast <256 x double> @llvm.ve.vl.vsral.vvvvl(<256 x double> %0, <256 x double> %1, <256 x double> %2, i32 128)
  ret <256 x double> %4
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vsral.vvvvl(<256 x double>, <256 x double>, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vsral_vvsl(<256 x double> %0, i64 %1) {
; CHECK-LABEL: vsral_vvsl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vsra.l %v0, %v0, %s0
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.vsral.vvsl(<256 x double> %0, i64 %1, i32 256)
  ret <256 x double> %3
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vsral.vvsl(<256 x double>, i64, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vsral_vvsvl(<256 x double> %0, i64 %1, <256 x double> %2) {
; CHECK-LABEL: vsral_vvsvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 128
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vsra.l %v1, %v0, %s0
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v1
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = tail call fast <256 x double> @llvm.ve.vl.vsral.vvsvl(<256 x double> %0, i64 %1, <256 x double> %2, i32 128)
  ret <256 x double> %4
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vsral.vvsvl(<256 x double>, i64, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vsral_vvsl_imm(<256 x double> %0) {
; CHECK-LABEL: vsral_vvsl_imm:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vsra.l %v0, %v0, 8
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call fast <256 x double> @llvm.ve.vl.vsral.vvsl(<256 x double> %0, i64 8, i32 256)
  ret <256 x double> %2
}

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vsral_vvsvl_imm(<256 x double> %0, <256 x double> %1) {
; CHECK-LABEL: vsral_vvsvl_imm:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 128
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vsra.l %v1, %v0, 8
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.vsral.vvsvl(<256 x double> %0, i64 8, <256 x double> %1, i32 128)
  ret <256 x double> %3
}

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vsral_vvvmvl(<256 x double> %0, <256 x double> %1, <256 x i1> %2, <256 x double> %3) {
; CHECK-LABEL: vsral_vvvmvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 128
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vsra.l %v2, %v0, %v1, %vm1
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v2
; CHECK-NEXT:    b.l.t (, %s10)
  %5 = tail call fast <256 x double> @llvm.ve.vl.vsral.vvvmvl(<256 x double> %0, <256 x double> %1, <256 x i1> %2, <256 x double> %3, i32 128)
  ret <256 x double> %5
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vsral.vvvmvl(<256 x double>, <256 x double>, <256 x i1>, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vsral_vvsmvl(<256 x double> %0, i64 %1, <256 x i1> %2, <256 x double> %3) {
; CHECK-LABEL: vsral_vvsmvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 128
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vsra.l %v1, %v0, %s0, %vm1
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v1
; CHECK-NEXT:    b.l.t (, %s10)
  %5 = tail call fast <256 x double> @llvm.ve.vl.vsral.vvsmvl(<256 x double> %0, i64 %1, <256 x i1> %2, <256 x double> %3, i32 128)
  ret <256 x double> %5
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vsral.vvsmvl(<256 x double>, i64, <256 x i1>, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vsral_vvsmvl_imm(<256 x double> %0, <256 x i1> %1, <256 x double> %2) {
; CHECK-LABEL: vsral_vvsmvl_imm:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 128
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vsra.l %v1, %v0, 8, %vm1
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v1
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = tail call fast <256 x double> @llvm.ve.vl.vsral.vvsmvl(<256 x double> %0, i64 8, <256 x i1> %1, <256 x double> %2, i32 128)
  ret <256 x double> %4
}

; Function Attrs: nounwind readnone
define fastcc <256 x double> @pvsra_vvvl(<256 x double> %0, <256 x double> %1) {
; CHECK-LABEL: pvsra_vvvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    pvsra %v0, %v0, %v1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.pvsra.vvvl(<256 x double> %0, <256 x double> %1, i32 256)
  ret <256 x double> %3
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.pvsra.vvvl(<256 x double>, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @pvsra_vvvvl(<256 x double> %0, <256 x double> %1, <256 x double> %2) {
; CHECK-LABEL: pvsra_vvvvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 128
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    pvsra %v2, %v0, %v1
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v2
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = tail call fast <256 x double> @llvm.ve.vl.pvsra.vvvvl(<256 x double> %0, <256 x double> %1, <256 x double> %2, i32 128)
  ret <256 x double> %4
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.pvsra.vvvvl(<256 x double>, <256 x double>, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @pvsra_vvsl(<256 x double> %0, i64 %1) {
; CHECK-LABEL: pvsra_vvsl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    pvsra %v0, %v0, %s0
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.pvsra.vvsl(<256 x double> %0, i64 %1, i32 256)
  ret <256 x double> %3
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.pvsra.vvsl(<256 x double>, i64, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @pvsra_vvsvl(<256 x double> %0, i64 %1, <256 x double> %2) {
; CHECK-LABEL: pvsra_vvsvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 128
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    pvsra %v1, %v0, %s0
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v1
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = tail call fast <256 x double> @llvm.ve.vl.pvsra.vvsvl(<256 x double> %0, i64 %1, <256 x double> %2, i32 128)
  ret <256 x double> %4
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.pvsra.vvsvl(<256 x double>, i64, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @pvsra_vvvMvl(<256 x double> %0, <256 x double> %1, <512 x i1> %2, <256 x double> %3) {
; CHECK-LABEL: pvsra_vvvMvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 128
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    pvsra %v2, %v0, %v1, %vm2
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v2
; CHECK-NEXT:    b.l.t (, %s10)
  %5 = tail call fast <256 x double> @llvm.ve.vl.pvsra.vvvMvl(<256 x double> %0, <256 x double> %1, <512 x i1> %2, <256 x double> %3, i32 128)
  ret <256 x double> %5
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.pvsra.vvvMvl(<256 x double>, <256 x double>, <512 x i1>, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @pvsra_vvsMvl(<256 x double> %0, i64 %1, <512 x i1> %2, <256 x double> %3) {
; CHECK-LABEL: pvsra_vvsMvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 128
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    pvsra %v1, %v0, %s0, %vm2
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v1
; CHECK-NEXT:    b.l.t (, %s10)
  %5 = tail call fast <256 x double> @llvm.ve.vl.pvsra.vvsMvl(<256 x double> %0, i64 %1, <512 x i1> %2, <256 x double> %3, i32 128)
  ret <256 x double> %5
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.pvsra.vvsMvl(<256 x double>, i64, <512 x i1>, <256 x double>, i32)
