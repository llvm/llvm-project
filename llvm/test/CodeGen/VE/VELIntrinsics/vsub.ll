; RUN: llc < %s -mtriple=ve -mattr=+vpu | FileCheck %s

;;; Test vector subtract intrinsic instructions
;;;
;;; Note:
;;;   We test VSUB*vvl, VSUB*vvl_v, VSUB*rvl, VSUB*rvl_v, VSUB*ivl, VSUB*ivl_v,
;;;   VSUB*vvml_v, VSUB*rvml_v, VSUB*ivml_v, PVSUB*vvl, PVSUB*vvl_v, PVSUB*rvl,
;;;   PVSUB*rvl_v, PVSUB*vvml_v, and PVSUB*rvml_v instructions.

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vsubul_vvvl(<256 x double> %0, <256 x double> %1) {
; CHECK-LABEL: vsubul_vvvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vsubu.l %v0, %v0, %v1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.vsubul.vvvl(<256 x double> %0, <256 x double> %1, i32 256)
  ret <256 x double> %3
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vsubul.vvvl(<256 x double>, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vsubul_vvvvl(<256 x double> %0, <256 x double> %1, <256 x double> %2) {
; CHECK-LABEL: vsubul_vvvvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 128
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vsubu.l %v2, %v0, %v1
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v2
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = tail call fast <256 x double> @llvm.ve.vl.vsubul.vvvvl(<256 x double> %0, <256 x double> %1, <256 x double> %2, i32 128)
  ret <256 x double> %4
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vsubul.vvvvl(<256 x double>, <256 x double>, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vsubul_vsvl(i64 %0, <256 x double> %1) {
; CHECK-LABEL: vsubul_vsvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vsubu.l %v0, %s0, %v0
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.vsubul.vsvl(i64 %0, <256 x double> %1, i32 256)
  ret <256 x double> %3
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vsubul.vsvl(i64, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vsubul_vsvvl(i64 %0, <256 x double> %1, <256 x double> %2) {
; CHECK-LABEL: vsubul_vsvvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 128
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vsubu.l %v1, %s0, %v0
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v1
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = tail call fast <256 x double> @llvm.ve.vl.vsubul.vsvvl(i64 %0, <256 x double> %1, <256 x double> %2, i32 128)
  ret <256 x double> %4
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vsubul.vsvvl(i64, <256 x double>, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vsubul_vsvl_imm(<256 x double> %0) {
; CHECK-LABEL: vsubul_vsvl_imm:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vsubu.l %v0, 8, %v0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call fast <256 x double> @llvm.ve.vl.vsubul.vsvl(i64 8, <256 x double> %0, i32 256)
  ret <256 x double> %2
}

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vsubul_vsvvl_imm(<256 x double> %0, <256 x double> %1) {
; CHECK-LABEL: vsubul_vsvvl_imm:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 128
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vsubu.l %v1, 8, %v0
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.vsubul.vsvvl(i64 8, <256 x double> %0, <256 x double> %1, i32 128)
  ret <256 x double> %3
}

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vsubul_vvvmvl(<256 x double> %0, <256 x double> %1, <256 x i1> %2, <256 x double> %3) {
; CHECK-LABEL: vsubul_vvvmvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 128
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vsubu.l %v2, %v0, %v1, %vm1
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v2
; CHECK-NEXT:    b.l.t (, %s10)
  %5 = tail call fast <256 x double> @llvm.ve.vl.vsubul.vvvmvl(<256 x double> %0, <256 x double> %1, <256 x i1> %2, <256 x double> %3, i32 128)
  ret <256 x double> %5
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vsubul.vvvmvl(<256 x double>, <256 x double>, <256 x i1>, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vsubul_vsvmvl(i64 %0, <256 x double> %1, <256 x i1> %2, <256 x double> %3) {
; CHECK-LABEL: vsubul_vsvmvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 128
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vsubu.l %v1, %s0, %v0, %vm1
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v1
; CHECK-NEXT:    b.l.t (, %s10)
  %5 = tail call fast <256 x double> @llvm.ve.vl.vsubul.vsvmvl(i64 %0, <256 x double> %1, <256 x i1> %2, <256 x double> %3, i32 128)
  ret <256 x double> %5
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vsubul.vsvmvl(i64, <256 x double>, <256 x i1>, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vsubul_vsvmvl_imm(<256 x double> %0, <256 x i1> %1, <256 x double> %2) {
; CHECK-LABEL: vsubul_vsvmvl_imm:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 128
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vsubu.l %v1, 8, %v0, %vm1
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v1
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = tail call fast <256 x double> @llvm.ve.vl.vsubul.vsvmvl(i64 8, <256 x double> %0, <256 x i1> %1, <256 x double> %2, i32 128)
  ret <256 x double> %4
}

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vsubuw_vvvl(<256 x double> %0, <256 x double> %1) {
; CHECK-LABEL: vsubuw_vvvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vsubu.w %v0, %v0, %v1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.vsubuw.vvvl(<256 x double> %0, <256 x double> %1, i32 256)
  ret <256 x double> %3
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vsubuw.vvvl(<256 x double>, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vsubuw_vvvvl(<256 x double> %0, <256 x double> %1, <256 x double> %2) {
; CHECK-LABEL: vsubuw_vvvvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 128
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vsubu.w %v2, %v0, %v1
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v2
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = tail call fast <256 x double> @llvm.ve.vl.vsubuw.vvvvl(<256 x double> %0, <256 x double> %1, <256 x double> %2, i32 128)
  ret <256 x double> %4
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vsubuw.vvvvl(<256 x double>, <256 x double>, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vsubuw_vsvl(i32 signext %0, <256 x double> %1) {
; CHECK-LABEL: vsubuw_vsvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vsubu.w %v0, %s0, %v0
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.vsubuw.vsvl(i32 %0, <256 x double> %1, i32 256)
  ret <256 x double> %3
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vsubuw.vsvl(i32, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vsubuw_vsvvl(i32 signext %0, <256 x double> %1, <256 x double> %2) {
; CHECK-LABEL: vsubuw_vsvvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    lea %s1, 128
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vsubu.w %v1, %s0, %v0
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v1
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = tail call fast <256 x double> @llvm.ve.vl.vsubuw.vsvvl(i32 %0, <256 x double> %1, <256 x double> %2, i32 128)
  ret <256 x double> %4
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vsubuw.vsvvl(i32, <256 x double>, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vsubuw_vsvl_imm(<256 x double> %0) {
; CHECK-LABEL: vsubuw_vsvl_imm:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vsubu.w %v0, 8, %v0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call fast <256 x double> @llvm.ve.vl.vsubuw.vsvl(i32 8, <256 x double> %0, i32 256)
  ret <256 x double> %2
}

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vsubuw_vsvvl_imm(<256 x double> %0, <256 x double> %1) {
; CHECK-LABEL: vsubuw_vsvvl_imm:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 128
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vsubu.w %v1, 8, %v0
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.vsubuw.vsvvl(i32 8, <256 x double> %0, <256 x double> %1, i32 128)
  ret <256 x double> %3
}

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vsubuw_vvvmvl(<256 x double> %0, <256 x double> %1, <256 x i1> %2, <256 x double> %3) {
; CHECK-LABEL: vsubuw_vvvmvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 128
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vsubu.w %v2, %v0, %v1, %vm1
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v2
; CHECK-NEXT:    b.l.t (, %s10)
  %5 = tail call fast <256 x double> @llvm.ve.vl.vsubuw.vvvmvl(<256 x double> %0, <256 x double> %1, <256 x i1> %2, <256 x double> %3, i32 128)
  ret <256 x double> %5
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vsubuw.vvvmvl(<256 x double>, <256 x double>, <256 x i1>, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vsubuw_vsvmvl(i32 signext %0, <256 x double> %1, <256 x i1> %2, <256 x double> %3) {
; CHECK-LABEL: vsubuw_vsvmvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    lea %s1, 128
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vsubu.w %v1, %s0, %v0, %vm1
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v1
; CHECK-NEXT:    b.l.t (, %s10)
  %5 = tail call fast <256 x double> @llvm.ve.vl.vsubuw.vsvmvl(i32 %0, <256 x double> %1, <256 x i1> %2, <256 x double> %3, i32 128)
  ret <256 x double> %5
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vsubuw.vsvmvl(i32, <256 x double>, <256 x i1>, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vsubuw_vsvmvl_imm(<256 x double> %0, <256 x i1> %1, <256 x double> %2) {
; CHECK-LABEL: vsubuw_vsvmvl_imm:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 128
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vsubu.w %v1, 8, %v0, %vm1
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v1
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = tail call fast <256 x double> @llvm.ve.vl.vsubuw.vsvmvl(i32 8, <256 x double> %0, <256 x i1> %1, <256 x double> %2, i32 128)
  ret <256 x double> %4
}

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vsubswsx_vvvl(<256 x double> %0, <256 x double> %1) {
; CHECK-LABEL: vsubswsx_vvvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vsubs.w.sx %v0, %v0, %v1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.vsubswsx.vvvl(<256 x double> %0, <256 x double> %1, i32 256)
  ret <256 x double> %3
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vsubswsx.vvvl(<256 x double>, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vsubswsx_vvvvl(<256 x double> %0, <256 x double> %1, <256 x double> %2) {
; CHECK-LABEL: vsubswsx_vvvvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 128
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vsubs.w.sx %v2, %v0, %v1
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v2
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = tail call fast <256 x double> @llvm.ve.vl.vsubswsx.vvvvl(<256 x double> %0, <256 x double> %1, <256 x double> %2, i32 128)
  ret <256 x double> %4
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vsubswsx.vvvvl(<256 x double>, <256 x double>, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vsubswsx_vsvl(i32 signext %0, <256 x double> %1) {
; CHECK-LABEL: vsubswsx_vsvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vsubs.w.sx %v0, %s0, %v0
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.vsubswsx.vsvl(i32 %0, <256 x double> %1, i32 256)
  ret <256 x double> %3
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vsubswsx.vsvl(i32, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vsubswsx_vsvvl(i32 signext %0, <256 x double> %1, <256 x double> %2) {
; CHECK-LABEL: vsubswsx_vsvvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    lea %s1, 128
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vsubs.w.sx %v1, %s0, %v0
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v1
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = tail call fast <256 x double> @llvm.ve.vl.vsubswsx.vsvvl(i32 %0, <256 x double> %1, <256 x double> %2, i32 128)
  ret <256 x double> %4
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vsubswsx.vsvvl(i32, <256 x double>, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vsubswsx_vsvl_imm(<256 x double> %0) {
; CHECK-LABEL: vsubswsx_vsvl_imm:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vsubs.w.sx %v0, 8, %v0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call fast <256 x double> @llvm.ve.vl.vsubswsx.vsvl(i32 8, <256 x double> %0, i32 256)
  ret <256 x double> %2
}

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vsubswsx_vsvvl_imm(<256 x double> %0, <256 x double> %1) {
; CHECK-LABEL: vsubswsx_vsvvl_imm:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 128
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vsubs.w.sx %v1, 8, %v0
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.vsubswsx.vsvvl(i32 8, <256 x double> %0, <256 x double> %1, i32 128)
  ret <256 x double> %3
}

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vsubswsx_vvvmvl(<256 x double> %0, <256 x double> %1, <256 x i1> %2, <256 x double> %3) {
; CHECK-LABEL: vsubswsx_vvvmvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 128
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vsubs.w.sx %v2, %v0, %v1, %vm1
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v2
; CHECK-NEXT:    b.l.t (, %s10)
  %5 = tail call fast <256 x double> @llvm.ve.vl.vsubswsx.vvvmvl(<256 x double> %0, <256 x double> %1, <256 x i1> %2, <256 x double> %3, i32 128)
  ret <256 x double> %5
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vsubswsx.vvvmvl(<256 x double>, <256 x double>, <256 x i1>, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vsubswsx_vsvmvl(i32 signext %0, <256 x double> %1, <256 x i1> %2, <256 x double> %3) {
; CHECK-LABEL: vsubswsx_vsvmvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    lea %s1, 128
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vsubs.w.sx %v1, %s0, %v0, %vm1
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v1
; CHECK-NEXT:    b.l.t (, %s10)
  %5 = tail call fast <256 x double> @llvm.ve.vl.vsubswsx.vsvmvl(i32 %0, <256 x double> %1, <256 x i1> %2, <256 x double> %3, i32 128)
  ret <256 x double> %5
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vsubswsx.vsvmvl(i32, <256 x double>, <256 x i1>, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vsubswsx_vsvmvl_imm(<256 x double> %0, <256 x i1> %1, <256 x double> %2) {
; CHECK-LABEL: vsubswsx_vsvmvl_imm:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 128
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vsubs.w.sx %v1, 8, %v0, %vm1
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v1
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = tail call fast <256 x double> @llvm.ve.vl.vsubswsx.vsvmvl(i32 8, <256 x double> %0, <256 x i1> %1, <256 x double> %2, i32 128)
  ret <256 x double> %4
}

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vsubswzx_vvvl(<256 x double> %0, <256 x double> %1) {
; CHECK-LABEL: vsubswzx_vvvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vsubs.w.zx %v0, %v0, %v1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.vsubswzx.vvvl(<256 x double> %0, <256 x double> %1, i32 256)
  ret <256 x double> %3
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vsubswzx.vvvl(<256 x double>, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vsubswzx_vvvvl(<256 x double> %0, <256 x double> %1, <256 x double> %2) {
; CHECK-LABEL: vsubswzx_vvvvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 128
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vsubs.w.zx %v2, %v0, %v1
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v2
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = tail call fast <256 x double> @llvm.ve.vl.vsubswzx.vvvvl(<256 x double> %0, <256 x double> %1, <256 x double> %2, i32 128)
  ret <256 x double> %4
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vsubswzx.vvvvl(<256 x double>, <256 x double>, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vsubswzx_vsvl(i32 signext %0, <256 x double> %1) {
; CHECK-LABEL: vsubswzx_vsvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vsubs.w.zx %v0, %s0, %v0
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.vsubswzx.vsvl(i32 %0, <256 x double> %1, i32 256)
  ret <256 x double> %3
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vsubswzx.vsvl(i32, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vsubswzx_vsvvl(i32 signext %0, <256 x double> %1, <256 x double> %2) {
; CHECK-LABEL: vsubswzx_vsvvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    lea %s1, 128
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vsubs.w.zx %v1, %s0, %v0
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v1
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = tail call fast <256 x double> @llvm.ve.vl.vsubswzx.vsvvl(i32 %0, <256 x double> %1, <256 x double> %2, i32 128)
  ret <256 x double> %4
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vsubswzx.vsvvl(i32, <256 x double>, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vsubswzx_vsvl_imm(<256 x double> %0) {
; CHECK-LABEL: vsubswzx_vsvl_imm:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vsubs.w.zx %v0, 8, %v0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call fast <256 x double> @llvm.ve.vl.vsubswzx.vsvl(i32 8, <256 x double> %0, i32 256)
  ret <256 x double> %2
}

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vsubswzx_vsvvl_imm(<256 x double> %0, <256 x double> %1) {
; CHECK-LABEL: vsubswzx_vsvvl_imm:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 128
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vsubs.w.zx %v1, 8, %v0
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.vsubswzx.vsvvl(i32 8, <256 x double> %0, <256 x double> %1, i32 128)
  ret <256 x double> %3
}

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vsubswzx_vvvmvl(<256 x double> %0, <256 x double> %1, <256 x i1> %2, <256 x double> %3) {
; CHECK-LABEL: vsubswzx_vvvmvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 128
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vsubs.w.zx %v2, %v0, %v1, %vm1
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v2
; CHECK-NEXT:    b.l.t (, %s10)
  %5 = tail call fast <256 x double> @llvm.ve.vl.vsubswzx.vvvmvl(<256 x double> %0, <256 x double> %1, <256 x i1> %2, <256 x double> %3, i32 128)
  ret <256 x double> %5
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vsubswzx.vvvmvl(<256 x double>, <256 x double>, <256 x i1>, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vsubswzx_vsvmvl(i32 signext %0, <256 x double> %1, <256 x i1> %2, <256 x double> %3) {
; CHECK-LABEL: vsubswzx_vsvmvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    lea %s1, 128
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vsubs.w.zx %v1, %s0, %v0, %vm1
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v1
; CHECK-NEXT:    b.l.t (, %s10)
  %5 = tail call fast <256 x double> @llvm.ve.vl.vsubswzx.vsvmvl(i32 %0, <256 x double> %1, <256 x i1> %2, <256 x double> %3, i32 128)
  ret <256 x double> %5
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vsubswzx.vsvmvl(i32, <256 x double>, <256 x i1>, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vsubswzx_vsvmvl_imm(<256 x double> %0, <256 x i1> %1, <256 x double> %2) {
; CHECK-LABEL: vsubswzx_vsvmvl_imm:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 128
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vsubs.w.zx %v1, 8, %v0, %vm1
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v1
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = tail call fast <256 x double> @llvm.ve.vl.vsubswzx.vsvmvl(i32 8, <256 x double> %0, <256 x i1> %1, <256 x double> %2, i32 128)
  ret <256 x double> %4
}

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vsubsl_vvvl(<256 x double> %0, <256 x double> %1) {
; CHECK-LABEL: vsubsl_vvvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vsubs.l %v0, %v0, %v1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.vsubsl.vvvl(<256 x double> %0, <256 x double> %1, i32 256)
  ret <256 x double> %3
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vsubsl.vvvl(<256 x double>, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vsubsl_vvvvl(<256 x double> %0, <256 x double> %1, <256 x double> %2) {
; CHECK-LABEL: vsubsl_vvvvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 128
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vsubs.l %v2, %v0, %v1
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v2
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = tail call fast <256 x double> @llvm.ve.vl.vsubsl.vvvvl(<256 x double> %0, <256 x double> %1, <256 x double> %2, i32 128)
  ret <256 x double> %4
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vsubsl.vvvvl(<256 x double>, <256 x double>, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vsubsl_vsvl(i64 %0, <256 x double> %1) {
; CHECK-LABEL: vsubsl_vsvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vsubs.l %v0, %s0, %v0
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.vsubsl.vsvl(i64 %0, <256 x double> %1, i32 256)
  ret <256 x double> %3
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vsubsl.vsvl(i64, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vsubsl_vsvvl(i64 %0, <256 x double> %1, <256 x double> %2) {
; CHECK-LABEL: vsubsl_vsvvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 128
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vsubs.l %v1, %s0, %v0
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v1
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = tail call fast <256 x double> @llvm.ve.vl.vsubsl.vsvvl(i64 %0, <256 x double> %1, <256 x double> %2, i32 128)
  ret <256 x double> %4
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vsubsl.vsvvl(i64, <256 x double>, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vsubsl_vsvl_imm(<256 x double> %0) {
; CHECK-LABEL: vsubsl_vsvl_imm:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vsubs.l %v0, 8, %v0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call fast <256 x double> @llvm.ve.vl.vsubsl.vsvl(i64 8, <256 x double> %0, i32 256)
  ret <256 x double> %2
}

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vsubsl_vsvvl_imm(<256 x double> %0, <256 x double> %1) {
; CHECK-LABEL: vsubsl_vsvvl_imm:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 128
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vsubs.l %v1, 8, %v0
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.vsubsl.vsvvl(i64 8, <256 x double> %0, <256 x double> %1, i32 128)
  ret <256 x double> %3
}

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vsubsl_vvvmvl(<256 x double> %0, <256 x double> %1, <256 x i1> %2, <256 x double> %3) {
; CHECK-LABEL: vsubsl_vvvmvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 128
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vsubs.l %v2, %v0, %v1, %vm1
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v2
; CHECK-NEXT:    b.l.t (, %s10)
  %5 = tail call fast <256 x double> @llvm.ve.vl.vsubsl.vvvmvl(<256 x double> %0, <256 x double> %1, <256 x i1> %2, <256 x double> %3, i32 128)
  ret <256 x double> %5
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vsubsl.vvvmvl(<256 x double>, <256 x double>, <256 x i1>, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vsubsl_vsvmvl(i64 %0, <256 x double> %1, <256 x i1> %2, <256 x double> %3) {
; CHECK-LABEL: vsubsl_vsvmvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 128
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vsubs.l %v1, %s0, %v0, %vm1
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v1
; CHECK-NEXT:    b.l.t (, %s10)
  %5 = tail call fast <256 x double> @llvm.ve.vl.vsubsl.vsvmvl(i64 %0, <256 x double> %1, <256 x i1> %2, <256 x double> %3, i32 128)
  ret <256 x double> %5
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vsubsl.vsvmvl(i64, <256 x double>, <256 x i1>, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vsubsl_vsvmvl_imm(<256 x double> %0, <256 x i1> %1, <256 x double> %2) {
; CHECK-LABEL: vsubsl_vsvmvl_imm:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 128
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vsubs.l %v1, 8, %v0, %vm1
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v1
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = tail call fast <256 x double> @llvm.ve.vl.vsubsl.vsvmvl(i64 8, <256 x double> %0, <256 x i1> %1, <256 x double> %2, i32 128)
  ret <256 x double> %4
}

; Function Attrs: nounwind readnone
define fastcc <256 x double> @pvsubu_vvvl(<256 x double> %0, <256 x double> %1) {
; CHECK-LABEL: pvsubu_vvvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    pvsubu %v0, %v0, %v1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.pvsubu.vvvl(<256 x double> %0, <256 x double> %1, i32 256)
  ret <256 x double> %3
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.pvsubu.vvvl(<256 x double>, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @pvsubu_vvvvl(<256 x double> %0, <256 x double> %1, <256 x double> %2) {
; CHECK-LABEL: pvsubu_vvvvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 128
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    pvsubu %v2, %v0, %v1
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v2
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = tail call fast <256 x double> @llvm.ve.vl.pvsubu.vvvvl(<256 x double> %0, <256 x double> %1, <256 x double> %2, i32 128)
  ret <256 x double> %4
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.pvsubu.vvvvl(<256 x double>, <256 x double>, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @pvsubu_vsvl(i64 %0, <256 x double> %1) {
; CHECK-LABEL: pvsubu_vsvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    pvsubu %v0, %s0, %v0
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.pvsubu.vsvl(i64 %0, <256 x double> %1, i32 256)
  ret <256 x double> %3
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.pvsubu.vsvl(i64, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @pvsubu_vsvvl(i64 %0, <256 x double> %1, <256 x double> %2) {
; CHECK-LABEL: pvsubu_vsvvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 128
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    pvsubu %v1, %s0, %v0
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v1
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = tail call fast <256 x double> @llvm.ve.vl.pvsubu.vsvvl(i64 %0, <256 x double> %1, <256 x double> %2, i32 128)
  ret <256 x double> %4
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.pvsubu.vsvvl(i64, <256 x double>, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @pvsubu_vvvMvl(<256 x double> %0, <256 x double> %1, <512 x i1> %2, <256 x double> %3) {
; CHECK-LABEL: pvsubu_vvvMvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 128
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    pvsubu %v2, %v0, %v1, %vm2
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v2
; CHECK-NEXT:    b.l.t (, %s10)
  %5 = tail call fast <256 x double> @llvm.ve.vl.pvsubu.vvvMvl(<256 x double> %0, <256 x double> %1, <512 x i1> %2, <256 x double> %3, i32 128)
  ret <256 x double> %5
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.pvsubu.vvvMvl(<256 x double>, <256 x double>, <512 x i1>, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @pvsubu_vsvMvl(i64 %0, <256 x double> %1, <512 x i1> %2, <256 x double> %3) {
; CHECK-LABEL: pvsubu_vsvMvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 128
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    pvsubu %v1, %s0, %v0, %vm2
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v1
; CHECK-NEXT:    b.l.t (, %s10)
  %5 = tail call fast <256 x double> @llvm.ve.vl.pvsubu.vsvMvl(i64 %0, <256 x double> %1, <512 x i1> %2, <256 x double> %3, i32 128)
  ret <256 x double> %5
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.pvsubu.vsvMvl(i64, <256 x double>, <512 x i1>, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @pvsubs_vvvl(<256 x double> %0, <256 x double> %1) {
; CHECK-LABEL: pvsubs_vvvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    pvsubs %v0, %v0, %v1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.pvsubs.vvvl(<256 x double> %0, <256 x double> %1, i32 256)
  ret <256 x double> %3
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.pvsubs.vvvl(<256 x double>, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @pvsubs_vvvvl(<256 x double> %0, <256 x double> %1, <256 x double> %2) {
; CHECK-LABEL: pvsubs_vvvvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 128
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    pvsubs %v2, %v0, %v1
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v2
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = tail call fast <256 x double> @llvm.ve.vl.pvsubs.vvvvl(<256 x double> %0, <256 x double> %1, <256 x double> %2, i32 128)
  ret <256 x double> %4
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.pvsubs.vvvvl(<256 x double>, <256 x double>, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @pvsubs_vsvl(i64 %0, <256 x double> %1) {
; CHECK-LABEL: pvsubs_vsvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    pvsubs %v0, %s0, %v0
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.pvsubs.vsvl(i64 %0, <256 x double> %1, i32 256)
  ret <256 x double> %3
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.pvsubs.vsvl(i64, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @pvsubs_vsvvl(i64 %0, <256 x double> %1, <256 x double> %2) {
; CHECK-LABEL: pvsubs_vsvvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 128
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    pvsubs %v1, %s0, %v0
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v1
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = tail call fast <256 x double> @llvm.ve.vl.pvsubs.vsvvl(i64 %0, <256 x double> %1, <256 x double> %2, i32 128)
  ret <256 x double> %4
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.pvsubs.vsvvl(i64, <256 x double>, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @pvsubs_vvvMvl(<256 x double> %0, <256 x double> %1, <512 x i1> %2, <256 x double> %3) {
; CHECK-LABEL: pvsubs_vvvMvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 128
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    pvsubs %v2, %v0, %v1, %vm2
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v2
; CHECK-NEXT:    b.l.t (, %s10)
  %5 = tail call fast <256 x double> @llvm.ve.vl.pvsubs.vvvMvl(<256 x double> %0, <256 x double> %1, <512 x i1> %2, <256 x double> %3, i32 128)
  ret <256 x double> %5
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.pvsubs.vvvMvl(<256 x double>, <256 x double>, <512 x i1>, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @pvsubs_vsvMvl(i64 %0, <256 x double> %1, <512 x i1> %2, <256 x double> %3) {
; CHECK-LABEL: pvsubs_vsvMvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 128
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    pvsubs %v1, %s0, %v0, %vm2
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v1
; CHECK-NEXT:    b.l.t (, %s10)
  %5 = tail call fast <256 x double> @llvm.ve.vl.pvsubs.vsvMvl(i64 %0, <256 x double> %1, <512 x i1> %2, <256 x double> %3, i32 128)
  ret <256 x double> %5
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.pvsubs.vsvMvl(i64, <256 x double>, <512 x i1>, <256 x double>, i32)
