; RUN: llc < %s -mtriple=ve -mattr=+vpu | FileCheck %s

;;; Test vector gather intrinsic instructions
;;;
;;; Note:
;;;   We test VGT*vrrl, VGT*vrrl_v, VGT*vrzl, VGT*vrzl_v, VGT*virl, VGT*virl_v,
;;;   VGT*vizl, VGT*vizl_v, VGT*vrrml, VGT*vrrml_v, VGT*vrzml, VGT*vrzml_v,
;;;   VGT*virml, VGT*virml_v, VGT*vizml, and VGT*vizml_v instructions.

; Function Attrs: nounwind readonly
define fastcc <256 x double> @vgt_vvssl(<256 x double> %0, i64 %1, i64 %2) {
; CHECK-LABEL: vgt_vvssl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s2, 256
; CHECK-NEXT:    lvl %s2
; CHECK-NEXT:    vgt %v0, %v0, %s0, %s1
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = tail call fast <256 x double> @llvm.ve.vl.vgt.vvssl(<256 x double> %0, i64 %1, i64 %2, i32 256)
  ret <256 x double> %4
}

; Function Attrs: nounwind readonly
declare <256 x double> @llvm.ve.vl.vgt.vvssl(<256 x double>, i64, i64, i32)

; Function Attrs: nounwind readonly
define fastcc <256 x double> @vgt_vvssvl(<256 x double> %0, i64 %1, i64 %2, <256 x double> %3) {
; CHECK-LABEL: vgt_vvssvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s2, 128
; CHECK-NEXT:    lvl %s2
; CHECK-NEXT:    vgt %v1, %v0, %s0, %s1
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v1
; CHECK-NEXT:    b.l.t (, %s10)
  %5 = tail call fast <256 x double> @llvm.ve.vl.vgt.vvssvl(<256 x double> %0, i64 %1, i64 %2, <256 x double> %3, i32 128)
  ret <256 x double> %5
}

; Function Attrs: nounwind readonly
declare <256 x double> @llvm.ve.vl.vgt.vvssvl(<256 x double>, i64, i64, <256 x double>, i32)

; Function Attrs: nounwind readonly
define fastcc <256 x double> @vgt_vvssl_imm_1(<256 x double> %0, i64 %1) {
; CHECK-LABEL: vgt_vvssl_imm_1:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vgt %v0, %v0, %s0, 0
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.vgt.vvssl(<256 x double> %0, i64 %1, i64 0, i32 256)
  ret <256 x double> %3
}

; Function Attrs: nounwind readonly
define fastcc <256 x double> @vgt_vvssvl_imm_1(<256 x double> %0, i64 %1, <256 x double> %2) {
; CHECK-LABEL: vgt_vvssvl_imm_1:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 128
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vgt %v1, %v0, %s0, 0
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v1
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = tail call fast <256 x double> @llvm.ve.vl.vgt.vvssvl(<256 x double> %0, i64 %1, i64 0, <256 x double> %2, i32 128)
  ret <256 x double> %4
}

; Function Attrs: nounwind readonly
define fastcc <256 x double> @vgt_vvssl_imm_2(<256 x double> %0, i64 %1) {
; CHECK-LABEL: vgt_vvssl_imm_2:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vgt %v0, %v0, 8, %s0
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.vgt.vvssl(<256 x double> %0, i64 8, i64 %1, i32 256)
  ret <256 x double> %3
}

; Function Attrs: nounwind readonly
define fastcc <256 x double> @vgt_vvssvl_imm_2(<256 x double> %0, i64 %1, <256 x double> %2) {
; CHECK-LABEL: vgt_vvssvl_imm_2:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 128
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vgt %v1, %v0, 8, %s0
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v1
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = tail call fast <256 x double> @llvm.ve.vl.vgt.vvssvl(<256 x double> %0, i64 8, i64 %1, <256 x double> %2, i32 128)
  ret <256 x double> %4
}

; Function Attrs: nounwind readonly
define fastcc <256 x double> @vgt_vvssl_imm_3(<256 x double> %0) {
; CHECK-LABEL: vgt_vvssl_imm_3:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vgt %v0, %v0, 8, 0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call fast <256 x double> @llvm.ve.vl.vgt.vvssl(<256 x double> %0, i64 8, i64 0, i32 256)
  ret <256 x double> %2
}

; Function Attrs: nounwind readonly
define fastcc <256 x double> @vgt_vvssvl_imm_3(<256 x double> %0, <256 x double> %1) {
; CHECK-LABEL: vgt_vvssvl_imm_3:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 128
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vgt %v1, %v0, 8, 0
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.vgt.vvssvl(<256 x double> %0, i64 8, i64 0, <256 x double> %1, i32 128)
  ret <256 x double> %3
}

; Function Attrs: nounwind readonly
define fastcc <256 x double> @vgt_vvssml(<256 x double> %0, i64 %1, i64 %2, <256 x i1> %3) {
; CHECK-LABEL: vgt_vvssml:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s2, 256
; CHECK-NEXT:    lvl %s2
; CHECK-NEXT:    vgt %v0, %v0, %s0, %s1, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %5 = tail call fast <256 x double> @llvm.ve.vl.vgt.vvssml(<256 x double> %0, i64 %1, i64 %2, <256 x i1> %3, i32 256)
  ret <256 x double> %5
}

; Function Attrs: nounwind readonly
declare <256 x double> @llvm.ve.vl.vgt.vvssml(<256 x double>, i64, i64, <256 x i1>, i32)

; Function Attrs: nounwind readonly
define fastcc <256 x double> @vgt_vvssmvl(<256 x double> %0, i64 %1, i64 %2, <256 x i1> %3, <256 x double> %4) {
; CHECK-LABEL: vgt_vvssmvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s2, 128
; CHECK-NEXT:    lvl %s2
; CHECK-NEXT:    vgt %v1, %v0, %s0, %s1, %vm1
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v1
; CHECK-NEXT:    b.l.t (, %s10)
  %6 = tail call fast <256 x double> @llvm.ve.vl.vgt.vvssmvl(<256 x double> %0, i64 %1, i64 %2, <256 x i1> %3, <256 x double> %4, i32 128)
  ret <256 x double> %6
}

; Function Attrs: nounwind readonly
declare <256 x double> @llvm.ve.vl.vgt.vvssmvl(<256 x double>, i64, i64, <256 x i1>, <256 x double>, i32)

; Function Attrs: nounwind readonly
define fastcc <256 x double> @vgt_vvssml_imm_1(<256 x double> %0, i64 %1, <256 x i1> %2) {
; CHECK-LABEL: vgt_vvssml_imm_1:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vgt %v0, %v0, %s0, 0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = tail call fast <256 x double> @llvm.ve.vl.vgt.vvssml(<256 x double> %0, i64 %1, i64 0, <256 x i1> %2, i32 256)
  ret <256 x double> %4
}

; Function Attrs: nounwind readonly
define fastcc <256 x double> @vgt_vvssmvl_imm_1(<256 x double> %0, i64 %1, <256 x i1> %2, <256 x double> %3) {
; CHECK-LABEL: vgt_vvssmvl_imm_1:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 128
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vgt %v1, %v0, %s0, 0, %vm1
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v1
; CHECK-NEXT:    b.l.t (, %s10)
  %5 = tail call fast <256 x double> @llvm.ve.vl.vgt.vvssmvl(<256 x double> %0, i64 %1, i64 0, <256 x i1> %2, <256 x double> %3, i32 128)
  ret <256 x double> %5
}

; Function Attrs: nounwind readonly
define fastcc <256 x double> @vgt_vvssml_imm_2(<256 x double> %0, i64 %1, <256 x i1> %2) {
; CHECK-LABEL: vgt_vvssml_imm_2:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vgt %v0, %v0, 8, %s0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = tail call fast <256 x double> @llvm.ve.vl.vgt.vvssml(<256 x double> %0, i64 8, i64 %1, <256 x i1> %2, i32 256)
  ret <256 x double> %4
}

; Function Attrs: nounwind readonly
define fastcc <256 x double> @vgt_vvssmvl_imm_2(<256 x double> %0, i64 %1, <256 x i1> %2, <256 x double> %3) {
; CHECK-LABEL: vgt_vvssmvl_imm_2:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 128
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vgt %v1, %v0, 8, %s0, %vm1
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v1
; CHECK-NEXT:    b.l.t (, %s10)
  %5 = tail call fast <256 x double> @llvm.ve.vl.vgt.vvssmvl(<256 x double> %0, i64 8, i64 %1, <256 x i1> %2, <256 x double> %3, i32 128)
  ret <256 x double> %5
}

; Function Attrs: nounwind readonly
define fastcc <256 x double> @vgt_vvssml_imm_3(<256 x double> %0, <256 x i1> %1) {
; CHECK-LABEL: vgt_vvssml_imm_3:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vgt %v0, %v0, 8, 0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.vgt.vvssml(<256 x double> %0, i64 8, i64 0, <256 x i1> %1, i32 256)
  ret <256 x double> %3
}

; Function Attrs: nounwind readonly
define fastcc <256 x double> @vgt_vvssmvl_imm_3(<256 x double> %0, <256 x i1> %1, <256 x double> %2) {
; CHECK-LABEL: vgt_vvssmvl_imm_3:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 128
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vgt %v1, %v0, 8, 0, %vm1
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v1
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = tail call fast <256 x double> @llvm.ve.vl.vgt.vvssmvl(<256 x double> %0, i64 8, i64 0, <256 x i1> %1, <256 x double> %2, i32 128)
  ret <256 x double> %4
}

; Function Attrs: nounwind readonly
define fastcc <256 x double> @vgt_vvssl_no_imm_1(<256 x double> %0, i64 %1) {
; CHECK-LABEL: vgt_vvssl_no_imm_1:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    or %s2, 8, (0)1
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vgt %v0, %v0, %s0, %s2
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.vgt.vvssl(<256 x double> %0, i64 %1, i64 8, i32 256)
  ret <256 x double> %3
}

; Function Attrs: nounwind readonly
define fastcc <256 x double> @vgtnc_vvssl(<256 x double> %0, i64 %1, i64 %2) {
; CHECK-LABEL: vgtnc_vvssl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s2, 256
; CHECK-NEXT:    lvl %s2
; CHECK-NEXT:    vgt.nc %v0, %v0, %s0, %s1
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = tail call fast <256 x double> @llvm.ve.vl.vgtnc.vvssl(<256 x double> %0, i64 %1, i64 %2, i32 256)
  ret <256 x double> %4
}

; Function Attrs: nounwind readonly
declare <256 x double> @llvm.ve.vl.vgtnc.vvssl(<256 x double>, i64, i64, i32)

; Function Attrs: nounwind readonly
define fastcc <256 x double> @vgtnc_vvssvl(<256 x double> %0, i64 %1, i64 %2, <256 x double> %3) {
; CHECK-LABEL: vgtnc_vvssvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s2, 128
; CHECK-NEXT:    lvl %s2
; CHECK-NEXT:    vgt.nc %v1, %v0, %s0, %s1
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v1
; CHECK-NEXT:    b.l.t (, %s10)
  %5 = tail call fast <256 x double> @llvm.ve.vl.vgtnc.vvssvl(<256 x double> %0, i64 %1, i64 %2, <256 x double> %3, i32 128)
  ret <256 x double> %5
}

; Function Attrs: nounwind readonly
declare <256 x double> @llvm.ve.vl.vgtnc.vvssvl(<256 x double>, i64, i64, <256 x double>, i32)

; Function Attrs: nounwind readonly
define fastcc <256 x double> @vgtnc_vvssl_imm_1(<256 x double> %0, i64 %1) {
; CHECK-LABEL: vgtnc_vvssl_imm_1:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vgt.nc %v0, %v0, %s0, 0
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.vgtnc.vvssl(<256 x double> %0, i64 %1, i64 0, i32 256)
  ret <256 x double> %3
}

; Function Attrs: nounwind readonly
define fastcc <256 x double> @vgtnc_vvssvl_imm_1(<256 x double> %0, i64 %1, <256 x double> %2) {
; CHECK-LABEL: vgtnc_vvssvl_imm_1:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 128
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vgt.nc %v1, %v0, %s0, 0
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v1
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = tail call fast <256 x double> @llvm.ve.vl.vgtnc.vvssvl(<256 x double> %0, i64 %1, i64 0, <256 x double> %2, i32 128)
  ret <256 x double> %4
}

; Function Attrs: nounwind readonly
define fastcc <256 x double> @vgtnc_vvssl_imm_2(<256 x double> %0, i64 %1) {
; CHECK-LABEL: vgtnc_vvssl_imm_2:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vgt.nc %v0, %v0, 8, %s0
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.vgtnc.vvssl(<256 x double> %0, i64 8, i64 %1, i32 256)
  ret <256 x double> %3
}

; Function Attrs: nounwind readonly
define fastcc <256 x double> @vgtnc_vvssvl_imm_2(<256 x double> %0, i64 %1, <256 x double> %2) {
; CHECK-LABEL: vgtnc_vvssvl_imm_2:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 128
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vgt.nc %v1, %v0, 8, %s0
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v1
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = tail call fast <256 x double> @llvm.ve.vl.vgtnc.vvssvl(<256 x double> %0, i64 8, i64 %1, <256 x double> %2, i32 128)
  ret <256 x double> %4
}

; Function Attrs: nounwind readonly
define fastcc <256 x double> @vgtnc_vvssl_imm_3(<256 x double> %0) {
; CHECK-LABEL: vgtnc_vvssl_imm_3:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vgt.nc %v0, %v0, 8, 0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call fast <256 x double> @llvm.ve.vl.vgtnc.vvssl(<256 x double> %0, i64 8, i64 0, i32 256)
  ret <256 x double> %2
}

; Function Attrs: nounwind readonly
define fastcc <256 x double> @vgtnc_vvssvl_imm_3(<256 x double> %0, <256 x double> %1) {
; CHECK-LABEL: vgtnc_vvssvl_imm_3:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 128
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vgt.nc %v1, %v0, 8, 0
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.vgtnc.vvssvl(<256 x double> %0, i64 8, i64 0, <256 x double> %1, i32 128)
  ret <256 x double> %3
}

; Function Attrs: nounwind readonly
define fastcc <256 x double> @vgtnc_vvssml(<256 x double> %0, i64 %1, i64 %2, <256 x i1> %3) {
; CHECK-LABEL: vgtnc_vvssml:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s2, 256
; CHECK-NEXT:    lvl %s2
; CHECK-NEXT:    vgt.nc %v0, %v0, %s0, %s1, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %5 = tail call fast <256 x double> @llvm.ve.vl.vgtnc.vvssml(<256 x double> %0, i64 %1, i64 %2, <256 x i1> %3, i32 256)
  ret <256 x double> %5
}

; Function Attrs: nounwind readonly
declare <256 x double> @llvm.ve.vl.vgtnc.vvssml(<256 x double>, i64, i64, <256 x i1>, i32)

; Function Attrs: nounwind readonly
define fastcc <256 x double> @vgtnc_vvssmvl(<256 x double> %0, i64 %1, i64 %2, <256 x i1> %3, <256 x double> %4) {
; CHECK-LABEL: vgtnc_vvssmvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s2, 128
; CHECK-NEXT:    lvl %s2
; CHECK-NEXT:    vgt.nc %v1, %v0, %s0, %s1, %vm1
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v1
; CHECK-NEXT:    b.l.t (, %s10)
  %6 = tail call fast <256 x double> @llvm.ve.vl.vgtnc.vvssmvl(<256 x double> %0, i64 %1, i64 %2, <256 x i1> %3, <256 x double> %4, i32 128)
  ret <256 x double> %6
}

; Function Attrs: nounwind readonly
declare <256 x double> @llvm.ve.vl.vgtnc.vvssmvl(<256 x double>, i64, i64, <256 x i1>, <256 x double>, i32)

; Function Attrs: nounwind readonly
define fastcc <256 x double> @vgtnc_vvssml_imm_1(<256 x double> %0, i64 %1, <256 x i1> %2) {
; CHECK-LABEL: vgtnc_vvssml_imm_1:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vgt.nc %v0, %v0, %s0, 0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = tail call fast <256 x double> @llvm.ve.vl.vgtnc.vvssml(<256 x double> %0, i64 %1, i64 0, <256 x i1> %2, i32 256)
  ret <256 x double> %4
}

; Function Attrs: nounwind readonly
define fastcc <256 x double> @vgtnc_vvssmvl_imm_1(<256 x double> %0, i64 %1, <256 x i1> %2, <256 x double> %3) {
; CHECK-LABEL: vgtnc_vvssmvl_imm_1:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 128
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vgt.nc %v1, %v0, %s0, 0, %vm1
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v1
; CHECK-NEXT:    b.l.t (, %s10)
  %5 = tail call fast <256 x double> @llvm.ve.vl.vgtnc.vvssmvl(<256 x double> %0, i64 %1, i64 0, <256 x i1> %2, <256 x double> %3, i32 128)
  ret <256 x double> %5
}

; Function Attrs: nounwind readonly
define fastcc <256 x double> @vgtnc_vvssml_imm_2(<256 x double> %0, i64 %1, <256 x i1> %2) {
; CHECK-LABEL: vgtnc_vvssml_imm_2:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vgt.nc %v0, %v0, 8, %s0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = tail call fast <256 x double> @llvm.ve.vl.vgtnc.vvssml(<256 x double> %0, i64 8, i64 %1, <256 x i1> %2, i32 256)
  ret <256 x double> %4
}

; Function Attrs: nounwind readonly
define fastcc <256 x double> @vgtnc_vvssmvl_imm_2(<256 x double> %0, i64 %1, <256 x i1> %2, <256 x double> %3) {
; CHECK-LABEL: vgtnc_vvssmvl_imm_2:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 128
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vgt.nc %v1, %v0, 8, %s0, %vm1
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v1
; CHECK-NEXT:    b.l.t (, %s10)
  %5 = tail call fast <256 x double> @llvm.ve.vl.vgtnc.vvssmvl(<256 x double> %0, i64 8, i64 %1, <256 x i1> %2, <256 x double> %3, i32 128)
  ret <256 x double> %5
}

; Function Attrs: nounwind readonly
define fastcc <256 x double> @vgtnc_vvssml_imm_3(<256 x double> %0, <256 x i1> %1) {
; CHECK-LABEL: vgtnc_vvssml_imm_3:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vgt.nc %v0, %v0, 8, 0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.vgtnc.vvssml(<256 x double> %0, i64 8, i64 0, <256 x i1> %1, i32 256)
  ret <256 x double> %3
}

; Function Attrs: nounwind readonly
define fastcc <256 x double> @vgtnc_vvssmvl_imm_3(<256 x double> %0, <256 x i1> %1, <256 x double> %2) {
; CHECK-LABEL: vgtnc_vvssmvl_imm_3:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 128
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vgt.nc %v1, %v0, 8, 0, %vm1
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v1
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = tail call fast <256 x double> @llvm.ve.vl.vgtnc.vvssmvl(<256 x double> %0, i64 8, i64 0, <256 x i1> %1, <256 x double> %2, i32 128)
  ret <256 x double> %4
}

; Function Attrs: nounwind readonly
define fastcc <256 x double> @vgtnc_vvssl_no_imm_1(<256 x double> %0, i64 %1) {
; CHECK-LABEL: vgtnc_vvssl_no_imm_1:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    or %s2, 8, (0)1
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vgt.nc %v0, %v0, %s0, %s2
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.vgtnc.vvssl(<256 x double> %0, i64 %1, i64 8, i32 256)
  ret <256 x double> %3
}

; Function Attrs: nounwind readonly
define fastcc <256 x double> @vgtu_vvssl(<256 x double> %0, i64 %1, i64 %2) {
; CHECK-LABEL: vgtu_vvssl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s2, 256
; CHECK-NEXT:    lvl %s2
; CHECK-NEXT:    vgtu %v0, %v0, %s0, %s1
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = tail call fast <256 x double> @llvm.ve.vl.vgtu.vvssl(<256 x double> %0, i64 %1, i64 %2, i32 256)
  ret <256 x double> %4
}

; Function Attrs: nounwind readonly
declare <256 x double> @llvm.ve.vl.vgtu.vvssl(<256 x double>, i64, i64, i32)

; Function Attrs: nounwind readonly
define fastcc <256 x double> @vgtu_vvssvl(<256 x double> %0, i64 %1, i64 %2, <256 x double> %3) {
; CHECK-LABEL: vgtu_vvssvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s2, 128
; CHECK-NEXT:    lvl %s2
; CHECK-NEXT:    vgtu %v1, %v0, %s0, %s1
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v1
; CHECK-NEXT:    b.l.t (, %s10)
  %5 = tail call fast <256 x double> @llvm.ve.vl.vgtu.vvssvl(<256 x double> %0, i64 %1, i64 %2, <256 x double> %3, i32 128)
  ret <256 x double> %5
}

; Function Attrs: nounwind readonly
declare <256 x double> @llvm.ve.vl.vgtu.vvssvl(<256 x double>, i64, i64, <256 x double>, i32)

; Function Attrs: nounwind readonly
define fastcc <256 x double> @vgtu_vvssl_imm_1(<256 x double> %0, i64 %1) {
; CHECK-LABEL: vgtu_vvssl_imm_1:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vgtu %v0, %v0, %s0, 0
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.vgtu.vvssl(<256 x double> %0, i64 %1, i64 0, i32 256)
  ret <256 x double> %3
}

; Function Attrs: nounwind readonly
define fastcc <256 x double> @vgtu_vvssvl_imm_1(<256 x double> %0, i64 %1, <256 x double> %2) {
; CHECK-LABEL: vgtu_vvssvl_imm_1:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 128
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vgtu %v1, %v0, %s0, 0
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v1
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = tail call fast <256 x double> @llvm.ve.vl.vgtu.vvssvl(<256 x double> %0, i64 %1, i64 0, <256 x double> %2, i32 128)
  ret <256 x double> %4
}

; Function Attrs: nounwind readonly
define fastcc <256 x double> @vgtu_vvssl_imm_2(<256 x double> %0, i64 %1) {
; CHECK-LABEL: vgtu_vvssl_imm_2:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vgtu %v0, %v0, 8, %s0
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.vgtu.vvssl(<256 x double> %0, i64 8, i64 %1, i32 256)
  ret <256 x double> %3
}

; Function Attrs: nounwind readonly
define fastcc <256 x double> @vgtu_vvssvl_imm_2(<256 x double> %0, i64 %1, <256 x double> %2) {
; CHECK-LABEL: vgtu_vvssvl_imm_2:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 128
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vgtu %v1, %v0, 8, %s0
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v1
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = tail call fast <256 x double> @llvm.ve.vl.vgtu.vvssvl(<256 x double> %0, i64 8, i64 %1, <256 x double> %2, i32 128)
  ret <256 x double> %4
}

; Function Attrs: nounwind readonly
define fastcc <256 x double> @vgtu_vvssl_imm_3(<256 x double> %0) {
; CHECK-LABEL: vgtu_vvssl_imm_3:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vgtu %v0, %v0, 8, 0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call fast <256 x double> @llvm.ve.vl.vgtu.vvssl(<256 x double> %0, i64 8, i64 0, i32 256)
  ret <256 x double> %2
}

; Function Attrs: nounwind readonly
define fastcc <256 x double> @vgtu_vvssvl_imm_3(<256 x double> %0, <256 x double> %1) {
; CHECK-LABEL: vgtu_vvssvl_imm_3:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 128
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vgtu %v1, %v0, 8, 0
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.vgtu.vvssvl(<256 x double> %0, i64 8, i64 0, <256 x double> %1, i32 128)
  ret <256 x double> %3
}

; Function Attrs: nounwind readonly
define fastcc <256 x double> @vgtu_vvssml(<256 x double> %0, i64 %1, i64 %2, <256 x i1> %3) {
; CHECK-LABEL: vgtu_vvssml:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s2, 256
; CHECK-NEXT:    lvl %s2
; CHECK-NEXT:    vgtu %v0, %v0, %s0, %s1, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %5 = tail call fast <256 x double> @llvm.ve.vl.vgtu.vvssml(<256 x double> %0, i64 %1, i64 %2, <256 x i1> %3, i32 256)
  ret <256 x double> %5
}

; Function Attrs: nounwind readonly
declare <256 x double> @llvm.ve.vl.vgtu.vvssml(<256 x double>, i64, i64, <256 x i1>, i32)

; Function Attrs: nounwind readonly
define fastcc <256 x double> @vgtu_vvssmvl(<256 x double> %0, i64 %1, i64 %2, <256 x i1> %3, <256 x double> %4) {
; CHECK-LABEL: vgtu_vvssmvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s2, 128
; CHECK-NEXT:    lvl %s2
; CHECK-NEXT:    vgtu %v1, %v0, %s0, %s1, %vm1
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v1
; CHECK-NEXT:    b.l.t (, %s10)
  %6 = tail call fast <256 x double> @llvm.ve.vl.vgtu.vvssmvl(<256 x double> %0, i64 %1, i64 %2, <256 x i1> %3, <256 x double> %4, i32 128)
  ret <256 x double> %6
}

; Function Attrs: nounwind readonly
declare <256 x double> @llvm.ve.vl.vgtu.vvssmvl(<256 x double>, i64, i64, <256 x i1>, <256 x double>, i32)

; Function Attrs: nounwind readonly
define fastcc <256 x double> @vgtu_vvssml_imm_1(<256 x double> %0, i64 %1, <256 x i1> %2) {
; CHECK-LABEL: vgtu_vvssml_imm_1:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vgtu %v0, %v0, %s0, 0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = tail call fast <256 x double> @llvm.ve.vl.vgtu.vvssml(<256 x double> %0, i64 %1, i64 0, <256 x i1> %2, i32 256)
  ret <256 x double> %4
}

; Function Attrs: nounwind readonly
define fastcc <256 x double> @vgtu_vvssmvl_imm_1(<256 x double> %0, i64 %1, <256 x i1> %2, <256 x double> %3) {
; CHECK-LABEL: vgtu_vvssmvl_imm_1:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 128
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vgtu %v1, %v0, %s0, 0, %vm1
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v1
; CHECK-NEXT:    b.l.t (, %s10)
  %5 = tail call fast <256 x double> @llvm.ve.vl.vgtu.vvssmvl(<256 x double> %0, i64 %1, i64 0, <256 x i1> %2, <256 x double> %3, i32 128)
  ret <256 x double> %5
}

; Function Attrs: nounwind readonly
define fastcc <256 x double> @vgtu_vvssml_imm_2(<256 x double> %0, i64 %1, <256 x i1> %2) {
; CHECK-LABEL: vgtu_vvssml_imm_2:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vgtu %v0, %v0, 8, %s0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = tail call fast <256 x double> @llvm.ve.vl.vgtu.vvssml(<256 x double> %0, i64 8, i64 %1, <256 x i1> %2, i32 256)
  ret <256 x double> %4
}

; Function Attrs: nounwind readonly
define fastcc <256 x double> @vgtu_vvssmvl_imm_2(<256 x double> %0, i64 %1, <256 x i1> %2, <256 x double> %3) {
; CHECK-LABEL: vgtu_vvssmvl_imm_2:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 128
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vgtu %v1, %v0, 8, %s0, %vm1
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v1
; CHECK-NEXT:    b.l.t (, %s10)
  %5 = tail call fast <256 x double> @llvm.ve.vl.vgtu.vvssmvl(<256 x double> %0, i64 8, i64 %1, <256 x i1> %2, <256 x double> %3, i32 128)
  ret <256 x double> %5
}

; Function Attrs: nounwind readonly
define fastcc <256 x double> @vgtu_vvssml_imm_3(<256 x double> %0, <256 x i1> %1) {
; CHECK-LABEL: vgtu_vvssml_imm_3:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vgtu %v0, %v0, 8, 0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.vgtu.vvssml(<256 x double> %0, i64 8, i64 0, <256 x i1> %1, i32 256)
  ret <256 x double> %3
}

; Function Attrs: nounwind readonly
define fastcc <256 x double> @vgtu_vvssmvl_imm_3(<256 x double> %0, <256 x i1> %1, <256 x double> %2) {
; CHECK-LABEL: vgtu_vvssmvl_imm_3:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 128
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vgtu %v1, %v0, 8, 0, %vm1
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v1
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = tail call fast <256 x double> @llvm.ve.vl.vgtu.vvssmvl(<256 x double> %0, i64 8, i64 0, <256 x i1> %1, <256 x double> %2, i32 128)
  ret <256 x double> %4
}

; Function Attrs: nounwind readonly
define fastcc <256 x double> @vgtu_vvssl_no_imm_1(<256 x double> %0, i64 %1) {
; CHECK-LABEL: vgtu_vvssl_no_imm_1:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    or %s2, 8, (0)1
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vgtu %v0, %v0, %s0, %s2
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.vgtu.vvssl(<256 x double> %0, i64 %1, i64 8, i32 256)
  ret <256 x double> %3
}

; Function Attrs: nounwind readonly
define fastcc <256 x double> @vgtunc_vvssl(<256 x double> %0, i64 %1, i64 %2) {
; CHECK-LABEL: vgtunc_vvssl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s2, 256
; CHECK-NEXT:    lvl %s2
; CHECK-NEXT:    vgtu.nc %v0, %v0, %s0, %s1
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = tail call fast <256 x double> @llvm.ve.vl.vgtunc.vvssl(<256 x double> %0, i64 %1, i64 %2, i32 256)
  ret <256 x double> %4
}

; Function Attrs: nounwind readonly
declare <256 x double> @llvm.ve.vl.vgtunc.vvssl(<256 x double>, i64, i64, i32)

; Function Attrs: nounwind readonly
define fastcc <256 x double> @vgtunc_vvssvl(<256 x double> %0, i64 %1, i64 %2, <256 x double> %3) {
; CHECK-LABEL: vgtunc_vvssvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s2, 128
; CHECK-NEXT:    lvl %s2
; CHECK-NEXT:    vgtu.nc %v1, %v0, %s0, %s1
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v1
; CHECK-NEXT:    b.l.t (, %s10)
  %5 = tail call fast <256 x double> @llvm.ve.vl.vgtunc.vvssvl(<256 x double> %0, i64 %1, i64 %2, <256 x double> %3, i32 128)
  ret <256 x double> %5
}

; Function Attrs: nounwind readonly
declare <256 x double> @llvm.ve.vl.vgtunc.vvssvl(<256 x double>, i64, i64, <256 x double>, i32)

; Function Attrs: nounwind readonly
define fastcc <256 x double> @vgtunc_vvssl_imm_1(<256 x double> %0, i64 %1) {
; CHECK-LABEL: vgtunc_vvssl_imm_1:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vgtu.nc %v0, %v0, %s0, 0
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.vgtunc.vvssl(<256 x double> %0, i64 %1, i64 0, i32 256)
  ret <256 x double> %3
}

; Function Attrs: nounwind readonly
define fastcc <256 x double> @vgtunc_vvssvl_imm_1(<256 x double> %0, i64 %1, <256 x double> %2) {
; CHECK-LABEL: vgtunc_vvssvl_imm_1:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 128
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vgtu.nc %v1, %v0, %s0, 0
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v1
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = tail call fast <256 x double> @llvm.ve.vl.vgtunc.vvssvl(<256 x double> %0, i64 %1, i64 0, <256 x double> %2, i32 128)
  ret <256 x double> %4
}

; Function Attrs: nounwind readonly
define fastcc <256 x double> @vgtunc_vvssl_imm_2(<256 x double> %0, i64 %1) {
; CHECK-LABEL: vgtunc_vvssl_imm_2:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vgtu.nc %v0, %v0, 8, %s0
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.vgtunc.vvssl(<256 x double> %0, i64 8, i64 %1, i32 256)
  ret <256 x double> %3
}

; Function Attrs: nounwind readonly
define fastcc <256 x double> @vgtunc_vvssvl_imm_2(<256 x double> %0, i64 %1, <256 x double> %2) {
; CHECK-LABEL: vgtunc_vvssvl_imm_2:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 128
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vgtu.nc %v1, %v0, 8, %s0
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v1
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = tail call fast <256 x double> @llvm.ve.vl.vgtunc.vvssvl(<256 x double> %0, i64 8, i64 %1, <256 x double> %2, i32 128)
  ret <256 x double> %4
}

; Function Attrs: nounwind readonly
define fastcc <256 x double> @vgtunc_vvssl_imm_3(<256 x double> %0) {
; CHECK-LABEL: vgtunc_vvssl_imm_3:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vgtu.nc %v0, %v0, 8, 0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call fast <256 x double> @llvm.ve.vl.vgtunc.vvssl(<256 x double> %0, i64 8, i64 0, i32 256)
  ret <256 x double> %2
}

; Function Attrs: nounwind readonly
define fastcc <256 x double> @vgtunc_vvssvl_imm_3(<256 x double> %0, <256 x double> %1) {
; CHECK-LABEL: vgtunc_vvssvl_imm_3:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 128
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vgtu.nc %v1, %v0, 8, 0
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.vgtunc.vvssvl(<256 x double> %0, i64 8, i64 0, <256 x double> %1, i32 128)
  ret <256 x double> %3
}

; Function Attrs: nounwind readonly
define fastcc <256 x double> @vgtunc_vvssml(<256 x double> %0, i64 %1, i64 %2, <256 x i1> %3) {
; CHECK-LABEL: vgtunc_vvssml:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s2, 256
; CHECK-NEXT:    lvl %s2
; CHECK-NEXT:    vgtu.nc %v0, %v0, %s0, %s1, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %5 = tail call fast <256 x double> @llvm.ve.vl.vgtunc.vvssml(<256 x double> %0, i64 %1, i64 %2, <256 x i1> %3, i32 256)
  ret <256 x double> %5
}

; Function Attrs: nounwind readonly
declare <256 x double> @llvm.ve.vl.vgtunc.vvssml(<256 x double>, i64, i64, <256 x i1>, i32)

; Function Attrs: nounwind readonly
define fastcc <256 x double> @vgtunc_vvssmvl(<256 x double> %0, i64 %1, i64 %2, <256 x i1> %3, <256 x double> %4) {
; CHECK-LABEL: vgtunc_vvssmvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s2, 128
; CHECK-NEXT:    lvl %s2
; CHECK-NEXT:    vgtu.nc %v1, %v0, %s0, %s1, %vm1
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v1
; CHECK-NEXT:    b.l.t (, %s10)
  %6 = tail call fast <256 x double> @llvm.ve.vl.vgtunc.vvssmvl(<256 x double> %0, i64 %1, i64 %2, <256 x i1> %3, <256 x double> %4, i32 128)
  ret <256 x double> %6
}

; Function Attrs: nounwind readonly
declare <256 x double> @llvm.ve.vl.vgtunc.vvssmvl(<256 x double>, i64, i64, <256 x i1>, <256 x double>, i32)

; Function Attrs: nounwind readonly
define fastcc <256 x double> @vgtunc_vvssml_imm_1(<256 x double> %0, i64 %1, <256 x i1> %2) {
; CHECK-LABEL: vgtunc_vvssml_imm_1:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vgtu.nc %v0, %v0, %s0, 0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = tail call fast <256 x double> @llvm.ve.vl.vgtunc.vvssml(<256 x double> %0, i64 %1, i64 0, <256 x i1> %2, i32 256)
  ret <256 x double> %4
}

; Function Attrs: nounwind readonly
define fastcc <256 x double> @vgtunc_vvssmvl_imm_1(<256 x double> %0, i64 %1, <256 x i1> %2, <256 x double> %3) {
; CHECK-LABEL: vgtunc_vvssmvl_imm_1:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 128
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vgtu.nc %v1, %v0, %s0, 0, %vm1
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v1
; CHECK-NEXT:    b.l.t (, %s10)
  %5 = tail call fast <256 x double> @llvm.ve.vl.vgtunc.vvssmvl(<256 x double> %0, i64 %1, i64 0, <256 x i1> %2, <256 x double> %3, i32 128)
  ret <256 x double> %5
}

; Function Attrs: nounwind readonly
define fastcc <256 x double> @vgtunc_vvssml_imm_2(<256 x double> %0, i64 %1, <256 x i1> %2) {
; CHECK-LABEL: vgtunc_vvssml_imm_2:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vgtu.nc %v0, %v0, 8, %s0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = tail call fast <256 x double> @llvm.ve.vl.vgtunc.vvssml(<256 x double> %0, i64 8, i64 %1, <256 x i1> %2, i32 256)
  ret <256 x double> %4
}

; Function Attrs: nounwind readonly
define fastcc <256 x double> @vgtunc_vvssmvl_imm_2(<256 x double> %0, i64 %1, <256 x i1> %2, <256 x double> %3) {
; CHECK-LABEL: vgtunc_vvssmvl_imm_2:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 128
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vgtu.nc %v1, %v0, 8, %s0, %vm1
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v1
; CHECK-NEXT:    b.l.t (, %s10)
  %5 = tail call fast <256 x double> @llvm.ve.vl.vgtunc.vvssmvl(<256 x double> %0, i64 8, i64 %1, <256 x i1> %2, <256 x double> %3, i32 128)
  ret <256 x double> %5
}

; Function Attrs: nounwind readonly
define fastcc <256 x double> @vgtunc_vvssml_imm_3(<256 x double> %0, <256 x i1> %1) {
; CHECK-LABEL: vgtunc_vvssml_imm_3:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vgtu.nc %v0, %v0, 8, 0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.vgtunc.vvssml(<256 x double> %0, i64 8, i64 0, <256 x i1> %1, i32 256)
  ret <256 x double> %3
}

; Function Attrs: nounwind readonly
define fastcc <256 x double> @vgtunc_vvssmvl_imm_3(<256 x double> %0, <256 x i1> %1, <256 x double> %2) {
; CHECK-LABEL: vgtunc_vvssmvl_imm_3:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 128
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vgtu.nc %v1, %v0, 8, 0, %vm1
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v1
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = tail call fast <256 x double> @llvm.ve.vl.vgtunc.vvssmvl(<256 x double> %0, i64 8, i64 0, <256 x i1> %1, <256 x double> %2, i32 128)
  ret <256 x double> %4
}

; Function Attrs: nounwind readonly
define fastcc <256 x double> @vgtunc_vvssl_no_imm_1(<256 x double> %0, i64 %1) {
; CHECK-LABEL: vgtunc_vvssl_no_imm_1:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    or %s2, 8, (0)1
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vgtu.nc %v0, %v0, %s0, %s2
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.vgtunc.vvssl(<256 x double> %0, i64 %1, i64 8, i32 256)
  ret <256 x double> %3
}

; Function Attrs: nounwind readonly
define fastcc <256 x double> @vgtlsx_vvssl(<256 x double> %0, i64 %1, i64 %2) {
; CHECK-LABEL: vgtlsx_vvssl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s2, 256
; CHECK-NEXT:    lvl %s2
; CHECK-NEXT:    vgtl.sx %v0, %v0, %s0, %s1
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = tail call fast <256 x double> @llvm.ve.vl.vgtlsx.vvssl(<256 x double> %0, i64 %1, i64 %2, i32 256)
  ret <256 x double> %4
}

; Function Attrs: nounwind readonly
declare <256 x double> @llvm.ve.vl.vgtlsx.vvssl(<256 x double>, i64, i64, i32)

; Function Attrs: nounwind readonly
define fastcc <256 x double> @vgtlsx_vvssvl(<256 x double> %0, i64 %1, i64 %2, <256 x double> %3) {
; CHECK-LABEL: vgtlsx_vvssvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s2, 128
; CHECK-NEXT:    lvl %s2
; CHECK-NEXT:    vgtl.sx %v1, %v0, %s0, %s1
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v1
; CHECK-NEXT:    b.l.t (, %s10)
  %5 = tail call fast <256 x double> @llvm.ve.vl.vgtlsx.vvssvl(<256 x double> %0, i64 %1, i64 %2, <256 x double> %3, i32 128)
  ret <256 x double> %5
}

; Function Attrs: nounwind readonly
declare <256 x double> @llvm.ve.vl.vgtlsx.vvssvl(<256 x double>, i64, i64, <256 x double>, i32)

; Function Attrs: nounwind readonly
define fastcc <256 x double> @vgtlsx_vvssl_imm_1(<256 x double> %0, i64 %1) {
; CHECK-LABEL: vgtlsx_vvssl_imm_1:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vgtl.sx %v0, %v0, %s0, 0
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.vgtlsx.vvssl(<256 x double> %0, i64 %1, i64 0, i32 256)
  ret <256 x double> %3
}

; Function Attrs: nounwind readonly
define fastcc <256 x double> @vgtlsx_vvssvl_imm_1(<256 x double> %0, i64 %1, <256 x double> %2) {
; CHECK-LABEL: vgtlsx_vvssvl_imm_1:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 128
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vgtl.sx %v1, %v0, %s0, 0
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v1
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = tail call fast <256 x double> @llvm.ve.vl.vgtlsx.vvssvl(<256 x double> %0, i64 %1, i64 0, <256 x double> %2, i32 128)
  ret <256 x double> %4
}

; Function Attrs: nounwind readonly
define fastcc <256 x double> @vgtlsx_vvssl_imm_2(<256 x double> %0, i64 %1) {
; CHECK-LABEL: vgtlsx_vvssl_imm_2:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vgtl.sx %v0, %v0, 8, %s0
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.vgtlsx.vvssl(<256 x double> %0, i64 8, i64 %1, i32 256)
  ret <256 x double> %3
}

; Function Attrs: nounwind readonly
define fastcc <256 x double> @vgtlsx_vvssvl_imm_2(<256 x double> %0, i64 %1, <256 x double> %2) {
; CHECK-LABEL: vgtlsx_vvssvl_imm_2:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 128
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vgtl.sx %v1, %v0, 8, %s0
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v1
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = tail call fast <256 x double> @llvm.ve.vl.vgtlsx.vvssvl(<256 x double> %0, i64 8, i64 %1, <256 x double> %2, i32 128)
  ret <256 x double> %4
}

; Function Attrs: nounwind readonly
define fastcc <256 x double> @vgtlsx_vvssl_imm_3(<256 x double> %0) {
; CHECK-LABEL: vgtlsx_vvssl_imm_3:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vgtl.sx %v0, %v0, 8, 0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call fast <256 x double> @llvm.ve.vl.vgtlsx.vvssl(<256 x double> %0, i64 8, i64 0, i32 256)
  ret <256 x double> %2
}

; Function Attrs: nounwind readonly
define fastcc <256 x double> @vgtlsx_vvssvl_imm_3(<256 x double> %0, <256 x double> %1) {
; CHECK-LABEL: vgtlsx_vvssvl_imm_3:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 128
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vgtl.sx %v1, %v0, 8, 0
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.vgtlsx.vvssvl(<256 x double> %0, i64 8, i64 0, <256 x double> %1, i32 128)
  ret <256 x double> %3
}

; Function Attrs: nounwind readonly
define fastcc <256 x double> @vgtlsx_vvssml(<256 x double> %0, i64 %1, i64 %2, <256 x i1> %3) {
; CHECK-LABEL: vgtlsx_vvssml:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s2, 256
; CHECK-NEXT:    lvl %s2
; CHECK-NEXT:    vgtl.sx %v0, %v0, %s0, %s1, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %5 = tail call fast <256 x double> @llvm.ve.vl.vgtlsx.vvssml(<256 x double> %0, i64 %1, i64 %2, <256 x i1> %3, i32 256)
  ret <256 x double> %5
}

; Function Attrs: nounwind readonly
declare <256 x double> @llvm.ve.vl.vgtlsx.vvssml(<256 x double>, i64, i64, <256 x i1>, i32)

; Function Attrs: nounwind readonly
define fastcc <256 x double> @vgtlsx_vvssmvl(<256 x double> %0, i64 %1, i64 %2, <256 x i1> %3, <256 x double> %4) {
; CHECK-LABEL: vgtlsx_vvssmvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s2, 128
; CHECK-NEXT:    lvl %s2
; CHECK-NEXT:    vgtl.sx %v1, %v0, %s0, %s1, %vm1
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v1
; CHECK-NEXT:    b.l.t (, %s10)
  %6 = tail call fast <256 x double> @llvm.ve.vl.vgtlsx.vvssmvl(<256 x double> %0, i64 %1, i64 %2, <256 x i1> %3, <256 x double> %4, i32 128)
  ret <256 x double> %6
}

; Function Attrs: nounwind readonly
declare <256 x double> @llvm.ve.vl.vgtlsx.vvssmvl(<256 x double>, i64, i64, <256 x i1>, <256 x double>, i32)

; Function Attrs: nounwind readonly
define fastcc <256 x double> @vgtlsx_vvssml_imm_1(<256 x double> %0, i64 %1, <256 x i1> %2) {
; CHECK-LABEL: vgtlsx_vvssml_imm_1:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vgtl.sx %v0, %v0, %s0, 0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = tail call fast <256 x double> @llvm.ve.vl.vgtlsx.vvssml(<256 x double> %0, i64 %1, i64 0, <256 x i1> %2, i32 256)
  ret <256 x double> %4
}

; Function Attrs: nounwind readonly
define fastcc <256 x double> @vgtlsx_vvssmvl_imm_1(<256 x double> %0, i64 %1, <256 x i1> %2, <256 x double> %3) {
; CHECK-LABEL: vgtlsx_vvssmvl_imm_1:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 128
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vgtl.sx %v1, %v0, %s0, 0, %vm1
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v1
; CHECK-NEXT:    b.l.t (, %s10)
  %5 = tail call fast <256 x double> @llvm.ve.vl.vgtlsx.vvssmvl(<256 x double> %0, i64 %1, i64 0, <256 x i1> %2, <256 x double> %3, i32 128)
  ret <256 x double> %5
}

; Function Attrs: nounwind readonly
define fastcc <256 x double> @vgtlsx_vvssml_imm_2(<256 x double> %0, i64 %1, <256 x i1> %2) {
; CHECK-LABEL: vgtlsx_vvssml_imm_2:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vgtl.sx %v0, %v0, 8, %s0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = tail call fast <256 x double> @llvm.ve.vl.vgtlsx.vvssml(<256 x double> %0, i64 8, i64 %1, <256 x i1> %2, i32 256)
  ret <256 x double> %4
}

; Function Attrs: nounwind readonly
define fastcc <256 x double> @vgtlsx_vvssmvl_imm_2(<256 x double> %0, i64 %1, <256 x i1> %2, <256 x double> %3) {
; CHECK-LABEL: vgtlsx_vvssmvl_imm_2:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 128
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vgtl.sx %v1, %v0, 8, %s0, %vm1
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v1
; CHECK-NEXT:    b.l.t (, %s10)
  %5 = tail call fast <256 x double> @llvm.ve.vl.vgtlsx.vvssmvl(<256 x double> %0, i64 8, i64 %1, <256 x i1> %2, <256 x double> %3, i32 128)
  ret <256 x double> %5
}

; Function Attrs: nounwind readonly
define fastcc <256 x double> @vgtlsx_vvssml_imm_3(<256 x double> %0, <256 x i1> %1) {
; CHECK-LABEL: vgtlsx_vvssml_imm_3:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vgtl.sx %v0, %v0, 8, 0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.vgtlsx.vvssml(<256 x double> %0, i64 8, i64 0, <256 x i1> %1, i32 256)
  ret <256 x double> %3
}

; Function Attrs: nounwind readonly
define fastcc <256 x double> @vgtlsx_vvssmvl_imm_3(<256 x double> %0, <256 x i1> %1, <256 x double> %2) {
; CHECK-LABEL: vgtlsx_vvssmvl_imm_3:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 128
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vgtl.sx %v1, %v0, 8, 0, %vm1
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v1
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = tail call fast <256 x double> @llvm.ve.vl.vgtlsx.vvssmvl(<256 x double> %0, i64 8, i64 0, <256 x i1> %1, <256 x double> %2, i32 128)
  ret <256 x double> %4
}

; Function Attrs: nounwind readonly
define fastcc <256 x double> @vgtlsx_vvssl_no_imm_1(<256 x double> %0, i64 %1) {
; CHECK-LABEL: vgtlsx_vvssl_no_imm_1:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    or %s2, 8, (0)1
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vgtl.sx %v0, %v0, %s0, %s2
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.vgtlsx.vvssl(<256 x double> %0, i64 %1, i64 8, i32 256)
  ret <256 x double> %3
}

; Function Attrs: nounwind readonly
define fastcc <256 x double> @vgtlsxnc_vvssl(<256 x double> %0, i64 %1, i64 %2) {
; CHECK-LABEL: vgtlsxnc_vvssl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s2, 256
; CHECK-NEXT:    lvl %s2
; CHECK-NEXT:    vgtl.sx.nc %v0, %v0, %s0, %s1
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = tail call fast <256 x double> @llvm.ve.vl.vgtlsxnc.vvssl(<256 x double> %0, i64 %1, i64 %2, i32 256)
  ret <256 x double> %4
}

; Function Attrs: nounwind readonly
declare <256 x double> @llvm.ve.vl.vgtlsxnc.vvssl(<256 x double>, i64, i64, i32)

; Function Attrs: nounwind readonly
define fastcc <256 x double> @vgtlsxnc_vvssvl(<256 x double> %0, i64 %1, i64 %2, <256 x double> %3) {
; CHECK-LABEL: vgtlsxnc_vvssvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s2, 128
; CHECK-NEXT:    lvl %s2
; CHECK-NEXT:    vgtl.sx.nc %v1, %v0, %s0, %s1
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v1
; CHECK-NEXT:    b.l.t (, %s10)
  %5 = tail call fast <256 x double> @llvm.ve.vl.vgtlsxnc.vvssvl(<256 x double> %0, i64 %1, i64 %2, <256 x double> %3, i32 128)
  ret <256 x double> %5
}

; Function Attrs: nounwind readonly
declare <256 x double> @llvm.ve.vl.vgtlsxnc.vvssvl(<256 x double>, i64, i64, <256 x double>, i32)

; Function Attrs: nounwind readonly
define fastcc <256 x double> @vgtlsxnc_vvssl_imm_1(<256 x double> %0, i64 %1) {
; CHECK-LABEL: vgtlsxnc_vvssl_imm_1:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vgtl.sx.nc %v0, %v0, %s0, 0
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.vgtlsxnc.vvssl(<256 x double> %0, i64 %1, i64 0, i32 256)
  ret <256 x double> %3
}

; Function Attrs: nounwind readonly
define fastcc <256 x double> @vgtlsxnc_vvssvl_imm_1(<256 x double> %0, i64 %1, <256 x double> %2) {
; CHECK-LABEL: vgtlsxnc_vvssvl_imm_1:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 128
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vgtl.sx.nc %v1, %v0, %s0, 0
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v1
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = tail call fast <256 x double> @llvm.ve.vl.vgtlsxnc.vvssvl(<256 x double> %0, i64 %1, i64 0, <256 x double> %2, i32 128)
  ret <256 x double> %4
}

; Function Attrs: nounwind readonly
define fastcc <256 x double> @vgtlsxnc_vvssl_imm_2(<256 x double> %0, i64 %1) {
; CHECK-LABEL: vgtlsxnc_vvssl_imm_2:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vgtl.sx.nc %v0, %v0, 8, %s0
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.vgtlsxnc.vvssl(<256 x double> %0, i64 8, i64 %1, i32 256)
  ret <256 x double> %3
}

; Function Attrs: nounwind readonly
define fastcc <256 x double> @vgtlsxnc_vvssvl_imm_2(<256 x double> %0, i64 %1, <256 x double> %2) {
; CHECK-LABEL: vgtlsxnc_vvssvl_imm_2:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 128
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vgtl.sx.nc %v1, %v0, 8, %s0
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v1
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = tail call fast <256 x double> @llvm.ve.vl.vgtlsxnc.vvssvl(<256 x double> %0, i64 8, i64 %1, <256 x double> %2, i32 128)
  ret <256 x double> %4
}

; Function Attrs: nounwind readonly
define fastcc <256 x double> @vgtlsxnc_vvssl_imm_3(<256 x double> %0) {
; CHECK-LABEL: vgtlsxnc_vvssl_imm_3:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vgtl.sx.nc %v0, %v0, 8, 0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call fast <256 x double> @llvm.ve.vl.vgtlsxnc.vvssl(<256 x double> %0, i64 8, i64 0, i32 256)
  ret <256 x double> %2
}

; Function Attrs: nounwind readonly
define fastcc <256 x double> @vgtlsxnc_vvssvl_imm_3(<256 x double> %0, <256 x double> %1) {
; CHECK-LABEL: vgtlsxnc_vvssvl_imm_3:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 128
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vgtl.sx.nc %v1, %v0, 8, 0
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.vgtlsxnc.vvssvl(<256 x double> %0, i64 8, i64 0, <256 x double> %1, i32 128)
  ret <256 x double> %3
}

; Function Attrs: nounwind readonly
define fastcc <256 x double> @vgtlsxnc_vvssml(<256 x double> %0, i64 %1, i64 %2, <256 x i1> %3) {
; CHECK-LABEL: vgtlsxnc_vvssml:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s2, 256
; CHECK-NEXT:    lvl %s2
; CHECK-NEXT:    vgtl.sx.nc %v0, %v0, %s0, %s1, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %5 = tail call fast <256 x double> @llvm.ve.vl.vgtlsxnc.vvssml(<256 x double> %0, i64 %1, i64 %2, <256 x i1> %3, i32 256)
  ret <256 x double> %5
}

; Function Attrs: nounwind readonly
declare <256 x double> @llvm.ve.vl.vgtlsxnc.vvssml(<256 x double>, i64, i64, <256 x i1>, i32)

; Function Attrs: nounwind readonly
define fastcc <256 x double> @vgtlsxnc_vvssmvl(<256 x double> %0, i64 %1, i64 %2, <256 x i1> %3, <256 x double> %4) {
; CHECK-LABEL: vgtlsxnc_vvssmvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s2, 128
; CHECK-NEXT:    lvl %s2
; CHECK-NEXT:    vgtl.sx.nc %v1, %v0, %s0, %s1, %vm1
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v1
; CHECK-NEXT:    b.l.t (, %s10)
  %6 = tail call fast <256 x double> @llvm.ve.vl.vgtlsxnc.vvssmvl(<256 x double> %0, i64 %1, i64 %2, <256 x i1> %3, <256 x double> %4, i32 128)
  ret <256 x double> %6
}

; Function Attrs: nounwind readonly
declare <256 x double> @llvm.ve.vl.vgtlsxnc.vvssmvl(<256 x double>, i64, i64, <256 x i1>, <256 x double>, i32)

; Function Attrs: nounwind readonly
define fastcc <256 x double> @vgtlsxnc_vvssml_imm_1(<256 x double> %0, i64 %1, <256 x i1> %2) {
; CHECK-LABEL: vgtlsxnc_vvssml_imm_1:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vgtl.sx.nc %v0, %v0, %s0, 0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = tail call fast <256 x double> @llvm.ve.vl.vgtlsxnc.vvssml(<256 x double> %0, i64 %1, i64 0, <256 x i1> %2, i32 256)
  ret <256 x double> %4
}

; Function Attrs: nounwind readonly
define fastcc <256 x double> @vgtlsxnc_vvssmvl_imm_1(<256 x double> %0, i64 %1, <256 x i1> %2, <256 x double> %3) {
; CHECK-LABEL: vgtlsxnc_vvssmvl_imm_1:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 128
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vgtl.sx.nc %v1, %v0, %s0, 0, %vm1
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v1
; CHECK-NEXT:    b.l.t (, %s10)
  %5 = tail call fast <256 x double> @llvm.ve.vl.vgtlsxnc.vvssmvl(<256 x double> %0, i64 %1, i64 0, <256 x i1> %2, <256 x double> %3, i32 128)
  ret <256 x double> %5
}

; Function Attrs: nounwind readonly
define fastcc <256 x double> @vgtlsxnc_vvssml_imm_2(<256 x double> %0, i64 %1, <256 x i1> %2) {
; CHECK-LABEL: vgtlsxnc_vvssml_imm_2:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vgtl.sx.nc %v0, %v0, 8, %s0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = tail call fast <256 x double> @llvm.ve.vl.vgtlsxnc.vvssml(<256 x double> %0, i64 8, i64 %1, <256 x i1> %2, i32 256)
  ret <256 x double> %4
}

; Function Attrs: nounwind readonly
define fastcc <256 x double> @vgtlsxnc_vvssmvl_imm_2(<256 x double> %0, i64 %1, <256 x i1> %2, <256 x double> %3) {
; CHECK-LABEL: vgtlsxnc_vvssmvl_imm_2:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 128
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vgtl.sx.nc %v1, %v0, 8, %s0, %vm1
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v1
; CHECK-NEXT:    b.l.t (, %s10)
  %5 = tail call fast <256 x double> @llvm.ve.vl.vgtlsxnc.vvssmvl(<256 x double> %0, i64 8, i64 %1, <256 x i1> %2, <256 x double> %3, i32 128)
  ret <256 x double> %5
}

; Function Attrs: nounwind readonly
define fastcc <256 x double> @vgtlsxnc_vvssml_imm_3(<256 x double> %0, <256 x i1> %1) {
; CHECK-LABEL: vgtlsxnc_vvssml_imm_3:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vgtl.sx.nc %v0, %v0, 8, 0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.vgtlsxnc.vvssml(<256 x double> %0, i64 8, i64 0, <256 x i1> %1, i32 256)
  ret <256 x double> %3
}

; Function Attrs: nounwind readonly
define fastcc <256 x double> @vgtlsxnc_vvssmvl_imm_3(<256 x double> %0, <256 x i1> %1, <256 x double> %2) {
; CHECK-LABEL: vgtlsxnc_vvssmvl_imm_3:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 128
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vgtl.sx.nc %v1, %v0, 8, 0, %vm1
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v1
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = tail call fast <256 x double> @llvm.ve.vl.vgtlsxnc.vvssmvl(<256 x double> %0, i64 8, i64 0, <256 x i1> %1, <256 x double> %2, i32 128)
  ret <256 x double> %4
}

; Function Attrs: nounwind readonly
define fastcc <256 x double> @vgtlsxnc_vvssl_no_imm_1(<256 x double> %0, i64 %1) {
; CHECK-LABEL: vgtlsxnc_vvssl_no_imm_1:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    or %s2, 8, (0)1
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vgtl.sx.nc %v0, %v0, %s0, %s2
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.vgtlsxnc.vvssl(<256 x double> %0, i64 %1, i64 8, i32 256)
  ret <256 x double> %3
}

; Function Attrs: nounwind readonly
define fastcc <256 x double> @vgtlzx_vvssl(<256 x double> %0, i64 %1, i64 %2) {
; CHECK-LABEL: vgtlzx_vvssl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s2, 256
; CHECK-NEXT:    lvl %s2
; CHECK-NEXT:    vgtl.zx %v0, %v0, %s0, %s1
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = tail call fast <256 x double> @llvm.ve.vl.vgtlzx.vvssl(<256 x double> %0, i64 %1, i64 %2, i32 256)
  ret <256 x double> %4
}

; Function Attrs: nounwind readonly
declare <256 x double> @llvm.ve.vl.vgtlzx.vvssl(<256 x double>, i64, i64, i32)

; Function Attrs: nounwind readonly
define fastcc <256 x double> @vgtlzx_vvssvl(<256 x double> %0, i64 %1, i64 %2, <256 x double> %3) {
; CHECK-LABEL: vgtlzx_vvssvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s2, 128
; CHECK-NEXT:    lvl %s2
; CHECK-NEXT:    vgtl.zx %v1, %v0, %s0, %s1
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v1
; CHECK-NEXT:    b.l.t (, %s10)
  %5 = tail call fast <256 x double> @llvm.ve.vl.vgtlzx.vvssvl(<256 x double> %0, i64 %1, i64 %2, <256 x double> %3, i32 128)
  ret <256 x double> %5
}

; Function Attrs: nounwind readonly
declare <256 x double> @llvm.ve.vl.vgtlzx.vvssvl(<256 x double>, i64, i64, <256 x double>, i32)

; Function Attrs: nounwind readonly
define fastcc <256 x double> @vgtlzx_vvssl_imm_1(<256 x double> %0, i64 %1) {
; CHECK-LABEL: vgtlzx_vvssl_imm_1:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vgtl.zx %v0, %v0, %s0, 0
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.vgtlzx.vvssl(<256 x double> %0, i64 %1, i64 0, i32 256)
  ret <256 x double> %3
}

; Function Attrs: nounwind readonly
define fastcc <256 x double> @vgtlzx_vvssvl_imm_1(<256 x double> %0, i64 %1, <256 x double> %2) {
; CHECK-LABEL: vgtlzx_vvssvl_imm_1:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 128
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vgtl.zx %v1, %v0, %s0, 0
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v1
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = tail call fast <256 x double> @llvm.ve.vl.vgtlzx.vvssvl(<256 x double> %0, i64 %1, i64 0, <256 x double> %2, i32 128)
  ret <256 x double> %4
}

; Function Attrs: nounwind readonly
define fastcc <256 x double> @vgtlzx_vvssl_imm_2(<256 x double> %0, i64 %1) {
; CHECK-LABEL: vgtlzx_vvssl_imm_2:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vgtl.zx %v0, %v0, 8, %s0
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.vgtlzx.vvssl(<256 x double> %0, i64 8, i64 %1, i32 256)
  ret <256 x double> %3
}

; Function Attrs: nounwind readonly
define fastcc <256 x double> @vgtlzx_vvssvl_imm_2(<256 x double> %0, i64 %1, <256 x double> %2) {
; CHECK-LABEL: vgtlzx_vvssvl_imm_2:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 128
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vgtl.zx %v1, %v0, 8, %s0
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v1
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = tail call fast <256 x double> @llvm.ve.vl.vgtlzx.vvssvl(<256 x double> %0, i64 8, i64 %1, <256 x double> %2, i32 128)
  ret <256 x double> %4
}

; Function Attrs: nounwind readonly
define fastcc <256 x double> @vgtlzx_vvssl_imm_3(<256 x double> %0) {
; CHECK-LABEL: vgtlzx_vvssl_imm_3:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vgtl.zx %v0, %v0, 8, 0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call fast <256 x double> @llvm.ve.vl.vgtlzx.vvssl(<256 x double> %0, i64 8, i64 0, i32 256)
  ret <256 x double> %2
}

; Function Attrs: nounwind readonly
define fastcc <256 x double> @vgtlzx_vvssvl_imm_3(<256 x double> %0, <256 x double> %1) {
; CHECK-LABEL: vgtlzx_vvssvl_imm_3:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 128
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vgtl.zx %v1, %v0, 8, 0
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.vgtlzx.vvssvl(<256 x double> %0, i64 8, i64 0, <256 x double> %1, i32 128)
  ret <256 x double> %3
}

; Function Attrs: nounwind readonly
define fastcc <256 x double> @vgtlzx_vvssml(<256 x double> %0, i64 %1, i64 %2, <256 x i1> %3) {
; CHECK-LABEL: vgtlzx_vvssml:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s2, 256
; CHECK-NEXT:    lvl %s2
; CHECK-NEXT:    vgtl.zx %v0, %v0, %s0, %s1, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %5 = tail call fast <256 x double> @llvm.ve.vl.vgtlzx.vvssml(<256 x double> %0, i64 %1, i64 %2, <256 x i1> %3, i32 256)
  ret <256 x double> %5
}

; Function Attrs: nounwind readonly
declare <256 x double> @llvm.ve.vl.vgtlzx.vvssml(<256 x double>, i64, i64, <256 x i1>, i32)

; Function Attrs: nounwind readonly
define fastcc <256 x double> @vgtlzx_vvssmvl(<256 x double> %0, i64 %1, i64 %2, <256 x i1> %3, <256 x double> %4) {
; CHECK-LABEL: vgtlzx_vvssmvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s2, 128
; CHECK-NEXT:    lvl %s2
; CHECK-NEXT:    vgtl.zx %v1, %v0, %s0, %s1, %vm1
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v1
; CHECK-NEXT:    b.l.t (, %s10)
  %6 = tail call fast <256 x double> @llvm.ve.vl.vgtlzx.vvssmvl(<256 x double> %0, i64 %1, i64 %2, <256 x i1> %3, <256 x double> %4, i32 128)
  ret <256 x double> %6
}

; Function Attrs: nounwind readonly
declare <256 x double> @llvm.ve.vl.vgtlzx.vvssmvl(<256 x double>, i64, i64, <256 x i1>, <256 x double>, i32)

; Function Attrs: nounwind readonly
define fastcc <256 x double> @vgtlzx_vvssml_imm_1(<256 x double> %0, i64 %1, <256 x i1> %2) {
; CHECK-LABEL: vgtlzx_vvssml_imm_1:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vgtl.zx %v0, %v0, %s0, 0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = tail call fast <256 x double> @llvm.ve.vl.vgtlzx.vvssml(<256 x double> %0, i64 %1, i64 0, <256 x i1> %2, i32 256)
  ret <256 x double> %4
}

; Function Attrs: nounwind readonly
define fastcc <256 x double> @vgtlzx_vvssmvl_imm_1(<256 x double> %0, i64 %1, <256 x i1> %2, <256 x double> %3) {
; CHECK-LABEL: vgtlzx_vvssmvl_imm_1:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 128
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vgtl.zx %v1, %v0, %s0, 0, %vm1
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v1
; CHECK-NEXT:    b.l.t (, %s10)
  %5 = tail call fast <256 x double> @llvm.ve.vl.vgtlzx.vvssmvl(<256 x double> %0, i64 %1, i64 0, <256 x i1> %2, <256 x double> %3, i32 128)
  ret <256 x double> %5
}

; Function Attrs: nounwind readonly
define fastcc <256 x double> @vgtlzx_vvssml_imm_2(<256 x double> %0, i64 %1, <256 x i1> %2) {
; CHECK-LABEL: vgtlzx_vvssml_imm_2:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vgtl.zx %v0, %v0, 8, %s0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = tail call fast <256 x double> @llvm.ve.vl.vgtlzx.vvssml(<256 x double> %0, i64 8, i64 %1, <256 x i1> %2, i32 256)
  ret <256 x double> %4
}

; Function Attrs: nounwind readonly
define fastcc <256 x double> @vgtlzx_vvssmvl_imm_2(<256 x double> %0, i64 %1, <256 x i1> %2, <256 x double> %3) {
; CHECK-LABEL: vgtlzx_vvssmvl_imm_2:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 128
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vgtl.zx %v1, %v0, 8, %s0, %vm1
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v1
; CHECK-NEXT:    b.l.t (, %s10)
  %5 = tail call fast <256 x double> @llvm.ve.vl.vgtlzx.vvssmvl(<256 x double> %0, i64 8, i64 %1, <256 x i1> %2, <256 x double> %3, i32 128)
  ret <256 x double> %5
}

; Function Attrs: nounwind readonly
define fastcc <256 x double> @vgtlzx_vvssml_imm_3(<256 x double> %0, <256 x i1> %1) {
; CHECK-LABEL: vgtlzx_vvssml_imm_3:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vgtl.zx %v0, %v0, 8, 0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.vgtlzx.vvssml(<256 x double> %0, i64 8, i64 0, <256 x i1> %1, i32 256)
  ret <256 x double> %3
}

; Function Attrs: nounwind readonly
define fastcc <256 x double> @vgtlzx_vvssmvl_imm_3(<256 x double> %0, <256 x i1> %1, <256 x double> %2) {
; CHECK-LABEL: vgtlzx_vvssmvl_imm_3:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 128
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vgtl.zx %v1, %v0, 8, 0, %vm1
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v1
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = tail call fast <256 x double> @llvm.ve.vl.vgtlzx.vvssmvl(<256 x double> %0, i64 8, i64 0, <256 x i1> %1, <256 x double> %2, i32 128)
  ret <256 x double> %4
}

; Function Attrs: nounwind readonly
define fastcc <256 x double> @vgtlzx_vvssl_no_imm_1(<256 x double> %0, i64 %1) {
; CHECK-LABEL: vgtlzx_vvssl_no_imm_1:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    or %s2, 8, (0)1
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vgtl.zx %v0, %v0, %s0, %s2
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.vgtlzx.vvssl(<256 x double> %0, i64 %1, i64 8, i32 256)
  ret <256 x double> %3
}

; Function Attrs: nounwind readonly
define fastcc <256 x double> @vgtlzxnc_vvssl(<256 x double> %0, i64 %1, i64 %2) {
; CHECK-LABEL: vgtlzxnc_vvssl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s2, 256
; CHECK-NEXT:    lvl %s2
; CHECK-NEXT:    vgtl.zx.nc %v0, %v0, %s0, %s1
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = tail call fast <256 x double> @llvm.ve.vl.vgtlzxnc.vvssl(<256 x double> %0, i64 %1, i64 %2, i32 256)
  ret <256 x double> %4
}

; Function Attrs: nounwind readonly
declare <256 x double> @llvm.ve.vl.vgtlzxnc.vvssl(<256 x double>, i64, i64, i32)

; Function Attrs: nounwind readonly
define fastcc <256 x double> @vgtlzxnc_vvssvl(<256 x double> %0, i64 %1, i64 %2, <256 x double> %3) {
; CHECK-LABEL: vgtlzxnc_vvssvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s2, 128
; CHECK-NEXT:    lvl %s2
; CHECK-NEXT:    vgtl.zx.nc %v1, %v0, %s0, %s1
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v1
; CHECK-NEXT:    b.l.t (, %s10)
  %5 = tail call fast <256 x double> @llvm.ve.vl.vgtlzxnc.vvssvl(<256 x double> %0, i64 %1, i64 %2, <256 x double> %3, i32 128)
  ret <256 x double> %5
}

; Function Attrs: nounwind readonly
declare <256 x double> @llvm.ve.vl.vgtlzxnc.vvssvl(<256 x double>, i64, i64, <256 x double>, i32)

; Function Attrs: nounwind readonly
define fastcc <256 x double> @vgtlzxnc_vvssl_imm_1(<256 x double> %0, i64 %1) {
; CHECK-LABEL: vgtlzxnc_vvssl_imm_1:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vgtl.zx.nc %v0, %v0, %s0, 0
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.vgtlzxnc.vvssl(<256 x double> %0, i64 %1, i64 0, i32 256)
  ret <256 x double> %3
}

; Function Attrs: nounwind readonly
define fastcc <256 x double> @vgtlzxnc_vvssvl_imm_1(<256 x double> %0, i64 %1, <256 x double> %2) {
; CHECK-LABEL: vgtlzxnc_vvssvl_imm_1:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 128
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vgtl.zx.nc %v1, %v0, %s0, 0
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v1
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = tail call fast <256 x double> @llvm.ve.vl.vgtlzxnc.vvssvl(<256 x double> %0, i64 %1, i64 0, <256 x double> %2, i32 128)
  ret <256 x double> %4
}

; Function Attrs: nounwind readonly
define fastcc <256 x double> @vgtlzxnc_vvssl_imm_2(<256 x double> %0, i64 %1) {
; CHECK-LABEL: vgtlzxnc_vvssl_imm_2:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vgtl.zx.nc %v0, %v0, 8, %s0
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.vgtlzxnc.vvssl(<256 x double> %0, i64 8, i64 %1, i32 256)
  ret <256 x double> %3
}

; Function Attrs: nounwind readonly
define fastcc <256 x double> @vgtlzxnc_vvssvl_imm_2(<256 x double> %0, i64 %1, <256 x double> %2) {
; CHECK-LABEL: vgtlzxnc_vvssvl_imm_2:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 128
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vgtl.zx.nc %v1, %v0, 8, %s0
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v1
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = tail call fast <256 x double> @llvm.ve.vl.vgtlzxnc.vvssvl(<256 x double> %0, i64 8, i64 %1, <256 x double> %2, i32 128)
  ret <256 x double> %4
}

; Function Attrs: nounwind readonly
define fastcc <256 x double> @vgtlzxnc_vvssl_imm_3(<256 x double> %0) {
; CHECK-LABEL: vgtlzxnc_vvssl_imm_3:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vgtl.zx.nc %v0, %v0, 8, 0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call fast <256 x double> @llvm.ve.vl.vgtlzxnc.vvssl(<256 x double> %0, i64 8, i64 0, i32 256)
  ret <256 x double> %2
}

; Function Attrs: nounwind readonly
define fastcc <256 x double> @vgtlzxnc_vvssvl_imm_3(<256 x double> %0, <256 x double> %1) {
; CHECK-LABEL: vgtlzxnc_vvssvl_imm_3:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 128
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vgtl.zx.nc %v1, %v0, 8, 0
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.vgtlzxnc.vvssvl(<256 x double> %0, i64 8, i64 0, <256 x double> %1, i32 128)
  ret <256 x double> %3
}

; Function Attrs: nounwind readonly
define fastcc <256 x double> @vgtlzxnc_vvssml(<256 x double> %0, i64 %1, i64 %2, <256 x i1> %3) {
; CHECK-LABEL: vgtlzxnc_vvssml:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s2, 256
; CHECK-NEXT:    lvl %s2
; CHECK-NEXT:    vgtl.zx.nc %v0, %v0, %s0, %s1, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %5 = tail call fast <256 x double> @llvm.ve.vl.vgtlzxnc.vvssml(<256 x double> %0, i64 %1, i64 %2, <256 x i1> %3, i32 256)
  ret <256 x double> %5
}

; Function Attrs: nounwind readonly
declare <256 x double> @llvm.ve.vl.vgtlzxnc.vvssml(<256 x double>, i64, i64, <256 x i1>, i32)

; Function Attrs: nounwind readonly
define fastcc <256 x double> @vgtlzxnc_vvssmvl(<256 x double> %0, i64 %1, i64 %2, <256 x i1> %3, <256 x double> %4) {
; CHECK-LABEL: vgtlzxnc_vvssmvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s2, 128
; CHECK-NEXT:    lvl %s2
; CHECK-NEXT:    vgtl.zx.nc %v1, %v0, %s0, %s1, %vm1
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v1
; CHECK-NEXT:    b.l.t (, %s10)
  %6 = tail call fast <256 x double> @llvm.ve.vl.vgtlzxnc.vvssmvl(<256 x double> %0, i64 %1, i64 %2, <256 x i1> %3, <256 x double> %4, i32 128)
  ret <256 x double> %6
}

; Function Attrs: nounwind readonly
declare <256 x double> @llvm.ve.vl.vgtlzxnc.vvssmvl(<256 x double>, i64, i64, <256 x i1>, <256 x double>, i32)

; Function Attrs: nounwind readonly
define fastcc <256 x double> @vgtlzxnc_vvssml_imm_1(<256 x double> %0, i64 %1, <256 x i1> %2) {
; CHECK-LABEL: vgtlzxnc_vvssml_imm_1:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vgtl.zx.nc %v0, %v0, %s0, 0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = tail call fast <256 x double> @llvm.ve.vl.vgtlzxnc.vvssml(<256 x double> %0, i64 %1, i64 0, <256 x i1> %2, i32 256)
  ret <256 x double> %4
}

; Function Attrs: nounwind readonly
define fastcc <256 x double> @vgtlzxnc_vvssmvl_imm_1(<256 x double> %0, i64 %1, <256 x i1> %2, <256 x double> %3) {
; CHECK-LABEL: vgtlzxnc_vvssmvl_imm_1:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 128
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vgtl.zx.nc %v1, %v0, %s0, 0, %vm1
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v1
; CHECK-NEXT:    b.l.t (, %s10)
  %5 = tail call fast <256 x double> @llvm.ve.vl.vgtlzxnc.vvssmvl(<256 x double> %0, i64 %1, i64 0, <256 x i1> %2, <256 x double> %3, i32 128)
  ret <256 x double> %5
}

; Function Attrs: nounwind readonly
define fastcc <256 x double> @vgtlzxnc_vvssml_imm_2(<256 x double> %0, i64 %1, <256 x i1> %2) {
; CHECK-LABEL: vgtlzxnc_vvssml_imm_2:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vgtl.zx.nc %v0, %v0, 8, %s0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = tail call fast <256 x double> @llvm.ve.vl.vgtlzxnc.vvssml(<256 x double> %0, i64 8, i64 %1, <256 x i1> %2, i32 256)
  ret <256 x double> %4
}

; Function Attrs: nounwind readonly
define fastcc <256 x double> @vgtlzxnc_vvssmvl_imm_2(<256 x double> %0, i64 %1, <256 x i1> %2, <256 x double> %3) {
; CHECK-LABEL: vgtlzxnc_vvssmvl_imm_2:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 128
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vgtl.zx.nc %v1, %v0, 8, %s0, %vm1
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v1
; CHECK-NEXT:    b.l.t (, %s10)
  %5 = tail call fast <256 x double> @llvm.ve.vl.vgtlzxnc.vvssmvl(<256 x double> %0, i64 8, i64 %1, <256 x i1> %2, <256 x double> %3, i32 128)
  ret <256 x double> %5
}

; Function Attrs: nounwind readonly
define fastcc <256 x double> @vgtlzxnc_vvssml_imm_3(<256 x double> %0, <256 x i1> %1) {
; CHECK-LABEL: vgtlzxnc_vvssml_imm_3:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vgtl.zx.nc %v0, %v0, 8, 0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.vgtlzxnc.vvssml(<256 x double> %0, i64 8, i64 0, <256 x i1> %1, i32 256)
  ret <256 x double> %3
}

; Function Attrs: nounwind readonly
define fastcc <256 x double> @vgtlzxnc_vvssmvl_imm_3(<256 x double> %0, <256 x i1> %1, <256 x double> %2) {
; CHECK-LABEL: vgtlzxnc_vvssmvl_imm_3:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 128
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vgtl.zx.nc %v1, %v0, 8, 0, %vm1
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v1
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = tail call fast <256 x double> @llvm.ve.vl.vgtlzxnc.vvssmvl(<256 x double> %0, i64 8, i64 0, <256 x i1> %1, <256 x double> %2, i32 128)
  ret <256 x double> %4
}

; Function Attrs: nounwind readonly
define fastcc <256 x double> @vgtlzxnc_vvssl_no_imm_1(<256 x double> %0, i64 %1) {
; CHECK-LABEL: vgtlzxnc_vvssl_no_imm_1:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    or %s2, 8, (0)1
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vgtl.zx.nc %v0, %v0, %s0, %s2
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.vgtlzxnc.vvssl(<256 x double> %0, i64 %1, i64 8, i32 256)
  ret <256 x double> %3
}
