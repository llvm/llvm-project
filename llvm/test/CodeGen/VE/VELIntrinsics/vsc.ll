; RUN: llc < %s -mtriple=ve -mattr=+vpu | FileCheck %s

;;; Test vector scatter intrinsic instructions
;;;
;;; Note:
;;;   We test VSC*vrrvl, VSC*vrzvl, VSC*virvl, VSC*vizvl, VSC*vrrvml,
;;;   VSC*vrzvml, VSC*virvml, and VSC*vizvml instructions.

; Function Attrs: nounwind writeonly
define fastcc void @vsc_vvssl(<256 x double> %0, <256 x double> %1, i64 %2, i64 %3) {
; CHECK-LABEL: vsc_vvssl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s2, 256
; CHECK-NEXT:    lvl %s2
; CHECK-NEXT:    vsc %v0, %v1, %s0, %s1
; CHECK-NEXT:    b.l.t (, %s10)
  tail call void @llvm.ve.vl.vsc.vvssl(<256 x double> %0, <256 x double> %1, i64 %2, i64 %3, i32 256)
  ret void
}

; Function Attrs: nounwind writeonly
declare void @llvm.ve.vl.vsc.vvssl(<256 x double>, <256 x double>, i64, i64, i32)

; Function Attrs: nounwind writeonly
define fastcc void @vsc_vvssl_imm_1(<256 x double> %0, <256 x double> %1, i64 %2) {
; CHECK-LABEL: vsc_vvssl_imm_1:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vsc %v0, %v1, %s0, 0
; CHECK-NEXT:    b.l.t (, %s10)
  tail call void @llvm.ve.vl.vsc.vvssl(<256 x double> %0, <256 x double> %1, i64 %2, i64 0, i32 256)
  ret void
}

; Function Attrs: nounwind writeonly
define fastcc void @vsc_vvssl_imm_2(<256 x double> %0, <256 x double> %1, i64 %2) {
; CHECK-LABEL: vsc_vvssl_imm_2:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vsc %v0, %v1, 8, %s0
; CHECK-NEXT:    b.l.t (, %s10)
  tail call void @llvm.ve.vl.vsc.vvssl(<256 x double> %0, <256 x double> %1, i64 8, i64 %2, i32 256)
  ret void
}

; Function Attrs: nounwind writeonly
define fastcc void @vsc_vvssl_imm_3(<256 x double> %0, <256 x double> %1) {
; CHECK-LABEL: vsc_vvssl_imm_3:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vsc %v0, %v1, 8, 0
; CHECK-NEXT:    b.l.t (, %s10)
  tail call void @llvm.ve.vl.vsc.vvssl(<256 x double> %0, <256 x double> %1, i64 8, i64 0, i32 256)
  ret void
}

; Function Attrs: nounwind writeonly
define fastcc void @vsc_vvssml(<256 x double> %0, <256 x double> %1, i64 %2, i64 %3, <256 x i1> %4) {
; CHECK-LABEL: vsc_vvssml:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s2, 256
; CHECK-NEXT:    lvl %s2
; CHECK-NEXT:    vsc %v0, %v1, %s0, %s1, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  tail call void @llvm.ve.vl.vsc.vvssml(<256 x double> %0, <256 x double> %1, i64 %2, i64 %3, <256 x i1> %4, i32 256)
  ret void
}

; Function Attrs: nounwind writeonly
declare void @llvm.ve.vl.vsc.vvssml(<256 x double>, <256 x double>, i64, i64, <256 x i1>, i32)

; Function Attrs: nounwind writeonly
define fastcc void @vsc_vvssml_imm_1(<256 x double> %0, <256 x double> %1, i64 %2, <256 x i1> %3) {
; CHECK-LABEL: vsc_vvssml_imm_1:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vsc %v0, %v1, %s0, 0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  tail call void @llvm.ve.vl.vsc.vvssml(<256 x double> %0, <256 x double> %1, i64 %2, i64 0, <256 x i1> %3, i32 256)
  ret void
}

; Function Attrs: nounwind writeonly
define fastcc void @vsc_vvssml_imm_2(<256 x double> %0, <256 x double> %1, i64 %2, <256 x i1> %3) {
; CHECK-LABEL: vsc_vvssml_imm_2:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vsc %v0, %v1, 8, %s0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  tail call void @llvm.ve.vl.vsc.vvssml(<256 x double> %0, <256 x double> %1, i64 8, i64 %2, <256 x i1> %3, i32 256)
  ret void
}

; Function Attrs: nounwind writeonly
define fastcc void @vsc_vvssml_imm_3(<256 x double> %0, <256 x double> %1, <256 x i1> %2) {
; CHECK-LABEL: vsc_vvssml_imm_3:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vsc %v0, %v1, 8, 0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  tail call void @llvm.ve.vl.vsc.vvssml(<256 x double> %0, <256 x double> %1, i64 8, i64 0, <256 x i1> %2, i32 256)
  ret void
}

; Function Attrs: nounwind writeonly
define fastcc void @vsc_vvssl_no_imm_1(<256 x double> %0, <256 x double> %1, i64 %2) {
; CHECK-LABEL: vsc_vvssl_no_imm_1:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    or %s2, 8, (0)1
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vsc %v0, %v1, %s0, %s2
; CHECK-NEXT:    b.l.t (, %s10)
  tail call void @llvm.ve.vl.vsc.vvssl(<256 x double> %0, <256 x double> %1, i64 %2, i64 8, i32 256)
  ret void
}

; Function Attrs: nounwind writeonly
define fastcc void @vscnc_vvssl(<256 x double> %0, <256 x double> %1, i64 %2, i64 %3) {
; CHECK-LABEL: vscnc_vvssl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s2, 256
; CHECK-NEXT:    lvl %s2
; CHECK-NEXT:    vsc.nc %v0, %v1, %s0, %s1
; CHECK-NEXT:    b.l.t (, %s10)
  tail call void @llvm.ve.vl.vscnc.vvssl(<256 x double> %0, <256 x double> %1, i64 %2, i64 %3, i32 256)
  ret void
}

; Function Attrs: nounwind writeonly
declare void @llvm.ve.vl.vscnc.vvssl(<256 x double>, <256 x double>, i64, i64, i32)

; Function Attrs: nounwind writeonly
define fastcc void @vscnc_vvssl_imm_1(<256 x double> %0, <256 x double> %1, i64 %2) {
; CHECK-LABEL: vscnc_vvssl_imm_1:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vsc.nc %v0, %v1, %s0, 0
; CHECK-NEXT:    b.l.t (, %s10)
  tail call void @llvm.ve.vl.vscnc.vvssl(<256 x double> %0, <256 x double> %1, i64 %2, i64 0, i32 256)
  ret void
}

; Function Attrs: nounwind writeonly
define fastcc void @vscnc_vvssl_imm_2(<256 x double> %0, <256 x double> %1, i64 %2) {
; CHECK-LABEL: vscnc_vvssl_imm_2:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vsc.nc %v0, %v1, 8, %s0
; CHECK-NEXT:    b.l.t (, %s10)
  tail call void @llvm.ve.vl.vscnc.vvssl(<256 x double> %0, <256 x double> %1, i64 8, i64 %2, i32 256)
  ret void
}

; Function Attrs: nounwind writeonly
define fastcc void @vscnc_vvssl_imm_3(<256 x double> %0, <256 x double> %1) {
; CHECK-LABEL: vscnc_vvssl_imm_3:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vsc.nc %v0, %v1, 8, 0
; CHECK-NEXT:    b.l.t (, %s10)
  tail call void @llvm.ve.vl.vscnc.vvssl(<256 x double> %0, <256 x double> %1, i64 8, i64 0, i32 256)
  ret void
}

; Function Attrs: nounwind writeonly
define fastcc void @vscnc_vvssml(<256 x double> %0, <256 x double> %1, i64 %2, i64 %3, <256 x i1> %4) {
; CHECK-LABEL: vscnc_vvssml:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s2, 256
; CHECK-NEXT:    lvl %s2
; CHECK-NEXT:    vsc.nc %v0, %v1, %s0, %s1, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  tail call void @llvm.ve.vl.vscnc.vvssml(<256 x double> %0, <256 x double> %1, i64 %2, i64 %3, <256 x i1> %4, i32 256)
  ret void
}

; Function Attrs: nounwind writeonly
declare void @llvm.ve.vl.vscnc.vvssml(<256 x double>, <256 x double>, i64, i64, <256 x i1>, i32)

; Function Attrs: nounwind writeonly
define fastcc void @vscnc_vvssml_imm_1(<256 x double> %0, <256 x double> %1, i64 %2, <256 x i1> %3) {
; CHECK-LABEL: vscnc_vvssml_imm_1:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vsc.nc %v0, %v1, %s0, 0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  tail call void @llvm.ve.vl.vscnc.vvssml(<256 x double> %0, <256 x double> %1, i64 %2, i64 0, <256 x i1> %3, i32 256)
  ret void
}

; Function Attrs: nounwind writeonly
define fastcc void @vscnc_vvssml_imm_2(<256 x double> %0, <256 x double> %1, i64 %2, <256 x i1> %3) {
; CHECK-LABEL: vscnc_vvssml_imm_2:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vsc.nc %v0, %v1, 8, %s0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  tail call void @llvm.ve.vl.vscnc.vvssml(<256 x double> %0, <256 x double> %1, i64 8, i64 %2, <256 x i1> %3, i32 256)
  ret void
}

; Function Attrs: nounwind writeonly
define fastcc void @vscnc_vvssml_imm_3(<256 x double> %0, <256 x double> %1, <256 x i1> %2) {
; CHECK-LABEL: vscnc_vvssml_imm_3:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vsc.nc %v0, %v1, 8, 0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  tail call void @llvm.ve.vl.vscnc.vvssml(<256 x double> %0, <256 x double> %1, i64 8, i64 0, <256 x i1> %2, i32 256)
  ret void
}

; Function Attrs: nounwind writeonly
define fastcc void @vscnc_vvssl_no_imm_1(<256 x double> %0, <256 x double> %1, i64 %2) {
; CHECK-LABEL: vscnc_vvssl_no_imm_1:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    or %s2, 8, (0)1
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vsc.nc %v0, %v1, %s0, %s2
; CHECK-NEXT:    b.l.t (, %s10)
  tail call void @llvm.ve.vl.vscnc.vvssl(<256 x double> %0, <256 x double> %1, i64 %2, i64 8, i32 256)
  ret void
}

; Function Attrs: nounwind writeonly
define fastcc void @vscot_vvssl(<256 x double> %0, <256 x double> %1, i64 %2, i64 %3) {
; CHECK-LABEL: vscot_vvssl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s2, 256
; CHECK-NEXT:    lvl %s2
; CHECK-NEXT:    vsc.ot %v0, %v1, %s0, %s1
; CHECK-NEXT:    b.l.t (, %s10)
  tail call void @llvm.ve.vl.vscot.vvssl(<256 x double> %0, <256 x double> %1, i64 %2, i64 %3, i32 256)
  ret void
}

; Function Attrs: nounwind writeonly
declare void @llvm.ve.vl.vscot.vvssl(<256 x double>, <256 x double>, i64, i64, i32)

; Function Attrs: nounwind writeonly
define fastcc void @vscot_vvssl_imm_1(<256 x double> %0, <256 x double> %1, i64 %2) {
; CHECK-LABEL: vscot_vvssl_imm_1:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vsc.ot %v0, %v1, %s0, 0
; CHECK-NEXT:    b.l.t (, %s10)
  tail call void @llvm.ve.vl.vscot.vvssl(<256 x double> %0, <256 x double> %1, i64 %2, i64 0, i32 256)
  ret void
}

; Function Attrs: nounwind writeonly
define fastcc void @vscot_vvssl_imm_2(<256 x double> %0, <256 x double> %1, i64 %2) {
; CHECK-LABEL: vscot_vvssl_imm_2:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vsc.ot %v0, %v1, 8, %s0
; CHECK-NEXT:    b.l.t (, %s10)
  tail call void @llvm.ve.vl.vscot.vvssl(<256 x double> %0, <256 x double> %1, i64 8, i64 %2, i32 256)
  ret void
}

; Function Attrs: nounwind writeonly
define fastcc void @vscot_vvssl_imm_3(<256 x double> %0, <256 x double> %1) {
; CHECK-LABEL: vscot_vvssl_imm_3:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vsc.ot %v0, %v1, 8, 0
; CHECK-NEXT:    b.l.t (, %s10)
  tail call void @llvm.ve.vl.vscot.vvssl(<256 x double> %0, <256 x double> %1, i64 8, i64 0, i32 256)
  ret void
}

; Function Attrs: nounwind writeonly
define fastcc void @vscot_vvssml(<256 x double> %0, <256 x double> %1, i64 %2, i64 %3, <256 x i1> %4) {
; CHECK-LABEL: vscot_vvssml:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s2, 256
; CHECK-NEXT:    lvl %s2
; CHECK-NEXT:    vsc.ot %v0, %v1, %s0, %s1, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  tail call void @llvm.ve.vl.vscot.vvssml(<256 x double> %0, <256 x double> %1, i64 %2, i64 %3, <256 x i1> %4, i32 256)
  ret void
}

; Function Attrs: nounwind writeonly
declare void @llvm.ve.vl.vscot.vvssml(<256 x double>, <256 x double>, i64, i64, <256 x i1>, i32)

; Function Attrs: nounwind writeonly
define fastcc void @vscot_vvssml_imm_1(<256 x double> %0, <256 x double> %1, i64 %2, <256 x i1> %3) {
; CHECK-LABEL: vscot_vvssml_imm_1:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vsc.ot %v0, %v1, %s0, 0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  tail call void @llvm.ve.vl.vscot.vvssml(<256 x double> %0, <256 x double> %1, i64 %2, i64 0, <256 x i1> %3, i32 256)
  ret void
}

; Function Attrs: nounwind writeonly
define fastcc void @vscot_vvssml_imm_2(<256 x double> %0, <256 x double> %1, i64 %2, <256 x i1> %3) {
; CHECK-LABEL: vscot_vvssml_imm_2:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vsc.ot %v0, %v1, 8, %s0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  tail call void @llvm.ve.vl.vscot.vvssml(<256 x double> %0, <256 x double> %1, i64 8, i64 %2, <256 x i1> %3, i32 256)
  ret void
}

; Function Attrs: nounwind writeonly
define fastcc void @vscot_vvssml_imm_3(<256 x double> %0, <256 x double> %1, <256 x i1> %2) {
; CHECK-LABEL: vscot_vvssml_imm_3:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vsc.ot %v0, %v1, 8, 0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  tail call void @llvm.ve.vl.vscot.vvssml(<256 x double> %0, <256 x double> %1, i64 8, i64 0, <256 x i1> %2, i32 256)
  ret void
}

; Function Attrs: nounwind writeonly
define fastcc void @vscot_vvssl_no_imm_1(<256 x double> %0, <256 x double> %1, i64 %2) {
; CHECK-LABEL: vscot_vvssl_no_imm_1:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    or %s2, 8, (0)1
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vsc.ot %v0, %v1, %s0, %s2
; CHECK-NEXT:    b.l.t (, %s10)
  tail call void @llvm.ve.vl.vscot.vvssl(<256 x double> %0, <256 x double> %1, i64 %2, i64 8, i32 256)
  ret void
}

; Function Attrs: nounwind writeonly
define fastcc void @vscncot_vvssl(<256 x double> %0, <256 x double> %1, i64 %2, i64 %3) {
; CHECK-LABEL: vscncot_vvssl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s2, 256
; CHECK-NEXT:    lvl %s2
; CHECK-NEXT:    vsc.nc.ot %v0, %v1, %s0, %s1
; CHECK-NEXT:    b.l.t (, %s10)
  tail call void @llvm.ve.vl.vscncot.vvssl(<256 x double> %0, <256 x double> %1, i64 %2, i64 %3, i32 256)
  ret void
}

; Function Attrs: nounwind writeonly
declare void @llvm.ve.vl.vscncot.vvssl(<256 x double>, <256 x double>, i64, i64, i32)

; Function Attrs: nounwind writeonly
define fastcc void @vscncot_vvssl_imm_1(<256 x double> %0, <256 x double> %1, i64 %2) {
; CHECK-LABEL: vscncot_vvssl_imm_1:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vsc.nc.ot %v0, %v1, %s0, 0
; CHECK-NEXT:    b.l.t (, %s10)
  tail call void @llvm.ve.vl.vscncot.vvssl(<256 x double> %0, <256 x double> %1, i64 %2, i64 0, i32 256)
  ret void
}

; Function Attrs: nounwind writeonly
define fastcc void @vscncot_vvssl_imm_2(<256 x double> %0, <256 x double> %1, i64 %2) {
; CHECK-LABEL: vscncot_vvssl_imm_2:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vsc.nc.ot %v0, %v1, 8, %s0
; CHECK-NEXT:    b.l.t (, %s10)
  tail call void @llvm.ve.vl.vscncot.vvssl(<256 x double> %0, <256 x double> %1, i64 8, i64 %2, i32 256)
  ret void
}

; Function Attrs: nounwind writeonly
define fastcc void @vscncot_vvssl_imm_3(<256 x double> %0, <256 x double> %1) {
; CHECK-LABEL: vscncot_vvssl_imm_3:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vsc.nc.ot %v0, %v1, 8, 0
; CHECK-NEXT:    b.l.t (, %s10)
  tail call void @llvm.ve.vl.vscncot.vvssl(<256 x double> %0, <256 x double> %1, i64 8, i64 0, i32 256)
  ret void
}

; Function Attrs: nounwind writeonly
define fastcc void @vscncot_vvssml(<256 x double> %0, <256 x double> %1, i64 %2, i64 %3, <256 x i1> %4) {
; CHECK-LABEL: vscncot_vvssml:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s2, 256
; CHECK-NEXT:    lvl %s2
; CHECK-NEXT:    vsc.nc.ot %v0, %v1, %s0, %s1, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  tail call void @llvm.ve.vl.vscncot.vvssml(<256 x double> %0, <256 x double> %1, i64 %2, i64 %3, <256 x i1> %4, i32 256)
  ret void
}

; Function Attrs: nounwind writeonly
declare void @llvm.ve.vl.vscncot.vvssml(<256 x double>, <256 x double>, i64, i64, <256 x i1>, i32)

; Function Attrs: nounwind writeonly
define fastcc void @vscncot_vvssml_imm_1(<256 x double> %0, <256 x double> %1, i64 %2, <256 x i1> %3) {
; CHECK-LABEL: vscncot_vvssml_imm_1:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vsc.nc.ot %v0, %v1, %s0, 0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  tail call void @llvm.ve.vl.vscncot.vvssml(<256 x double> %0, <256 x double> %1, i64 %2, i64 0, <256 x i1> %3, i32 256)
  ret void
}

; Function Attrs: nounwind writeonly
define fastcc void @vscncot_vvssml_imm_2(<256 x double> %0, <256 x double> %1, i64 %2, <256 x i1> %3) {
; CHECK-LABEL: vscncot_vvssml_imm_2:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vsc.nc.ot %v0, %v1, 8, %s0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  tail call void @llvm.ve.vl.vscncot.vvssml(<256 x double> %0, <256 x double> %1, i64 8, i64 %2, <256 x i1> %3, i32 256)
  ret void
}

; Function Attrs: nounwind writeonly
define fastcc void @vscncot_vvssml_imm_3(<256 x double> %0, <256 x double> %1, <256 x i1> %2) {
; CHECK-LABEL: vscncot_vvssml_imm_3:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vsc.nc.ot %v0, %v1, 8, 0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  tail call void @llvm.ve.vl.vscncot.vvssml(<256 x double> %0, <256 x double> %1, i64 8, i64 0, <256 x i1> %2, i32 256)
  ret void
}

; Function Attrs: nounwind writeonly
define fastcc void @vscncot_vvssl_no_imm_1(<256 x double> %0, <256 x double> %1, i64 %2) {
; CHECK-LABEL: vscncot_vvssl_no_imm_1:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    or %s2, 8, (0)1
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vsc.nc.ot %v0, %v1, %s0, %s2
; CHECK-NEXT:    b.l.t (, %s10)
  tail call void @llvm.ve.vl.vscncot.vvssl(<256 x double> %0, <256 x double> %1, i64 %2, i64 8, i32 256)
  ret void
}

; Function Attrs: nounwind writeonly
define fastcc void @vscu_vvssl(<256 x double> %0, <256 x double> %1, i64 %2, i64 %3) {
; CHECK-LABEL: vscu_vvssl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s2, 256
; CHECK-NEXT:    lvl %s2
; CHECK-NEXT:    vscu %v0, %v1, %s0, %s1
; CHECK-NEXT:    b.l.t (, %s10)
  tail call void @llvm.ve.vl.vscu.vvssl(<256 x double> %0, <256 x double> %1, i64 %2, i64 %3, i32 256)
  ret void
}

; Function Attrs: nounwind writeonly
declare void @llvm.ve.vl.vscu.vvssl(<256 x double>, <256 x double>, i64, i64, i32)

; Function Attrs: nounwind writeonly
define fastcc void @vscu_vvssl_imm_1(<256 x double> %0, <256 x double> %1, i64 %2) {
; CHECK-LABEL: vscu_vvssl_imm_1:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vscu %v0, %v1, %s0, 0
; CHECK-NEXT:    b.l.t (, %s10)
  tail call void @llvm.ve.vl.vscu.vvssl(<256 x double> %0, <256 x double> %1, i64 %2, i64 0, i32 256)
  ret void
}

; Function Attrs: nounwind writeonly
define fastcc void @vscu_vvssl_imm_2(<256 x double> %0, <256 x double> %1, i64 %2) {
; CHECK-LABEL: vscu_vvssl_imm_2:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vscu %v0, %v1, 8, %s0
; CHECK-NEXT:    b.l.t (, %s10)
  tail call void @llvm.ve.vl.vscu.vvssl(<256 x double> %0, <256 x double> %1, i64 8, i64 %2, i32 256)
  ret void
}

; Function Attrs: nounwind writeonly
define fastcc void @vscu_vvssl_imm_3(<256 x double> %0, <256 x double> %1) {
; CHECK-LABEL: vscu_vvssl_imm_3:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vscu %v0, %v1, 8, 0
; CHECK-NEXT:    b.l.t (, %s10)
  tail call void @llvm.ve.vl.vscu.vvssl(<256 x double> %0, <256 x double> %1, i64 8, i64 0, i32 256)
  ret void
}

; Function Attrs: nounwind writeonly
define fastcc void @vscu_vvssml(<256 x double> %0, <256 x double> %1, i64 %2, i64 %3, <256 x i1> %4) {
; CHECK-LABEL: vscu_vvssml:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s2, 256
; CHECK-NEXT:    lvl %s2
; CHECK-NEXT:    vscu %v0, %v1, %s0, %s1, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  tail call void @llvm.ve.vl.vscu.vvssml(<256 x double> %0, <256 x double> %1, i64 %2, i64 %3, <256 x i1> %4, i32 256)
  ret void
}

; Function Attrs: nounwind writeonly
declare void @llvm.ve.vl.vscu.vvssml(<256 x double>, <256 x double>, i64, i64, <256 x i1>, i32)

; Function Attrs: nounwind writeonly
define fastcc void @vscu_vvssml_imm_1(<256 x double> %0, <256 x double> %1, i64 %2, <256 x i1> %3) {
; CHECK-LABEL: vscu_vvssml_imm_1:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vscu %v0, %v1, %s0, 0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  tail call void @llvm.ve.vl.vscu.vvssml(<256 x double> %0, <256 x double> %1, i64 %2, i64 0, <256 x i1> %3, i32 256)
  ret void
}

; Function Attrs: nounwind writeonly
define fastcc void @vscu_vvssml_imm_2(<256 x double> %0, <256 x double> %1, i64 %2, <256 x i1> %3) {
; CHECK-LABEL: vscu_vvssml_imm_2:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vscu %v0, %v1, 8, %s0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  tail call void @llvm.ve.vl.vscu.vvssml(<256 x double> %0, <256 x double> %1, i64 8, i64 %2, <256 x i1> %3, i32 256)
  ret void
}

; Function Attrs: nounwind writeonly
define fastcc void @vscu_vvssml_imm_3(<256 x double> %0, <256 x double> %1, <256 x i1> %2) {
; CHECK-LABEL: vscu_vvssml_imm_3:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vscu %v0, %v1, 8, 0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  tail call void @llvm.ve.vl.vscu.vvssml(<256 x double> %0, <256 x double> %1, i64 8, i64 0, <256 x i1> %2, i32 256)
  ret void
}

; Function Attrs: nounwind writeonly
define fastcc void @vscu_vvssl_no_imm_1(<256 x double> %0, <256 x double> %1, i64 %2) {
; CHECK-LABEL: vscu_vvssl_no_imm_1:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    or %s2, 8, (0)1
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vscu %v0, %v1, %s0, %s2
; CHECK-NEXT:    b.l.t (, %s10)
  tail call void @llvm.ve.vl.vscu.vvssl(<256 x double> %0, <256 x double> %1, i64 %2, i64 8, i32 256)
  ret void
}

; Function Attrs: nounwind writeonly
define fastcc void @vscunc_vvssl(<256 x double> %0, <256 x double> %1, i64 %2, i64 %3) {
; CHECK-LABEL: vscunc_vvssl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s2, 256
; CHECK-NEXT:    lvl %s2
; CHECK-NEXT:    vscu.nc %v0, %v1, %s0, %s1
; CHECK-NEXT:    b.l.t (, %s10)
  tail call void @llvm.ve.vl.vscunc.vvssl(<256 x double> %0, <256 x double> %1, i64 %2, i64 %3, i32 256)
  ret void
}

; Function Attrs: nounwind writeonly
declare void @llvm.ve.vl.vscunc.vvssl(<256 x double>, <256 x double>, i64, i64, i32)

; Function Attrs: nounwind writeonly
define fastcc void @vscunc_vvssl_imm_1(<256 x double> %0, <256 x double> %1, i64 %2) {
; CHECK-LABEL: vscunc_vvssl_imm_1:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vscu.nc %v0, %v1, %s0, 0
; CHECK-NEXT:    b.l.t (, %s10)
  tail call void @llvm.ve.vl.vscunc.vvssl(<256 x double> %0, <256 x double> %1, i64 %2, i64 0, i32 256)
  ret void
}

; Function Attrs: nounwind writeonly
define fastcc void @vscunc_vvssl_imm_2(<256 x double> %0, <256 x double> %1, i64 %2) {
; CHECK-LABEL: vscunc_vvssl_imm_2:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vscu.nc %v0, %v1, 8, %s0
; CHECK-NEXT:    b.l.t (, %s10)
  tail call void @llvm.ve.vl.vscunc.vvssl(<256 x double> %0, <256 x double> %1, i64 8, i64 %2, i32 256)
  ret void
}

; Function Attrs: nounwind writeonly
define fastcc void @vscunc_vvssl_imm_3(<256 x double> %0, <256 x double> %1) {
; CHECK-LABEL: vscunc_vvssl_imm_3:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vscu.nc %v0, %v1, 8, 0
; CHECK-NEXT:    b.l.t (, %s10)
  tail call void @llvm.ve.vl.vscunc.vvssl(<256 x double> %0, <256 x double> %1, i64 8, i64 0, i32 256)
  ret void
}

; Function Attrs: nounwind writeonly
define fastcc void @vscunc_vvssml(<256 x double> %0, <256 x double> %1, i64 %2, i64 %3, <256 x i1> %4) {
; CHECK-LABEL: vscunc_vvssml:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s2, 256
; CHECK-NEXT:    lvl %s2
; CHECK-NEXT:    vscu.nc %v0, %v1, %s0, %s1, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  tail call void @llvm.ve.vl.vscunc.vvssml(<256 x double> %0, <256 x double> %1, i64 %2, i64 %3, <256 x i1> %4, i32 256)
  ret void
}

; Function Attrs: nounwind writeonly
declare void @llvm.ve.vl.vscunc.vvssml(<256 x double>, <256 x double>, i64, i64, <256 x i1>, i32)

; Function Attrs: nounwind writeonly
define fastcc void @vscunc_vvssml_imm_1(<256 x double> %0, <256 x double> %1, i64 %2, <256 x i1> %3) {
; CHECK-LABEL: vscunc_vvssml_imm_1:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vscu.nc %v0, %v1, %s0, 0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  tail call void @llvm.ve.vl.vscunc.vvssml(<256 x double> %0, <256 x double> %1, i64 %2, i64 0, <256 x i1> %3, i32 256)
  ret void
}

; Function Attrs: nounwind writeonly
define fastcc void @vscunc_vvssml_imm_2(<256 x double> %0, <256 x double> %1, i64 %2, <256 x i1> %3) {
; CHECK-LABEL: vscunc_vvssml_imm_2:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vscu.nc %v0, %v1, 8, %s0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  tail call void @llvm.ve.vl.vscunc.vvssml(<256 x double> %0, <256 x double> %1, i64 8, i64 %2, <256 x i1> %3, i32 256)
  ret void
}

; Function Attrs: nounwind writeonly
define fastcc void @vscunc_vvssml_imm_3(<256 x double> %0, <256 x double> %1, <256 x i1> %2) {
; CHECK-LABEL: vscunc_vvssml_imm_3:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vscu.nc %v0, %v1, 8, 0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  tail call void @llvm.ve.vl.vscunc.vvssml(<256 x double> %0, <256 x double> %1, i64 8, i64 0, <256 x i1> %2, i32 256)
  ret void
}

; Function Attrs: nounwind writeonly
define fastcc void @vscunc_vvssl_no_imm_1(<256 x double> %0, <256 x double> %1, i64 %2) {
; CHECK-LABEL: vscunc_vvssl_no_imm_1:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    or %s2, 8, (0)1
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vscu.nc %v0, %v1, %s0, %s2
; CHECK-NEXT:    b.l.t (, %s10)
  tail call void @llvm.ve.vl.vscunc.vvssl(<256 x double> %0, <256 x double> %1, i64 %2, i64 8, i32 256)
  ret void
}

; Function Attrs: nounwind writeonly
define fastcc void @vscuot_vvssl(<256 x double> %0, <256 x double> %1, i64 %2, i64 %3) {
; CHECK-LABEL: vscuot_vvssl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s2, 256
; CHECK-NEXT:    lvl %s2
; CHECK-NEXT:    vscu.ot %v0, %v1, %s0, %s1
; CHECK-NEXT:    b.l.t (, %s10)
  tail call void @llvm.ve.vl.vscuot.vvssl(<256 x double> %0, <256 x double> %1, i64 %2, i64 %3, i32 256)
  ret void
}

; Function Attrs: nounwind writeonly
declare void @llvm.ve.vl.vscuot.vvssl(<256 x double>, <256 x double>, i64, i64, i32)

; Function Attrs: nounwind writeonly
define fastcc void @vscuot_vvssl_imm_1(<256 x double> %0, <256 x double> %1, i64 %2) {
; CHECK-LABEL: vscuot_vvssl_imm_1:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vscu.ot %v0, %v1, %s0, 0
; CHECK-NEXT:    b.l.t (, %s10)
  tail call void @llvm.ve.vl.vscuot.vvssl(<256 x double> %0, <256 x double> %1, i64 %2, i64 0, i32 256)
  ret void
}

; Function Attrs: nounwind writeonly
define fastcc void @vscuot_vvssl_imm_2(<256 x double> %0, <256 x double> %1, i64 %2) {
; CHECK-LABEL: vscuot_vvssl_imm_2:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vscu.ot %v0, %v1, 8, %s0
; CHECK-NEXT:    b.l.t (, %s10)
  tail call void @llvm.ve.vl.vscuot.vvssl(<256 x double> %0, <256 x double> %1, i64 8, i64 %2, i32 256)
  ret void
}

; Function Attrs: nounwind writeonly
define fastcc void @vscuot_vvssl_imm_3(<256 x double> %0, <256 x double> %1) {
; CHECK-LABEL: vscuot_vvssl_imm_3:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vscu.ot %v0, %v1, 8, 0
; CHECK-NEXT:    b.l.t (, %s10)
  tail call void @llvm.ve.vl.vscuot.vvssl(<256 x double> %0, <256 x double> %1, i64 8, i64 0, i32 256)
  ret void
}

; Function Attrs: nounwind writeonly
define fastcc void @vscuot_vvssml(<256 x double> %0, <256 x double> %1, i64 %2, i64 %3, <256 x i1> %4) {
; CHECK-LABEL: vscuot_vvssml:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s2, 256
; CHECK-NEXT:    lvl %s2
; CHECK-NEXT:    vscu.ot %v0, %v1, %s0, %s1, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  tail call void @llvm.ve.vl.vscuot.vvssml(<256 x double> %0, <256 x double> %1, i64 %2, i64 %3, <256 x i1> %4, i32 256)
  ret void
}

; Function Attrs: nounwind writeonly
declare void @llvm.ve.vl.vscuot.vvssml(<256 x double>, <256 x double>, i64, i64, <256 x i1>, i32)

; Function Attrs: nounwind writeonly
define fastcc void @vscuot_vvssml_imm_1(<256 x double> %0, <256 x double> %1, i64 %2, <256 x i1> %3) {
; CHECK-LABEL: vscuot_vvssml_imm_1:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vscu.ot %v0, %v1, %s0, 0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  tail call void @llvm.ve.vl.vscuot.vvssml(<256 x double> %0, <256 x double> %1, i64 %2, i64 0, <256 x i1> %3, i32 256)
  ret void
}

; Function Attrs: nounwind writeonly
define fastcc void @vscuot_vvssml_imm_2(<256 x double> %0, <256 x double> %1, i64 %2, <256 x i1> %3) {
; CHECK-LABEL: vscuot_vvssml_imm_2:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vscu.ot %v0, %v1, 8, %s0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  tail call void @llvm.ve.vl.vscuot.vvssml(<256 x double> %0, <256 x double> %1, i64 8, i64 %2, <256 x i1> %3, i32 256)
  ret void
}

; Function Attrs: nounwind writeonly
define fastcc void @vscuot_vvssml_imm_3(<256 x double> %0, <256 x double> %1, <256 x i1> %2) {
; CHECK-LABEL: vscuot_vvssml_imm_3:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vscu.ot %v0, %v1, 8, 0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  tail call void @llvm.ve.vl.vscuot.vvssml(<256 x double> %0, <256 x double> %1, i64 8, i64 0, <256 x i1> %2, i32 256)
  ret void
}

; Function Attrs: nounwind writeonly
define fastcc void @vscuot_vvssl_no_imm_1(<256 x double> %0, <256 x double> %1, i64 %2) {
; CHECK-LABEL: vscuot_vvssl_no_imm_1:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    or %s2, 8, (0)1
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vscu.ot %v0, %v1, %s0, %s2
; CHECK-NEXT:    b.l.t (, %s10)
  tail call void @llvm.ve.vl.vscuot.vvssl(<256 x double> %0, <256 x double> %1, i64 %2, i64 8, i32 256)
  ret void
}

; Function Attrs: nounwind writeonly
define fastcc void @vscuncot_vvssl(<256 x double> %0, <256 x double> %1, i64 %2, i64 %3) {
; CHECK-LABEL: vscuncot_vvssl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s2, 256
; CHECK-NEXT:    lvl %s2
; CHECK-NEXT:    vscu.nc.ot %v0, %v1, %s0, %s1
; CHECK-NEXT:    b.l.t (, %s10)
  tail call void @llvm.ve.vl.vscuncot.vvssl(<256 x double> %0, <256 x double> %1, i64 %2, i64 %3, i32 256)
  ret void
}

; Function Attrs: nounwind writeonly
declare void @llvm.ve.vl.vscuncot.vvssl(<256 x double>, <256 x double>, i64, i64, i32)

; Function Attrs: nounwind writeonly
define fastcc void @vscuncot_vvssl_imm_1(<256 x double> %0, <256 x double> %1, i64 %2) {
; CHECK-LABEL: vscuncot_vvssl_imm_1:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vscu.nc.ot %v0, %v1, %s0, 0
; CHECK-NEXT:    b.l.t (, %s10)
  tail call void @llvm.ve.vl.vscuncot.vvssl(<256 x double> %0, <256 x double> %1, i64 %2, i64 0, i32 256)
  ret void
}

; Function Attrs: nounwind writeonly
define fastcc void @vscuncot_vvssl_imm_2(<256 x double> %0, <256 x double> %1, i64 %2) {
; CHECK-LABEL: vscuncot_vvssl_imm_2:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vscu.nc.ot %v0, %v1, 8, %s0
; CHECK-NEXT:    b.l.t (, %s10)
  tail call void @llvm.ve.vl.vscuncot.vvssl(<256 x double> %0, <256 x double> %1, i64 8, i64 %2, i32 256)
  ret void
}

; Function Attrs: nounwind writeonly
define fastcc void @vscuncot_vvssl_imm_3(<256 x double> %0, <256 x double> %1) {
; CHECK-LABEL: vscuncot_vvssl_imm_3:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vscu.nc.ot %v0, %v1, 8, 0
; CHECK-NEXT:    b.l.t (, %s10)
  tail call void @llvm.ve.vl.vscuncot.vvssl(<256 x double> %0, <256 x double> %1, i64 8, i64 0, i32 256)
  ret void
}

; Function Attrs: nounwind writeonly
define fastcc void @vscuncot_vvssml(<256 x double> %0, <256 x double> %1, i64 %2, i64 %3, <256 x i1> %4) {
; CHECK-LABEL: vscuncot_vvssml:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s2, 256
; CHECK-NEXT:    lvl %s2
; CHECK-NEXT:    vscu.nc.ot %v0, %v1, %s0, %s1, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  tail call void @llvm.ve.vl.vscuncot.vvssml(<256 x double> %0, <256 x double> %1, i64 %2, i64 %3, <256 x i1> %4, i32 256)
  ret void
}

; Function Attrs: nounwind writeonly
declare void @llvm.ve.vl.vscuncot.vvssml(<256 x double>, <256 x double>, i64, i64, <256 x i1>, i32)

; Function Attrs: nounwind writeonly
define fastcc void @vscuncot_vvssml_imm_1(<256 x double> %0, <256 x double> %1, i64 %2, <256 x i1> %3) {
; CHECK-LABEL: vscuncot_vvssml_imm_1:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vscu.nc.ot %v0, %v1, %s0, 0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  tail call void @llvm.ve.vl.vscuncot.vvssml(<256 x double> %0, <256 x double> %1, i64 %2, i64 0, <256 x i1> %3, i32 256)
  ret void
}

; Function Attrs: nounwind writeonly
define fastcc void @vscuncot_vvssml_imm_2(<256 x double> %0, <256 x double> %1, i64 %2, <256 x i1> %3) {
; CHECK-LABEL: vscuncot_vvssml_imm_2:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vscu.nc.ot %v0, %v1, 8, %s0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  tail call void @llvm.ve.vl.vscuncot.vvssml(<256 x double> %0, <256 x double> %1, i64 8, i64 %2, <256 x i1> %3, i32 256)
  ret void
}

; Function Attrs: nounwind writeonly
define fastcc void @vscuncot_vvssml_imm_3(<256 x double> %0, <256 x double> %1, <256 x i1> %2) {
; CHECK-LABEL: vscuncot_vvssml_imm_3:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vscu.nc.ot %v0, %v1, 8, 0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  tail call void @llvm.ve.vl.vscuncot.vvssml(<256 x double> %0, <256 x double> %1, i64 8, i64 0, <256 x i1> %2, i32 256)
  ret void
}

; Function Attrs: nounwind writeonly
define fastcc void @vscuncot_vvssl_no_imm_1(<256 x double> %0, <256 x double> %1, i64 %2) {
; CHECK-LABEL: vscuncot_vvssl_no_imm_1:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    or %s2, 8, (0)1
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vscu.nc.ot %v0, %v1, %s0, %s2
; CHECK-NEXT:    b.l.t (, %s10)
  tail call void @llvm.ve.vl.vscuncot.vvssl(<256 x double> %0, <256 x double> %1, i64 %2, i64 8, i32 256)
  ret void
}

; Function Attrs: nounwind writeonly
define fastcc void @vscl_vvssl(<256 x double> %0, <256 x double> %1, i64 %2, i64 %3) {
; CHECK-LABEL: vscl_vvssl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s2, 256
; CHECK-NEXT:    lvl %s2
; CHECK-NEXT:    vscl %v0, %v1, %s0, %s1
; CHECK-NEXT:    b.l.t (, %s10)
  tail call void @llvm.ve.vl.vscl.vvssl(<256 x double> %0, <256 x double> %1, i64 %2, i64 %3, i32 256)
  ret void
}

; Function Attrs: nounwind writeonly
declare void @llvm.ve.vl.vscl.vvssl(<256 x double>, <256 x double>, i64, i64, i32)

; Function Attrs: nounwind writeonly
define fastcc void @vscl_vvssl_imm_1(<256 x double> %0, <256 x double> %1, i64 %2) {
; CHECK-LABEL: vscl_vvssl_imm_1:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vscl %v0, %v1, %s0, 0
; CHECK-NEXT:    b.l.t (, %s10)
  tail call void @llvm.ve.vl.vscl.vvssl(<256 x double> %0, <256 x double> %1, i64 %2, i64 0, i32 256)
  ret void
}

; Function Attrs: nounwind writeonly
define fastcc void @vscl_vvssl_imm_2(<256 x double> %0, <256 x double> %1, i64 %2) {
; CHECK-LABEL: vscl_vvssl_imm_2:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vscl %v0, %v1, 8, %s0
; CHECK-NEXT:    b.l.t (, %s10)
  tail call void @llvm.ve.vl.vscl.vvssl(<256 x double> %0, <256 x double> %1, i64 8, i64 %2, i32 256)
  ret void
}

; Function Attrs: nounwind writeonly
define fastcc void @vscl_vvssl_imm_3(<256 x double> %0, <256 x double> %1) {
; CHECK-LABEL: vscl_vvssl_imm_3:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vscl %v0, %v1, 8, 0
; CHECK-NEXT:    b.l.t (, %s10)
  tail call void @llvm.ve.vl.vscl.vvssl(<256 x double> %0, <256 x double> %1, i64 8, i64 0, i32 256)
  ret void
}

; Function Attrs: nounwind writeonly
define fastcc void @vscl_vvssml(<256 x double> %0, <256 x double> %1, i64 %2, i64 %3, <256 x i1> %4) {
; CHECK-LABEL: vscl_vvssml:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s2, 256
; CHECK-NEXT:    lvl %s2
; CHECK-NEXT:    vscl %v0, %v1, %s0, %s1, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  tail call void @llvm.ve.vl.vscl.vvssml(<256 x double> %0, <256 x double> %1, i64 %2, i64 %3, <256 x i1> %4, i32 256)
  ret void
}

; Function Attrs: nounwind writeonly
declare void @llvm.ve.vl.vscl.vvssml(<256 x double>, <256 x double>, i64, i64, <256 x i1>, i32)

; Function Attrs: nounwind writeonly
define fastcc void @vscl_vvssml_imm_1(<256 x double> %0, <256 x double> %1, i64 %2, <256 x i1> %3) {
; CHECK-LABEL: vscl_vvssml_imm_1:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vscl %v0, %v1, %s0, 0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  tail call void @llvm.ve.vl.vscl.vvssml(<256 x double> %0, <256 x double> %1, i64 %2, i64 0, <256 x i1> %3, i32 256)
  ret void
}

; Function Attrs: nounwind writeonly
define fastcc void @vscl_vvssml_imm_2(<256 x double> %0, <256 x double> %1, i64 %2, <256 x i1> %3) {
; CHECK-LABEL: vscl_vvssml_imm_2:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vscl %v0, %v1, 8, %s0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  tail call void @llvm.ve.vl.vscl.vvssml(<256 x double> %0, <256 x double> %1, i64 8, i64 %2, <256 x i1> %3, i32 256)
  ret void
}

; Function Attrs: nounwind writeonly
define fastcc void @vscl_vvssml_imm_3(<256 x double> %0, <256 x double> %1, <256 x i1> %2) {
; CHECK-LABEL: vscl_vvssml_imm_3:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vscl %v0, %v1, 8, 0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  tail call void @llvm.ve.vl.vscl.vvssml(<256 x double> %0, <256 x double> %1, i64 8, i64 0, <256 x i1> %2, i32 256)
  ret void
}

; Function Attrs: nounwind writeonly
define fastcc void @vscl_vvssl_no_imm_1(<256 x double> %0, <256 x double> %1, i64 %2) {
; CHECK-LABEL: vscl_vvssl_no_imm_1:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    or %s2, 8, (0)1
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vscl %v0, %v1, %s0, %s2
; CHECK-NEXT:    b.l.t (, %s10)
  tail call void @llvm.ve.vl.vscl.vvssl(<256 x double> %0, <256 x double> %1, i64 %2, i64 8, i32 256)
  ret void
}

; Function Attrs: nounwind writeonly
define fastcc void @vsclnc_vvssl(<256 x double> %0, <256 x double> %1, i64 %2, i64 %3) {
; CHECK-LABEL: vsclnc_vvssl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s2, 256
; CHECK-NEXT:    lvl %s2
; CHECK-NEXT:    vscl.nc %v0, %v1, %s0, %s1
; CHECK-NEXT:    b.l.t (, %s10)
  tail call void @llvm.ve.vl.vsclnc.vvssl(<256 x double> %0, <256 x double> %1, i64 %2, i64 %3, i32 256)
  ret void
}

; Function Attrs: nounwind writeonly
declare void @llvm.ve.vl.vsclnc.vvssl(<256 x double>, <256 x double>, i64, i64, i32)

; Function Attrs: nounwind writeonly
define fastcc void @vsclnc_vvssl_imm_1(<256 x double> %0, <256 x double> %1, i64 %2) {
; CHECK-LABEL: vsclnc_vvssl_imm_1:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vscl.nc %v0, %v1, %s0, 0
; CHECK-NEXT:    b.l.t (, %s10)
  tail call void @llvm.ve.vl.vsclnc.vvssl(<256 x double> %0, <256 x double> %1, i64 %2, i64 0, i32 256)
  ret void
}

; Function Attrs: nounwind writeonly
define fastcc void @vsclnc_vvssl_imm_2(<256 x double> %0, <256 x double> %1, i64 %2) {
; CHECK-LABEL: vsclnc_vvssl_imm_2:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vscl.nc %v0, %v1, 8, %s0
; CHECK-NEXT:    b.l.t (, %s10)
  tail call void @llvm.ve.vl.vsclnc.vvssl(<256 x double> %0, <256 x double> %1, i64 8, i64 %2, i32 256)
  ret void
}

; Function Attrs: nounwind writeonly
define fastcc void @vsclnc_vvssl_imm_3(<256 x double> %0, <256 x double> %1) {
; CHECK-LABEL: vsclnc_vvssl_imm_3:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vscl.nc %v0, %v1, 8, 0
; CHECK-NEXT:    b.l.t (, %s10)
  tail call void @llvm.ve.vl.vsclnc.vvssl(<256 x double> %0, <256 x double> %1, i64 8, i64 0, i32 256)
  ret void
}

; Function Attrs: nounwind writeonly
define fastcc void @vsclnc_vvssml(<256 x double> %0, <256 x double> %1, i64 %2, i64 %3, <256 x i1> %4) {
; CHECK-LABEL: vsclnc_vvssml:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s2, 256
; CHECK-NEXT:    lvl %s2
; CHECK-NEXT:    vscl.nc %v0, %v1, %s0, %s1, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  tail call void @llvm.ve.vl.vsclnc.vvssml(<256 x double> %0, <256 x double> %1, i64 %2, i64 %3, <256 x i1> %4, i32 256)
  ret void
}

; Function Attrs: nounwind writeonly
declare void @llvm.ve.vl.vsclnc.vvssml(<256 x double>, <256 x double>, i64, i64, <256 x i1>, i32)

; Function Attrs: nounwind writeonly
define fastcc void @vsclnc_vvssml_imm_1(<256 x double> %0, <256 x double> %1, i64 %2, <256 x i1> %3) {
; CHECK-LABEL: vsclnc_vvssml_imm_1:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vscl.nc %v0, %v1, %s0, 0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  tail call void @llvm.ve.vl.vsclnc.vvssml(<256 x double> %0, <256 x double> %1, i64 %2, i64 0, <256 x i1> %3, i32 256)
  ret void
}

; Function Attrs: nounwind writeonly
define fastcc void @vsclnc_vvssml_imm_2(<256 x double> %0, <256 x double> %1, i64 %2, <256 x i1> %3) {
; CHECK-LABEL: vsclnc_vvssml_imm_2:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vscl.nc %v0, %v1, 8, %s0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  tail call void @llvm.ve.vl.vsclnc.vvssml(<256 x double> %0, <256 x double> %1, i64 8, i64 %2, <256 x i1> %3, i32 256)
  ret void
}

; Function Attrs: nounwind writeonly
define fastcc void @vsclnc_vvssml_imm_3(<256 x double> %0, <256 x double> %1, <256 x i1> %2) {
; CHECK-LABEL: vsclnc_vvssml_imm_3:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vscl.nc %v0, %v1, 8, 0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  tail call void @llvm.ve.vl.vsclnc.vvssml(<256 x double> %0, <256 x double> %1, i64 8, i64 0, <256 x i1> %2, i32 256)
  ret void
}

; Function Attrs: nounwind writeonly
define fastcc void @vsclnc_vvssl_no_imm_1(<256 x double> %0, <256 x double> %1, i64 %2) {
; CHECK-LABEL: vsclnc_vvssl_no_imm_1:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    or %s2, 8, (0)1
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vscl.nc %v0, %v1, %s0, %s2
; CHECK-NEXT:    b.l.t (, %s10)
  tail call void @llvm.ve.vl.vsclnc.vvssl(<256 x double> %0, <256 x double> %1, i64 %2, i64 8, i32 256)
  ret void
}

; Function Attrs: nounwind writeonly
define fastcc void @vsclot_vvssl(<256 x double> %0, <256 x double> %1, i64 %2, i64 %3) {
; CHECK-LABEL: vsclot_vvssl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s2, 256
; CHECK-NEXT:    lvl %s2
; CHECK-NEXT:    vscl.ot %v0, %v1, %s0, %s1
; CHECK-NEXT:    b.l.t (, %s10)
  tail call void @llvm.ve.vl.vsclot.vvssl(<256 x double> %0, <256 x double> %1, i64 %2, i64 %3, i32 256)
  ret void
}

; Function Attrs: nounwind writeonly
declare void @llvm.ve.vl.vsclot.vvssl(<256 x double>, <256 x double>, i64, i64, i32)

; Function Attrs: nounwind writeonly
define fastcc void @vsclot_vvssl_imm_1(<256 x double> %0, <256 x double> %1, i64 %2) {
; CHECK-LABEL: vsclot_vvssl_imm_1:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vscl.ot %v0, %v1, %s0, 0
; CHECK-NEXT:    b.l.t (, %s10)
  tail call void @llvm.ve.vl.vsclot.vvssl(<256 x double> %0, <256 x double> %1, i64 %2, i64 0, i32 256)
  ret void
}

; Function Attrs: nounwind writeonly
define fastcc void @vsclot_vvssl_imm_2(<256 x double> %0, <256 x double> %1, i64 %2) {
; CHECK-LABEL: vsclot_vvssl_imm_2:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vscl.ot %v0, %v1, 8, %s0
; CHECK-NEXT:    b.l.t (, %s10)
  tail call void @llvm.ve.vl.vsclot.vvssl(<256 x double> %0, <256 x double> %1, i64 8, i64 %2, i32 256)
  ret void
}

; Function Attrs: nounwind writeonly
define fastcc void @vsclot_vvssl_imm_3(<256 x double> %0, <256 x double> %1) {
; CHECK-LABEL: vsclot_vvssl_imm_3:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vscl.ot %v0, %v1, 8, 0
; CHECK-NEXT:    b.l.t (, %s10)
  tail call void @llvm.ve.vl.vsclot.vvssl(<256 x double> %0, <256 x double> %1, i64 8, i64 0, i32 256)
  ret void
}

; Function Attrs: nounwind writeonly
define fastcc void @vsclot_vvssml(<256 x double> %0, <256 x double> %1, i64 %2, i64 %3, <256 x i1> %4) {
; CHECK-LABEL: vsclot_vvssml:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s2, 256
; CHECK-NEXT:    lvl %s2
; CHECK-NEXT:    vscl.ot %v0, %v1, %s0, %s1, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  tail call void @llvm.ve.vl.vsclot.vvssml(<256 x double> %0, <256 x double> %1, i64 %2, i64 %3, <256 x i1> %4, i32 256)
  ret void
}

; Function Attrs: nounwind writeonly
declare void @llvm.ve.vl.vsclot.vvssml(<256 x double>, <256 x double>, i64, i64, <256 x i1>, i32)

; Function Attrs: nounwind writeonly
define fastcc void @vsclot_vvssml_imm_1(<256 x double> %0, <256 x double> %1, i64 %2, <256 x i1> %3) {
; CHECK-LABEL: vsclot_vvssml_imm_1:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vscl.ot %v0, %v1, %s0, 0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  tail call void @llvm.ve.vl.vsclot.vvssml(<256 x double> %0, <256 x double> %1, i64 %2, i64 0, <256 x i1> %3, i32 256)
  ret void
}

; Function Attrs: nounwind writeonly
define fastcc void @vsclot_vvssml_imm_2(<256 x double> %0, <256 x double> %1, i64 %2, <256 x i1> %3) {
; CHECK-LABEL: vsclot_vvssml_imm_2:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vscl.ot %v0, %v1, 8, %s0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  tail call void @llvm.ve.vl.vsclot.vvssml(<256 x double> %0, <256 x double> %1, i64 8, i64 %2, <256 x i1> %3, i32 256)
  ret void
}

; Function Attrs: nounwind writeonly
define fastcc void @vsclot_vvssml_imm_3(<256 x double> %0, <256 x double> %1, <256 x i1> %2) {
; CHECK-LABEL: vsclot_vvssml_imm_3:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vscl.ot %v0, %v1, 8, 0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  tail call void @llvm.ve.vl.vsclot.vvssml(<256 x double> %0, <256 x double> %1, i64 8, i64 0, <256 x i1> %2, i32 256)
  ret void
}

; Function Attrs: nounwind writeonly
define fastcc void @vsclot_vvssl_no_imm_1(<256 x double> %0, <256 x double> %1, i64 %2) {
; CHECK-LABEL: vsclot_vvssl_no_imm_1:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    or %s2, 8, (0)1
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vscl.ot %v0, %v1, %s0, %s2
; CHECK-NEXT:    b.l.t (, %s10)
  tail call void @llvm.ve.vl.vsclot.vvssl(<256 x double> %0, <256 x double> %1, i64 %2, i64 8, i32 256)
  ret void
}

; Function Attrs: nounwind writeonly
define fastcc void @vsclncot_vvssl(<256 x double> %0, <256 x double> %1, i64 %2, i64 %3) {
; CHECK-LABEL: vsclncot_vvssl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s2, 256
; CHECK-NEXT:    lvl %s2
; CHECK-NEXT:    vscl.nc.ot %v0, %v1, %s0, %s1
; CHECK-NEXT:    b.l.t (, %s10)
  tail call void @llvm.ve.vl.vsclncot.vvssl(<256 x double> %0, <256 x double> %1, i64 %2, i64 %3, i32 256)
  ret void
}

; Function Attrs: nounwind writeonly
declare void @llvm.ve.vl.vsclncot.vvssl(<256 x double>, <256 x double>, i64, i64, i32)

; Function Attrs: nounwind writeonly
define fastcc void @vsclncot_vvssl_imm_1(<256 x double> %0, <256 x double> %1, i64 %2) {
; CHECK-LABEL: vsclncot_vvssl_imm_1:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vscl.nc.ot %v0, %v1, %s0, 0
; CHECK-NEXT:    b.l.t (, %s10)
  tail call void @llvm.ve.vl.vsclncot.vvssl(<256 x double> %0, <256 x double> %1, i64 %2, i64 0, i32 256)
  ret void
}

; Function Attrs: nounwind writeonly
define fastcc void @vsclncot_vvssl_imm_2(<256 x double> %0, <256 x double> %1, i64 %2) {
; CHECK-LABEL: vsclncot_vvssl_imm_2:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vscl.nc.ot %v0, %v1, 8, %s0
; CHECK-NEXT:    b.l.t (, %s10)
  tail call void @llvm.ve.vl.vsclncot.vvssl(<256 x double> %0, <256 x double> %1, i64 8, i64 %2, i32 256)
  ret void
}

; Function Attrs: nounwind writeonly
define fastcc void @vsclncot_vvssl_imm_3(<256 x double> %0, <256 x double> %1) {
; CHECK-LABEL: vsclncot_vvssl_imm_3:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vscl.nc.ot %v0, %v1, 8, 0
; CHECK-NEXT:    b.l.t (, %s10)
  tail call void @llvm.ve.vl.vsclncot.vvssl(<256 x double> %0, <256 x double> %1, i64 8, i64 0, i32 256)
  ret void
}

; Function Attrs: nounwind writeonly
define fastcc void @vsclncot_vvssml(<256 x double> %0, <256 x double> %1, i64 %2, i64 %3, <256 x i1> %4) {
; CHECK-LABEL: vsclncot_vvssml:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s2, 256
; CHECK-NEXT:    lvl %s2
; CHECK-NEXT:    vscl.nc.ot %v0, %v1, %s0, %s1, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  tail call void @llvm.ve.vl.vsclncot.vvssml(<256 x double> %0, <256 x double> %1, i64 %2, i64 %3, <256 x i1> %4, i32 256)
  ret void
}

; Function Attrs: nounwind writeonly
declare void @llvm.ve.vl.vsclncot.vvssml(<256 x double>, <256 x double>, i64, i64, <256 x i1>, i32)

; Function Attrs: nounwind writeonly
define fastcc void @vsclncot_vvssml_imm_1(<256 x double> %0, <256 x double> %1, i64 %2, <256 x i1> %3) {
; CHECK-LABEL: vsclncot_vvssml_imm_1:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vscl.nc.ot %v0, %v1, %s0, 0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  tail call void @llvm.ve.vl.vsclncot.vvssml(<256 x double> %0, <256 x double> %1, i64 %2, i64 0, <256 x i1> %3, i32 256)
  ret void
}

; Function Attrs: nounwind writeonly
define fastcc void @vsclncot_vvssml_imm_2(<256 x double> %0, <256 x double> %1, i64 %2, <256 x i1> %3) {
; CHECK-LABEL: vsclncot_vvssml_imm_2:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vscl.nc.ot %v0, %v1, 8, %s0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  tail call void @llvm.ve.vl.vsclncot.vvssml(<256 x double> %0, <256 x double> %1, i64 8, i64 %2, <256 x i1> %3, i32 256)
  ret void
}

; Function Attrs: nounwind writeonly
define fastcc void @vsclncot_vvssml_imm_3(<256 x double> %0, <256 x double> %1, <256 x i1> %2) {
; CHECK-LABEL: vsclncot_vvssml_imm_3:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vscl.nc.ot %v0, %v1, 8, 0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  tail call void @llvm.ve.vl.vsclncot.vvssml(<256 x double> %0, <256 x double> %1, i64 8, i64 0, <256 x i1> %2, i32 256)
  ret void
}

; Function Attrs: nounwind writeonly
define fastcc void @vsclncot_vvssl_no_imm_1(<256 x double> %0, <256 x double> %1, i64 %2) {
; CHECK-LABEL: vsclncot_vvssl_no_imm_1:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    or %s2, 8, (0)1
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vscl.nc.ot %v0, %v1, %s0, %s2
; CHECK-NEXT:    b.l.t (, %s10)
  tail call void @llvm.ve.vl.vsclncot.vvssl(<256 x double> %0, <256 x double> %1, i64 %2, i64 8, i32 256)
  ret void
}
