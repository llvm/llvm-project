; RUN: llc < %s -mtriple=ve -mattr=+vpu | FileCheck %s

;;; Test vector bit reverse intrinsic instructions
;;;
;;; Note:
;;;   We test VBRV*vl, VBRV*vl_v, VBRV*vml_v, PVBRV*vl, PVBRV*vl_v, PVBRV*vml_v instructions.

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vbrv_vvl(<256 x double> %0) {
; CHECK-LABEL: vbrv_vvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vbrv %v0, %v0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call fast <256 x double> @llvm.ve.vl.vbrv.vvl(<256 x double> %0, i32 256)
  ret <256 x double> %2
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vbrv.vvl(<256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vbrv_vvvl(<256 x double> %0, <256 x double> %1) {
; CHECK-LABEL: vbrv_vvvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 128
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vbrv %v1, %v0
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.vbrv.vvvl(<256 x double> %0, <256 x double> %1, i32 128)
  ret <256 x double> %3
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vbrv.vvvl(<256 x double>, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vbrv_vvmvl(<256 x double> %0, <256 x i1> %1, <256 x double> %2) {
; CHECK-LABEL: vbrv_vvmvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 128
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vbrv %v1, %v0, %vm1
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v1
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = tail call fast <256 x double> @llvm.ve.vl.vbrv.vvmvl(<256 x double> %0, <256 x i1> %1, <256 x double> %2, i32 128)
  ret <256 x double> %4
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vbrv.vvmvl(<256 x double>, <256 x i1>, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @pvbrvlo_vvl(<256 x double> %0) {
; CHECK-LABEL: pvbrvlo_vvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    pvbrv.lo %v0, %v0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call fast <256 x double> @llvm.ve.vl.pvbrvlo.vvl(<256 x double> %0, i32 256)
  ret <256 x double> %2
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.pvbrvlo.vvl(<256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @pvbrvlo_vvvl(<256 x double> %0, <256 x double> %1) {
; CHECK-LABEL: pvbrvlo_vvvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 128
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    pvbrv.lo %v1, %v0
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.pvbrvlo.vvvl(<256 x double> %0, <256 x double> %1, i32 128)
  ret <256 x double> %3
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.pvbrvlo.vvvl(<256 x double>, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @pvbrvlo_vvmvl(<256 x double> %0, <256 x i1> %1, <256 x double> %2) {
; CHECK-LABEL: pvbrvlo_vvmvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 128
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    pvbrv.lo %v1, %v0, %vm1
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v1
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = tail call fast <256 x double> @llvm.ve.vl.pvbrvlo.vvmvl(<256 x double> %0, <256 x i1> %1, <256 x double> %2, i32 128)
  ret <256 x double> %4
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.pvbrvlo.vvmvl(<256 x double>, <256 x i1>, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @pvbrvup_vvl(<256 x double> %0) {
; CHECK-LABEL: pvbrvup_vvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    pvbrv.up %v0, %v0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call fast <256 x double> @llvm.ve.vl.pvbrvup.vvl(<256 x double> %0, i32 256)
  ret <256 x double> %2
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.pvbrvup.vvl(<256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @pvbrvup_vvvl(<256 x double> %0, <256 x double> %1) {
; CHECK-LABEL: pvbrvup_vvvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 128
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    pvbrv.up %v1, %v0
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.pvbrvup.vvvl(<256 x double> %0, <256 x double> %1, i32 128)
  ret <256 x double> %3
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.pvbrvup.vvvl(<256 x double>, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @pvbrvup_vvmvl(<256 x double> %0, <256 x i1> %1, <256 x double> %2) {
; CHECK-LABEL: pvbrvup_vvmvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 128
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    pvbrv.up %v1, %v0, %vm1
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v1
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = tail call fast <256 x double> @llvm.ve.vl.pvbrvup.vvmvl(<256 x double> %0, <256 x i1> %1, <256 x double> %2, i32 128)
  ret <256 x double> %4
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.pvbrvup.vvmvl(<256 x double>, <256 x i1>, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @pvbrv_vvl(<256 x double> %0) {
; CHECK-LABEL: pvbrv_vvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    pvbrv %v0, %v0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call fast <256 x double> @llvm.ve.vl.pvbrv.vvl(<256 x double> %0, i32 256)
  ret <256 x double> %2
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.pvbrv.vvl(<256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @pvbrv_vvvl(<256 x double> %0, <256 x double> %1) {
; CHECK-LABEL: pvbrv_vvvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 128
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    pvbrv %v1, %v0
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.pvbrv.vvvl(<256 x double> %0, <256 x double> %1, i32 128)
  ret <256 x double> %3
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.pvbrv.vvvl(<256 x double>, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @pvbrv_vvMvl(<256 x double> %0, <512 x i1> %1, <256 x double> %2) {
; CHECK-LABEL: pvbrv_vvMvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 128
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    pvbrv %v1, %v0, %vm2
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v1
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = tail call fast <256 x double> @llvm.ve.vl.pvbrv.vvMvl(<256 x double> %0, <512 x i1> %1, <256 x double> %2, i32 128)
  ret <256 x double> %4
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.pvbrv.vvMvl(<256 x double>, <512 x i1>, <256 x double>, i32)
