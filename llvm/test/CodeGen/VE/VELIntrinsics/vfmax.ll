; RUN: llc < %s -mtriple=ve -mattr=+vpu | FileCheck %s

;;; Test vector floating compare and select maximum intrinsic instructions
;;;
;;; Note:
;;;   We test VFMAX*vvl, VFMAX*vvl_v, VFMAX*rvl, VFMAX*rvl_v, VFMAX*vvml_v,
;;;   VFMAX*rvml_v, PVFMAX*vvl, PVFMAX*vvl_v, PVFMAX*rvl, PVFMAX*rvl_v,
;;;   PVFMAX*vvml_v, and PVFMAX*rvml_v instructions.

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vfmaxd_vvvl(<256 x double> %0, <256 x double> %1) {
; CHECK-LABEL: vfmaxd_vvvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vfmax.d %v0, %v0, %v1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.vfmaxd.vvvl(<256 x double> %0, <256 x double> %1, i32 256)
  ret <256 x double> %3
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vfmaxd.vvvl(<256 x double>, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vfmaxd_vvvvl(<256 x double> %0, <256 x double> %1, <256 x double> %2) {
; CHECK-LABEL: vfmaxd_vvvvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 128
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vfmax.d %v2, %v0, %v1
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v2
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = tail call fast <256 x double> @llvm.ve.vl.vfmaxd.vvvvl(<256 x double> %0, <256 x double> %1, <256 x double> %2, i32 128)
  ret <256 x double> %4
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vfmaxd.vvvvl(<256 x double>, <256 x double>, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vfmaxd_vsvl(double %0, <256 x double> %1) {
; CHECK-LABEL: vfmaxd_vsvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vfmax.d %v0, %s0, %v0
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.vfmaxd.vsvl(double %0, <256 x double> %1, i32 256)
  ret <256 x double> %3
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vfmaxd.vsvl(double, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vfmaxd_vsvvl(double %0, <256 x double> %1, <256 x double> %2) {
; CHECK-LABEL: vfmaxd_vsvvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 128
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vfmax.d %v1, %s0, %v0
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v1
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = tail call fast <256 x double> @llvm.ve.vl.vfmaxd.vsvvl(double %0, <256 x double> %1, <256 x double> %2, i32 128)
  ret <256 x double> %4
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vfmaxd.vsvvl(double, <256 x double>, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vfmaxd_vvvmvl(<256 x double> %0, <256 x double> %1, <256 x i1> %2, <256 x double> %3) {
; CHECK-LABEL: vfmaxd_vvvmvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 128
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vfmax.d %v2, %v0, %v1, %vm1
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v2
; CHECK-NEXT:    b.l.t (, %s10)
  %5 = tail call fast <256 x double> @llvm.ve.vl.vfmaxd.vvvmvl(<256 x double> %0, <256 x double> %1, <256 x i1> %2, <256 x double> %3, i32 128)
  ret <256 x double> %5
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vfmaxd.vvvmvl(<256 x double>, <256 x double>, <256 x i1>, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vfmaxd_vsvmvl(double %0, <256 x double> %1, <256 x i1> %2, <256 x double> %3) {
; CHECK-LABEL: vfmaxd_vsvmvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 128
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vfmax.d %v1, %s0, %v0, %vm1
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v1
; CHECK-NEXT:    b.l.t (, %s10)
  %5 = tail call fast <256 x double> @llvm.ve.vl.vfmaxd.vsvmvl(double %0, <256 x double> %1, <256 x i1> %2, <256 x double> %3, i32 128)
  ret <256 x double> %5
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vfmaxd.vsvmvl(double, <256 x double>, <256 x i1>, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vfmaxs_vvvl(<256 x double> %0, <256 x double> %1) {
; CHECK-LABEL: vfmaxs_vvvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vfmax.s %v0, %v0, %v1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.vfmaxs.vvvl(<256 x double> %0, <256 x double> %1, i32 256)
  ret <256 x double> %3
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vfmaxs.vvvl(<256 x double>, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vfmaxs_vvvvl(<256 x double> %0, <256 x double> %1, <256 x double> %2) {
; CHECK-LABEL: vfmaxs_vvvvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 128
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vfmax.s %v2, %v0, %v1
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v2
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = tail call fast <256 x double> @llvm.ve.vl.vfmaxs.vvvvl(<256 x double> %0, <256 x double> %1, <256 x double> %2, i32 128)
  ret <256 x double> %4
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vfmaxs.vvvvl(<256 x double>, <256 x double>, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vfmaxs_vsvl(float %0, <256 x double> %1) {
; CHECK-LABEL: vfmaxs_vsvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vfmax.s %v0, %s0, %v0
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.vfmaxs.vsvl(float %0, <256 x double> %1, i32 256)
  ret <256 x double> %3
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vfmaxs.vsvl(float, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vfmaxs_vsvvl(float %0, <256 x double> %1, <256 x double> %2) {
; CHECK-LABEL: vfmaxs_vsvvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 128
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vfmax.s %v1, %s0, %v0
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v1
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = tail call fast <256 x double> @llvm.ve.vl.vfmaxs.vsvvl(float %0, <256 x double> %1, <256 x double> %2, i32 128)
  ret <256 x double> %4
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vfmaxs.vsvvl(float, <256 x double>, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vfmaxs_vvvmvl(<256 x double> %0, <256 x double> %1, <256 x i1> %2, <256 x double> %3) {
; CHECK-LABEL: vfmaxs_vvvmvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 128
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vfmax.s %v2, %v0, %v1, %vm1
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v2
; CHECK-NEXT:    b.l.t (, %s10)
  %5 = tail call fast <256 x double> @llvm.ve.vl.vfmaxs.vvvmvl(<256 x double> %0, <256 x double> %1, <256 x i1> %2, <256 x double> %3, i32 128)
  ret <256 x double> %5
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vfmaxs.vvvmvl(<256 x double>, <256 x double>, <256 x i1>, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vfmaxs_vsvmvl(float %0, <256 x double> %1, <256 x i1> %2, <256 x double> %3) {
; CHECK-LABEL: vfmaxs_vsvmvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 128
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vfmax.s %v1, %s0, %v0, %vm1
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v1
; CHECK-NEXT:    b.l.t (, %s10)
  %5 = tail call fast <256 x double> @llvm.ve.vl.vfmaxs.vsvmvl(float %0, <256 x double> %1, <256 x i1> %2, <256 x double> %3, i32 128)
  ret <256 x double> %5
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vfmaxs.vsvmvl(float, <256 x double>, <256 x i1>, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @pvfmax_vvvl(<256 x double> %0, <256 x double> %1) {
; CHECK-LABEL: pvfmax_vvvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    pvfmax %v0, %v0, %v1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.pvfmax.vvvl(<256 x double> %0, <256 x double> %1, i32 256)
  ret <256 x double> %3
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.pvfmax.vvvl(<256 x double>, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @pvfmax_vvvvl(<256 x double> %0, <256 x double> %1, <256 x double> %2) {
; CHECK-LABEL: pvfmax_vvvvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 128
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    pvfmax %v2, %v0, %v1
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v2
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = tail call fast <256 x double> @llvm.ve.vl.pvfmax.vvvvl(<256 x double> %0, <256 x double> %1, <256 x double> %2, i32 128)
  ret <256 x double> %4
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.pvfmax.vvvvl(<256 x double>, <256 x double>, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @pvfmax_vsvl(i64 %0, <256 x double> %1) {
; CHECK-LABEL: pvfmax_vsvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    pvfmax %v0, %s0, %v0
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.pvfmax.vsvl(i64 %0, <256 x double> %1, i32 256)
  ret <256 x double> %3
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.pvfmax.vsvl(i64, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @pvfmax_vsvvl(i64 %0, <256 x double> %1, <256 x double> %2) {
; CHECK-LABEL: pvfmax_vsvvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 128
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    pvfmax %v1, %s0, %v0
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v1
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = tail call fast <256 x double> @llvm.ve.vl.pvfmax.vsvvl(i64 %0, <256 x double> %1, <256 x double> %2, i32 128)
  ret <256 x double> %4
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.pvfmax.vsvvl(i64, <256 x double>, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @pvfmax_vvvMvl(<256 x double> %0, <256 x double> %1, <512 x i1> %2, <256 x double> %3) {
; CHECK-LABEL: pvfmax_vvvMvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 128
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    pvfmax %v2, %v0, %v1, %vm2
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v2
; CHECK-NEXT:    b.l.t (, %s10)
  %5 = tail call fast <256 x double> @llvm.ve.vl.pvfmax.vvvMvl(<256 x double> %0, <256 x double> %1, <512 x i1> %2, <256 x double> %3, i32 128)
  ret <256 x double> %5
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.pvfmax.vvvMvl(<256 x double>, <256 x double>, <512 x i1>, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @pvfmax_vsvMvl(i64 %0, <256 x double> %1, <512 x i1> %2, <256 x double> %3) {
; CHECK-LABEL: pvfmax_vsvMvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 128
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    pvfmax %v1, %s0, %v0, %vm2
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v1
; CHECK-NEXT:    b.l.t (, %s10)
  %5 = tail call fast <256 x double> @llvm.ve.vl.pvfmax.vsvMvl(i64 %0, <256 x double> %1, <512 x i1> %2, <256 x double> %3, i32 128)
  ret <256 x double> %5
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.pvfmax.vsvMvl(i64, <256 x double>, <512 x i1>, <256 x double>, i32)
