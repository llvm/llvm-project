; RUN: llc < %s -mtriple=ve -mattr=+vpu | FileCheck %s

;;; Test vector floating add intrinsic instructions
;;;
;;; Note:
;;;   We test VFADD*vvl, VFADD*vvl_v, VFADD*rvl, VFADD*rvl_v, VFADD*vvml_v,
;;;   VFADD*rvml_v, PVFADD*vvl, PVFADD*vvl_v, PVFADD*rvl, PVFADD*rvl_v,
;;;   PVFADD*vvml_v, and PVFADD*rvml_v instructions.

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vfaddd_vvvl(<256 x double> %0, <256 x double> %1) {
; CHECK-LABEL: vfaddd_vvvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vfadd.d %v0, %v0, %v1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.vfaddd.vvvl(<256 x double> %0, <256 x double> %1, i32 256)
  ret <256 x double> %3
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vfaddd.vvvl(<256 x double>, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vfaddd_vvvvl(<256 x double> %0, <256 x double> %1, <256 x double> %2) {
; CHECK-LABEL: vfaddd_vvvvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 128
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vfadd.d %v2, %v0, %v1
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v2
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = tail call fast <256 x double> @llvm.ve.vl.vfaddd.vvvvl(<256 x double> %0, <256 x double> %1, <256 x double> %2, i32 128)
  ret <256 x double> %4
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vfaddd.vvvvl(<256 x double>, <256 x double>, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vfaddd_vsvl(double %0, <256 x double> %1) {
; CHECK-LABEL: vfaddd_vsvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vfadd.d %v0, %s0, %v0
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.vfaddd.vsvl(double %0, <256 x double> %1, i32 256)
  ret <256 x double> %3
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vfaddd.vsvl(double, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vfaddd_vsvvl(double %0, <256 x double> %1, <256 x double> %2) {
; CHECK-LABEL: vfaddd_vsvvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 128
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vfadd.d %v1, %s0, %v0
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v1
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = tail call fast <256 x double> @llvm.ve.vl.vfaddd.vsvvl(double %0, <256 x double> %1, <256 x double> %2, i32 128)
  ret <256 x double> %4
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vfaddd.vsvvl(double, <256 x double>, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vfaddd_vvvmvl(<256 x double> %0, <256 x double> %1, <256 x i1> %2, <256 x double> %3) {
; CHECK-LABEL: vfaddd_vvvmvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 128
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vfadd.d %v2, %v0, %v1, %vm1
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v2
; CHECK-NEXT:    b.l.t (, %s10)
  %5 = tail call fast <256 x double> @llvm.ve.vl.vfaddd.vvvmvl(<256 x double> %0, <256 x double> %1, <256 x i1> %2, <256 x double> %3, i32 128)
  ret <256 x double> %5
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vfaddd.vvvmvl(<256 x double>, <256 x double>, <256 x i1>, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vfaddd_vsvmvl(double %0, <256 x double> %1, <256 x i1> %2, <256 x double> %3) {
; CHECK-LABEL: vfaddd_vsvmvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 128
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vfadd.d %v1, %s0, %v0, %vm1
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v1
; CHECK-NEXT:    b.l.t (, %s10)
  %5 = tail call fast <256 x double> @llvm.ve.vl.vfaddd.vsvmvl(double %0, <256 x double> %1, <256 x i1> %2, <256 x double> %3, i32 128)
  ret <256 x double> %5
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vfaddd.vsvmvl(double, <256 x double>, <256 x i1>, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vfadds_vvvl(<256 x double> %0, <256 x double> %1) {
; CHECK-LABEL: vfadds_vvvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vfadd.s %v0, %v0, %v1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.vfadds.vvvl(<256 x double> %0, <256 x double> %1, i32 256)
  ret <256 x double> %3
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vfadds.vvvl(<256 x double>, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vfadds_vvvvl(<256 x double> %0, <256 x double> %1, <256 x double> %2) {
; CHECK-LABEL: vfadds_vvvvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 128
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vfadd.s %v2, %v0, %v1
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v2
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = tail call fast <256 x double> @llvm.ve.vl.vfadds.vvvvl(<256 x double> %0, <256 x double> %1, <256 x double> %2, i32 128)
  ret <256 x double> %4
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vfadds.vvvvl(<256 x double>, <256 x double>, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vfadds_vsvl(float %0, <256 x double> %1) {
; CHECK-LABEL: vfadds_vsvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vfadd.s %v0, %s0, %v0
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.vfadds.vsvl(float %0, <256 x double> %1, i32 256)
  ret <256 x double> %3
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vfadds.vsvl(float, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vfadds_vsvvl(float %0, <256 x double> %1, <256 x double> %2) {
; CHECK-LABEL: vfadds_vsvvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 128
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vfadd.s %v1, %s0, %v0
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v1
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = tail call fast <256 x double> @llvm.ve.vl.vfadds.vsvvl(float %0, <256 x double> %1, <256 x double> %2, i32 128)
  ret <256 x double> %4
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vfadds.vsvvl(float, <256 x double>, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vfadds_vvvmvl(<256 x double> %0, <256 x double> %1, <256 x i1> %2, <256 x double> %3) {
; CHECK-LABEL: vfadds_vvvmvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 128
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vfadd.s %v2, %v0, %v1, %vm1
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v2
; CHECK-NEXT:    b.l.t (, %s10)
  %5 = tail call fast <256 x double> @llvm.ve.vl.vfadds.vvvmvl(<256 x double> %0, <256 x double> %1, <256 x i1> %2, <256 x double> %3, i32 128)
  ret <256 x double> %5
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vfadds.vvvmvl(<256 x double>, <256 x double>, <256 x i1>, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vfadds_vsvmvl(float %0, <256 x double> %1, <256 x i1> %2, <256 x double> %3) {
; CHECK-LABEL: vfadds_vsvmvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 128
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vfadd.s %v1, %s0, %v0, %vm1
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v1
; CHECK-NEXT:    b.l.t (, %s10)
  %5 = tail call fast <256 x double> @llvm.ve.vl.vfadds.vsvmvl(float %0, <256 x double> %1, <256 x i1> %2, <256 x double> %3, i32 128)
  ret <256 x double> %5
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vfadds.vsvmvl(float, <256 x double>, <256 x i1>, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @pvfadd_vvvl(<256 x double> %0, <256 x double> %1) {
; CHECK-LABEL: pvfadd_vvvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    pvfadd %v0, %v0, %v1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.pvfadd.vvvl(<256 x double> %0, <256 x double> %1, i32 256)
  ret <256 x double> %3
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.pvfadd.vvvl(<256 x double>, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @pvfadd_vvvvl(<256 x double> %0, <256 x double> %1, <256 x double> %2) {
; CHECK-LABEL: pvfadd_vvvvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 128
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    pvfadd %v2, %v0, %v1
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v2
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = tail call fast <256 x double> @llvm.ve.vl.pvfadd.vvvvl(<256 x double> %0, <256 x double> %1, <256 x double> %2, i32 128)
  ret <256 x double> %4
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.pvfadd.vvvvl(<256 x double>, <256 x double>, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @pvfadd_vsvl(i64 %0, <256 x double> %1) {
; CHECK-LABEL: pvfadd_vsvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    pvfadd %v0, %s0, %v0
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.pvfadd.vsvl(i64 %0, <256 x double> %1, i32 256)
  ret <256 x double> %3
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.pvfadd.vsvl(i64, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @pvfadd_vsvvl(i64 %0, <256 x double> %1, <256 x double> %2) {
; CHECK-LABEL: pvfadd_vsvvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 128
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    pvfadd %v1, %s0, %v0
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v1
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = tail call fast <256 x double> @llvm.ve.vl.pvfadd.vsvvl(i64 %0, <256 x double> %1, <256 x double> %2, i32 128)
  ret <256 x double> %4
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.pvfadd.vsvvl(i64, <256 x double>, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @pvfadd_vvvMvl(<256 x double> %0, <256 x double> %1, <512 x i1> %2, <256 x double> %3) {
; CHECK-LABEL: pvfadd_vvvMvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 128
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    pvfadd %v2, %v0, %v1, %vm2
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v2
; CHECK-NEXT:    b.l.t (, %s10)
  %5 = tail call fast <256 x double> @llvm.ve.vl.pvfadd.vvvMvl(<256 x double> %0, <256 x double> %1, <512 x i1> %2, <256 x double> %3, i32 128)
  ret <256 x double> %5
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.pvfadd.vvvMvl(<256 x double>, <256 x double>, <512 x i1>, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @pvfadd_vsvMvl(i64 %0, <256 x double> %1, <512 x i1> %2, <256 x double> %3) {
; CHECK-LABEL: pvfadd_vsvMvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 128
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    pvfadd %v1, %s0, %v0, %vm2
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v1
; CHECK-NEXT:    b.l.t (, %s10)
  %5 = tail call fast <256 x double> @llvm.ve.vl.pvfadd.vsvMvl(i64 %0, <256 x double> %1, <512 x i1> %2, <256 x double> %3, i32 128)
  ret <256 x double> %5
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.pvfadd.vsvMvl(i64, <256 x double>, <512 x i1>, <256 x double>, i32)
