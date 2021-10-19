; RUN: llc < %s -mtriple=ve -mattr=+vpu | FileCheck %s

;;; Test vector floating fused multiply add intrinsic instructions
;;;
;;; Note:
;;;   We test VFMAD*vvvl, VFMAD*vvvl_v, VFMAD*rvvl, VFMAD*rvvl_v, VFMAD*vrvl,
;;;   VFMAD*vrvl_v, VFMAD*vvvml_v, VFMAD*rvvml_v, VFMAD*vrvml_v, PVFMAD*vvvl,
;;;   PVFMAD*vvvl_v, PVFMAD*rvvl, PVFMAD*rvvl_v, PVFMAD*vrvl, PVFMAD*vrvl_v,
;;;   PVFMAD*vvvml_v, PVFMAD*rvvml_v, and PVFMAD*vrvml_v instructions.

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vfmadd_vvvvl(<256 x double> %0, <256 x double> %1, <256 x double> %2) {
; CHECK-LABEL: vfmadd_vvvvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vfmad.d %v0, %v0, %v1, %v2
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = tail call fast <256 x double> @llvm.ve.vl.vfmadd.vvvvl(<256 x double> %0, <256 x double> %1, <256 x double> %2, i32 256)
  ret <256 x double> %4
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vfmadd.vvvvl(<256 x double>, <256 x double>, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vfmadd_vvvvvl(<256 x double> %0, <256 x double> %1, <256 x double> %2, <256 x double> %3) {
; CHECK-LABEL: vfmadd_vvvvvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 128
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vfmad.d %v3, %v0, %v1, %v2
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v3
; CHECK-NEXT:    b.l.t (, %s10)
  %5 = tail call fast <256 x double> @llvm.ve.vl.vfmadd.vvvvvl(<256 x double> %0, <256 x double> %1, <256 x double> %2, <256 x double> %3, i32 128)
  ret <256 x double> %5
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vfmadd.vvvvvl(<256 x double>, <256 x double>, <256 x double>, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vfmadd_vsvvl(double %0, <256 x double> %1, <256 x double> %2) {
; CHECK-LABEL: vfmadd_vsvvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vfmad.d %v0, %s0, %v0, %v1
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = tail call fast <256 x double> @llvm.ve.vl.vfmadd.vsvvl(double %0, <256 x double> %1, <256 x double> %2, i32 256)
  ret <256 x double> %4
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vfmadd.vsvvl(double, <256 x double>, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vfmadd_vsvvvl(double %0, <256 x double> %1, <256 x double> %2, <256 x double> %3) {
; CHECK-LABEL: vfmadd_vsvvvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 128
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vfmad.d %v2, %s0, %v0, %v1
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v2
; CHECK-NEXT:    b.l.t (, %s10)
  %5 = tail call fast <256 x double> @llvm.ve.vl.vfmadd.vsvvvl(double %0, <256 x double> %1, <256 x double> %2, <256 x double> %3, i32 128)
  ret <256 x double> %5
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vfmadd.vsvvvl(double, <256 x double>, <256 x double>, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vfmadd_vvsvl(<256 x double> %0, double %1, <256 x double> %2) {
; CHECK-LABEL: vfmadd_vvsvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vfmad.d %v0, %v0, %s0, %v1
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = tail call fast <256 x double> @llvm.ve.vl.vfmadd.vvsvl(<256 x double> %0, double %1, <256 x double> %2, i32 256)
  ret <256 x double> %4
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vfmadd.vvsvl(<256 x double>, double, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vfmadd_vvsvvl(<256 x double> %0, double %1, <256 x double> %2, <256 x double> %3) {
; CHECK-LABEL: vfmadd_vvsvvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 128
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vfmad.d %v2, %v0, %s0, %v1
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v2
; CHECK-NEXT:    b.l.t (, %s10)
  %5 = tail call fast <256 x double> @llvm.ve.vl.vfmadd.vvsvvl(<256 x double> %0, double %1, <256 x double> %2, <256 x double> %3, i32 128)
  ret <256 x double> %5
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vfmadd.vvsvvl(<256 x double>, double, <256 x double>, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vfmadd_vvvvmvl(<256 x double> %0, <256 x double> %1, <256 x double> %2, <256 x i1> %3, <256 x double> %4) {
; CHECK-LABEL: vfmadd_vvvvmvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 128
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vfmad.d %v3, %v0, %v1, %v2, %vm1
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v3
; CHECK-NEXT:    b.l.t (, %s10)
  %6 = tail call fast <256 x double> @llvm.ve.vl.vfmadd.vvvvmvl(<256 x double> %0, <256 x double> %1, <256 x double> %2, <256 x i1> %3, <256 x double> %4, i32 128)
  ret <256 x double> %6
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vfmadd.vvvvmvl(<256 x double>, <256 x double>, <256 x double>, <256 x i1>, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vfmadd_vsvvmvl(double %0, <256 x double> %1, <256 x double> %2, <256 x i1> %3, <256 x double> %4) {
; CHECK-LABEL: vfmadd_vsvvmvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 128
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vfmad.d %v2, %s0, %v0, %v1, %vm1
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v2
; CHECK-NEXT:    b.l.t (, %s10)
  %6 = tail call fast <256 x double> @llvm.ve.vl.vfmadd.vsvvmvl(double %0, <256 x double> %1, <256 x double> %2, <256 x i1> %3, <256 x double> %4, i32 128)
  ret <256 x double> %6
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vfmadd.vsvvmvl(double, <256 x double>, <256 x double>, <256 x i1>, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vfmadd_vvsvmvl(<256 x double> %0, double %1, <256 x double> %2, <256 x i1> %3, <256 x double> %4) {
; CHECK-LABEL: vfmadd_vvsvmvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 128
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vfmad.d %v2, %v0, %s0, %v1, %vm1
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v2
; CHECK-NEXT:    b.l.t (, %s10)
  %6 = tail call fast <256 x double> @llvm.ve.vl.vfmadd.vvsvmvl(<256 x double> %0, double %1, <256 x double> %2, <256 x i1> %3, <256 x double> %4, i32 128)
  ret <256 x double> %6
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vfmadd.vvsvmvl(<256 x double>, double, <256 x double>, <256 x i1>, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vfmads_vvvvl(<256 x double> %0, <256 x double> %1, <256 x double> %2) {
; CHECK-LABEL: vfmads_vvvvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vfmad.s %v0, %v0, %v1, %v2
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = tail call fast <256 x double> @llvm.ve.vl.vfmads.vvvvl(<256 x double> %0, <256 x double> %1, <256 x double> %2, i32 256)
  ret <256 x double> %4
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vfmads.vvvvl(<256 x double>, <256 x double>, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vfmads_vvvvvl(<256 x double> %0, <256 x double> %1, <256 x double> %2, <256 x double> %3) {
; CHECK-LABEL: vfmads_vvvvvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 128
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vfmad.s %v3, %v0, %v1, %v2
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v3
; CHECK-NEXT:    b.l.t (, %s10)
  %5 = tail call fast <256 x double> @llvm.ve.vl.vfmads.vvvvvl(<256 x double> %0, <256 x double> %1, <256 x double> %2, <256 x double> %3, i32 128)
  ret <256 x double> %5
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vfmads.vvvvvl(<256 x double>, <256 x double>, <256 x double>, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vfmads_vsvvl(float %0, <256 x double> %1, <256 x double> %2) {
; CHECK-LABEL: vfmads_vsvvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vfmad.s %v0, %s0, %v0, %v1
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = tail call fast <256 x double> @llvm.ve.vl.vfmads.vsvvl(float %0, <256 x double> %1, <256 x double> %2, i32 256)
  ret <256 x double> %4
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vfmads.vsvvl(float, <256 x double>, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vfmads_vsvvvl(float %0, <256 x double> %1, <256 x double> %2, <256 x double> %3) {
; CHECK-LABEL: vfmads_vsvvvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 128
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vfmad.s %v2, %s0, %v0, %v1
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v2
; CHECK-NEXT:    b.l.t (, %s10)
  %5 = tail call fast <256 x double> @llvm.ve.vl.vfmads.vsvvvl(float %0, <256 x double> %1, <256 x double> %2, <256 x double> %3, i32 128)
  ret <256 x double> %5
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vfmads.vsvvvl(float, <256 x double>, <256 x double>, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vfmads_vvsvl(<256 x double> %0, float %1, <256 x double> %2) {
; CHECK-LABEL: vfmads_vvsvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vfmad.s %v0, %v0, %s0, %v1
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = tail call fast <256 x double> @llvm.ve.vl.vfmads.vvsvl(<256 x double> %0, float %1, <256 x double> %2, i32 256)
  ret <256 x double> %4
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vfmads.vvsvl(<256 x double>, float, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vfmads_vvsvvl(<256 x double> %0, float %1, <256 x double> %2, <256 x double> %3) {
; CHECK-LABEL: vfmads_vvsvvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 128
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vfmad.s %v2, %v0, %s0, %v1
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v2
; CHECK-NEXT:    b.l.t (, %s10)
  %5 = tail call fast <256 x double> @llvm.ve.vl.vfmads.vvsvvl(<256 x double> %0, float %1, <256 x double> %2, <256 x double> %3, i32 128)
  ret <256 x double> %5
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vfmads.vvsvvl(<256 x double>, float, <256 x double>, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vfmads_vvvvmvl(<256 x double> %0, <256 x double> %1, <256 x double> %2, <256 x i1> %3, <256 x double> %4) {
; CHECK-LABEL: vfmads_vvvvmvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 128
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vfmad.s %v3, %v0, %v1, %v2, %vm1
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v3
; CHECK-NEXT:    b.l.t (, %s10)
  %6 = tail call fast <256 x double> @llvm.ve.vl.vfmads.vvvvmvl(<256 x double> %0, <256 x double> %1, <256 x double> %2, <256 x i1> %3, <256 x double> %4, i32 128)
  ret <256 x double> %6
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vfmads.vvvvmvl(<256 x double>, <256 x double>, <256 x double>, <256 x i1>, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vfmads_vsvvmvl(float %0, <256 x double> %1, <256 x double> %2, <256 x i1> %3, <256 x double> %4) {
; CHECK-LABEL: vfmads_vsvvmvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 128
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vfmad.s %v2, %s0, %v0, %v1, %vm1
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v2
; CHECK-NEXT:    b.l.t (, %s10)
  %6 = tail call fast <256 x double> @llvm.ve.vl.vfmads.vsvvmvl(float %0, <256 x double> %1, <256 x double> %2, <256 x i1> %3, <256 x double> %4, i32 128)
  ret <256 x double> %6
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vfmads.vsvvmvl(float, <256 x double>, <256 x double>, <256 x i1>, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vfmads_vvsvmvl(<256 x double> %0, float %1, <256 x double> %2, <256 x i1> %3, <256 x double> %4) {
; CHECK-LABEL: vfmads_vvsvmvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 128
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vfmad.s %v2, %v0, %s0, %v1, %vm1
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v2
; CHECK-NEXT:    b.l.t (, %s10)
  %6 = tail call fast <256 x double> @llvm.ve.vl.vfmads.vvsvmvl(<256 x double> %0, float %1, <256 x double> %2, <256 x i1> %3, <256 x double> %4, i32 128)
  ret <256 x double> %6
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vfmads.vvsvmvl(<256 x double>, float, <256 x double>, <256 x i1>, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @pvfmad_vvvvl(<256 x double> %0, <256 x double> %1, <256 x double> %2) {
; CHECK-LABEL: pvfmad_vvvvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    pvfmad %v0, %v0, %v1, %v2
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = tail call fast <256 x double> @llvm.ve.vl.pvfmad.vvvvl(<256 x double> %0, <256 x double> %1, <256 x double> %2, i32 256)
  ret <256 x double> %4
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.pvfmad.vvvvl(<256 x double>, <256 x double>, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @pvfmad_vvvvvl(<256 x double> %0, <256 x double> %1, <256 x double> %2, <256 x double> %3) {
; CHECK-LABEL: pvfmad_vvvvvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 128
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    pvfmad %v3, %v0, %v1, %v2
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v3
; CHECK-NEXT:    b.l.t (, %s10)
  %5 = tail call fast <256 x double> @llvm.ve.vl.pvfmad.vvvvvl(<256 x double> %0, <256 x double> %1, <256 x double> %2, <256 x double> %3, i32 128)
  ret <256 x double> %5
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.pvfmad.vvvvvl(<256 x double>, <256 x double>, <256 x double>, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @pvfmad_vsvvl(i64 %0, <256 x double> %1, <256 x double> %2) {
; CHECK-LABEL: pvfmad_vsvvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    pvfmad %v0, %s0, %v0, %v1
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = tail call fast <256 x double> @llvm.ve.vl.pvfmad.vsvvl(i64 %0, <256 x double> %1, <256 x double> %2, i32 256)
  ret <256 x double> %4
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.pvfmad.vsvvl(i64, <256 x double>, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @pvfmad_vsvvvl(i64 %0, <256 x double> %1, <256 x double> %2, <256 x double> %3) {
; CHECK-LABEL: pvfmad_vsvvvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 128
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    pvfmad %v2, %s0, %v0, %v1
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v2
; CHECK-NEXT:    b.l.t (, %s10)
  %5 = tail call fast <256 x double> @llvm.ve.vl.pvfmad.vsvvvl(i64 %0, <256 x double> %1, <256 x double> %2, <256 x double> %3, i32 128)
  ret <256 x double> %5
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.pvfmad.vsvvvl(i64, <256 x double>, <256 x double>, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @pvfmad_vvsvl(<256 x double> %0, i64 %1, <256 x double> %2) {
; CHECK-LABEL: pvfmad_vvsvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    pvfmad %v0, %v0, %s0, %v1
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = tail call fast <256 x double> @llvm.ve.vl.pvfmad.vvsvl(<256 x double> %0, i64 %1, <256 x double> %2, i32 256)
  ret <256 x double> %4
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.pvfmad.vvsvl(<256 x double>, i64, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @pvfmad_vvsvvl(<256 x double> %0, i64 %1, <256 x double> %2, <256 x double> %3) {
; CHECK-LABEL: pvfmad_vvsvvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 128
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    pvfmad %v2, %v0, %s0, %v1
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v2
; CHECK-NEXT:    b.l.t (, %s10)
  %5 = tail call fast <256 x double> @llvm.ve.vl.pvfmad.vvsvvl(<256 x double> %0, i64 %1, <256 x double> %2, <256 x double> %3, i32 128)
  ret <256 x double> %5
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.pvfmad.vvsvvl(<256 x double>, i64, <256 x double>, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @pvfmad_vvvvMvl(<256 x double> %0, <256 x double> %1, <256 x double> %2, <512 x i1> %3, <256 x double> %4) {
; CHECK-LABEL: pvfmad_vvvvMvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 128
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    pvfmad %v3, %v0, %v1, %v2, %vm2
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v3
; CHECK-NEXT:    b.l.t (, %s10)
  %6 = tail call fast <256 x double> @llvm.ve.vl.pvfmad.vvvvMvl(<256 x double> %0, <256 x double> %1, <256 x double> %2, <512 x i1> %3, <256 x double> %4, i32 128)
  ret <256 x double> %6
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.pvfmad.vvvvMvl(<256 x double>, <256 x double>, <256 x double>, <512 x i1>, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @pvfmad_vsvvMvl(i64 %0, <256 x double> %1, <256 x double> %2, <512 x i1> %3, <256 x double> %4) {
; CHECK-LABEL: pvfmad_vsvvMvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 128
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    pvfmad %v2, %s0, %v0, %v1, %vm2
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v2
; CHECK-NEXT:    b.l.t (, %s10)
  %6 = tail call fast <256 x double> @llvm.ve.vl.pvfmad.vsvvMvl(i64 %0, <256 x double> %1, <256 x double> %2, <512 x i1> %3, <256 x double> %4, i32 128)
  ret <256 x double> %6
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.pvfmad.vsvvMvl(i64, <256 x double>, <256 x double>, <512 x i1>, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @pvfmad_vvsvMvl(<256 x double> %0, i64 %1, <256 x double> %2, <512 x i1> %3, <256 x double> %4) {
; CHECK-LABEL: pvfmad_vvsvMvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 128
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    pvfmad %v2, %v0, %s0, %v1, %vm2
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v2
; CHECK-NEXT:    b.l.t (, %s10)
  %6 = tail call fast <256 x double> @llvm.ve.vl.pvfmad.vvsvMvl(<256 x double> %0, i64 %1, <256 x double> %2, <512 x i1> %3, <256 x double> %4, i32 128)
  ret <256 x double> %6
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.pvfmad.vvsvMvl(<256 x double>, i64, <256 x double>, <512 x i1>, <256 x double>, i32)
