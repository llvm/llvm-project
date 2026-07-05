; RUN: llc < %s -mtriple=ve -mattr=+vpu | FileCheck %s

;;; Test vector floating fused negative multiply add intrinsic instructions
;;;
;;; Note:
;;;   We test VFNMAD*vvvl, VFNMAD*vvvl_v, VFNMAD*rvvl, VFNMAD*rvvl_v,
;;;   VFNMAD*vrvl, VFNMAD*vrvl_v, VFNMAD*vvvml_v, VFNMAD*rvvml_v,
;;;   VFNMAD*vrvml_v, PVFNMAD*vvvl, PVFNMAD*vvvl_v, PVFNMAD*rvvl,
;;;   PVFNMAD*rvvl_v, PVFNMAD*vrvl, PVFNMAD*vrvl_v, PVFNMAD*vvvml_v,
;;;   PVFNMAD*rvvml_v, and PVFNMAD*vrvml_v instructions.

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vfnmadd_vvvvl(<256 x double> %0, <256 x double> %1, <256 x double> %2) {
; CHECK-LABEL: vfnmadd_vvvvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vfnmad.d %v0, %v0, %v1, %v2
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = tail call fast <256 x double> @llvm.ve.vl.vfnmadd.vvvvl(<256 x double> %0, <256 x double> %1, <256 x double> %2, i32 256)
  ret <256 x double> %4
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vfnmadd.vvvvl(<256 x double>, <256 x double>, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vfnmadd_vvvvvl(<256 x double> %0, <256 x double> %1, <256 x double> %2, <256 x double> %3) {
; CHECK-LABEL: vfnmadd_vvvvvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 128
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vfnmad.d %v3, %v0, %v1, %v2
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v3
; CHECK-NEXT:    b.l.t (, %s10)
  %5 = tail call fast <256 x double> @llvm.ve.vl.vfnmadd.vvvvvl(<256 x double> %0, <256 x double> %1, <256 x double> %2, <256 x double> %3, i32 128)
  ret <256 x double> %5
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vfnmadd.vvvvvl(<256 x double>, <256 x double>, <256 x double>, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vfnmadd_vsvvl(double %0, <256 x double> %1, <256 x double> %2) {
; CHECK-LABEL: vfnmadd_vsvvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vfnmad.d %v0, %s0, %v0, %v1
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = tail call fast <256 x double> @llvm.ve.vl.vfnmadd.vsvvl(double %0, <256 x double> %1, <256 x double> %2, i32 256)
  ret <256 x double> %4
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vfnmadd.vsvvl(double, <256 x double>, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vfnmadd_vsvvvl(double %0, <256 x double> %1, <256 x double> %2, <256 x double> %3) {
; CHECK-LABEL: vfnmadd_vsvvvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 128
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vfnmad.d %v2, %s0, %v0, %v1
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v2
; CHECK-NEXT:    b.l.t (, %s10)
  %5 = tail call fast <256 x double> @llvm.ve.vl.vfnmadd.vsvvvl(double %0, <256 x double> %1, <256 x double> %2, <256 x double> %3, i32 128)
  ret <256 x double> %5
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vfnmadd.vsvvvl(double, <256 x double>, <256 x double>, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vfnmadd_vvsvl(<256 x double> %0, double %1, <256 x double> %2) {
; CHECK-LABEL: vfnmadd_vvsvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vfnmad.d %v0, %v0, %s0, %v1
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = tail call fast <256 x double> @llvm.ve.vl.vfnmadd.vvsvl(<256 x double> %0, double %1, <256 x double> %2, i32 256)
  ret <256 x double> %4
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vfnmadd.vvsvl(<256 x double>, double, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vfnmadd_vvsvvl(<256 x double> %0, double %1, <256 x double> %2, <256 x double> %3) {
; CHECK-LABEL: vfnmadd_vvsvvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 128
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vfnmad.d %v2, %v0, %s0, %v1
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v2
; CHECK-NEXT:    b.l.t (, %s10)
  %5 = tail call fast <256 x double> @llvm.ve.vl.vfnmadd.vvsvvl(<256 x double> %0, double %1, <256 x double> %2, <256 x double> %3, i32 128)
  ret <256 x double> %5
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vfnmadd.vvsvvl(<256 x double>, double, <256 x double>, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vfnmadd_vvvvmvl(<256 x double> %0, <256 x double> %1, <256 x double> %2, <256 x i1> %3, <256 x double> %4) {
; CHECK-LABEL: vfnmadd_vvvvmvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 128
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vfnmad.d %v3, %v0, %v1, %v2, %vm1
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v3
; CHECK-NEXT:    b.l.t (, %s10)
  %6 = tail call fast <256 x double> @llvm.ve.vl.vfnmadd.vvvvmvl(<256 x double> %0, <256 x double> %1, <256 x double> %2, <256 x i1> %3, <256 x double> %4, i32 128)
  ret <256 x double> %6
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vfnmadd.vvvvmvl(<256 x double>, <256 x double>, <256 x double>, <256 x i1>, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vfnmadd_vsvvmvl(double %0, <256 x double> %1, <256 x double> %2, <256 x i1> %3, <256 x double> %4) {
; CHECK-LABEL: vfnmadd_vsvvmvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 128
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vfnmad.d %v2, %s0, %v0, %v1, %vm1
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v2
; CHECK-NEXT:    b.l.t (, %s10)
  %6 = tail call fast <256 x double> @llvm.ve.vl.vfnmadd.vsvvmvl(double %0, <256 x double> %1, <256 x double> %2, <256 x i1> %3, <256 x double> %4, i32 128)
  ret <256 x double> %6
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vfnmadd.vsvvmvl(double, <256 x double>, <256 x double>, <256 x i1>, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vfnmadd_vvsvmvl(<256 x double> %0, double %1, <256 x double> %2, <256 x i1> %3, <256 x double> %4) {
; CHECK-LABEL: vfnmadd_vvsvmvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 128
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vfnmad.d %v2, %v0, %s0, %v1, %vm1
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v2
; CHECK-NEXT:    b.l.t (, %s10)
  %6 = tail call fast <256 x double> @llvm.ve.vl.vfnmadd.vvsvmvl(<256 x double> %0, double %1, <256 x double> %2, <256 x i1> %3, <256 x double> %4, i32 128)
  ret <256 x double> %6
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vfnmadd.vvsvmvl(<256 x double>, double, <256 x double>, <256 x i1>, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vfnmads_vvvvl(<256 x double> %0, <256 x double> %1, <256 x double> %2) {
; CHECK-LABEL: vfnmads_vvvvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vfnmad.s %v0, %v0, %v1, %v2
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = tail call fast <256 x double> @llvm.ve.vl.vfnmads.vvvvl(<256 x double> %0, <256 x double> %1, <256 x double> %2, i32 256)
  ret <256 x double> %4
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vfnmads.vvvvl(<256 x double>, <256 x double>, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vfnmads_vvvvvl(<256 x double> %0, <256 x double> %1, <256 x double> %2, <256 x double> %3) {
; CHECK-LABEL: vfnmads_vvvvvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 128
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vfnmad.s %v3, %v0, %v1, %v2
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v3
; CHECK-NEXT:    b.l.t (, %s10)
  %5 = tail call fast <256 x double> @llvm.ve.vl.vfnmads.vvvvvl(<256 x double> %0, <256 x double> %1, <256 x double> %2, <256 x double> %3, i32 128)
  ret <256 x double> %5
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vfnmads.vvvvvl(<256 x double>, <256 x double>, <256 x double>, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vfnmads_vsvvl(float %0, <256 x double> %1, <256 x double> %2) {
; CHECK-LABEL: vfnmads_vsvvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vfnmad.s %v0, %s0, %v0, %v1
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = tail call fast <256 x double> @llvm.ve.vl.vfnmads.vsvvl(float %0, <256 x double> %1, <256 x double> %2, i32 256)
  ret <256 x double> %4
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vfnmads.vsvvl(float, <256 x double>, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vfnmads_vsvvvl(float %0, <256 x double> %1, <256 x double> %2, <256 x double> %3) {
; CHECK-LABEL: vfnmads_vsvvvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 128
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vfnmad.s %v2, %s0, %v0, %v1
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v2
; CHECK-NEXT:    b.l.t (, %s10)
  %5 = tail call fast <256 x double> @llvm.ve.vl.vfnmads.vsvvvl(float %0, <256 x double> %1, <256 x double> %2, <256 x double> %3, i32 128)
  ret <256 x double> %5
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vfnmads.vsvvvl(float, <256 x double>, <256 x double>, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vfnmads_vvsvl(<256 x double> %0, float %1, <256 x double> %2) {
; CHECK-LABEL: vfnmads_vvsvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vfnmad.s %v0, %v0, %s0, %v1
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = tail call fast <256 x double> @llvm.ve.vl.vfnmads.vvsvl(<256 x double> %0, float %1, <256 x double> %2, i32 256)
  ret <256 x double> %4
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vfnmads.vvsvl(<256 x double>, float, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vfnmads_vvsvvl(<256 x double> %0, float %1, <256 x double> %2, <256 x double> %3) {
; CHECK-LABEL: vfnmads_vvsvvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 128
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vfnmad.s %v2, %v0, %s0, %v1
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v2
; CHECK-NEXT:    b.l.t (, %s10)
  %5 = tail call fast <256 x double> @llvm.ve.vl.vfnmads.vvsvvl(<256 x double> %0, float %1, <256 x double> %2, <256 x double> %3, i32 128)
  ret <256 x double> %5
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vfnmads.vvsvvl(<256 x double>, float, <256 x double>, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vfnmads_vvvvmvl(<256 x double> %0, <256 x double> %1, <256 x double> %2, <256 x i1> %3, <256 x double> %4) {
; CHECK-LABEL: vfnmads_vvvvmvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 128
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vfnmad.s %v3, %v0, %v1, %v2, %vm1
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v3
; CHECK-NEXT:    b.l.t (, %s10)
  %6 = tail call fast <256 x double> @llvm.ve.vl.vfnmads.vvvvmvl(<256 x double> %0, <256 x double> %1, <256 x double> %2, <256 x i1> %3, <256 x double> %4, i32 128)
  ret <256 x double> %6
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vfnmads.vvvvmvl(<256 x double>, <256 x double>, <256 x double>, <256 x i1>, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vfnmads_vsvvmvl(float %0, <256 x double> %1, <256 x double> %2, <256 x i1> %3, <256 x double> %4) {
; CHECK-LABEL: vfnmads_vsvvmvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 128
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vfnmad.s %v2, %s0, %v0, %v1, %vm1
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v2
; CHECK-NEXT:    b.l.t (, %s10)
  %6 = tail call fast <256 x double> @llvm.ve.vl.vfnmads.vsvvmvl(float %0, <256 x double> %1, <256 x double> %2, <256 x i1> %3, <256 x double> %4, i32 128)
  ret <256 x double> %6
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vfnmads.vsvvmvl(float, <256 x double>, <256 x double>, <256 x i1>, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vfnmads_vvsvmvl(<256 x double> %0, float %1, <256 x double> %2, <256 x i1> %3, <256 x double> %4) {
; CHECK-LABEL: vfnmads_vvsvmvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 128
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vfnmad.s %v2, %v0, %s0, %v1, %vm1
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v2
; CHECK-NEXT:    b.l.t (, %s10)
  %6 = tail call fast <256 x double> @llvm.ve.vl.vfnmads.vvsvmvl(<256 x double> %0, float %1, <256 x double> %2, <256 x i1> %3, <256 x double> %4, i32 128)
  ret <256 x double> %6
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vfnmads.vvsvmvl(<256 x double>, float, <256 x double>, <256 x i1>, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @pvfnmad_vvvvl(<256 x double> %0, <256 x double> %1, <256 x double> %2) {
; CHECK-LABEL: pvfnmad_vvvvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    pvfnmad %v0, %v0, %v1, %v2
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = tail call fast <256 x double> @llvm.ve.vl.pvfnmad.vvvvl(<256 x double> %0, <256 x double> %1, <256 x double> %2, i32 256)
  ret <256 x double> %4
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.pvfnmad.vvvvl(<256 x double>, <256 x double>, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @pvfnmad_vvvvvl(<256 x double> %0, <256 x double> %1, <256 x double> %2, <256 x double> %3) {
; CHECK-LABEL: pvfnmad_vvvvvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 128
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    pvfnmad %v3, %v0, %v1, %v2
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v3
; CHECK-NEXT:    b.l.t (, %s10)
  %5 = tail call fast <256 x double> @llvm.ve.vl.pvfnmad.vvvvvl(<256 x double> %0, <256 x double> %1, <256 x double> %2, <256 x double> %3, i32 128)
  ret <256 x double> %5
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.pvfnmad.vvvvvl(<256 x double>, <256 x double>, <256 x double>, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @pvfnmad_vsvvl(i64 %0, <256 x double> %1, <256 x double> %2) {
; CHECK-LABEL: pvfnmad_vsvvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    pvfnmad %v0, %s0, %v0, %v1
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = tail call fast <256 x double> @llvm.ve.vl.pvfnmad.vsvvl(i64 %0, <256 x double> %1, <256 x double> %2, i32 256)
  ret <256 x double> %4
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.pvfnmad.vsvvl(i64, <256 x double>, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @pvfnmad_vsvvvl(i64 %0, <256 x double> %1, <256 x double> %2, <256 x double> %3) {
; CHECK-LABEL: pvfnmad_vsvvvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 128
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    pvfnmad %v2, %s0, %v0, %v1
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v2
; CHECK-NEXT:    b.l.t (, %s10)
  %5 = tail call fast <256 x double> @llvm.ve.vl.pvfnmad.vsvvvl(i64 %0, <256 x double> %1, <256 x double> %2, <256 x double> %3, i32 128)
  ret <256 x double> %5
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.pvfnmad.vsvvvl(i64, <256 x double>, <256 x double>, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @pvfnmad_vvsvl(<256 x double> %0, i64 %1, <256 x double> %2) {
; CHECK-LABEL: pvfnmad_vvsvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    pvfnmad %v0, %v0, %s0, %v1
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = tail call fast <256 x double> @llvm.ve.vl.pvfnmad.vvsvl(<256 x double> %0, i64 %1, <256 x double> %2, i32 256)
  ret <256 x double> %4
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.pvfnmad.vvsvl(<256 x double>, i64, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @pvfnmad_vvsvvl(<256 x double> %0, i64 %1, <256 x double> %2, <256 x double> %3) {
; CHECK-LABEL: pvfnmad_vvsvvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 128
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    pvfnmad %v2, %v0, %s0, %v1
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v2
; CHECK-NEXT:    b.l.t (, %s10)
  %5 = tail call fast <256 x double> @llvm.ve.vl.pvfnmad.vvsvvl(<256 x double> %0, i64 %1, <256 x double> %2, <256 x double> %3, i32 128)
  ret <256 x double> %5
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.pvfnmad.vvsvvl(<256 x double>, i64, <256 x double>, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @pvfnmad_vvvvMvl(<256 x double> %0, <256 x double> %1, <256 x double> %2, <512 x i1> %3, <256 x double> %4) {
; CHECK-LABEL: pvfnmad_vvvvMvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 128
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    pvfnmad %v3, %v0, %v1, %v2, %vm2
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v3
; CHECK-NEXT:    b.l.t (, %s10)
  %6 = tail call fast <256 x double> @llvm.ve.vl.pvfnmad.vvvvMvl(<256 x double> %0, <256 x double> %1, <256 x double> %2, <512 x i1> %3, <256 x double> %4, i32 128)
  ret <256 x double> %6
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.pvfnmad.vvvvMvl(<256 x double>, <256 x double>, <256 x double>, <512 x i1>, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @pvfnmad_vsvvMvl(i64 %0, <256 x double> %1, <256 x double> %2, <512 x i1> %3, <256 x double> %4) {
; CHECK-LABEL: pvfnmad_vsvvMvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 128
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    pvfnmad %v2, %s0, %v0, %v1, %vm2
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v2
; CHECK-NEXT:    b.l.t (, %s10)
  %6 = tail call fast <256 x double> @llvm.ve.vl.pvfnmad.vsvvMvl(i64 %0, <256 x double> %1, <256 x double> %2, <512 x i1> %3, <256 x double> %4, i32 128)
  ret <256 x double> %6
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.pvfnmad.vsvvMvl(i64, <256 x double>, <256 x double>, <512 x i1>, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @pvfnmad_vvsvMvl(<256 x double> %0, i64 %1, <256 x double> %2, <512 x i1> %3, <256 x double> %4) {
; CHECK-LABEL: pvfnmad_vvsvMvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 128
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    pvfnmad %v2, %v0, %s0, %v1, %vm2
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v2
; CHECK-NEXT:    b.l.t (, %s10)
  %6 = tail call fast <256 x double> @llvm.ve.vl.pvfnmad.vvsvMvl(<256 x double> %0, i64 %1, <256 x double> %2, <512 x i1> %3, <256 x double> %4, i32 128)
  ret <256 x double> %6
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.pvfnmad.vvsvMvl(<256 x double>, i64, <256 x double>, <512 x i1>, <256 x double>, i32)
