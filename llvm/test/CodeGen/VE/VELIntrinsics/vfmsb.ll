; RUN: llc < %s -mtriple=ve -mattr=+vpu | FileCheck %s

;;; Test vector floating fused multiply subtract intrinsic instructions
;;;
;;; Note:
;;;   We test VFMSB*vvvl, VFMSB*vvvl_v, VFMSB*rvvl, VFMSB*rvvl_v, VFMSB*vrvl,
;;;   VFMSB*vrvl_v, VFMSB*vvvml_v, VFMSB*rvvml_v, VFMSB*vrvml_v, PVFMSB*vvvl,
;;;   PVFMSB*vvvl_v, PVFMSB*rvvl, PVFMSB*rvvl_v, PVFMSB*vrvl, PVFMSB*vrvl_v,
;;;   PVFMSB*vvvml_v, PVFMSB*rvvml_v, and PVFMSB*vrvml_v instructions.

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vfmsbd_vvvvl(<256 x double> %0, <256 x double> %1, <256 x double> %2) {
; CHECK-LABEL: vfmsbd_vvvvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vfmsb.d %v0, %v0, %v1, %v2
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = tail call fast <256 x double> @llvm.ve.vl.vfmsbd.vvvvl(<256 x double> %0, <256 x double> %1, <256 x double> %2, i32 256)
  ret <256 x double> %4
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vfmsbd.vvvvl(<256 x double>, <256 x double>, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vfmsbd_vvvvvl(<256 x double> %0, <256 x double> %1, <256 x double> %2, <256 x double> %3) {
; CHECK-LABEL: vfmsbd_vvvvvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 128
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vfmsb.d %v3, %v0, %v1, %v2
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v3
; CHECK-NEXT:    b.l.t (, %s10)
  %5 = tail call fast <256 x double> @llvm.ve.vl.vfmsbd.vvvvvl(<256 x double> %0, <256 x double> %1, <256 x double> %2, <256 x double> %3, i32 128)
  ret <256 x double> %5
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vfmsbd.vvvvvl(<256 x double>, <256 x double>, <256 x double>, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vfmsbd_vsvvl(double %0, <256 x double> %1, <256 x double> %2) {
; CHECK-LABEL: vfmsbd_vsvvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vfmsb.d %v0, %s0, %v0, %v1
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = tail call fast <256 x double> @llvm.ve.vl.vfmsbd.vsvvl(double %0, <256 x double> %1, <256 x double> %2, i32 256)
  ret <256 x double> %4
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vfmsbd.vsvvl(double, <256 x double>, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vfmsbd_vsvvvl(double %0, <256 x double> %1, <256 x double> %2, <256 x double> %3) {
; CHECK-LABEL: vfmsbd_vsvvvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 128
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vfmsb.d %v2, %s0, %v0, %v1
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v2
; CHECK-NEXT:    b.l.t (, %s10)
  %5 = tail call fast <256 x double> @llvm.ve.vl.vfmsbd.vsvvvl(double %0, <256 x double> %1, <256 x double> %2, <256 x double> %3, i32 128)
  ret <256 x double> %5
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vfmsbd.vsvvvl(double, <256 x double>, <256 x double>, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vfmsbd_vvsvl(<256 x double> %0, double %1, <256 x double> %2) {
; CHECK-LABEL: vfmsbd_vvsvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vfmsb.d %v0, %v0, %s0, %v1
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = tail call fast <256 x double> @llvm.ve.vl.vfmsbd.vvsvl(<256 x double> %0, double %1, <256 x double> %2, i32 256)
  ret <256 x double> %4
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vfmsbd.vvsvl(<256 x double>, double, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vfmsbd_vvsvvl(<256 x double> %0, double %1, <256 x double> %2, <256 x double> %3) {
; CHECK-LABEL: vfmsbd_vvsvvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 128
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vfmsb.d %v2, %v0, %s0, %v1
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v2
; CHECK-NEXT:    b.l.t (, %s10)
  %5 = tail call fast <256 x double> @llvm.ve.vl.vfmsbd.vvsvvl(<256 x double> %0, double %1, <256 x double> %2, <256 x double> %3, i32 128)
  ret <256 x double> %5
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vfmsbd.vvsvvl(<256 x double>, double, <256 x double>, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vfmsbd_vvvvmvl(<256 x double> %0, <256 x double> %1, <256 x double> %2, <256 x i1> %3, <256 x double> %4) {
; CHECK-LABEL: vfmsbd_vvvvmvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 128
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vfmsb.d %v3, %v0, %v1, %v2, %vm1
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v3
; CHECK-NEXT:    b.l.t (, %s10)
  %6 = tail call fast <256 x double> @llvm.ve.vl.vfmsbd.vvvvmvl(<256 x double> %0, <256 x double> %1, <256 x double> %2, <256 x i1> %3, <256 x double> %4, i32 128)
  ret <256 x double> %6
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vfmsbd.vvvvmvl(<256 x double>, <256 x double>, <256 x double>, <256 x i1>, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vfmsbd_vsvvmvl(double %0, <256 x double> %1, <256 x double> %2, <256 x i1> %3, <256 x double> %4) {
; CHECK-LABEL: vfmsbd_vsvvmvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 128
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vfmsb.d %v2, %s0, %v0, %v1, %vm1
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v2
; CHECK-NEXT:    b.l.t (, %s10)
  %6 = tail call fast <256 x double> @llvm.ve.vl.vfmsbd.vsvvmvl(double %0, <256 x double> %1, <256 x double> %2, <256 x i1> %3, <256 x double> %4, i32 128)
  ret <256 x double> %6
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vfmsbd.vsvvmvl(double, <256 x double>, <256 x double>, <256 x i1>, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vfmsbd_vvsvmvl(<256 x double> %0, double %1, <256 x double> %2, <256 x i1> %3, <256 x double> %4) {
; CHECK-LABEL: vfmsbd_vvsvmvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 128
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vfmsb.d %v2, %v0, %s0, %v1, %vm1
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v2
; CHECK-NEXT:    b.l.t (, %s10)
  %6 = tail call fast <256 x double> @llvm.ve.vl.vfmsbd.vvsvmvl(<256 x double> %0, double %1, <256 x double> %2, <256 x i1> %3, <256 x double> %4, i32 128)
  ret <256 x double> %6
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vfmsbd.vvsvmvl(<256 x double>, double, <256 x double>, <256 x i1>, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vfmsbs_vvvvl(<256 x double> %0, <256 x double> %1, <256 x double> %2) {
; CHECK-LABEL: vfmsbs_vvvvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vfmsb.s %v0, %v0, %v1, %v2
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = tail call fast <256 x double> @llvm.ve.vl.vfmsbs.vvvvl(<256 x double> %0, <256 x double> %1, <256 x double> %2, i32 256)
  ret <256 x double> %4
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vfmsbs.vvvvl(<256 x double>, <256 x double>, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vfmsbs_vvvvvl(<256 x double> %0, <256 x double> %1, <256 x double> %2, <256 x double> %3) {
; CHECK-LABEL: vfmsbs_vvvvvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 128
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vfmsb.s %v3, %v0, %v1, %v2
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v3
; CHECK-NEXT:    b.l.t (, %s10)
  %5 = tail call fast <256 x double> @llvm.ve.vl.vfmsbs.vvvvvl(<256 x double> %0, <256 x double> %1, <256 x double> %2, <256 x double> %3, i32 128)
  ret <256 x double> %5
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vfmsbs.vvvvvl(<256 x double>, <256 x double>, <256 x double>, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vfmsbs_vsvvl(float %0, <256 x double> %1, <256 x double> %2) {
; CHECK-LABEL: vfmsbs_vsvvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vfmsb.s %v0, %s0, %v0, %v1
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = tail call fast <256 x double> @llvm.ve.vl.vfmsbs.vsvvl(float %0, <256 x double> %1, <256 x double> %2, i32 256)
  ret <256 x double> %4
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vfmsbs.vsvvl(float, <256 x double>, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vfmsbs_vsvvvl(float %0, <256 x double> %1, <256 x double> %2, <256 x double> %3) {
; CHECK-LABEL: vfmsbs_vsvvvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 128
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vfmsb.s %v2, %s0, %v0, %v1
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v2
; CHECK-NEXT:    b.l.t (, %s10)
  %5 = tail call fast <256 x double> @llvm.ve.vl.vfmsbs.vsvvvl(float %0, <256 x double> %1, <256 x double> %2, <256 x double> %3, i32 128)
  ret <256 x double> %5
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vfmsbs.vsvvvl(float, <256 x double>, <256 x double>, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vfmsbs_vvsvl(<256 x double> %0, float %1, <256 x double> %2) {
; CHECK-LABEL: vfmsbs_vvsvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vfmsb.s %v0, %v0, %s0, %v1
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = tail call fast <256 x double> @llvm.ve.vl.vfmsbs.vvsvl(<256 x double> %0, float %1, <256 x double> %2, i32 256)
  ret <256 x double> %4
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vfmsbs.vvsvl(<256 x double>, float, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vfmsbs_vvsvvl(<256 x double> %0, float %1, <256 x double> %2, <256 x double> %3) {
; CHECK-LABEL: vfmsbs_vvsvvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 128
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vfmsb.s %v2, %v0, %s0, %v1
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v2
; CHECK-NEXT:    b.l.t (, %s10)
  %5 = tail call fast <256 x double> @llvm.ve.vl.vfmsbs.vvsvvl(<256 x double> %0, float %1, <256 x double> %2, <256 x double> %3, i32 128)
  ret <256 x double> %5
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vfmsbs.vvsvvl(<256 x double>, float, <256 x double>, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vfmsbs_vvvvmvl(<256 x double> %0, <256 x double> %1, <256 x double> %2, <256 x i1> %3, <256 x double> %4) {
; CHECK-LABEL: vfmsbs_vvvvmvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 128
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vfmsb.s %v3, %v0, %v1, %v2, %vm1
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v3
; CHECK-NEXT:    b.l.t (, %s10)
  %6 = tail call fast <256 x double> @llvm.ve.vl.vfmsbs.vvvvmvl(<256 x double> %0, <256 x double> %1, <256 x double> %2, <256 x i1> %3, <256 x double> %4, i32 128)
  ret <256 x double> %6
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vfmsbs.vvvvmvl(<256 x double>, <256 x double>, <256 x double>, <256 x i1>, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vfmsbs_vsvvmvl(float %0, <256 x double> %1, <256 x double> %2, <256 x i1> %3, <256 x double> %4) {
; CHECK-LABEL: vfmsbs_vsvvmvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 128
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vfmsb.s %v2, %s0, %v0, %v1, %vm1
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v2
; CHECK-NEXT:    b.l.t (, %s10)
  %6 = tail call fast <256 x double> @llvm.ve.vl.vfmsbs.vsvvmvl(float %0, <256 x double> %1, <256 x double> %2, <256 x i1> %3, <256 x double> %4, i32 128)
  ret <256 x double> %6
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vfmsbs.vsvvmvl(float, <256 x double>, <256 x double>, <256 x i1>, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vfmsbs_vvsvmvl(<256 x double> %0, float %1, <256 x double> %2, <256 x i1> %3, <256 x double> %4) {
; CHECK-LABEL: vfmsbs_vvsvmvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 128
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vfmsb.s %v2, %v0, %s0, %v1, %vm1
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v2
; CHECK-NEXT:    b.l.t (, %s10)
  %6 = tail call fast <256 x double> @llvm.ve.vl.vfmsbs.vvsvmvl(<256 x double> %0, float %1, <256 x double> %2, <256 x i1> %3, <256 x double> %4, i32 128)
  ret <256 x double> %6
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vfmsbs.vvsvmvl(<256 x double>, float, <256 x double>, <256 x i1>, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @pvfmsb_vvvvl(<256 x double> %0, <256 x double> %1, <256 x double> %2) {
; CHECK-LABEL: pvfmsb_vvvvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    pvfmsb %v0, %v0, %v1, %v2
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = tail call fast <256 x double> @llvm.ve.vl.pvfmsb.vvvvl(<256 x double> %0, <256 x double> %1, <256 x double> %2, i32 256)
  ret <256 x double> %4
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.pvfmsb.vvvvl(<256 x double>, <256 x double>, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @pvfmsb_vvvvvl(<256 x double> %0, <256 x double> %1, <256 x double> %2, <256 x double> %3) {
; CHECK-LABEL: pvfmsb_vvvvvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 128
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    pvfmsb %v3, %v0, %v1, %v2
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v3
; CHECK-NEXT:    b.l.t (, %s10)
  %5 = tail call fast <256 x double> @llvm.ve.vl.pvfmsb.vvvvvl(<256 x double> %0, <256 x double> %1, <256 x double> %2, <256 x double> %3, i32 128)
  ret <256 x double> %5
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.pvfmsb.vvvvvl(<256 x double>, <256 x double>, <256 x double>, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @pvfmsb_vsvvl(i64 %0, <256 x double> %1, <256 x double> %2) {
; CHECK-LABEL: pvfmsb_vsvvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    pvfmsb %v0, %s0, %v0, %v1
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = tail call fast <256 x double> @llvm.ve.vl.pvfmsb.vsvvl(i64 %0, <256 x double> %1, <256 x double> %2, i32 256)
  ret <256 x double> %4
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.pvfmsb.vsvvl(i64, <256 x double>, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @pvfmsb_vsvvvl(i64 %0, <256 x double> %1, <256 x double> %2, <256 x double> %3) {
; CHECK-LABEL: pvfmsb_vsvvvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 128
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    pvfmsb %v2, %s0, %v0, %v1
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v2
; CHECK-NEXT:    b.l.t (, %s10)
  %5 = tail call fast <256 x double> @llvm.ve.vl.pvfmsb.vsvvvl(i64 %0, <256 x double> %1, <256 x double> %2, <256 x double> %3, i32 128)
  ret <256 x double> %5
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.pvfmsb.vsvvvl(i64, <256 x double>, <256 x double>, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @pvfmsb_vvsvl(<256 x double> %0, i64 %1, <256 x double> %2) {
; CHECK-LABEL: pvfmsb_vvsvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    pvfmsb %v0, %v0, %s0, %v1
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = tail call fast <256 x double> @llvm.ve.vl.pvfmsb.vvsvl(<256 x double> %0, i64 %1, <256 x double> %2, i32 256)
  ret <256 x double> %4
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.pvfmsb.vvsvl(<256 x double>, i64, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @pvfmsb_vvsvvl(<256 x double> %0, i64 %1, <256 x double> %2, <256 x double> %3) {
; CHECK-LABEL: pvfmsb_vvsvvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 128
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    pvfmsb %v2, %v0, %s0, %v1
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v2
; CHECK-NEXT:    b.l.t (, %s10)
  %5 = tail call fast <256 x double> @llvm.ve.vl.pvfmsb.vvsvvl(<256 x double> %0, i64 %1, <256 x double> %2, <256 x double> %3, i32 128)
  ret <256 x double> %5
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.pvfmsb.vvsvvl(<256 x double>, i64, <256 x double>, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @pvfmsb_vvvvMvl(<256 x double> %0, <256 x double> %1, <256 x double> %2, <512 x i1> %3, <256 x double> %4) {
; CHECK-LABEL: pvfmsb_vvvvMvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 128
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    pvfmsb %v3, %v0, %v1, %v2, %vm2
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v3
; CHECK-NEXT:    b.l.t (, %s10)
  %6 = tail call fast <256 x double> @llvm.ve.vl.pvfmsb.vvvvMvl(<256 x double> %0, <256 x double> %1, <256 x double> %2, <512 x i1> %3, <256 x double> %4, i32 128)
  ret <256 x double> %6
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.pvfmsb.vvvvMvl(<256 x double>, <256 x double>, <256 x double>, <512 x i1>, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @pvfmsb_vsvvMvl(i64 %0, <256 x double> %1, <256 x double> %2, <512 x i1> %3, <256 x double> %4) {
; CHECK-LABEL: pvfmsb_vsvvMvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 128
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    pvfmsb %v2, %s0, %v0, %v1, %vm2
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v2
; CHECK-NEXT:    b.l.t (, %s10)
  %6 = tail call fast <256 x double> @llvm.ve.vl.pvfmsb.vsvvMvl(i64 %0, <256 x double> %1, <256 x double> %2, <512 x i1> %3, <256 x double> %4, i32 128)
  ret <256 x double> %6
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.pvfmsb.vsvvMvl(i64, <256 x double>, <256 x double>, <512 x i1>, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @pvfmsb_vvsvMvl(<256 x double> %0, i64 %1, <256 x double> %2, <512 x i1> %3, <256 x double> %4) {
; CHECK-LABEL: pvfmsb_vvsvMvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 128
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    pvfmsb %v2, %v0, %s0, %v1, %vm2
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v2
; CHECK-NEXT:    b.l.t (, %s10)
  %6 = tail call fast <256 x double> @llvm.ve.vl.pvfmsb.vvsvMvl(<256 x double> %0, i64 %1, <256 x double> %2, <512 x i1> %3, <256 x double> %4, i32 128)
  ret <256 x double> %6
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.pvfmsb.vvsvMvl(<256 x double>, i64, <256 x double>, <512 x i1>, <256 x double>, i32)
