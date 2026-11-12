; RUN: llc < %s -mtriple=ve -mattr=+vpu | FileCheck %s

;;; Test vector floating fused negative multiply subtract intrinsic instructions
;;;
;;; Note:
;;;   We test VFNMSB*vvvl, VFNMSB*vvvl_v, VFNMSB*rvvl, VFNMSB*rvvl_v,
;;;   VFNMSB*vrvl, VFNMSB*vrvl_v, VFNMSB*vvvml_v, VFNMSB*rvvml_v,
;;;   VFNMSB*vrvml_v, PVFNMSB*vvvl, PVFNMSB*vvvl_v, PVFNMSB*rvvl,
;;;   PVFNMSB*rvvl_v, PVFNMSB*vrvl, PVFNMSB*vrvl_v, PVFNMSB*vvvml_v,
;;;   PVFNMSB*rvvml_v, and PVFNMSB*vrvml_v instructions.

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vfnmsbd_vvvvl(<256 x double> %0, <256 x double> %1, <256 x double> %2) {
; CHECK-LABEL: vfnmsbd_vvvvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vfnmsb.d %v0, %v0, %v1, %v2
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = tail call fast <256 x double> @llvm.ve.vl.vfnmsbd.vvvvl(<256 x double> %0, <256 x double> %1, <256 x double> %2, i32 256)
  ret <256 x double> %4
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vfnmsbd.vvvvl(<256 x double>, <256 x double>, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vfnmsbd_vvvvvl(<256 x double> %0, <256 x double> %1, <256 x double> %2, <256 x double> %3) {
; CHECK-LABEL: vfnmsbd_vvvvvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 128
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vfnmsb.d %v3, %v0, %v1, %v2
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v3
; CHECK-NEXT:    b.l.t (, %s10)
  %5 = tail call fast <256 x double> @llvm.ve.vl.vfnmsbd.vvvvvl(<256 x double> %0, <256 x double> %1, <256 x double> %2, <256 x double> %3, i32 128)
  ret <256 x double> %5
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vfnmsbd.vvvvvl(<256 x double>, <256 x double>, <256 x double>, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vfnmsbd_vsvvl(double %0, <256 x double> %1, <256 x double> %2) {
; CHECK-LABEL: vfnmsbd_vsvvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vfnmsb.d %v0, %s0, %v0, %v1
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = tail call fast <256 x double> @llvm.ve.vl.vfnmsbd.vsvvl(double %0, <256 x double> %1, <256 x double> %2, i32 256)
  ret <256 x double> %4
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vfnmsbd.vsvvl(double, <256 x double>, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vfnmsbd_vsvvvl(double %0, <256 x double> %1, <256 x double> %2, <256 x double> %3) {
; CHECK-LABEL: vfnmsbd_vsvvvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 128
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vfnmsb.d %v2, %s0, %v0, %v1
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v2
; CHECK-NEXT:    b.l.t (, %s10)
  %5 = tail call fast <256 x double> @llvm.ve.vl.vfnmsbd.vsvvvl(double %0, <256 x double> %1, <256 x double> %2, <256 x double> %3, i32 128)
  ret <256 x double> %5
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vfnmsbd.vsvvvl(double, <256 x double>, <256 x double>, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vfnmsbd_vvsvl(<256 x double> %0, double %1, <256 x double> %2) {
; CHECK-LABEL: vfnmsbd_vvsvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vfnmsb.d %v0, %v0, %s0, %v1
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = tail call fast <256 x double> @llvm.ve.vl.vfnmsbd.vvsvl(<256 x double> %0, double %1, <256 x double> %2, i32 256)
  ret <256 x double> %4
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vfnmsbd.vvsvl(<256 x double>, double, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vfnmsbd_vvsvvl(<256 x double> %0, double %1, <256 x double> %2, <256 x double> %3) {
; CHECK-LABEL: vfnmsbd_vvsvvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 128
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vfnmsb.d %v2, %v0, %s0, %v1
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v2
; CHECK-NEXT:    b.l.t (, %s10)
  %5 = tail call fast <256 x double> @llvm.ve.vl.vfnmsbd.vvsvvl(<256 x double> %0, double %1, <256 x double> %2, <256 x double> %3, i32 128)
  ret <256 x double> %5
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vfnmsbd.vvsvvl(<256 x double>, double, <256 x double>, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vfnmsbd_vvvvmvl(<256 x double> %0, <256 x double> %1, <256 x double> %2, <256 x i1> %3, <256 x double> %4) {
; CHECK-LABEL: vfnmsbd_vvvvmvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 128
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vfnmsb.d %v3, %v0, %v1, %v2, %vm1
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v3
; CHECK-NEXT:    b.l.t (, %s10)
  %6 = tail call fast <256 x double> @llvm.ve.vl.vfnmsbd.vvvvmvl(<256 x double> %0, <256 x double> %1, <256 x double> %2, <256 x i1> %3, <256 x double> %4, i32 128)
  ret <256 x double> %6
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vfnmsbd.vvvvmvl(<256 x double>, <256 x double>, <256 x double>, <256 x i1>, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vfnmsbd_vsvvmvl(double %0, <256 x double> %1, <256 x double> %2, <256 x i1> %3, <256 x double> %4) {
; CHECK-LABEL: vfnmsbd_vsvvmvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 128
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vfnmsb.d %v2, %s0, %v0, %v1, %vm1
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v2
; CHECK-NEXT:    b.l.t (, %s10)
  %6 = tail call fast <256 x double> @llvm.ve.vl.vfnmsbd.vsvvmvl(double %0, <256 x double> %1, <256 x double> %2, <256 x i1> %3, <256 x double> %4, i32 128)
  ret <256 x double> %6
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vfnmsbd.vsvvmvl(double, <256 x double>, <256 x double>, <256 x i1>, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vfnmsbd_vvsvmvl(<256 x double> %0, double %1, <256 x double> %2, <256 x i1> %3, <256 x double> %4) {
; CHECK-LABEL: vfnmsbd_vvsvmvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 128
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vfnmsb.d %v2, %v0, %s0, %v1, %vm1
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v2
; CHECK-NEXT:    b.l.t (, %s10)
  %6 = tail call fast <256 x double> @llvm.ve.vl.vfnmsbd.vvsvmvl(<256 x double> %0, double %1, <256 x double> %2, <256 x i1> %3, <256 x double> %4, i32 128)
  ret <256 x double> %6
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vfnmsbd.vvsvmvl(<256 x double>, double, <256 x double>, <256 x i1>, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vfnmsbs_vvvvl(<256 x double> %0, <256 x double> %1, <256 x double> %2) {
; CHECK-LABEL: vfnmsbs_vvvvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vfnmsb.s %v0, %v0, %v1, %v2
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = tail call fast <256 x double> @llvm.ve.vl.vfnmsbs.vvvvl(<256 x double> %0, <256 x double> %1, <256 x double> %2, i32 256)
  ret <256 x double> %4
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vfnmsbs.vvvvl(<256 x double>, <256 x double>, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vfnmsbs_vvvvvl(<256 x double> %0, <256 x double> %1, <256 x double> %2, <256 x double> %3) {
; CHECK-LABEL: vfnmsbs_vvvvvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 128
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vfnmsb.s %v3, %v0, %v1, %v2
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v3
; CHECK-NEXT:    b.l.t (, %s10)
  %5 = tail call fast <256 x double> @llvm.ve.vl.vfnmsbs.vvvvvl(<256 x double> %0, <256 x double> %1, <256 x double> %2, <256 x double> %3, i32 128)
  ret <256 x double> %5
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vfnmsbs.vvvvvl(<256 x double>, <256 x double>, <256 x double>, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vfnmsbs_vsvvl(float %0, <256 x double> %1, <256 x double> %2) {
; CHECK-LABEL: vfnmsbs_vsvvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vfnmsb.s %v0, %s0, %v0, %v1
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = tail call fast <256 x double> @llvm.ve.vl.vfnmsbs.vsvvl(float %0, <256 x double> %1, <256 x double> %2, i32 256)
  ret <256 x double> %4
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vfnmsbs.vsvvl(float, <256 x double>, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vfnmsbs_vsvvvl(float %0, <256 x double> %1, <256 x double> %2, <256 x double> %3) {
; CHECK-LABEL: vfnmsbs_vsvvvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 128
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vfnmsb.s %v2, %s0, %v0, %v1
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v2
; CHECK-NEXT:    b.l.t (, %s10)
  %5 = tail call fast <256 x double> @llvm.ve.vl.vfnmsbs.vsvvvl(float %0, <256 x double> %1, <256 x double> %2, <256 x double> %3, i32 128)
  ret <256 x double> %5
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vfnmsbs.vsvvvl(float, <256 x double>, <256 x double>, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vfnmsbs_vvsvl(<256 x double> %0, float %1, <256 x double> %2) {
; CHECK-LABEL: vfnmsbs_vvsvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vfnmsb.s %v0, %v0, %s0, %v1
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = tail call fast <256 x double> @llvm.ve.vl.vfnmsbs.vvsvl(<256 x double> %0, float %1, <256 x double> %2, i32 256)
  ret <256 x double> %4
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vfnmsbs.vvsvl(<256 x double>, float, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vfnmsbs_vvsvvl(<256 x double> %0, float %1, <256 x double> %2, <256 x double> %3) {
; CHECK-LABEL: vfnmsbs_vvsvvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 128
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vfnmsb.s %v2, %v0, %s0, %v1
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v2
; CHECK-NEXT:    b.l.t (, %s10)
  %5 = tail call fast <256 x double> @llvm.ve.vl.vfnmsbs.vvsvvl(<256 x double> %0, float %1, <256 x double> %2, <256 x double> %3, i32 128)
  ret <256 x double> %5
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vfnmsbs.vvsvvl(<256 x double>, float, <256 x double>, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vfnmsbs_vvvvmvl(<256 x double> %0, <256 x double> %1, <256 x double> %2, <256 x i1> %3, <256 x double> %4) {
; CHECK-LABEL: vfnmsbs_vvvvmvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 128
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vfnmsb.s %v3, %v0, %v1, %v2, %vm1
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v3
; CHECK-NEXT:    b.l.t (, %s10)
  %6 = tail call fast <256 x double> @llvm.ve.vl.vfnmsbs.vvvvmvl(<256 x double> %0, <256 x double> %1, <256 x double> %2, <256 x i1> %3, <256 x double> %4, i32 128)
  ret <256 x double> %6
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vfnmsbs.vvvvmvl(<256 x double>, <256 x double>, <256 x double>, <256 x i1>, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vfnmsbs_vsvvmvl(float %0, <256 x double> %1, <256 x double> %2, <256 x i1> %3, <256 x double> %4) {
; CHECK-LABEL: vfnmsbs_vsvvmvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 128
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vfnmsb.s %v2, %s0, %v0, %v1, %vm1
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v2
; CHECK-NEXT:    b.l.t (, %s10)
  %6 = tail call fast <256 x double> @llvm.ve.vl.vfnmsbs.vsvvmvl(float %0, <256 x double> %1, <256 x double> %2, <256 x i1> %3, <256 x double> %4, i32 128)
  ret <256 x double> %6
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vfnmsbs.vsvvmvl(float, <256 x double>, <256 x double>, <256 x i1>, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vfnmsbs_vvsvmvl(<256 x double> %0, float %1, <256 x double> %2, <256 x i1> %3, <256 x double> %4) {
; CHECK-LABEL: vfnmsbs_vvsvmvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 128
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vfnmsb.s %v2, %v0, %s0, %v1, %vm1
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v2
; CHECK-NEXT:    b.l.t (, %s10)
  %6 = tail call fast <256 x double> @llvm.ve.vl.vfnmsbs.vvsvmvl(<256 x double> %0, float %1, <256 x double> %2, <256 x i1> %3, <256 x double> %4, i32 128)
  ret <256 x double> %6
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vfnmsbs.vvsvmvl(<256 x double>, float, <256 x double>, <256 x i1>, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @pvfnmsb_vvvvl(<256 x double> %0, <256 x double> %1, <256 x double> %2) {
; CHECK-LABEL: pvfnmsb_vvvvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    pvfnmsb %v0, %v0, %v1, %v2
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = tail call fast <256 x double> @llvm.ve.vl.pvfnmsb.vvvvl(<256 x double> %0, <256 x double> %1, <256 x double> %2, i32 256)
  ret <256 x double> %4
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.pvfnmsb.vvvvl(<256 x double>, <256 x double>, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @pvfnmsb_vvvvvl(<256 x double> %0, <256 x double> %1, <256 x double> %2, <256 x double> %3) {
; CHECK-LABEL: pvfnmsb_vvvvvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 128
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    pvfnmsb %v3, %v0, %v1, %v2
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v3
; CHECK-NEXT:    b.l.t (, %s10)
  %5 = tail call fast <256 x double> @llvm.ve.vl.pvfnmsb.vvvvvl(<256 x double> %0, <256 x double> %1, <256 x double> %2, <256 x double> %3, i32 128)
  ret <256 x double> %5
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.pvfnmsb.vvvvvl(<256 x double>, <256 x double>, <256 x double>, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @pvfnmsb_vsvvl(i64 %0, <256 x double> %1, <256 x double> %2) {
; CHECK-LABEL: pvfnmsb_vsvvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    pvfnmsb %v0, %s0, %v0, %v1
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = tail call fast <256 x double> @llvm.ve.vl.pvfnmsb.vsvvl(i64 %0, <256 x double> %1, <256 x double> %2, i32 256)
  ret <256 x double> %4
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.pvfnmsb.vsvvl(i64, <256 x double>, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @pvfnmsb_vsvvvl(i64 %0, <256 x double> %1, <256 x double> %2, <256 x double> %3) {
; CHECK-LABEL: pvfnmsb_vsvvvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 128
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    pvfnmsb %v2, %s0, %v0, %v1
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v2
; CHECK-NEXT:    b.l.t (, %s10)
  %5 = tail call fast <256 x double> @llvm.ve.vl.pvfnmsb.vsvvvl(i64 %0, <256 x double> %1, <256 x double> %2, <256 x double> %3, i32 128)
  ret <256 x double> %5
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.pvfnmsb.vsvvvl(i64, <256 x double>, <256 x double>, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @pvfnmsb_vvsvl(<256 x double> %0, i64 %1, <256 x double> %2) {
; CHECK-LABEL: pvfnmsb_vvsvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    pvfnmsb %v0, %v0, %s0, %v1
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = tail call fast <256 x double> @llvm.ve.vl.pvfnmsb.vvsvl(<256 x double> %0, i64 %1, <256 x double> %2, i32 256)
  ret <256 x double> %4
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.pvfnmsb.vvsvl(<256 x double>, i64, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @pvfnmsb_vvsvvl(<256 x double> %0, i64 %1, <256 x double> %2, <256 x double> %3) {
; CHECK-LABEL: pvfnmsb_vvsvvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 128
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    pvfnmsb %v2, %v0, %s0, %v1
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v2
; CHECK-NEXT:    b.l.t (, %s10)
  %5 = tail call fast <256 x double> @llvm.ve.vl.pvfnmsb.vvsvvl(<256 x double> %0, i64 %1, <256 x double> %2, <256 x double> %3, i32 128)
  ret <256 x double> %5
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.pvfnmsb.vvsvvl(<256 x double>, i64, <256 x double>, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @pvfnmsb_vvvvMvl(<256 x double> %0, <256 x double> %1, <256 x double> %2, <512 x i1> %3, <256 x double> %4) {
; CHECK-LABEL: pvfnmsb_vvvvMvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 128
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    pvfnmsb %v3, %v0, %v1, %v2, %vm2
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v3
; CHECK-NEXT:    b.l.t (, %s10)
  %6 = tail call fast <256 x double> @llvm.ve.vl.pvfnmsb.vvvvMvl(<256 x double> %0, <256 x double> %1, <256 x double> %2, <512 x i1> %3, <256 x double> %4, i32 128)
  ret <256 x double> %6
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.pvfnmsb.vvvvMvl(<256 x double>, <256 x double>, <256 x double>, <512 x i1>, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @pvfnmsb_vsvvMvl(i64 %0, <256 x double> %1, <256 x double> %2, <512 x i1> %3, <256 x double> %4) {
; CHECK-LABEL: pvfnmsb_vsvvMvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 128
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    pvfnmsb %v2, %s0, %v0, %v1, %vm2
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v2
; CHECK-NEXT:    b.l.t (, %s10)
  %6 = tail call fast <256 x double> @llvm.ve.vl.pvfnmsb.vsvvMvl(i64 %0, <256 x double> %1, <256 x double> %2, <512 x i1> %3, <256 x double> %4, i32 128)
  ret <256 x double> %6
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.pvfnmsb.vsvvMvl(i64, <256 x double>, <256 x double>, <512 x i1>, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @pvfnmsb_vvsvMvl(<256 x double> %0, i64 %1, <256 x double> %2, <512 x i1> %3, <256 x double> %4) {
; CHECK-LABEL: pvfnmsb_vvsvMvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 128
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    pvfnmsb %v2, %v0, %s0, %v1, %vm2
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v2
; CHECK-NEXT:    b.l.t (, %s10)
  %6 = tail call fast <256 x double> @llvm.ve.vl.pvfnmsb.vvsvMvl(<256 x double> %0, i64 %1, <256 x double> %2, <512 x i1> %3, <256 x double> %4, i32 128)
  ret <256 x double> %6
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.pvfnmsb.vvsvMvl(<256 x double>, i64, <256 x double>, <512 x i1>, <256 x double>, i32)
