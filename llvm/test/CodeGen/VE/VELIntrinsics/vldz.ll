; RUN: llc < %s -mtriple=ve -mattr=+vpu | FileCheck %s

;;; Test vector leading zero count intrinsic instructions
;;;
;;; Note:
;;;   We test VLDZ*vl, VLDZ*vl_v, VLDZ*vml_v, PVLDZ*vl, PVLDZ*vl_v, PVLDZ*vml_v instructions.

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vldz_vvl(<256 x double> %0) {
; CHECK-LABEL: vldz_vvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vldz %v0, %v0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call fast <256 x double> @llvm.ve.vl.vldz.vvl(<256 x double> %0, i32 256)
  ret <256 x double> %2
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vldz.vvl(<256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vldz_vvvl(<256 x double> %0, <256 x double> %1) {
; CHECK-LABEL: vldz_vvvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 128
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vldz %v1, %v0
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.vldz.vvvl(<256 x double> %0, <256 x double> %1, i32 128)
  ret <256 x double> %3
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vldz.vvvl(<256 x double>, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vldz_vvmvl(<256 x double> %0, <256 x i1> %1, <256 x double> %2) {
; CHECK-LABEL: vldz_vvmvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 128
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vldz %v1, %v0, %vm1
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v1
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = tail call fast <256 x double> @llvm.ve.vl.vldz.vvmvl(<256 x double> %0, <256 x i1> %1, <256 x double> %2, i32 128)
  ret <256 x double> %4
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vldz.vvmvl(<256 x double>, <256 x i1>, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @pvldzlo_vvl(<256 x double> %0) {
; CHECK-LABEL: pvldzlo_vvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    pvldz.lo %v0, %v0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call fast <256 x double> @llvm.ve.vl.pvldzlo.vvl(<256 x double> %0, i32 256)
  ret <256 x double> %2
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.pvldzlo.vvl(<256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @pvldzlo_vvvl(<256 x double> %0, <256 x double> %1) {
; CHECK-LABEL: pvldzlo_vvvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 128
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    pvldz.lo %v1, %v0
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.pvldzlo.vvvl(<256 x double> %0, <256 x double> %1, i32 128)
  ret <256 x double> %3
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.pvldzlo.vvvl(<256 x double>, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @pvldzlo_vvmvl(<256 x double> %0, <256 x i1> %1, <256 x double> %2) {
; CHECK-LABEL: pvldzlo_vvmvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 128
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    pvldz.lo %v1, %v0, %vm1
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v1
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = tail call fast <256 x double> @llvm.ve.vl.pvldzlo.vvmvl(<256 x double> %0, <256 x i1> %1, <256 x double> %2, i32 128)
  ret <256 x double> %4
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.pvldzlo.vvmvl(<256 x double>, <256 x i1>, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @pvldzup_vvl(<256 x double> %0) {
; CHECK-LABEL: pvldzup_vvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    pvldz.up %v0, %v0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call fast <256 x double> @llvm.ve.vl.pvldzup.vvl(<256 x double> %0, i32 256)
  ret <256 x double> %2
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.pvldzup.vvl(<256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @pvldzup_vvvl(<256 x double> %0, <256 x double> %1) {
; CHECK-LABEL: pvldzup_vvvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 128
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    pvldz.up %v1, %v0
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.pvldzup.vvvl(<256 x double> %0, <256 x double> %1, i32 128)
  ret <256 x double> %3
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.pvldzup.vvvl(<256 x double>, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @pvldzup_vvmvl(<256 x double> %0, <256 x i1> %1, <256 x double> %2) {
; CHECK-LABEL: pvldzup_vvmvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 128
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    pvldz.up %v1, %v0, %vm1
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v1
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = tail call fast <256 x double> @llvm.ve.vl.pvldzup.vvmvl(<256 x double> %0, <256 x i1> %1, <256 x double> %2, i32 128)
  ret <256 x double> %4
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.pvldzup.vvmvl(<256 x double>, <256 x i1>, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @pvldz_vvl(<256 x double> %0) {
; CHECK-LABEL: pvldz_vvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    pvldz %v0, %v0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call fast <256 x double> @llvm.ve.vl.pvldz.vvl(<256 x double> %0, i32 256)
  ret <256 x double> %2
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.pvldz.vvl(<256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @pvldz_vvvl(<256 x double> %0, <256 x double> %1) {
; CHECK-LABEL: pvldz_vvvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 128
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    pvldz %v1, %v0
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.pvldz.vvvl(<256 x double> %0, <256 x double> %1, i32 128)
  ret <256 x double> %3
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.pvldz.vvvl(<256 x double>, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @pvldz_vvMvl(<256 x double> %0, <512 x i1> %1, <256 x double> %2) {
; CHECK-LABEL: pvldz_vvMvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 128
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    pvldz %v1, %v0, %vm2
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v1
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = tail call fast <256 x double> @llvm.ve.vl.pvldz.vvMvl(<256 x double> %0, <512 x i1> %1, <256 x double> %2, i32 128)
  ret <256 x double> %4
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.pvldz.vvMvl(<256 x double>, <512 x i1>, <256 x double>, i32)
