; RUN: llc < %s -mtriple=ve -mattr=+vpu | FileCheck %s

;;; Test vector floating reciprocal square root intrinsic instructions
;;;
;;; Note:
;;;   We test VRSQRT*vl, VRSQRT*vl_v, PVRSQRT*vl, and PVRSQRT*vl_v
;;;   instructions.

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vrsqrtd_vvl(<256 x double> %0) {
; CHECK-LABEL: vrsqrtd_vvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vrsqrt.d %v0, %v0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call fast <256 x double> @llvm.ve.vl.vrsqrtd.vvl(<256 x double> %0, i32 256)
  ret <256 x double> %2
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vrsqrtd.vvl(<256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vrsqrtd_vvvl(<256 x double> %0, <256 x double> %1) {
; CHECK-LABEL: vrsqrtd_vvvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 128
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vrsqrt.d %v1, %v0
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.vrsqrtd.vvvl(<256 x double> %0, <256 x double> %1, i32 128)
  ret <256 x double> %3
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vrsqrtd.vvvl(<256 x double>, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vrsqrts_vvl(<256 x double> %0) {
; CHECK-LABEL: vrsqrts_vvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vrsqrt.s %v0, %v0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call fast <256 x double> @llvm.ve.vl.vrsqrts.vvl(<256 x double> %0, i32 256)
  ret <256 x double> %2
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vrsqrts.vvl(<256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vrsqrts_vvvl(<256 x double> %0, <256 x double> %1) {
; CHECK-LABEL: vrsqrts_vvvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 128
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vrsqrt.s %v1, %v0
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.vrsqrts.vvvl(<256 x double> %0, <256 x double> %1, i32 128)
  ret <256 x double> %3
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vrsqrts.vvvl(<256 x double>, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vrsqrtdnex_vvl(<256 x double> %0) {
; CHECK-LABEL: vrsqrtdnex_vvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vrsqrt.d.nex %v0, %v0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call fast <256 x double> @llvm.ve.vl.vrsqrtdnex.vvl(<256 x double> %0, i32 256)
  ret <256 x double> %2
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vrsqrtdnex.vvl(<256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vrsqrtdnex_vvvl(<256 x double> %0, <256 x double> %1) {
; CHECK-LABEL: vrsqrtdnex_vvvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 128
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vrsqrt.d.nex %v1, %v0
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.vrsqrtdnex.vvvl(<256 x double> %0, <256 x double> %1, i32 128)
  ret <256 x double> %3
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vrsqrtdnex.vvvl(<256 x double>, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vrsqrtsnex_vvl(<256 x double> %0) {
; CHECK-LABEL: vrsqrtsnex_vvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vrsqrt.s.nex %v0, %v0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call fast <256 x double> @llvm.ve.vl.vrsqrtsnex.vvl(<256 x double> %0, i32 256)
  ret <256 x double> %2
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vrsqrtsnex.vvl(<256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vrsqrtsnex_vvvl(<256 x double> %0, <256 x double> %1) {
; CHECK-LABEL: vrsqrtsnex_vvvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 128
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vrsqrt.s.nex %v1, %v0
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.vrsqrtsnex.vvvl(<256 x double> %0, <256 x double> %1, i32 128)
  ret <256 x double> %3
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vrsqrtsnex.vvvl(<256 x double>, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @pvrsqrt_vvl(<256 x double> %0) {
; CHECK-LABEL: pvrsqrt_vvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    pvrsqrt %v0, %v0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call fast <256 x double> @llvm.ve.vl.pvrsqrt.vvl(<256 x double> %0, i32 256)
  ret <256 x double> %2
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.pvrsqrt.vvl(<256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @pvrsqrt_vvvl(<256 x double> %0, <256 x double> %1) {
; CHECK-LABEL: pvrsqrt_vvvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 128
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    pvrsqrt %v1, %v0
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.pvrsqrt.vvvl(<256 x double> %0, <256 x double> %1, i32 128)
  ret <256 x double> %3
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.pvrsqrt.vvvl(<256 x double>, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @pvrsqrtnex_vvl(<256 x double> %0) {
; CHECK-LABEL: pvrsqrtnex_vvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    pvrsqrt.nex %v0, %v0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call fast <256 x double> @llvm.ve.vl.pvrsqrtnex.vvl(<256 x double> %0, i32 256)
  ret <256 x double> %2
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.pvrsqrtnex.vvl(<256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @pvrsqrtnex_vvvl(<256 x double> %0, <256 x double> %1) {
; CHECK-LABEL: pvrsqrtnex_vvvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 128
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    pvrsqrt.nex %v1, %v0
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.pvrsqrtnex.vvvl(<256 x double> %0, <256 x double> %1, i32 128)
  ret <256 x double> %3
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.pvrsqrtnex.vvvl(<256 x double>, <256 x double>, i32)
