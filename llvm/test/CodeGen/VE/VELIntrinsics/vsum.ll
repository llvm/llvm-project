; RUN: llc < %s -mtriple=ve -mattr=+vpu | FileCheck %s

;;; Test vector sum intrinsic instructions
;;;
;;; Note:
;;;   We test VSUM*vl and VSUM*vml instructions.

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vsumwsx_vvl(<256 x double> %0) {
; CHECK-LABEL: vsumwsx_vvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vsum.w.sx %v0, %v0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call fast <256 x double> @llvm.ve.vl.vsumwsx.vvl(<256 x double> %0, i32 256)
  ret <256 x double> %2
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vsumwsx.vvl(<256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vsumwsx_vvml(<256 x double> %0, <256 x i1> %1) {
; CHECK-LABEL: vsumwsx_vvml:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vsum.w.sx %v0, %v0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.vsumwsx.vvml(<256 x double> %0, <256 x i1> %1, i32 256)
  ret <256 x double> %3
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vsumwsx.vvml(<256 x double>, <256 x i1>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vsumwzx_vvl(<256 x double> %0) {
; CHECK-LABEL: vsumwzx_vvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vsum.w.zx %v0, %v0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call fast <256 x double> @llvm.ve.vl.vsumwzx.vvl(<256 x double> %0, i32 256)
  ret <256 x double> %2
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vsumwzx.vvl(<256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vsumwzx_vvml(<256 x double> %0, <256 x i1> %1) {
; CHECK-LABEL: vsumwzx_vvml:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vsum.w.zx %v0, %v0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.vsumwzx.vvml(<256 x double> %0, <256 x i1> %1, i32 256)
  ret <256 x double> %3
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vsumwzx.vvml(<256 x double>, <256 x i1>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vsuml_vvl(<256 x double> %0) {
; CHECK-LABEL: vsuml_vvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vsum.l %v0, %v0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call fast <256 x double> @llvm.ve.vl.vsuml.vvl(<256 x double> %0, i32 256)
  ret <256 x double> %2
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vsuml.vvl(<256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vsuml_vvml(<256 x double> %0, <256 x i1> %1) {
; CHECK-LABEL: vsuml_vvml:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vsum.l %v0, %v0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.vsuml.vvml(<256 x double> %0, <256 x i1> %1, i32 256)
  ret <256 x double> %3
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vsuml.vvml(<256 x double>, <256 x i1>, i32)
