; RUN: llc < %s -mtriple=ve -mattr=+vpu | FileCheck %s

;;; Test vector floating reciprocal intrinsic instructions
;;;
;;; Note:
;;;   We test VRCP*vl, VRCP*vl_v, PVRCP*vl, and PVRCP*vl_v instructions.

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vrcpd_vvl(<256 x double> %0) {
; CHECK-LABEL: vrcpd_vvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vrcp.d %v0, %v0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call fast <256 x double> @llvm.ve.vl.vrcpd.vvl(<256 x double> %0, i32 256)
  ret <256 x double> %2
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vrcpd.vvl(<256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vrcpd_vvvl(<256 x double> %0, <256 x double> %1) {
; CHECK-LABEL: vrcpd_vvvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 128
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vrcp.d %v1, %v0
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.vrcpd.vvvl(<256 x double> %0, <256 x double> %1, i32 128)
  ret <256 x double> %3
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vrcpd.vvvl(<256 x double>, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vrcps_vvl(<256 x double> %0) {
; CHECK-LABEL: vrcps_vvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vrcp.s %v0, %v0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call fast <256 x double> @llvm.ve.vl.vrcps.vvl(<256 x double> %0, i32 256)
  ret <256 x double> %2
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vrcps.vvl(<256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vrcps_vvvl(<256 x double> %0, <256 x double> %1) {
; CHECK-LABEL: vrcps_vvvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 128
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vrcp.s %v1, %v0
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.vrcps.vvvl(<256 x double> %0, <256 x double> %1, i32 128)
  ret <256 x double> %3
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vrcps.vvvl(<256 x double>, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @pvrcp_vvl(<256 x double> %0) {
; CHECK-LABEL: pvrcp_vvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    pvrcp %v0, %v0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call fast <256 x double> @llvm.ve.vl.pvrcp.vvl(<256 x double> %0, i32 256)
  ret <256 x double> %2
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.pvrcp.vvl(<256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @pvrcp_vvvl(<256 x double> %0, <256 x double> %1) {
; CHECK-LABEL: pvrcp_vvvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 128
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    pvrcp %v1, %v0
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.pvrcp.vvvl(<256 x double> %0, <256 x double> %1, i32 128)
  ret <256 x double> %3
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.pvrcp.vvvl(<256 x double>, <256 x double>, i32)
