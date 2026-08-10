; RUN: llc < %s -mtriple=ve -mattr=+vpu | FileCheck %s

;;; Test vector sequential number intrinsic instructions
;;;
;;; Note:
;;;   We test VSEQ*l, VSEQ*l_v, PVSEQ*l, and PVSEQ*l_v instructions.

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vseq_vl() {
; CHECK-LABEL: vseq_vl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vseq %v0
; CHECK-NEXT:    b.l.t (, %s10)
  %1 = tail call fast <256 x double> @llvm.ve.vl.vseq.vl(i32 256)
  ret <256 x double> %1
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vseq.vl(i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vseq_vvl(<256 x double> %0) {
; CHECK-LABEL: vseq_vvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vseq %v0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call fast <256 x double> @llvm.ve.vl.vseq.vvl(<256 x double> %0, i32 256)
  ret <256 x double> %2
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vseq.vvl(<256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @pvseqlo_vl() {
; CHECK-LABEL: pvseqlo_vl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    pvseq.lo %v0
; CHECK-NEXT:    b.l.t (, %s10)
  %1 = tail call fast <256 x double> @llvm.ve.vl.pvseqlo.vl(i32 256)
  ret <256 x double> %1
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.pvseqlo.vl(i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @pvseqlo_vvl(<256 x double> %0) {
; CHECK-LABEL: pvseqlo_vvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    pvseq.lo %v0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call fast <256 x double> @llvm.ve.vl.pvseqlo.vvl(<256 x double> %0, i32 256)
  ret <256 x double> %2
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.pvseqlo.vvl(<256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @pvsequp_vl() {
; CHECK-LABEL: pvsequp_vl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    pvseq.up %v0
; CHECK-NEXT:    b.l.t (, %s10)
  %1 = tail call fast <256 x double> @llvm.ve.vl.pvsequp.vl(i32 256)
  ret <256 x double> %1
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.pvsequp.vl(i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @pvsequp_vvl(<256 x double> %0) {
; CHECK-LABEL: pvsequp_vvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    pvseq.up %v0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call fast <256 x double> @llvm.ve.vl.pvsequp.vvl(<256 x double> %0, i32 256)
  ret <256 x double> %2
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.pvsequp.vvl(<256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @pvseq_vl() {
; CHECK-LABEL: pvseq_vl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    pvseq %v0
; CHECK-NEXT:    b.l.t (, %s10)
  %1 = tail call fast <256 x double> @llvm.ve.vl.pvseq.vl(i32 256)
  ret <256 x double> %1
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.pvseq.vl(i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @pvseq_vvl(<256 x double> %0) {
; CHECK-LABEL: pvseq_vvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    pvseq %v0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call fast <256 x double> @llvm.ve.vl.pvseq.vvl(<256 x double> %0, i32 256)
  ret <256 x double> %2
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.pvseq.vvl(<256 x double>, i32)
