; RUN: llc < %s -mtriple=ve -mattr=+vpu | FileCheck %s

;;; Test vector form mask intrinsic instructions
;;;
;;; Note:
;;;   We test VFMK*al, VFMK*nal, VFMK*vl, VFMK*vml, PVFMK*yal, PVFMK*ynal,
;;;   PVFMK*vl, PVFMK*vml, PVFMK*yvl, and PVFMK*yvyl instructions.

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @vfmklat_ml() {
; CHECK-LABEL: vfmklat_ml:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vfmk.l.at %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %1 = tail call <256 x i1> @llvm.ve.vl.vfmklat.ml(i32 256)
  ret <256 x i1> %1
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.vfmklat.ml(i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @vfmklaf_ml() {
; CHECK-LABEL: vfmklaf_ml:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vfmk.l.af %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %1 = tail call <256 x i1> @llvm.ve.vl.vfmklaf.ml(i32 256)
  ret <256 x i1> %1
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.vfmklaf.ml(i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @vfmklgt_mvl(<256 x double> %0) {
; CHECK-LABEL: vfmklgt_mvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vfmk.l.gt %vm1, %v0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call <256 x i1> @llvm.ve.vl.vfmklgt.mvl(<256 x double> %0, i32 256)
  ret <256 x i1> %2
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.vfmklgt.mvl(<256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @vfmklgt_mvml(<256 x double> %0, <256 x i1> %1) {
; CHECK-LABEL: vfmklgt_mvml:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vfmk.l.gt %vm1, %v0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call <256 x i1> @llvm.ve.vl.vfmklgt.mvml(<256 x double> %0, <256 x i1> %1, i32 256)
  ret <256 x i1> %3
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.vfmklgt.mvml(<256 x double>, <256 x i1>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @vfmkllt_mvl(<256 x double> %0) {
; CHECK-LABEL: vfmkllt_mvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vfmk.l.lt %vm1, %v0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call <256 x i1> @llvm.ve.vl.vfmkllt.mvl(<256 x double> %0, i32 256)
  ret <256 x i1> %2
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.vfmkllt.mvl(<256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @vfmkllt_mvml(<256 x double> %0, <256 x i1> %1) {
; CHECK-LABEL: vfmkllt_mvml:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vfmk.l.lt %vm1, %v0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call <256 x i1> @llvm.ve.vl.vfmkllt.mvml(<256 x double> %0, <256 x i1> %1, i32 256)
  ret <256 x i1> %3
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.vfmkllt.mvml(<256 x double>, <256 x i1>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @vfmklne_mvl(<256 x double> %0) {
; CHECK-LABEL: vfmklne_mvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vfmk.l.ne %vm1, %v0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call <256 x i1> @llvm.ve.vl.vfmklne.mvl(<256 x double> %0, i32 256)
  ret <256 x i1> %2
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.vfmklne.mvl(<256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @vfmklne_mvml(<256 x double> %0, <256 x i1> %1) {
; CHECK-LABEL: vfmklne_mvml:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vfmk.l.ne %vm1, %v0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call <256 x i1> @llvm.ve.vl.vfmklne.mvml(<256 x double> %0, <256 x i1> %1, i32 256)
  ret <256 x i1> %3
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.vfmklne.mvml(<256 x double>, <256 x i1>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @vfmkleq_mvl(<256 x double> %0) {
; CHECK-LABEL: vfmkleq_mvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vfmk.l.eq %vm1, %v0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call <256 x i1> @llvm.ve.vl.vfmkleq.mvl(<256 x double> %0, i32 256)
  ret <256 x i1> %2
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.vfmkleq.mvl(<256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @vfmkleq_mvml(<256 x double> %0, <256 x i1> %1) {
; CHECK-LABEL: vfmkleq_mvml:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vfmk.l.eq %vm1, %v0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call <256 x i1> @llvm.ve.vl.vfmkleq.mvml(<256 x double> %0, <256 x i1> %1, i32 256)
  ret <256 x i1> %3
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.vfmkleq.mvml(<256 x double>, <256 x i1>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @vfmklge_mvl(<256 x double> %0) {
; CHECK-LABEL: vfmklge_mvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vfmk.l.ge %vm1, %v0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call <256 x i1> @llvm.ve.vl.vfmklge.mvl(<256 x double> %0, i32 256)
  ret <256 x i1> %2
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.vfmklge.mvl(<256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @vfmklge_mvml(<256 x double> %0, <256 x i1> %1) {
; CHECK-LABEL: vfmklge_mvml:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vfmk.l.ge %vm1, %v0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call <256 x i1> @llvm.ve.vl.vfmklge.mvml(<256 x double> %0, <256 x i1> %1, i32 256)
  ret <256 x i1> %3
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.vfmklge.mvml(<256 x double>, <256 x i1>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @vfmklle_mvl(<256 x double> %0) {
; CHECK-LABEL: vfmklle_mvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vfmk.l.le %vm1, %v0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call <256 x i1> @llvm.ve.vl.vfmklle.mvl(<256 x double> %0, i32 256)
  ret <256 x i1> %2
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.vfmklle.mvl(<256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @vfmklle_mvml(<256 x double> %0, <256 x i1> %1) {
; CHECK-LABEL: vfmklle_mvml:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vfmk.l.le %vm1, %v0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call <256 x i1> @llvm.ve.vl.vfmklle.mvml(<256 x double> %0, <256 x i1> %1, i32 256)
  ret <256 x i1> %3
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.vfmklle.mvml(<256 x double>, <256 x i1>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @vfmklnum_mvl(<256 x double> %0) {
; CHECK-LABEL: vfmklnum_mvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vfmk.l.num %vm1, %v0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call <256 x i1> @llvm.ve.vl.vfmklnum.mvl(<256 x double> %0, i32 256)
  ret <256 x i1> %2
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.vfmklnum.mvl(<256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @vfmklnum_mvml(<256 x double> %0, <256 x i1> %1) {
; CHECK-LABEL: vfmklnum_mvml:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vfmk.l.num %vm1, %v0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call <256 x i1> @llvm.ve.vl.vfmklnum.mvml(<256 x double> %0, <256 x i1> %1, i32 256)
  ret <256 x i1> %3
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.vfmklnum.mvml(<256 x double>, <256 x i1>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @vfmklnan_mvl(<256 x double> %0) {
; CHECK-LABEL: vfmklnan_mvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vfmk.l.nan %vm1, %v0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call <256 x i1> @llvm.ve.vl.vfmklnan.mvl(<256 x double> %0, i32 256)
  ret <256 x i1> %2
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.vfmklnan.mvl(<256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @vfmklnan_mvml(<256 x double> %0, <256 x i1> %1) {
; CHECK-LABEL: vfmklnan_mvml:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vfmk.l.nan %vm1, %v0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call <256 x i1> @llvm.ve.vl.vfmklnan.mvml(<256 x double> %0, <256 x i1> %1, i32 256)
  ret <256 x i1> %3
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.vfmklnan.mvml(<256 x double>, <256 x i1>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @vfmklgtnan_mvl(<256 x double> %0) {
; CHECK-LABEL: vfmklgtnan_mvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vfmk.l.gtnan %vm1, %v0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call <256 x i1> @llvm.ve.vl.vfmklgtnan.mvl(<256 x double> %0, i32 256)
  ret <256 x i1> %2
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.vfmklgtnan.mvl(<256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @vfmklgtnan_mvml(<256 x double> %0, <256 x i1> %1) {
; CHECK-LABEL: vfmklgtnan_mvml:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vfmk.l.gtnan %vm1, %v0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call <256 x i1> @llvm.ve.vl.vfmklgtnan.mvml(<256 x double> %0, <256 x i1> %1, i32 256)
  ret <256 x i1> %3
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.vfmklgtnan.mvml(<256 x double>, <256 x i1>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @vfmklltnan_mvl(<256 x double> %0) {
; CHECK-LABEL: vfmklltnan_mvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vfmk.l.ltnan %vm1, %v0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call <256 x i1> @llvm.ve.vl.vfmklltnan.mvl(<256 x double> %0, i32 256)
  ret <256 x i1> %2
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.vfmklltnan.mvl(<256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @vfmklltnan_mvml(<256 x double> %0, <256 x i1> %1) {
; CHECK-LABEL: vfmklltnan_mvml:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vfmk.l.ltnan %vm1, %v0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call <256 x i1> @llvm.ve.vl.vfmklltnan.mvml(<256 x double> %0, <256 x i1> %1, i32 256)
  ret <256 x i1> %3
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.vfmklltnan.mvml(<256 x double>, <256 x i1>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @vfmklnenan_mvl(<256 x double> %0) {
; CHECK-LABEL: vfmklnenan_mvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vfmk.l.nenan %vm1, %v0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call <256 x i1> @llvm.ve.vl.vfmklnenan.mvl(<256 x double> %0, i32 256)
  ret <256 x i1> %2
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.vfmklnenan.mvl(<256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @vfmklnenan_mvml(<256 x double> %0, <256 x i1> %1) {
; CHECK-LABEL: vfmklnenan_mvml:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vfmk.l.nenan %vm1, %v0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call <256 x i1> @llvm.ve.vl.vfmklnenan.mvml(<256 x double> %0, <256 x i1> %1, i32 256)
  ret <256 x i1> %3
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.vfmklnenan.mvml(<256 x double>, <256 x i1>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @vfmkleqnan_mvl(<256 x double> %0) {
; CHECK-LABEL: vfmkleqnan_mvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vfmk.l.eqnan %vm1, %v0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call <256 x i1> @llvm.ve.vl.vfmkleqnan.mvl(<256 x double> %0, i32 256)
  ret <256 x i1> %2
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.vfmkleqnan.mvl(<256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @vfmkleqnan_mvml(<256 x double> %0, <256 x i1> %1) {
; CHECK-LABEL: vfmkleqnan_mvml:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vfmk.l.eqnan %vm1, %v0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call <256 x i1> @llvm.ve.vl.vfmkleqnan.mvml(<256 x double> %0, <256 x i1> %1, i32 256)
  ret <256 x i1> %3
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.vfmkleqnan.mvml(<256 x double>, <256 x i1>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @vfmklgenan_mvl(<256 x double> %0) {
; CHECK-LABEL: vfmklgenan_mvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vfmk.l.genan %vm1, %v0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call <256 x i1> @llvm.ve.vl.vfmklgenan.mvl(<256 x double> %0, i32 256)
  ret <256 x i1> %2
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.vfmklgenan.mvl(<256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @vfmklgenan_mvml(<256 x double> %0, <256 x i1> %1) {
; CHECK-LABEL: vfmklgenan_mvml:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vfmk.l.genan %vm1, %v0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call <256 x i1> @llvm.ve.vl.vfmklgenan.mvml(<256 x double> %0, <256 x i1> %1, i32 256)
  ret <256 x i1> %3
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.vfmklgenan.mvml(<256 x double>, <256 x i1>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @vfmkllenan_mvl(<256 x double> %0) {
; CHECK-LABEL: vfmkllenan_mvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vfmk.l.lenan %vm1, %v0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call <256 x i1> @llvm.ve.vl.vfmkllenan.mvl(<256 x double> %0, i32 256)
  ret <256 x i1> %2
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.vfmkllenan.mvl(<256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @vfmkllenan_mvml(<256 x double> %0, <256 x i1> %1) {
; CHECK-LABEL: vfmkllenan_mvml:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vfmk.l.lenan %vm1, %v0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call <256 x i1> @llvm.ve.vl.vfmkllenan.mvml(<256 x double> %0, <256 x i1> %1, i32 256)
  ret <256 x i1> %3
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.vfmkllenan.mvml(<256 x double>, <256 x i1>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @vfmkwgt_mvl(<256 x double> %0) {
; CHECK-LABEL: vfmkwgt_mvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vfmk.w.gt %vm1, %v0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call <256 x i1> @llvm.ve.vl.vfmkwgt.mvl(<256 x double> %0, i32 256)
  ret <256 x i1> %2
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.vfmkwgt.mvl(<256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @vfmkwgt_mvml(<256 x double> %0, <256 x i1> %1) {
; CHECK-LABEL: vfmkwgt_mvml:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vfmk.w.gt %vm1, %v0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call <256 x i1> @llvm.ve.vl.vfmkwgt.mvml(<256 x double> %0, <256 x i1> %1, i32 256)
  ret <256 x i1> %3
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.vfmkwgt.mvml(<256 x double>, <256 x i1>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @vfmkwlt_mvl(<256 x double> %0) {
; CHECK-LABEL: vfmkwlt_mvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vfmk.w.lt %vm1, %v0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call <256 x i1> @llvm.ve.vl.vfmkwlt.mvl(<256 x double> %0, i32 256)
  ret <256 x i1> %2
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.vfmkwlt.mvl(<256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @vfmkwlt_mvml(<256 x double> %0, <256 x i1> %1) {
; CHECK-LABEL: vfmkwlt_mvml:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vfmk.w.lt %vm1, %v0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call <256 x i1> @llvm.ve.vl.vfmkwlt.mvml(<256 x double> %0, <256 x i1> %1, i32 256)
  ret <256 x i1> %3
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.vfmkwlt.mvml(<256 x double>, <256 x i1>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @vfmkwne_mvl(<256 x double> %0) {
; CHECK-LABEL: vfmkwne_mvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vfmk.w.ne %vm1, %v0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call <256 x i1> @llvm.ve.vl.vfmkwne.mvl(<256 x double> %0, i32 256)
  ret <256 x i1> %2
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.vfmkwne.mvl(<256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @vfmkwne_mvml(<256 x double> %0, <256 x i1> %1) {
; CHECK-LABEL: vfmkwne_mvml:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vfmk.w.ne %vm1, %v0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call <256 x i1> @llvm.ve.vl.vfmkwne.mvml(<256 x double> %0, <256 x i1> %1, i32 256)
  ret <256 x i1> %3
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.vfmkwne.mvml(<256 x double>, <256 x i1>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @vfmkweq_mvl(<256 x double> %0) {
; CHECK-LABEL: vfmkweq_mvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vfmk.w.eq %vm1, %v0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call <256 x i1> @llvm.ve.vl.vfmkweq.mvl(<256 x double> %0, i32 256)
  ret <256 x i1> %2
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.vfmkweq.mvl(<256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @vfmkweq_mvml(<256 x double> %0, <256 x i1> %1) {
; CHECK-LABEL: vfmkweq_mvml:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vfmk.w.eq %vm1, %v0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call <256 x i1> @llvm.ve.vl.vfmkweq.mvml(<256 x double> %0, <256 x i1> %1, i32 256)
  ret <256 x i1> %3
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.vfmkweq.mvml(<256 x double>, <256 x i1>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @vfmkwge_mvl(<256 x double> %0) {
; CHECK-LABEL: vfmkwge_mvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vfmk.w.ge %vm1, %v0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call <256 x i1> @llvm.ve.vl.vfmkwge.mvl(<256 x double> %0, i32 256)
  ret <256 x i1> %2
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.vfmkwge.mvl(<256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @vfmkwge_mvml(<256 x double> %0, <256 x i1> %1) {
; CHECK-LABEL: vfmkwge_mvml:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vfmk.w.ge %vm1, %v0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call <256 x i1> @llvm.ve.vl.vfmkwge.mvml(<256 x double> %0, <256 x i1> %1, i32 256)
  ret <256 x i1> %3
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.vfmkwge.mvml(<256 x double>, <256 x i1>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @vfmkwle_mvl(<256 x double> %0) {
; CHECK-LABEL: vfmkwle_mvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vfmk.w.le %vm1, %v0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call <256 x i1> @llvm.ve.vl.vfmkwle.mvl(<256 x double> %0, i32 256)
  ret <256 x i1> %2
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.vfmkwle.mvl(<256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @vfmkwle_mvml(<256 x double> %0, <256 x i1> %1) {
; CHECK-LABEL: vfmkwle_mvml:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vfmk.w.le %vm1, %v0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call <256 x i1> @llvm.ve.vl.vfmkwle.mvml(<256 x double> %0, <256 x i1> %1, i32 256)
  ret <256 x i1> %3
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.vfmkwle.mvml(<256 x double>, <256 x i1>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @vfmkwnum_mvl(<256 x double> %0) {
; CHECK-LABEL: vfmkwnum_mvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vfmk.w.num %vm1, %v0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call <256 x i1> @llvm.ve.vl.vfmkwnum.mvl(<256 x double> %0, i32 256)
  ret <256 x i1> %2
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.vfmkwnum.mvl(<256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @vfmkwnum_mvml(<256 x double> %0, <256 x i1> %1) {
; CHECK-LABEL: vfmkwnum_mvml:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vfmk.w.num %vm1, %v0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call <256 x i1> @llvm.ve.vl.vfmkwnum.mvml(<256 x double> %0, <256 x i1> %1, i32 256)
  ret <256 x i1> %3
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.vfmkwnum.mvml(<256 x double>, <256 x i1>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @vfmkwnan_mvl(<256 x double> %0) {
; CHECK-LABEL: vfmkwnan_mvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vfmk.w.nan %vm1, %v0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call <256 x i1> @llvm.ve.vl.vfmkwnan.mvl(<256 x double> %0, i32 256)
  ret <256 x i1> %2
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.vfmkwnan.mvl(<256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @vfmkwnan_mvml(<256 x double> %0, <256 x i1> %1) {
; CHECK-LABEL: vfmkwnan_mvml:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vfmk.w.nan %vm1, %v0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call <256 x i1> @llvm.ve.vl.vfmkwnan.mvml(<256 x double> %0, <256 x i1> %1, i32 256)
  ret <256 x i1> %3
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.vfmkwnan.mvml(<256 x double>, <256 x i1>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @vfmkwgtnan_mvl(<256 x double> %0) {
; CHECK-LABEL: vfmkwgtnan_mvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vfmk.w.gtnan %vm1, %v0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call <256 x i1> @llvm.ve.vl.vfmkwgtnan.mvl(<256 x double> %0, i32 256)
  ret <256 x i1> %2
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.vfmkwgtnan.mvl(<256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @vfmkwgtnan_mvml(<256 x double> %0, <256 x i1> %1) {
; CHECK-LABEL: vfmkwgtnan_mvml:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vfmk.w.gtnan %vm1, %v0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call <256 x i1> @llvm.ve.vl.vfmkwgtnan.mvml(<256 x double> %0, <256 x i1> %1, i32 256)
  ret <256 x i1> %3
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.vfmkwgtnan.mvml(<256 x double>, <256 x i1>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @vfmkwltnan_mvl(<256 x double> %0) {
; CHECK-LABEL: vfmkwltnan_mvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vfmk.w.ltnan %vm1, %v0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call <256 x i1> @llvm.ve.vl.vfmkwltnan.mvl(<256 x double> %0, i32 256)
  ret <256 x i1> %2
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.vfmkwltnan.mvl(<256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @vfmkwltnan_mvml(<256 x double> %0, <256 x i1> %1) {
; CHECK-LABEL: vfmkwltnan_mvml:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vfmk.w.ltnan %vm1, %v0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call <256 x i1> @llvm.ve.vl.vfmkwltnan.mvml(<256 x double> %0, <256 x i1> %1, i32 256)
  ret <256 x i1> %3
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.vfmkwltnan.mvml(<256 x double>, <256 x i1>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @vfmkwnenan_mvl(<256 x double> %0) {
; CHECK-LABEL: vfmkwnenan_mvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vfmk.w.nenan %vm1, %v0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call <256 x i1> @llvm.ve.vl.vfmkwnenan.mvl(<256 x double> %0, i32 256)
  ret <256 x i1> %2
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.vfmkwnenan.mvl(<256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @vfmkwnenan_mvml(<256 x double> %0, <256 x i1> %1) {
; CHECK-LABEL: vfmkwnenan_mvml:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vfmk.w.nenan %vm1, %v0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call <256 x i1> @llvm.ve.vl.vfmkwnenan.mvml(<256 x double> %0, <256 x i1> %1, i32 256)
  ret <256 x i1> %3
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.vfmkwnenan.mvml(<256 x double>, <256 x i1>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @vfmkweqnan_mvl(<256 x double> %0) {
; CHECK-LABEL: vfmkweqnan_mvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vfmk.w.eqnan %vm1, %v0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call <256 x i1> @llvm.ve.vl.vfmkweqnan.mvl(<256 x double> %0, i32 256)
  ret <256 x i1> %2
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.vfmkweqnan.mvl(<256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @vfmkweqnan_mvml(<256 x double> %0, <256 x i1> %1) {
; CHECK-LABEL: vfmkweqnan_mvml:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vfmk.w.eqnan %vm1, %v0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call <256 x i1> @llvm.ve.vl.vfmkweqnan.mvml(<256 x double> %0, <256 x i1> %1, i32 256)
  ret <256 x i1> %3
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.vfmkweqnan.mvml(<256 x double>, <256 x i1>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @vfmkwgenan_mvl(<256 x double> %0) {
; CHECK-LABEL: vfmkwgenan_mvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vfmk.w.genan %vm1, %v0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call <256 x i1> @llvm.ve.vl.vfmkwgenan.mvl(<256 x double> %0, i32 256)
  ret <256 x i1> %2
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.vfmkwgenan.mvl(<256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @vfmkwgenan_mvml(<256 x double> %0, <256 x i1> %1) {
; CHECK-LABEL: vfmkwgenan_mvml:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vfmk.w.genan %vm1, %v0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call <256 x i1> @llvm.ve.vl.vfmkwgenan.mvml(<256 x double> %0, <256 x i1> %1, i32 256)
  ret <256 x i1> %3
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.vfmkwgenan.mvml(<256 x double>, <256 x i1>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @vfmkwlenan_mvl(<256 x double> %0) {
; CHECK-LABEL: vfmkwlenan_mvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vfmk.w.lenan %vm1, %v0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call <256 x i1> @llvm.ve.vl.vfmkwlenan.mvl(<256 x double> %0, i32 256)
  ret <256 x i1> %2
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.vfmkwlenan.mvl(<256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @vfmkwlenan_mvml(<256 x double> %0, <256 x i1> %1) {
; CHECK-LABEL: vfmkwlenan_mvml:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vfmk.w.lenan %vm1, %v0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call <256 x i1> @llvm.ve.vl.vfmkwlenan.mvml(<256 x double> %0, <256 x i1> %1, i32 256)
  ret <256 x i1> %3
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.vfmkwlenan.mvml(<256 x double>, <256 x i1>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @vfmkdgt_mvl(<256 x double> %0) {
; CHECK-LABEL: vfmkdgt_mvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vfmk.d.gt %vm1, %v0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call <256 x i1> @llvm.ve.vl.vfmkdgt.mvl(<256 x double> %0, i32 256)
  ret <256 x i1> %2
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.vfmkdgt.mvl(<256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @vfmkdgt_mvml(<256 x double> %0, <256 x i1> %1) {
; CHECK-LABEL: vfmkdgt_mvml:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vfmk.d.gt %vm1, %v0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call <256 x i1> @llvm.ve.vl.vfmkdgt.mvml(<256 x double> %0, <256 x i1> %1, i32 256)
  ret <256 x i1> %3
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.vfmkdgt.mvml(<256 x double>, <256 x i1>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @vfmkdlt_mvl(<256 x double> %0) {
; CHECK-LABEL: vfmkdlt_mvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vfmk.d.lt %vm1, %v0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call <256 x i1> @llvm.ve.vl.vfmkdlt.mvl(<256 x double> %0, i32 256)
  ret <256 x i1> %2
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.vfmkdlt.mvl(<256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @vfmkdlt_mvml(<256 x double> %0, <256 x i1> %1) {
; CHECK-LABEL: vfmkdlt_mvml:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vfmk.d.lt %vm1, %v0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call <256 x i1> @llvm.ve.vl.vfmkdlt.mvml(<256 x double> %0, <256 x i1> %1, i32 256)
  ret <256 x i1> %3
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.vfmkdlt.mvml(<256 x double>, <256 x i1>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @vfmkdne_mvl(<256 x double> %0) {
; CHECK-LABEL: vfmkdne_mvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vfmk.d.ne %vm1, %v0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call <256 x i1> @llvm.ve.vl.vfmkdne.mvl(<256 x double> %0, i32 256)
  ret <256 x i1> %2
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.vfmkdne.mvl(<256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @vfmkdne_mvml(<256 x double> %0, <256 x i1> %1) {
; CHECK-LABEL: vfmkdne_mvml:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vfmk.d.ne %vm1, %v0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call <256 x i1> @llvm.ve.vl.vfmkdne.mvml(<256 x double> %0, <256 x i1> %1, i32 256)
  ret <256 x i1> %3
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.vfmkdne.mvml(<256 x double>, <256 x i1>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @vfmkdeq_mvl(<256 x double> %0) {
; CHECK-LABEL: vfmkdeq_mvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vfmk.d.eq %vm1, %v0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call <256 x i1> @llvm.ve.vl.vfmkdeq.mvl(<256 x double> %0, i32 256)
  ret <256 x i1> %2
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.vfmkdeq.mvl(<256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @vfmkdeq_mvml(<256 x double> %0, <256 x i1> %1) {
; CHECK-LABEL: vfmkdeq_mvml:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vfmk.d.eq %vm1, %v0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call <256 x i1> @llvm.ve.vl.vfmkdeq.mvml(<256 x double> %0, <256 x i1> %1, i32 256)
  ret <256 x i1> %3
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.vfmkdeq.mvml(<256 x double>, <256 x i1>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @vfmkdge_mvl(<256 x double> %0) {
; CHECK-LABEL: vfmkdge_mvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vfmk.d.ge %vm1, %v0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call <256 x i1> @llvm.ve.vl.vfmkdge.mvl(<256 x double> %0, i32 256)
  ret <256 x i1> %2
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.vfmkdge.mvl(<256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @vfmkdge_mvml(<256 x double> %0, <256 x i1> %1) {
; CHECK-LABEL: vfmkdge_mvml:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vfmk.d.ge %vm1, %v0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call <256 x i1> @llvm.ve.vl.vfmkdge.mvml(<256 x double> %0, <256 x i1> %1, i32 256)
  ret <256 x i1> %3
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.vfmkdge.mvml(<256 x double>, <256 x i1>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @vfmkdle_mvl(<256 x double> %0) {
; CHECK-LABEL: vfmkdle_mvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vfmk.d.le %vm1, %v0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call <256 x i1> @llvm.ve.vl.vfmkdle.mvl(<256 x double> %0, i32 256)
  ret <256 x i1> %2
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.vfmkdle.mvl(<256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @vfmkdle_mvml(<256 x double> %0, <256 x i1> %1) {
; CHECK-LABEL: vfmkdle_mvml:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vfmk.d.le %vm1, %v0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call <256 x i1> @llvm.ve.vl.vfmkdle.mvml(<256 x double> %0, <256 x i1> %1, i32 256)
  ret <256 x i1> %3
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.vfmkdle.mvml(<256 x double>, <256 x i1>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @vfmkdnum_mvl(<256 x double> %0) {
; CHECK-LABEL: vfmkdnum_mvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vfmk.d.num %vm1, %v0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call <256 x i1> @llvm.ve.vl.vfmkdnum.mvl(<256 x double> %0, i32 256)
  ret <256 x i1> %2
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.vfmkdnum.mvl(<256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @vfmkdnum_mvml(<256 x double> %0, <256 x i1> %1) {
; CHECK-LABEL: vfmkdnum_mvml:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vfmk.d.num %vm1, %v0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call <256 x i1> @llvm.ve.vl.vfmkdnum.mvml(<256 x double> %0, <256 x i1> %1, i32 256)
  ret <256 x i1> %3
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.vfmkdnum.mvml(<256 x double>, <256 x i1>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @vfmkdnan_mvl(<256 x double> %0) {
; CHECK-LABEL: vfmkdnan_mvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vfmk.d.nan %vm1, %v0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call <256 x i1> @llvm.ve.vl.vfmkdnan.mvl(<256 x double> %0, i32 256)
  ret <256 x i1> %2
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.vfmkdnan.mvl(<256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @vfmkdnan_mvml(<256 x double> %0, <256 x i1> %1) {
; CHECK-LABEL: vfmkdnan_mvml:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vfmk.d.nan %vm1, %v0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call <256 x i1> @llvm.ve.vl.vfmkdnan.mvml(<256 x double> %0, <256 x i1> %1, i32 256)
  ret <256 x i1> %3
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.vfmkdnan.mvml(<256 x double>, <256 x i1>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @vfmkdgtnan_mvl(<256 x double> %0) {
; CHECK-LABEL: vfmkdgtnan_mvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vfmk.d.gtnan %vm1, %v0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call <256 x i1> @llvm.ve.vl.vfmkdgtnan.mvl(<256 x double> %0, i32 256)
  ret <256 x i1> %2
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.vfmkdgtnan.mvl(<256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @vfmkdgtnan_mvml(<256 x double> %0, <256 x i1> %1) {
; CHECK-LABEL: vfmkdgtnan_mvml:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vfmk.d.gtnan %vm1, %v0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call <256 x i1> @llvm.ve.vl.vfmkdgtnan.mvml(<256 x double> %0, <256 x i1> %1, i32 256)
  ret <256 x i1> %3
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.vfmkdgtnan.mvml(<256 x double>, <256 x i1>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @vfmkdltnan_mvl(<256 x double> %0) {
; CHECK-LABEL: vfmkdltnan_mvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vfmk.d.ltnan %vm1, %v0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call <256 x i1> @llvm.ve.vl.vfmkdltnan.mvl(<256 x double> %0, i32 256)
  ret <256 x i1> %2
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.vfmkdltnan.mvl(<256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @vfmkdltnan_mvml(<256 x double> %0, <256 x i1> %1) {
; CHECK-LABEL: vfmkdltnan_mvml:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vfmk.d.ltnan %vm1, %v0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call <256 x i1> @llvm.ve.vl.vfmkdltnan.mvml(<256 x double> %0, <256 x i1> %1, i32 256)
  ret <256 x i1> %3
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.vfmkdltnan.mvml(<256 x double>, <256 x i1>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @vfmkdnenan_mvl(<256 x double> %0) {
; CHECK-LABEL: vfmkdnenan_mvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vfmk.d.nenan %vm1, %v0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call <256 x i1> @llvm.ve.vl.vfmkdnenan.mvl(<256 x double> %0, i32 256)
  ret <256 x i1> %2
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.vfmkdnenan.mvl(<256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @vfmkdnenan_mvml(<256 x double> %0, <256 x i1> %1) {
; CHECK-LABEL: vfmkdnenan_mvml:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vfmk.d.nenan %vm1, %v0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call <256 x i1> @llvm.ve.vl.vfmkdnenan.mvml(<256 x double> %0, <256 x i1> %1, i32 256)
  ret <256 x i1> %3
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.vfmkdnenan.mvml(<256 x double>, <256 x i1>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @vfmkdeqnan_mvl(<256 x double> %0) {
; CHECK-LABEL: vfmkdeqnan_mvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vfmk.d.eqnan %vm1, %v0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call <256 x i1> @llvm.ve.vl.vfmkdeqnan.mvl(<256 x double> %0, i32 256)
  ret <256 x i1> %2
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.vfmkdeqnan.mvl(<256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @vfmkdeqnan_mvml(<256 x double> %0, <256 x i1> %1) {
; CHECK-LABEL: vfmkdeqnan_mvml:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vfmk.d.eqnan %vm1, %v0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call <256 x i1> @llvm.ve.vl.vfmkdeqnan.mvml(<256 x double> %0, <256 x i1> %1, i32 256)
  ret <256 x i1> %3
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.vfmkdeqnan.mvml(<256 x double>, <256 x i1>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @vfmkdgenan_mvl(<256 x double> %0) {
; CHECK-LABEL: vfmkdgenan_mvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vfmk.d.genan %vm1, %v0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call <256 x i1> @llvm.ve.vl.vfmkdgenan.mvl(<256 x double> %0, i32 256)
  ret <256 x i1> %2
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.vfmkdgenan.mvl(<256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @vfmkdgenan_mvml(<256 x double> %0, <256 x i1> %1) {
; CHECK-LABEL: vfmkdgenan_mvml:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vfmk.d.genan %vm1, %v0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call <256 x i1> @llvm.ve.vl.vfmkdgenan.mvml(<256 x double> %0, <256 x i1> %1, i32 256)
  ret <256 x i1> %3
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.vfmkdgenan.mvml(<256 x double>, <256 x i1>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @vfmkdlenan_mvl(<256 x double> %0) {
; CHECK-LABEL: vfmkdlenan_mvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vfmk.d.lenan %vm1, %v0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call <256 x i1> @llvm.ve.vl.vfmkdlenan.mvl(<256 x double> %0, i32 256)
  ret <256 x i1> %2
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.vfmkdlenan.mvl(<256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @vfmkdlenan_mvml(<256 x double> %0, <256 x i1> %1) {
; CHECK-LABEL: vfmkdlenan_mvml:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vfmk.d.lenan %vm1, %v0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call <256 x i1> @llvm.ve.vl.vfmkdlenan.mvml(<256 x double> %0, <256 x i1> %1, i32 256)
  ret <256 x i1> %3
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.vfmkdlenan.mvml(<256 x double>, <256 x i1>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @vfmksgt_mvl(<256 x double> %0) {
; CHECK-LABEL: vfmksgt_mvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vfmk.s.gt %vm1, %v0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call <256 x i1> @llvm.ve.vl.vfmksgt.mvl(<256 x double> %0, i32 256)
  ret <256 x i1> %2
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.vfmksgt.mvl(<256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @vfmksgt_mvml(<256 x double> %0, <256 x i1> %1) {
; CHECK-LABEL: vfmksgt_mvml:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vfmk.s.gt %vm1, %v0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call <256 x i1> @llvm.ve.vl.vfmksgt.mvml(<256 x double> %0, <256 x i1> %1, i32 256)
  ret <256 x i1> %3
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.vfmksgt.mvml(<256 x double>, <256 x i1>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @vfmkslt_mvl(<256 x double> %0) {
; CHECK-LABEL: vfmkslt_mvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vfmk.s.lt %vm1, %v0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call <256 x i1> @llvm.ve.vl.vfmkslt.mvl(<256 x double> %0, i32 256)
  ret <256 x i1> %2
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.vfmkslt.mvl(<256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @vfmkslt_mvml(<256 x double> %0, <256 x i1> %1) {
; CHECK-LABEL: vfmkslt_mvml:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vfmk.s.lt %vm1, %v0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call <256 x i1> @llvm.ve.vl.vfmkslt.mvml(<256 x double> %0, <256 x i1> %1, i32 256)
  ret <256 x i1> %3
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.vfmkslt.mvml(<256 x double>, <256 x i1>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @vfmksne_mvl(<256 x double> %0) {
; CHECK-LABEL: vfmksne_mvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vfmk.s.ne %vm1, %v0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call <256 x i1> @llvm.ve.vl.vfmksne.mvl(<256 x double> %0, i32 256)
  ret <256 x i1> %2
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.vfmksne.mvl(<256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @vfmksne_mvml(<256 x double> %0, <256 x i1> %1) {
; CHECK-LABEL: vfmksne_mvml:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vfmk.s.ne %vm1, %v0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call <256 x i1> @llvm.ve.vl.vfmksne.mvml(<256 x double> %0, <256 x i1> %1, i32 256)
  ret <256 x i1> %3
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.vfmksne.mvml(<256 x double>, <256 x i1>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @vfmkseq_mvl(<256 x double> %0) {
; CHECK-LABEL: vfmkseq_mvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vfmk.s.eq %vm1, %v0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call <256 x i1> @llvm.ve.vl.vfmkseq.mvl(<256 x double> %0, i32 256)
  ret <256 x i1> %2
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.vfmkseq.mvl(<256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @vfmkseq_mvml(<256 x double> %0, <256 x i1> %1) {
; CHECK-LABEL: vfmkseq_mvml:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vfmk.s.eq %vm1, %v0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call <256 x i1> @llvm.ve.vl.vfmkseq.mvml(<256 x double> %0, <256 x i1> %1, i32 256)
  ret <256 x i1> %3
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.vfmkseq.mvml(<256 x double>, <256 x i1>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @vfmksge_mvl(<256 x double> %0) {
; CHECK-LABEL: vfmksge_mvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vfmk.s.ge %vm1, %v0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call <256 x i1> @llvm.ve.vl.vfmksge.mvl(<256 x double> %0, i32 256)
  ret <256 x i1> %2
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.vfmksge.mvl(<256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @vfmksge_mvml(<256 x double> %0, <256 x i1> %1) {
; CHECK-LABEL: vfmksge_mvml:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vfmk.s.ge %vm1, %v0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call <256 x i1> @llvm.ve.vl.vfmksge.mvml(<256 x double> %0, <256 x i1> %1, i32 256)
  ret <256 x i1> %3
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.vfmksge.mvml(<256 x double>, <256 x i1>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @vfmksle_mvl(<256 x double> %0) {
; CHECK-LABEL: vfmksle_mvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vfmk.s.le %vm1, %v0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call <256 x i1> @llvm.ve.vl.vfmksle.mvl(<256 x double> %0, i32 256)
  ret <256 x i1> %2
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.vfmksle.mvl(<256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @vfmksle_mvml(<256 x double> %0, <256 x i1> %1) {
; CHECK-LABEL: vfmksle_mvml:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vfmk.s.le %vm1, %v0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call <256 x i1> @llvm.ve.vl.vfmksle.mvml(<256 x double> %0, <256 x i1> %1, i32 256)
  ret <256 x i1> %3
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.vfmksle.mvml(<256 x double>, <256 x i1>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @vfmksnum_mvl(<256 x double> %0) {
; CHECK-LABEL: vfmksnum_mvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vfmk.s.num %vm1, %v0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call <256 x i1> @llvm.ve.vl.vfmksnum.mvl(<256 x double> %0, i32 256)
  ret <256 x i1> %2
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.vfmksnum.mvl(<256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @vfmksnum_mvml(<256 x double> %0, <256 x i1> %1) {
; CHECK-LABEL: vfmksnum_mvml:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vfmk.s.num %vm1, %v0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call <256 x i1> @llvm.ve.vl.vfmksnum.mvml(<256 x double> %0, <256 x i1> %1, i32 256)
  ret <256 x i1> %3
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.vfmksnum.mvml(<256 x double>, <256 x i1>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @vfmksnan_mvl(<256 x double> %0) {
; CHECK-LABEL: vfmksnan_mvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vfmk.s.nan %vm1, %v0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call <256 x i1> @llvm.ve.vl.vfmksnan.mvl(<256 x double> %0, i32 256)
  ret <256 x i1> %2
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.vfmksnan.mvl(<256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @vfmksnan_mvml(<256 x double> %0, <256 x i1> %1) {
; CHECK-LABEL: vfmksnan_mvml:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vfmk.s.nan %vm1, %v0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call <256 x i1> @llvm.ve.vl.vfmksnan.mvml(<256 x double> %0, <256 x i1> %1, i32 256)
  ret <256 x i1> %3
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.vfmksnan.mvml(<256 x double>, <256 x i1>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @vfmksgtnan_mvl(<256 x double> %0) {
; CHECK-LABEL: vfmksgtnan_mvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vfmk.s.gtnan %vm1, %v0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call <256 x i1> @llvm.ve.vl.vfmksgtnan.mvl(<256 x double> %0, i32 256)
  ret <256 x i1> %2
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.vfmksgtnan.mvl(<256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @vfmksgtnan_mvml(<256 x double> %0, <256 x i1> %1) {
; CHECK-LABEL: vfmksgtnan_mvml:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vfmk.s.gtnan %vm1, %v0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call <256 x i1> @llvm.ve.vl.vfmksgtnan.mvml(<256 x double> %0, <256 x i1> %1, i32 256)
  ret <256 x i1> %3
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.vfmksgtnan.mvml(<256 x double>, <256 x i1>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @vfmksltnan_mvl(<256 x double> %0) {
; CHECK-LABEL: vfmksltnan_mvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vfmk.s.ltnan %vm1, %v0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call <256 x i1> @llvm.ve.vl.vfmksltnan.mvl(<256 x double> %0, i32 256)
  ret <256 x i1> %2
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.vfmksltnan.mvl(<256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @vfmksltnan_mvml(<256 x double> %0, <256 x i1> %1) {
; CHECK-LABEL: vfmksltnan_mvml:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vfmk.s.ltnan %vm1, %v0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call <256 x i1> @llvm.ve.vl.vfmksltnan.mvml(<256 x double> %0, <256 x i1> %1, i32 256)
  ret <256 x i1> %3
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.vfmksltnan.mvml(<256 x double>, <256 x i1>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @vfmksnenan_mvl(<256 x double> %0) {
; CHECK-LABEL: vfmksnenan_mvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vfmk.s.nenan %vm1, %v0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call <256 x i1> @llvm.ve.vl.vfmksnenan.mvl(<256 x double> %0, i32 256)
  ret <256 x i1> %2
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.vfmksnenan.mvl(<256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @vfmksnenan_mvml(<256 x double> %0, <256 x i1> %1) {
; CHECK-LABEL: vfmksnenan_mvml:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vfmk.s.nenan %vm1, %v0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call <256 x i1> @llvm.ve.vl.vfmksnenan.mvml(<256 x double> %0, <256 x i1> %1, i32 256)
  ret <256 x i1> %3
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.vfmksnenan.mvml(<256 x double>, <256 x i1>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @vfmkseqnan_mvl(<256 x double> %0) {
; CHECK-LABEL: vfmkseqnan_mvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vfmk.s.eqnan %vm1, %v0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call <256 x i1> @llvm.ve.vl.vfmkseqnan.mvl(<256 x double> %0, i32 256)
  ret <256 x i1> %2
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.vfmkseqnan.mvl(<256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @vfmkseqnan_mvml(<256 x double> %0, <256 x i1> %1) {
; CHECK-LABEL: vfmkseqnan_mvml:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vfmk.s.eqnan %vm1, %v0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call <256 x i1> @llvm.ve.vl.vfmkseqnan.mvml(<256 x double> %0, <256 x i1> %1, i32 256)
  ret <256 x i1> %3
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.vfmkseqnan.mvml(<256 x double>, <256 x i1>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @vfmksgenan_mvl(<256 x double> %0) {
; CHECK-LABEL: vfmksgenan_mvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vfmk.s.genan %vm1, %v0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call <256 x i1> @llvm.ve.vl.vfmksgenan.mvl(<256 x double> %0, i32 256)
  ret <256 x i1> %2
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.vfmksgenan.mvl(<256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @vfmksgenan_mvml(<256 x double> %0, <256 x i1> %1) {
; CHECK-LABEL: vfmksgenan_mvml:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vfmk.s.genan %vm1, %v0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call <256 x i1> @llvm.ve.vl.vfmksgenan.mvml(<256 x double> %0, <256 x i1> %1, i32 256)
  ret <256 x i1> %3
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.vfmksgenan.mvml(<256 x double>, <256 x i1>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @vfmkslenan_mvl(<256 x double> %0) {
; CHECK-LABEL: vfmkslenan_mvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vfmk.s.lenan %vm1, %v0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call <256 x i1> @llvm.ve.vl.vfmkslenan.mvl(<256 x double> %0, i32 256)
  ret <256 x i1> %2
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.vfmkslenan.mvl(<256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @vfmkslenan_mvml(<256 x double> %0, <256 x i1> %1) {
; CHECK-LABEL: vfmkslenan_mvml:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vfmk.s.lenan %vm1, %v0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call <256 x i1> @llvm.ve.vl.vfmkslenan.mvml(<256 x double> %0, <256 x i1> %1, i32 256)
  ret <256 x i1> %3
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.vfmkslenan.mvml(<256 x double>, <256 x i1>, i32)

; Function Attrs: nounwind readnone
define fastcc <512 x i1> @pvfmkat_Ml() {
; CHECK-LABEL: pvfmkat_Ml:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vfmk.l.at %vm2
; CHECK-NEXT:    vfmk.l.at %vm3
; CHECK-NEXT:    b.l.t (, %s10)
  %1 = tail call <512 x i1> @llvm.ve.vl.pvfmkat.Ml(i32 256)
  ret <512 x i1> %1
}

; Function Attrs: nounwind readnone
declare <512 x i1> @llvm.ve.vl.pvfmkat.Ml(i32)

; Function Attrs: nounwind readnone
define fastcc <512 x i1> @pvfmkaf_Ml() {
; CHECK-LABEL: pvfmkaf_Ml:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vfmk.l.af %vm2
; CHECK-NEXT:    vfmk.l.af %vm3
; CHECK-NEXT:    b.l.t (, %s10)
  %1 = tail call <512 x i1> @llvm.ve.vl.pvfmkaf.Ml(i32 256)
  ret <512 x i1> %1
}

; Function Attrs: nounwind readnone
declare <512 x i1> @llvm.ve.vl.pvfmkaf.Ml(i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @pvfmkslogt_mvl(<256 x double> %0) {
; CHECK-LABEL: pvfmkslogt_mvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    pvfmk.s.lo.gt %vm1, %v0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call <256 x i1> @llvm.ve.vl.pvfmkslogt.mvl(<256 x double> %0, i32 256)
  ret <256 x i1> %2
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.pvfmkslogt.mvl(<256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @pvfmkslogt_mvml(<256 x double> %0, <256 x i1> %1) {
; CHECK-LABEL: pvfmkslogt_mvml:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    pvfmk.s.lo.gt %vm1, %v0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call <256 x i1> @llvm.ve.vl.pvfmkslogt.mvml(<256 x double> %0, <256 x i1> %1, i32 256)
  ret <256 x i1> %3
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.pvfmkslogt.mvml(<256 x double>, <256 x i1>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @pvfmkslolt_mvl(<256 x double> %0) {
; CHECK-LABEL: pvfmkslolt_mvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    pvfmk.s.lo.lt %vm1, %v0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call <256 x i1> @llvm.ve.vl.pvfmkslolt.mvl(<256 x double> %0, i32 256)
  ret <256 x i1> %2
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.pvfmkslolt.mvl(<256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @pvfmkslolt_mvml(<256 x double> %0, <256 x i1> %1) {
; CHECK-LABEL: pvfmkslolt_mvml:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    pvfmk.s.lo.lt %vm1, %v0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call <256 x i1> @llvm.ve.vl.pvfmkslolt.mvml(<256 x double> %0, <256 x i1> %1, i32 256)
  ret <256 x i1> %3
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.pvfmkslolt.mvml(<256 x double>, <256 x i1>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @pvfmkslone_mvl(<256 x double> %0) {
; CHECK-LABEL: pvfmkslone_mvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    pvfmk.s.lo.ne %vm1, %v0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call <256 x i1> @llvm.ve.vl.pvfmkslone.mvl(<256 x double> %0, i32 256)
  ret <256 x i1> %2
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.pvfmkslone.mvl(<256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @pvfmkslone_mvml(<256 x double> %0, <256 x i1> %1) {
; CHECK-LABEL: pvfmkslone_mvml:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    pvfmk.s.lo.ne %vm1, %v0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call <256 x i1> @llvm.ve.vl.pvfmkslone.mvml(<256 x double> %0, <256 x i1> %1, i32 256)
  ret <256 x i1> %3
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.pvfmkslone.mvml(<256 x double>, <256 x i1>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @pvfmksloeq_mvl(<256 x double> %0) {
; CHECK-LABEL: pvfmksloeq_mvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    pvfmk.s.lo.eq %vm1, %v0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call <256 x i1> @llvm.ve.vl.pvfmksloeq.mvl(<256 x double> %0, i32 256)
  ret <256 x i1> %2
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.pvfmksloeq.mvl(<256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @pvfmksloeq_mvml(<256 x double> %0, <256 x i1> %1) {
; CHECK-LABEL: pvfmksloeq_mvml:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    pvfmk.s.lo.eq %vm1, %v0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call <256 x i1> @llvm.ve.vl.pvfmksloeq.mvml(<256 x double> %0, <256 x i1> %1, i32 256)
  ret <256 x i1> %3
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.pvfmksloeq.mvml(<256 x double>, <256 x i1>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @pvfmksloge_mvl(<256 x double> %0) {
; CHECK-LABEL: pvfmksloge_mvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    pvfmk.s.lo.ge %vm1, %v0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call <256 x i1> @llvm.ve.vl.pvfmksloge.mvl(<256 x double> %0, i32 256)
  ret <256 x i1> %2
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.pvfmksloge.mvl(<256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @pvfmksloge_mvml(<256 x double> %0, <256 x i1> %1) {
; CHECK-LABEL: pvfmksloge_mvml:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    pvfmk.s.lo.ge %vm1, %v0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call <256 x i1> @llvm.ve.vl.pvfmksloge.mvml(<256 x double> %0, <256 x i1> %1, i32 256)
  ret <256 x i1> %3
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.pvfmksloge.mvml(<256 x double>, <256 x i1>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @pvfmkslole_mvl(<256 x double> %0) {
; CHECK-LABEL: pvfmkslole_mvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    pvfmk.s.lo.le %vm1, %v0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call <256 x i1> @llvm.ve.vl.pvfmkslole.mvl(<256 x double> %0, i32 256)
  ret <256 x i1> %2
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.pvfmkslole.mvl(<256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @pvfmkslole_mvml(<256 x double> %0, <256 x i1> %1) {
; CHECK-LABEL: pvfmkslole_mvml:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    pvfmk.s.lo.le %vm1, %v0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call <256 x i1> @llvm.ve.vl.pvfmkslole.mvml(<256 x double> %0, <256 x i1> %1, i32 256)
  ret <256 x i1> %3
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.pvfmkslole.mvml(<256 x double>, <256 x i1>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @pvfmkslonum_mvl(<256 x double> %0) {
; CHECK-LABEL: pvfmkslonum_mvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    pvfmk.s.lo.num %vm1, %v0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call <256 x i1> @llvm.ve.vl.pvfmkslonum.mvl(<256 x double> %0, i32 256)
  ret <256 x i1> %2
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.pvfmkslonum.mvl(<256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @pvfmkslonum_mvml(<256 x double> %0, <256 x i1> %1) {
; CHECK-LABEL: pvfmkslonum_mvml:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    pvfmk.s.lo.num %vm1, %v0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call <256 x i1> @llvm.ve.vl.pvfmkslonum.mvml(<256 x double> %0, <256 x i1> %1, i32 256)
  ret <256 x i1> %3
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.pvfmkslonum.mvml(<256 x double>, <256 x i1>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @pvfmkslonan_mvl(<256 x double> %0) {
; CHECK-LABEL: pvfmkslonan_mvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    pvfmk.s.lo.nan %vm1, %v0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call <256 x i1> @llvm.ve.vl.pvfmkslonan.mvl(<256 x double> %0, i32 256)
  ret <256 x i1> %2
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.pvfmkslonan.mvl(<256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @pvfmkslonan_mvml(<256 x double> %0, <256 x i1> %1) {
; CHECK-LABEL: pvfmkslonan_mvml:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    pvfmk.s.lo.nan %vm1, %v0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call <256 x i1> @llvm.ve.vl.pvfmkslonan.mvml(<256 x double> %0, <256 x i1> %1, i32 256)
  ret <256 x i1> %3
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.pvfmkslonan.mvml(<256 x double>, <256 x i1>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @pvfmkslogtnan_mvl(<256 x double> %0) {
; CHECK-LABEL: pvfmkslogtnan_mvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    pvfmk.s.lo.gtnan %vm1, %v0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call <256 x i1> @llvm.ve.vl.pvfmkslogtnan.mvl(<256 x double> %0, i32 256)
  ret <256 x i1> %2
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.pvfmkslogtnan.mvl(<256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @pvfmkslogtnan_mvml(<256 x double> %0, <256 x i1> %1) {
; CHECK-LABEL: pvfmkslogtnan_mvml:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    pvfmk.s.lo.gtnan %vm1, %v0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call <256 x i1> @llvm.ve.vl.pvfmkslogtnan.mvml(<256 x double> %0, <256 x i1> %1, i32 256)
  ret <256 x i1> %3
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.pvfmkslogtnan.mvml(<256 x double>, <256 x i1>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @pvfmksloltnan_mvl(<256 x double> %0) {
; CHECK-LABEL: pvfmksloltnan_mvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    pvfmk.s.lo.ltnan %vm1, %v0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call <256 x i1> @llvm.ve.vl.pvfmksloltnan.mvl(<256 x double> %0, i32 256)
  ret <256 x i1> %2
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.pvfmksloltnan.mvl(<256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @pvfmksloltnan_mvml(<256 x double> %0, <256 x i1> %1) {
; CHECK-LABEL: pvfmksloltnan_mvml:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    pvfmk.s.lo.ltnan %vm1, %v0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call <256 x i1> @llvm.ve.vl.pvfmksloltnan.mvml(<256 x double> %0, <256 x i1> %1, i32 256)
  ret <256 x i1> %3
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.pvfmksloltnan.mvml(<256 x double>, <256 x i1>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @pvfmkslonenan_mvl(<256 x double> %0) {
; CHECK-LABEL: pvfmkslonenan_mvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    pvfmk.s.lo.nenan %vm1, %v0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call <256 x i1> @llvm.ve.vl.pvfmkslonenan.mvl(<256 x double> %0, i32 256)
  ret <256 x i1> %2
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.pvfmkslonenan.mvl(<256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @pvfmkslonenan_mvml(<256 x double> %0, <256 x i1> %1) {
; CHECK-LABEL: pvfmkslonenan_mvml:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    pvfmk.s.lo.nenan %vm1, %v0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call <256 x i1> @llvm.ve.vl.pvfmkslonenan.mvml(<256 x double> %0, <256 x i1> %1, i32 256)
  ret <256 x i1> %3
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.pvfmkslonenan.mvml(<256 x double>, <256 x i1>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @pvfmksloeqnan_mvl(<256 x double> %0) {
; CHECK-LABEL: pvfmksloeqnan_mvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    pvfmk.s.lo.eqnan %vm1, %v0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call <256 x i1> @llvm.ve.vl.pvfmksloeqnan.mvl(<256 x double> %0, i32 256)
  ret <256 x i1> %2
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.pvfmksloeqnan.mvl(<256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @pvfmksloeqnan_mvml(<256 x double> %0, <256 x i1> %1) {
; CHECK-LABEL: pvfmksloeqnan_mvml:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    pvfmk.s.lo.eqnan %vm1, %v0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call <256 x i1> @llvm.ve.vl.pvfmksloeqnan.mvml(<256 x double> %0, <256 x i1> %1, i32 256)
  ret <256 x i1> %3
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.pvfmksloeqnan.mvml(<256 x double>, <256 x i1>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @pvfmkslogenan_mvl(<256 x double> %0) {
; CHECK-LABEL: pvfmkslogenan_mvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    pvfmk.s.lo.genan %vm1, %v0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call <256 x i1> @llvm.ve.vl.pvfmkslogenan.mvl(<256 x double> %0, i32 256)
  ret <256 x i1> %2
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.pvfmkslogenan.mvl(<256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @pvfmkslogenan_mvml(<256 x double> %0, <256 x i1> %1) {
; CHECK-LABEL: pvfmkslogenan_mvml:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    pvfmk.s.lo.genan %vm1, %v0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call <256 x i1> @llvm.ve.vl.pvfmkslogenan.mvml(<256 x double> %0, <256 x i1> %1, i32 256)
  ret <256 x i1> %3
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.pvfmkslogenan.mvml(<256 x double>, <256 x i1>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @pvfmkslolenan_mvl(<256 x double> %0) {
; CHECK-LABEL: pvfmkslolenan_mvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    pvfmk.s.lo.lenan %vm1, %v0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call <256 x i1> @llvm.ve.vl.pvfmkslolenan.mvl(<256 x double> %0, i32 256)
  ret <256 x i1> %2
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.pvfmkslolenan.mvl(<256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @pvfmkslolenan_mvml(<256 x double> %0, <256 x i1> %1) {
; CHECK-LABEL: pvfmkslolenan_mvml:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    pvfmk.s.lo.lenan %vm1, %v0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call <256 x i1> @llvm.ve.vl.pvfmkslolenan.mvml(<256 x double> %0, <256 x i1> %1, i32 256)
  ret <256 x i1> %3
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.pvfmkslolenan.mvml(<256 x double>, <256 x i1>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @pvfmksupgt_mvl(<256 x double> %0) {
; CHECK-LABEL: pvfmksupgt_mvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    pvfmk.s.up.gt %vm1, %v0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call <256 x i1> @llvm.ve.vl.pvfmksupgt.mvl(<256 x double> %0, i32 256)
  ret <256 x i1> %2
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.pvfmksupgt.mvl(<256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @pvfmksupgt_mvml(<256 x double> %0, <256 x i1> %1) {
; CHECK-LABEL: pvfmksupgt_mvml:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    pvfmk.s.up.gt %vm1, %v0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call <256 x i1> @llvm.ve.vl.pvfmksupgt.mvml(<256 x double> %0, <256 x i1> %1, i32 256)
  ret <256 x i1> %3
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.pvfmksupgt.mvml(<256 x double>, <256 x i1>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @pvfmksuplt_mvl(<256 x double> %0) {
; CHECK-LABEL: pvfmksuplt_mvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    pvfmk.s.up.lt %vm1, %v0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call <256 x i1> @llvm.ve.vl.pvfmksuplt.mvl(<256 x double> %0, i32 256)
  ret <256 x i1> %2
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.pvfmksuplt.mvl(<256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @pvfmksuplt_mvml(<256 x double> %0, <256 x i1> %1) {
; CHECK-LABEL: pvfmksuplt_mvml:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    pvfmk.s.up.lt %vm1, %v0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call <256 x i1> @llvm.ve.vl.pvfmksuplt.mvml(<256 x double> %0, <256 x i1> %1, i32 256)
  ret <256 x i1> %3
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.pvfmksuplt.mvml(<256 x double>, <256 x i1>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @pvfmksupne_mvl(<256 x double> %0) {
; CHECK-LABEL: pvfmksupne_mvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    pvfmk.s.up.ne %vm1, %v0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call <256 x i1> @llvm.ve.vl.pvfmksupne.mvl(<256 x double> %0, i32 256)
  ret <256 x i1> %2
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.pvfmksupne.mvl(<256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @pvfmksupne_mvml(<256 x double> %0, <256 x i1> %1) {
; CHECK-LABEL: pvfmksupne_mvml:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    pvfmk.s.up.ne %vm1, %v0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call <256 x i1> @llvm.ve.vl.pvfmksupne.mvml(<256 x double> %0, <256 x i1> %1, i32 256)
  ret <256 x i1> %3
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.pvfmksupne.mvml(<256 x double>, <256 x i1>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @pvfmksupeq_mvl(<256 x double> %0) {
; CHECK-LABEL: pvfmksupeq_mvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    pvfmk.s.up.eq %vm1, %v0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call <256 x i1> @llvm.ve.vl.pvfmksupeq.mvl(<256 x double> %0, i32 256)
  ret <256 x i1> %2
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.pvfmksupeq.mvl(<256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @pvfmksupeq_mvml(<256 x double> %0, <256 x i1> %1) {
; CHECK-LABEL: pvfmksupeq_mvml:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    pvfmk.s.up.eq %vm1, %v0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call <256 x i1> @llvm.ve.vl.pvfmksupeq.mvml(<256 x double> %0, <256 x i1> %1, i32 256)
  ret <256 x i1> %3
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.pvfmksupeq.mvml(<256 x double>, <256 x i1>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @pvfmksupge_mvl(<256 x double> %0) {
; CHECK-LABEL: pvfmksupge_mvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    pvfmk.s.up.ge %vm1, %v0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call <256 x i1> @llvm.ve.vl.pvfmksupge.mvl(<256 x double> %0, i32 256)
  ret <256 x i1> %2
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.pvfmksupge.mvl(<256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @pvfmksupge_mvml(<256 x double> %0, <256 x i1> %1) {
; CHECK-LABEL: pvfmksupge_mvml:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    pvfmk.s.up.ge %vm1, %v0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call <256 x i1> @llvm.ve.vl.pvfmksupge.mvml(<256 x double> %0, <256 x i1> %1, i32 256)
  ret <256 x i1> %3
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.pvfmksupge.mvml(<256 x double>, <256 x i1>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @pvfmksuple_mvl(<256 x double> %0) {
; CHECK-LABEL: pvfmksuple_mvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    pvfmk.s.up.le %vm1, %v0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call <256 x i1> @llvm.ve.vl.pvfmksuple.mvl(<256 x double> %0, i32 256)
  ret <256 x i1> %2
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.pvfmksuple.mvl(<256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @pvfmksuple_mvml(<256 x double> %0, <256 x i1> %1) {
; CHECK-LABEL: pvfmksuple_mvml:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    pvfmk.s.up.le %vm1, %v0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call <256 x i1> @llvm.ve.vl.pvfmksuple.mvml(<256 x double> %0, <256 x i1> %1, i32 256)
  ret <256 x i1> %3
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.pvfmksuple.mvml(<256 x double>, <256 x i1>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @pvfmksupnum_mvl(<256 x double> %0) {
; CHECK-LABEL: pvfmksupnum_mvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    pvfmk.s.up.num %vm1, %v0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call <256 x i1> @llvm.ve.vl.pvfmksupnum.mvl(<256 x double> %0, i32 256)
  ret <256 x i1> %2
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.pvfmksupnum.mvl(<256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @pvfmksupnum_mvml(<256 x double> %0, <256 x i1> %1) {
; CHECK-LABEL: pvfmksupnum_mvml:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    pvfmk.s.up.num %vm1, %v0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call <256 x i1> @llvm.ve.vl.pvfmksupnum.mvml(<256 x double> %0, <256 x i1> %1, i32 256)
  ret <256 x i1> %3
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.pvfmksupnum.mvml(<256 x double>, <256 x i1>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @pvfmksupnan_mvl(<256 x double> %0) {
; CHECK-LABEL: pvfmksupnan_mvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    pvfmk.s.up.nan %vm1, %v0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call <256 x i1> @llvm.ve.vl.pvfmksupnan.mvl(<256 x double> %0, i32 256)
  ret <256 x i1> %2
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.pvfmksupnan.mvl(<256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @pvfmksupnan_mvml(<256 x double> %0, <256 x i1> %1) {
; CHECK-LABEL: pvfmksupnan_mvml:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    pvfmk.s.up.nan %vm1, %v0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call <256 x i1> @llvm.ve.vl.pvfmksupnan.mvml(<256 x double> %0, <256 x i1> %1, i32 256)
  ret <256 x i1> %3
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.pvfmksupnan.mvml(<256 x double>, <256 x i1>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @pvfmksupgtnan_mvl(<256 x double> %0) {
; CHECK-LABEL: pvfmksupgtnan_mvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    pvfmk.s.up.gtnan %vm1, %v0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call <256 x i1> @llvm.ve.vl.pvfmksupgtnan.mvl(<256 x double> %0, i32 256)
  ret <256 x i1> %2
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.pvfmksupgtnan.mvl(<256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @pvfmksupgtnan_mvml(<256 x double> %0, <256 x i1> %1) {
; CHECK-LABEL: pvfmksupgtnan_mvml:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    pvfmk.s.up.gtnan %vm1, %v0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call <256 x i1> @llvm.ve.vl.pvfmksupgtnan.mvml(<256 x double> %0, <256 x i1> %1, i32 256)
  ret <256 x i1> %3
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.pvfmksupgtnan.mvml(<256 x double>, <256 x i1>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @pvfmksupltnan_mvl(<256 x double> %0) {
; CHECK-LABEL: pvfmksupltnan_mvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    pvfmk.s.up.ltnan %vm1, %v0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call <256 x i1> @llvm.ve.vl.pvfmksupltnan.mvl(<256 x double> %0, i32 256)
  ret <256 x i1> %2
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.pvfmksupltnan.mvl(<256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @pvfmksupltnan_mvml(<256 x double> %0, <256 x i1> %1) {
; CHECK-LABEL: pvfmksupltnan_mvml:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    pvfmk.s.up.ltnan %vm1, %v0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call <256 x i1> @llvm.ve.vl.pvfmksupltnan.mvml(<256 x double> %0, <256 x i1> %1, i32 256)
  ret <256 x i1> %3
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.pvfmksupltnan.mvml(<256 x double>, <256 x i1>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @pvfmksupnenan_mvl(<256 x double> %0) {
; CHECK-LABEL: pvfmksupnenan_mvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    pvfmk.s.up.nenan %vm1, %v0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call <256 x i1> @llvm.ve.vl.pvfmksupnenan.mvl(<256 x double> %0, i32 256)
  ret <256 x i1> %2
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.pvfmksupnenan.mvl(<256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @pvfmksupnenan_mvml(<256 x double> %0, <256 x i1> %1) {
; CHECK-LABEL: pvfmksupnenan_mvml:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    pvfmk.s.up.nenan %vm1, %v0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call <256 x i1> @llvm.ve.vl.pvfmksupnenan.mvml(<256 x double> %0, <256 x i1> %1, i32 256)
  ret <256 x i1> %3
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.pvfmksupnenan.mvml(<256 x double>, <256 x i1>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @pvfmksupeqnan_mvl(<256 x double> %0) {
; CHECK-LABEL: pvfmksupeqnan_mvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    pvfmk.s.up.eqnan %vm1, %v0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call <256 x i1> @llvm.ve.vl.pvfmksupeqnan.mvl(<256 x double> %0, i32 256)
  ret <256 x i1> %2
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.pvfmksupeqnan.mvl(<256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @pvfmksupeqnan_mvml(<256 x double> %0, <256 x i1> %1) {
; CHECK-LABEL: pvfmksupeqnan_mvml:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    pvfmk.s.up.eqnan %vm1, %v0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call <256 x i1> @llvm.ve.vl.pvfmksupeqnan.mvml(<256 x double> %0, <256 x i1> %1, i32 256)
  ret <256 x i1> %3
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.pvfmksupeqnan.mvml(<256 x double>, <256 x i1>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @pvfmksupgenan_mvl(<256 x double> %0) {
; CHECK-LABEL: pvfmksupgenan_mvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    pvfmk.s.up.genan %vm1, %v0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call <256 x i1> @llvm.ve.vl.pvfmksupgenan.mvl(<256 x double> %0, i32 256)
  ret <256 x i1> %2
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.pvfmksupgenan.mvl(<256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @pvfmksupgenan_mvml(<256 x double> %0, <256 x i1> %1) {
; CHECK-LABEL: pvfmksupgenan_mvml:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    pvfmk.s.up.genan %vm1, %v0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call <256 x i1> @llvm.ve.vl.pvfmksupgenan.mvml(<256 x double> %0, <256 x i1> %1, i32 256)
  ret <256 x i1> %3
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.pvfmksupgenan.mvml(<256 x double>, <256 x i1>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @pvfmksuplenan_mvl(<256 x double> %0) {
; CHECK-LABEL: pvfmksuplenan_mvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    pvfmk.s.up.lenan %vm1, %v0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call <256 x i1> @llvm.ve.vl.pvfmksuplenan.mvl(<256 x double> %0, i32 256)
  ret <256 x i1> %2
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.pvfmksuplenan.mvl(<256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @pvfmksuplenan_mvml(<256 x double> %0, <256 x i1> %1) {
; CHECK-LABEL: pvfmksuplenan_mvml:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    pvfmk.s.up.lenan %vm1, %v0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call <256 x i1> @llvm.ve.vl.pvfmksuplenan.mvml(<256 x double> %0, <256 x i1> %1, i32 256)
  ret <256 x i1> %3
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.pvfmksuplenan.mvml(<256 x double>, <256 x i1>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @pvfmkwlogt_mvl(<256 x double> %0) {
; CHECK-LABEL: pvfmkwlogt_mvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vfmk.w.gt %vm1, %v0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call <256 x i1> @llvm.ve.vl.pvfmkwlogt.mvl(<256 x double> %0, i32 256)
  ret <256 x i1> %2
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.pvfmkwlogt.mvl(<256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @pvfmkwlogt_mvml(<256 x double> %0, <256 x i1> %1) {
; CHECK-LABEL: pvfmkwlogt_mvml:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vfmk.w.gt %vm1, %v0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call <256 x i1> @llvm.ve.vl.pvfmkwlogt.mvml(<256 x double> %0, <256 x i1> %1, i32 256)
  ret <256 x i1> %3
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.pvfmkwlogt.mvml(<256 x double>, <256 x i1>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @pvfmkwlolt_mvl(<256 x double> %0) {
; CHECK-LABEL: pvfmkwlolt_mvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vfmk.w.lt %vm1, %v0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call <256 x i1> @llvm.ve.vl.pvfmkwlolt.mvl(<256 x double> %0, i32 256)
  ret <256 x i1> %2
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.pvfmkwlolt.mvl(<256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @pvfmkwlolt_mvml(<256 x double> %0, <256 x i1> %1) {
; CHECK-LABEL: pvfmkwlolt_mvml:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vfmk.w.lt %vm1, %v0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call <256 x i1> @llvm.ve.vl.pvfmkwlolt.mvml(<256 x double> %0, <256 x i1> %1, i32 256)
  ret <256 x i1> %3
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.pvfmkwlolt.mvml(<256 x double>, <256 x i1>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @pvfmkwlone_mvl(<256 x double> %0) {
; CHECK-LABEL: pvfmkwlone_mvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vfmk.w.ne %vm1, %v0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call <256 x i1> @llvm.ve.vl.pvfmkwlone.mvl(<256 x double> %0, i32 256)
  ret <256 x i1> %2
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.pvfmkwlone.mvl(<256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @pvfmkwlone_mvml(<256 x double> %0, <256 x i1> %1) {
; CHECK-LABEL: pvfmkwlone_mvml:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vfmk.w.ne %vm1, %v0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call <256 x i1> @llvm.ve.vl.pvfmkwlone.mvml(<256 x double> %0, <256 x i1> %1, i32 256)
  ret <256 x i1> %3
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.pvfmkwlone.mvml(<256 x double>, <256 x i1>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @pvfmkwloeq_mvl(<256 x double> %0) {
; CHECK-LABEL: pvfmkwloeq_mvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vfmk.w.eq %vm1, %v0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call <256 x i1> @llvm.ve.vl.pvfmkwloeq.mvl(<256 x double> %0, i32 256)
  ret <256 x i1> %2
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.pvfmkwloeq.mvl(<256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @pvfmkwloeq_mvml(<256 x double> %0, <256 x i1> %1) {
; CHECK-LABEL: pvfmkwloeq_mvml:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vfmk.w.eq %vm1, %v0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call <256 x i1> @llvm.ve.vl.pvfmkwloeq.mvml(<256 x double> %0, <256 x i1> %1, i32 256)
  ret <256 x i1> %3
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.pvfmkwloeq.mvml(<256 x double>, <256 x i1>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @pvfmkwloge_mvl(<256 x double> %0) {
; CHECK-LABEL: pvfmkwloge_mvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vfmk.w.ge %vm1, %v0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call <256 x i1> @llvm.ve.vl.pvfmkwloge.mvl(<256 x double> %0, i32 256)
  ret <256 x i1> %2
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.pvfmkwloge.mvl(<256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @pvfmkwloge_mvml(<256 x double> %0, <256 x i1> %1) {
; CHECK-LABEL: pvfmkwloge_mvml:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vfmk.w.ge %vm1, %v0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call <256 x i1> @llvm.ve.vl.pvfmkwloge.mvml(<256 x double> %0, <256 x i1> %1, i32 256)
  ret <256 x i1> %3
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.pvfmkwloge.mvml(<256 x double>, <256 x i1>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @pvfmkwlole_mvl(<256 x double> %0) {
; CHECK-LABEL: pvfmkwlole_mvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vfmk.w.le %vm1, %v0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call <256 x i1> @llvm.ve.vl.pvfmkwlole.mvl(<256 x double> %0, i32 256)
  ret <256 x i1> %2
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.pvfmkwlole.mvl(<256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @pvfmkwlole_mvml(<256 x double> %0, <256 x i1> %1) {
; CHECK-LABEL: pvfmkwlole_mvml:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vfmk.w.le %vm1, %v0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call <256 x i1> @llvm.ve.vl.pvfmkwlole.mvml(<256 x double> %0, <256 x i1> %1, i32 256)
  ret <256 x i1> %3
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.pvfmkwlole.mvml(<256 x double>, <256 x i1>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @pvfmkwlonum_mvl(<256 x double> %0) {
; CHECK-LABEL: pvfmkwlonum_mvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vfmk.w.num %vm1, %v0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call <256 x i1> @llvm.ve.vl.pvfmkwlonum.mvl(<256 x double> %0, i32 256)
  ret <256 x i1> %2
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.pvfmkwlonum.mvl(<256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @pvfmkwlonum_mvml(<256 x double> %0, <256 x i1> %1) {
; CHECK-LABEL: pvfmkwlonum_mvml:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vfmk.w.num %vm1, %v0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call <256 x i1> @llvm.ve.vl.pvfmkwlonum.mvml(<256 x double> %0, <256 x i1> %1, i32 256)
  ret <256 x i1> %3
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.pvfmkwlonum.mvml(<256 x double>, <256 x i1>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @pvfmkwlonan_mvl(<256 x double> %0) {
; CHECK-LABEL: pvfmkwlonan_mvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vfmk.w.nan %vm1, %v0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call <256 x i1> @llvm.ve.vl.pvfmkwlonan.mvl(<256 x double> %0, i32 256)
  ret <256 x i1> %2
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.pvfmkwlonan.mvl(<256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @pvfmkwlonan_mvml(<256 x double> %0, <256 x i1> %1) {
; CHECK-LABEL: pvfmkwlonan_mvml:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vfmk.w.nan %vm1, %v0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call <256 x i1> @llvm.ve.vl.pvfmkwlonan.mvml(<256 x double> %0, <256 x i1> %1, i32 256)
  ret <256 x i1> %3
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.pvfmkwlonan.mvml(<256 x double>, <256 x i1>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @pvfmkwlogtnan_mvl(<256 x double> %0) {
; CHECK-LABEL: pvfmkwlogtnan_mvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vfmk.w.gtnan %vm1, %v0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call <256 x i1> @llvm.ve.vl.pvfmkwlogtnan.mvl(<256 x double> %0, i32 256)
  ret <256 x i1> %2
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.pvfmkwlogtnan.mvl(<256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @pvfmkwlogtnan_mvml(<256 x double> %0, <256 x i1> %1) {
; CHECK-LABEL: pvfmkwlogtnan_mvml:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vfmk.w.gtnan %vm1, %v0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call <256 x i1> @llvm.ve.vl.pvfmkwlogtnan.mvml(<256 x double> %0, <256 x i1> %1, i32 256)
  ret <256 x i1> %3
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.pvfmkwlogtnan.mvml(<256 x double>, <256 x i1>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @pvfmkwloltnan_mvl(<256 x double> %0) {
; CHECK-LABEL: pvfmkwloltnan_mvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vfmk.w.ltnan %vm1, %v0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call <256 x i1> @llvm.ve.vl.pvfmkwloltnan.mvl(<256 x double> %0, i32 256)
  ret <256 x i1> %2
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.pvfmkwloltnan.mvl(<256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @pvfmkwloltnan_mvml(<256 x double> %0, <256 x i1> %1) {
; CHECK-LABEL: pvfmkwloltnan_mvml:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vfmk.w.ltnan %vm1, %v0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call <256 x i1> @llvm.ve.vl.pvfmkwloltnan.mvml(<256 x double> %0, <256 x i1> %1, i32 256)
  ret <256 x i1> %3
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.pvfmkwloltnan.mvml(<256 x double>, <256 x i1>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @pvfmkwlonenan_mvl(<256 x double> %0) {
; CHECK-LABEL: pvfmkwlonenan_mvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vfmk.w.nenan %vm1, %v0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call <256 x i1> @llvm.ve.vl.pvfmkwlonenan.mvl(<256 x double> %0, i32 256)
  ret <256 x i1> %2
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.pvfmkwlonenan.mvl(<256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @pvfmkwlonenan_mvml(<256 x double> %0, <256 x i1> %1) {
; CHECK-LABEL: pvfmkwlonenan_mvml:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vfmk.w.nenan %vm1, %v0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call <256 x i1> @llvm.ve.vl.pvfmkwlonenan.mvml(<256 x double> %0, <256 x i1> %1, i32 256)
  ret <256 x i1> %3
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.pvfmkwlonenan.mvml(<256 x double>, <256 x i1>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @pvfmkwloeqnan_mvl(<256 x double> %0) {
; CHECK-LABEL: pvfmkwloeqnan_mvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vfmk.w.eqnan %vm1, %v0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call <256 x i1> @llvm.ve.vl.pvfmkwloeqnan.mvl(<256 x double> %0, i32 256)
  ret <256 x i1> %2
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.pvfmkwloeqnan.mvl(<256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @pvfmkwloeqnan_mvml(<256 x double> %0, <256 x i1> %1) {
; CHECK-LABEL: pvfmkwloeqnan_mvml:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vfmk.w.eqnan %vm1, %v0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call <256 x i1> @llvm.ve.vl.pvfmkwloeqnan.mvml(<256 x double> %0, <256 x i1> %1, i32 256)
  ret <256 x i1> %3
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.pvfmkwloeqnan.mvml(<256 x double>, <256 x i1>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @pvfmkwlogenan_mvl(<256 x double> %0) {
; CHECK-LABEL: pvfmkwlogenan_mvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vfmk.w.genan %vm1, %v0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call <256 x i1> @llvm.ve.vl.pvfmkwlogenan.mvl(<256 x double> %0, i32 256)
  ret <256 x i1> %2
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.pvfmkwlogenan.mvl(<256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @pvfmkwlogenan_mvml(<256 x double> %0, <256 x i1> %1) {
; CHECK-LABEL: pvfmkwlogenan_mvml:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vfmk.w.genan %vm1, %v0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call <256 x i1> @llvm.ve.vl.pvfmkwlogenan.mvml(<256 x double> %0, <256 x i1> %1, i32 256)
  ret <256 x i1> %3
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.pvfmkwlogenan.mvml(<256 x double>, <256 x i1>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @pvfmkwlolenan_mvl(<256 x double> %0) {
; CHECK-LABEL: pvfmkwlolenan_mvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vfmk.w.lenan %vm1, %v0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call <256 x i1> @llvm.ve.vl.pvfmkwlolenan.mvl(<256 x double> %0, i32 256)
  ret <256 x i1> %2
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.pvfmkwlolenan.mvl(<256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @pvfmkwlolenan_mvml(<256 x double> %0, <256 x i1> %1) {
; CHECK-LABEL: pvfmkwlolenan_mvml:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vfmk.w.lenan %vm1, %v0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call <256 x i1> @llvm.ve.vl.pvfmkwlolenan.mvml(<256 x double> %0, <256 x i1> %1, i32 256)
  ret <256 x i1> %3
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.pvfmkwlolenan.mvml(<256 x double>, <256 x i1>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @pvfmkwupgt_mvl(<256 x double> %0) {
; CHECK-LABEL: pvfmkwupgt_mvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    pvfmk.w.up.gt %vm1, %v0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call <256 x i1> @llvm.ve.vl.pvfmkwupgt.mvl(<256 x double> %0, i32 256)
  ret <256 x i1> %2
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.pvfmkwupgt.mvl(<256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @pvfmkwupgt_mvml(<256 x double> %0, <256 x i1> %1) {
; CHECK-LABEL: pvfmkwupgt_mvml:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    pvfmk.w.up.gt %vm1, %v0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call <256 x i1> @llvm.ve.vl.pvfmkwupgt.mvml(<256 x double> %0, <256 x i1> %1, i32 256)
  ret <256 x i1> %3
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.pvfmkwupgt.mvml(<256 x double>, <256 x i1>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @pvfmkwuplt_mvl(<256 x double> %0) {
; CHECK-LABEL: pvfmkwuplt_mvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    pvfmk.w.up.lt %vm1, %v0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call <256 x i1> @llvm.ve.vl.pvfmkwuplt.mvl(<256 x double> %0, i32 256)
  ret <256 x i1> %2
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.pvfmkwuplt.mvl(<256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @pvfmkwuplt_mvml(<256 x double> %0, <256 x i1> %1) {
; CHECK-LABEL: pvfmkwuplt_mvml:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    pvfmk.w.up.lt %vm1, %v0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call <256 x i1> @llvm.ve.vl.pvfmkwuplt.mvml(<256 x double> %0, <256 x i1> %1, i32 256)
  ret <256 x i1> %3
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.pvfmkwuplt.mvml(<256 x double>, <256 x i1>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @pvfmkwupne_mvl(<256 x double> %0) {
; CHECK-LABEL: pvfmkwupne_mvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    pvfmk.w.up.ne %vm1, %v0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call <256 x i1> @llvm.ve.vl.pvfmkwupne.mvl(<256 x double> %0, i32 256)
  ret <256 x i1> %2
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.pvfmkwupne.mvl(<256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @pvfmkwupne_mvml(<256 x double> %0, <256 x i1> %1) {
; CHECK-LABEL: pvfmkwupne_mvml:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    pvfmk.w.up.ne %vm1, %v0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call <256 x i1> @llvm.ve.vl.pvfmkwupne.mvml(<256 x double> %0, <256 x i1> %1, i32 256)
  ret <256 x i1> %3
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.pvfmkwupne.mvml(<256 x double>, <256 x i1>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @pvfmkwupeq_mvl(<256 x double> %0) {
; CHECK-LABEL: pvfmkwupeq_mvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    pvfmk.w.up.eq %vm1, %v0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call <256 x i1> @llvm.ve.vl.pvfmkwupeq.mvl(<256 x double> %0, i32 256)
  ret <256 x i1> %2
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.pvfmkwupeq.mvl(<256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @pvfmkwupeq_mvml(<256 x double> %0, <256 x i1> %1) {
; CHECK-LABEL: pvfmkwupeq_mvml:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    pvfmk.w.up.eq %vm1, %v0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call <256 x i1> @llvm.ve.vl.pvfmkwupeq.mvml(<256 x double> %0, <256 x i1> %1, i32 256)
  ret <256 x i1> %3
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.pvfmkwupeq.mvml(<256 x double>, <256 x i1>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @pvfmkwupge_mvl(<256 x double> %0) {
; CHECK-LABEL: pvfmkwupge_mvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    pvfmk.w.up.ge %vm1, %v0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call <256 x i1> @llvm.ve.vl.pvfmkwupge.mvl(<256 x double> %0, i32 256)
  ret <256 x i1> %2
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.pvfmkwupge.mvl(<256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @pvfmkwupge_mvml(<256 x double> %0, <256 x i1> %1) {
; CHECK-LABEL: pvfmkwupge_mvml:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    pvfmk.w.up.ge %vm1, %v0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call <256 x i1> @llvm.ve.vl.pvfmkwupge.mvml(<256 x double> %0, <256 x i1> %1, i32 256)
  ret <256 x i1> %3
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.pvfmkwupge.mvml(<256 x double>, <256 x i1>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @pvfmkwuple_mvl(<256 x double> %0) {
; CHECK-LABEL: pvfmkwuple_mvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    pvfmk.w.up.le %vm1, %v0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call <256 x i1> @llvm.ve.vl.pvfmkwuple.mvl(<256 x double> %0, i32 256)
  ret <256 x i1> %2
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.pvfmkwuple.mvl(<256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @pvfmkwuple_mvml(<256 x double> %0, <256 x i1> %1) {
; CHECK-LABEL: pvfmkwuple_mvml:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    pvfmk.w.up.le %vm1, %v0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call <256 x i1> @llvm.ve.vl.pvfmkwuple.mvml(<256 x double> %0, <256 x i1> %1, i32 256)
  ret <256 x i1> %3
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.pvfmkwuple.mvml(<256 x double>, <256 x i1>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @pvfmkwupnum_mvl(<256 x double> %0) {
; CHECK-LABEL: pvfmkwupnum_mvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    pvfmk.w.up.num %vm1, %v0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call <256 x i1> @llvm.ve.vl.pvfmkwupnum.mvl(<256 x double> %0, i32 256)
  ret <256 x i1> %2
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.pvfmkwupnum.mvl(<256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @pvfmkwupnum_mvml(<256 x double> %0, <256 x i1> %1) {
; CHECK-LABEL: pvfmkwupnum_mvml:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    pvfmk.w.up.num %vm1, %v0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call <256 x i1> @llvm.ve.vl.pvfmkwupnum.mvml(<256 x double> %0, <256 x i1> %1, i32 256)
  ret <256 x i1> %3
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.pvfmkwupnum.mvml(<256 x double>, <256 x i1>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @pvfmkwupnan_mvl(<256 x double> %0) {
; CHECK-LABEL: pvfmkwupnan_mvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    pvfmk.w.up.nan %vm1, %v0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call <256 x i1> @llvm.ve.vl.pvfmkwupnan.mvl(<256 x double> %0, i32 256)
  ret <256 x i1> %2
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.pvfmkwupnan.mvl(<256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @pvfmkwupnan_mvml(<256 x double> %0, <256 x i1> %1) {
; CHECK-LABEL: pvfmkwupnan_mvml:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    pvfmk.w.up.nan %vm1, %v0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call <256 x i1> @llvm.ve.vl.pvfmkwupnan.mvml(<256 x double> %0, <256 x i1> %1, i32 256)
  ret <256 x i1> %3
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.pvfmkwupnan.mvml(<256 x double>, <256 x i1>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @pvfmkwupgtnan_mvl(<256 x double> %0) {
; CHECK-LABEL: pvfmkwupgtnan_mvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    pvfmk.w.up.gtnan %vm1, %v0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call <256 x i1> @llvm.ve.vl.pvfmkwupgtnan.mvl(<256 x double> %0, i32 256)
  ret <256 x i1> %2
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.pvfmkwupgtnan.mvl(<256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @pvfmkwupgtnan_mvml(<256 x double> %0, <256 x i1> %1) {
; CHECK-LABEL: pvfmkwupgtnan_mvml:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    pvfmk.w.up.gtnan %vm1, %v0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call <256 x i1> @llvm.ve.vl.pvfmkwupgtnan.mvml(<256 x double> %0, <256 x i1> %1, i32 256)
  ret <256 x i1> %3
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.pvfmkwupgtnan.mvml(<256 x double>, <256 x i1>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @pvfmkwupltnan_mvl(<256 x double> %0) {
; CHECK-LABEL: pvfmkwupltnan_mvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    pvfmk.w.up.ltnan %vm1, %v0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call <256 x i1> @llvm.ve.vl.pvfmkwupltnan.mvl(<256 x double> %0, i32 256)
  ret <256 x i1> %2
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.pvfmkwupltnan.mvl(<256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @pvfmkwupltnan_mvml(<256 x double> %0, <256 x i1> %1) {
; CHECK-LABEL: pvfmkwupltnan_mvml:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    pvfmk.w.up.ltnan %vm1, %v0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call <256 x i1> @llvm.ve.vl.pvfmkwupltnan.mvml(<256 x double> %0, <256 x i1> %1, i32 256)
  ret <256 x i1> %3
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.pvfmkwupltnan.mvml(<256 x double>, <256 x i1>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @pvfmkwupnenan_mvl(<256 x double> %0) {
; CHECK-LABEL: pvfmkwupnenan_mvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    pvfmk.w.up.nenan %vm1, %v0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call <256 x i1> @llvm.ve.vl.pvfmkwupnenan.mvl(<256 x double> %0, i32 256)
  ret <256 x i1> %2
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.pvfmkwupnenan.mvl(<256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @pvfmkwupnenan_mvml(<256 x double> %0, <256 x i1> %1) {
; CHECK-LABEL: pvfmkwupnenan_mvml:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    pvfmk.w.up.nenan %vm1, %v0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call <256 x i1> @llvm.ve.vl.pvfmkwupnenan.mvml(<256 x double> %0, <256 x i1> %1, i32 256)
  ret <256 x i1> %3
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.pvfmkwupnenan.mvml(<256 x double>, <256 x i1>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @pvfmkwupeqnan_mvl(<256 x double> %0) {
; CHECK-LABEL: pvfmkwupeqnan_mvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    pvfmk.w.up.eqnan %vm1, %v0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call <256 x i1> @llvm.ve.vl.pvfmkwupeqnan.mvl(<256 x double> %0, i32 256)
  ret <256 x i1> %2
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.pvfmkwupeqnan.mvl(<256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @pvfmkwupeqnan_mvml(<256 x double> %0, <256 x i1> %1) {
; CHECK-LABEL: pvfmkwupeqnan_mvml:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    pvfmk.w.up.eqnan %vm1, %v0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call <256 x i1> @llvm.ve.vl.pvfmkwupeqnan.mvml(<256 x double> %0, <256 x i1> %1, i32 256)
  ret <256 x i1> %3
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.pvfmkwupeqnan.mvml(<256 x double>, <256 x i1>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @pvfmkwupgenan_mvl(<256 x double> %0) {
; CHECK-LABEL: pvfmkwupgenan_mvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    pvfmk.w.up.genan %vm1, %v0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call <256 x i1> @llvm.ve.vl.pvfmkwupgenan.mvl(<256 x double> %0, i32 256)
  ret <256 x i1> %2
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.pvfmkwupgenan.mvl(<256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @pvfmkwupgenan_mvml(<256 x double> %0, <256 x i1> %1) {
; CHECK-LABEL: pvfmkwupgenan_mvml:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    pvfmk.w.up.genan %vm1, %v0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call <256 x i1> @llvm.ve.vl.pvfmkwupgenan.mvml(<256 x double> %0, <256 x i1> %1, i32 256)
  ret <256 x i1> %3
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.pvfmkwupgenan.mvml(<256 x double>, <256 x i1>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @pvfmkwuplenan_mvl(<256 x double> %0) {
; CHECK-LABEL: pvfmkwuplenan_mvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    pvfmk.w.up.lenan %vm1, %v0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call <256 x i1> @llvm.ve.vl.pvfmkwuplenan.mvl(<256 x double> %0, i32 256)
  ret <256 x i1> %2
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.pvfmkwuplenan.mvl(<256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x i1> @pvfmkwuplenan_mvml(<256 x double> %0, <256 x i1> %1) {
; CHECK-LABEL: pvfmkwuplenan_mvml:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    pvfmk.w.up.lenan %vm1, %v0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call <256 x i1> @llvm.ve.vl.pvfmkwuplenan.mvml(<256 x double> %0, <256 x i1> %1, i32 256)
  ret <256 x i1> %3
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.pvfmkwuplenan.mvml(<256 x double>, <256 x i1>, i32)

; Function Attrs: nounwind readnone
define fastcc <512 x i1> @pvfmksgt_Mvl(<256 x double> %0) {
; CHECK-LABEL: pvfmksgt_Mvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    pvfmk.s.up.gt %vm2, %v0
; CHECK-NEXT:    pvfmk.s.lo.gt %vm3, %v0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call <512 x i1> @llvm.ve.vl.pvfmksgt.Mvl(<256 x double> %0, i32 256)
  ret <512 x i1> %2
}

; Function Attrs: nounwind readnone
declare <512 x i1> @llvm.ve.vl.pvfmksgt.Mvl(<256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <512 x i1> @pvfmksgt_MvMl(<256 x double> %0, <512 x i1> %1) {
; CHECK-LABEL: pvfmksgt_MvMl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    pvfmk.s.up.gt %vm2, %v0, %vm2
; CHECK-NEXT:    pvfmk.s.lo.gt %vm3, %v0, %vm3
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call <512 x i1> @llvm.ve.vl.pvfmksgt.MvMl(<256 x double> %0, <512 x i1> %1, i32 256)
  ret <512 x i1> %3
}

; Function Attrs: nounwind readnone
declare <512 x i1> @llvm.ve.vl.pvfmksgt.MvMl(<256 x double>, <512 x i1>, i32)

; Function Attrs: nounwind readnone
define fastcc <512 x i1> @pvfmkslt_Mvl(<256 x double> %0) {
; CHECK-LABEL: pvfmkslt_Mvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    pvfmk.s.up.lt %vm2, %v0
; CHECK-NEXT:    pvfmk.s.lo.lt %vm3, %v0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call <512 x i1> @llvm.ve.vl.pvfmkslt.Mvl(<256 x double> %0, i32 256)
  ret <512 x i1> %2
}

; Function Attrs: nounwind readnone
declare <512 x i1> @llvm.ve.vl.pvfmkslt.Mvl(<256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <512 x i1> @pvfmkslt_MvMl(<256 x double> %0, <512 x i1> %1) {
; CHECK-LABEL: pvfmkslt_MvMl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    pvfmk.s.up.lt %vm2, %v0, %vm2
; CHECK-NEXT:    pvfmk.s.lo.lt %vm3, %v0, %vm3
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call <512 x i1> @llvm.ve.vl.pvfmkslt.MvMl(<256 x double> %0, <512 x i1> %1, i32 256)
  ret <512 x i1> %3
}

; Function Attrs: nounwind readnone
declare <512 x i1> @llvm.ve.vl.pvfmkslt.MvMl(<256 x double>, <512 x i1>, i32)

; Function Attrs: nounwind readnone
define fastcc <512 x i1> @pvfmksne_Mvl(<256 x double> %0) {
; CHECK-LABEL: pvfmksne_Mvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    pvfmk.s.up.ne %vm2, %v0
; CHECK-NEXT:    pvfmk.s.lo.ne %vm3, %v0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call <512 x i1> @llvm.ve.vl.pvfmksne.Mvl(<256 x double> %0, i32 256)
  ret <512 x i1> %2
}

; Function Attrs: nounwind readnone
declare <512 x i1> @llvm.ve.vl.pvfmksne.Mvl(<256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <512 x i1> @pvfmksne_MvMl(<256 x double> %0, <512 x i1> %1) {
; CHECK-LABEL: pvfmksne_MvMl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    pvfmk.s.up.ne %vm2, %v0, %vm2
; CHECK-NEXT:    pvfmk.s.lo.ne %vm3, %v0, %vm3
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call <512 x i1> @llvm.ve.vl.pvfmksne.MvMl(<256 x double> %0, <512 x i1> %1, i32 256)
  ret <512 x i1> %3
}

; Function Attrs: nounwind readnone
declare <512 x i1> @llvm.ve.vl.pvfmksne.MvMl(<256 x double>, <512 x i1>, i32)

; Function Attrs: nounwind readnone
define fastcc <512 x i1> @pvfmkseq_Mvl(<256 x double> %0) {
; CHECK-LABEL: pvfmkseq_Mvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    pvfmk.s.up.eq %vm2, %v0
; CHECK-NEXT:    pvfmk.s.lo.eq %vm3, %v0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call <512 x i1> @llvm.ve.vl.pvfmkseq.Mvl(<256 x double> %0, i32 256)
  ret <512 x i1> %2
}

; Function Attrs: nounwind readnone
declare <512 x i1> @llvm.ve.vl.pvfmkseq.Mvl(<256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <512 x i1> @pvfmkseq_MvMl(<256 x double> %0, <512 x i1> %1) {
; CHECK-LABEL: pvfmkseq_MvMl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    pvfmk.s.up.eq %vm2, %v0, %vm2
; CHECK-NEXT:    pvfmk.s.lo.eq %vm3, %v0, %vm3
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call <512 x i1> @llvm.ve.vl.pvfmkseq.MvMl(<256 x double> %0, <512 x i1> %1, i32 256)
  ret <512 x i1> %3
}

; Function Attrs: nounwind readnone
declare <512 x i1> @llvm.ve.vl.pvfmkseq.MvMl(<256 x double>, <512 x i1>, i32)

; Function Attrs: nounwind readnone
define fastcc <512 x i1> @pvfmksge_Mvl(<256 x double> %0) {
; CHECK-LABEL: pvfmksge_Mvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    pvfmk.s.up.ge %vm2, %v0
; CHECK-NEXT:    pvfmk.s.lo.ge %vm3, %v0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call <512 x i1> @llvm.ve.vl.pvfmksge.Mvl(<256 x double> %0, i32 256)
  ret <512 x i1> %2
}

; Function Attrs: nounwind readnone
declare <512 x i1> @llvm.ve.vl.pvfmksge.Mvl(<256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <512 x i1> @pvfmksge_MvMl(<256 x double> %0, <512 x i1> %1) {
; CHECK-LABEL: pvfmksge_MvMl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    pvfmk.s.up.ge %vm2, %v0, %vm2
; CHECK-NEXT:    pvfmk.s.lo.ge %vm3, %v0, %vm3
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call <512 x i1> @llvm.ve.vl.pvfmksge.MvMl(<256 x double> %0, <512 x i1> %1, i32 256)
  ret <512 x i1> %3
}

; Function Attrs: nounwind readnone
declare <512 x i1> @llvm.ve.vl.pvfmksge.MvMl(<256 x double>, <512 x i1>, i32)

; Function Attrs: nounwind readnone
define fastcc <512 x i1> @pvfmksle_Mvl(<256 x double> %0) {
; CHECK-LABEL: pvfmksle_Mvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    pvfmk.s.up.le %vm2, %v0
; CHECK-NEXT:    pvfmk.s.lo.le %vm3, %v0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call <512 x i1> @llvm.ve.vl.pvfmksle.Mvl(<256 x double> %0, i32 256)
  ret <512 x i1> %2
}

; Function Attrs: nounwind readnone
declare <512 x i1> @llvm.ve.vl.pvfmksle.Mvl(<256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <512 x i1> @pvfmksle_MvMl(<256 x double> %0, <512 x i1> %1) {
; CHECK-LABEL: pvfmksle_MvMl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    pvfmk.s.up.le %vm2, %v0, %vm2
; CHECK-NEXT:    pvfmk.s.lo.le %vm3, %v0, %vm3
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call <512 x i1> @llvm.ve.vl.pvfmksle.MvMl(<256 x double> %0, <512 x i1> %1, i32 256)
  ret <512 x i1> %3
}

; Function Attrs: nounwind readnone
declare <512 x i1> @llvm.ve.vl.pvfmksle.MvMl(<256 x double>, <512 x i1>, i32)

; Function Attrs: nounwind readnone
define fastcc <512 x i1> @pvfmksnum_Mvl(<256 x double> %0) {
; CHECK-LABEL: pvfmksnum_Mvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    pvfmk.s.up.num %vm2, %v0
; CHECK-NEXT:    pvfmk.s.lo.num %vm3, %v0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call <512 x i1> @llvm.ve.vl.pvfmksnum.Mvl(<256 x double> %0, i32 256)
  ret <512 x i1> %2
}

; Function Attrs: nounwind readnone
declare <512 x i1> @llvm.ve.vl.pvfmksnum.Mvl(<256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <512 x i1> @pvfmksnum_MvMl(<256 x double> %0, <512 x i1> %1) {
; CHECK-LABEL: pvfmksnum_MvMl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    pvfmk.s.up.num %vm2, %v0, %vm2
; CHECK-NEXT:    pvfmk.s.lo.num %vm3, %v0, %vm3
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call <512 x i1> @llvm.ve.vl.pvfmksnum.MvMl(<256 x double> %0, <512 x i1> %1, i32 256)
  ret <512 x i1> %3
}

; Function Attrs: nounwind readnone
declare <512 x i1> @llvm.ve.vl.pvfmksnum.MvMl(<256 x double>, <512 x i1>, i32)

; Function Attrs: nounwind readnone
define fastcc <512 x i1> @pvfmksnan_Mvl(<256 x double> %0) {
; CHECK-LABEL: pvfmksnan_Mvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    pvfmk.s.up.nan %vm2, %v0
; CHECK-NEXT:    pvfmk.s.lo.nan %vm3, %v0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call <512 x i1> @llvm.ve.vl.pvfmksnan.Mvl(<256 x double> %0, i32 256)
  ret <512 x i1> %2
}

; Function Attrs: nounwind readnone
declare <512 x i1> @llvm.ve.vl.pvfmksnan.Mvl(<256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <512 x i1> @pvfmksnan_MvMl(<256 x double> %0, <512 x i1> %1) {
; CHECK-LABEL: pvfmksnan_MvMl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    pvfmk.s.up.nan %vm2, %v0, %vm2
; CHECK-NEXT:    pvfmk.s.lo.nan %vm3, %v0, %vm3
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call <512 x i1> @llvm.ve.vl.pvfmksnan.MvMl(<256 x double> %0, <512 x i1> %1, i32 256)
  ret <512 x i1> %3
}

; Function Attrs: nounwind readnone
declare <512 x i1> @llvm.ve.vl.pvfmksnan.MvMl(<256 x double>, <512 x i1>, i32)

; Function Attrs: nounwind readnone
define fastcc <512 x i1> @pvfmksgtnan_Mvl(<256 x double> %0) {
; CHECK-LABEL: pvfmksgtnan_Mvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    pvfmk.s.up.gtnan %vm2, %v0
; CHECK-NEXT:    pvfmk.s.lo.gtnan %vm3, %v0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call <512 x i1> @llvm.ve.vl.pvfmksgtnan.Mvl(<256 x double> %0, i32 256)
  ret <512 x i1> %2
}

; Function Attrs: nounwind readnone
declare <512 x i1> @llvm.ve.vl.pvfmksgtnan.Mvl(<256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <512 x i1> @pvfmksgtnan_MvMl(<256 x double> %0, <512 x i1> %1) {
; CHECK-LABEL: pvfmksgtnan_MvMl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    pvfmk.s.up.gtnan %vm2, %v0, %vm2
; CHECK-NEXT:    pvfmk.s.lo.gtnan %vm3, %v0, %vm3
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call <512 x i1> @llvm.ve.vl.pvfmksgtnan.MvMl(<256 x double> %0, <512 x i1> %1, i32 256)
  ret <512 x i1> %3
}

; Function Attrs: nounwind readnone
declare <512 x i1> @llvm.ve.vl.pvfmksgtnan.MvMl(<256 x double>, <512 x i1>, i32)

; Function Attrs: nounwind readnone
define fastcc <512 x i1> @pvfmksltnan_Mvl(<256 x double> %0) {
; CHECK-LABEL: pvfmksltnan_Mvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    pvfmk.s.up.ltnan %vm2, %v0
; CHECK-NEXT:    pvfmk.s.lo.ltnan %vm3, %v0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call <512 x i1> @llvm.ve.vl.pvfmksltnan.Mvl(<256 x double> %0, i32 256)
  ret <512 x i1> %2
}

; Function Attrs: nounwind readnone
declare <512 x i1> @llvm.ve.vl.pvfmksltnan.Mvl(<256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <512 x i1> @pvfmksltnan_MvMl(<256 x double> %0, <512 x i1> %1) {
; CHECK-LABEL: pvfmksltnan_MvMl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    pvfmk.s.up.ltnan %vm2, %v0, %vm2
; CHECK-NEXT:    pvfmk.s.lo.ltnan %vm3, %v0, %vm3
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call <512 x i1> @llvm.ve.vl.pvfmksltnan.MvMl(<256 x double> %0, <512 x i1> %1, i32 256)
  ret <512 x i1> %3
}

; Function Attrs: nounwind readnone
declare <512 x i1> @llvm.ve.vl.pvfmksltnan.MvMl(<256 x double>, <512 x i1>, i32)

; Function Attrs: nounwind readnone
define fastcc <512 x i1> @pvfmksnenan_Mvl(<256 x double> %0) {
; CHECK-LABEL: pvfmksnenan_Mvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    pvfmk.s.up.nenan %vm2, %v0
; CHECK-NEXT:    pvfmk.s.lo.nenan %vm3, %v0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call <512 x i1> @llvm.ve.vl.pvfmksnenan.Mvl(<256 x double> %0, i32 256)
  ret <512 x i1> %2
}

; Function Attrs: nounwind readnone
declare <512 x i1> @llvm.ve.vl.pvfmksnenan.Mvl(<256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <512 x i1> @pvfmksnenan_MvMl(<256 x double> %0, <512 x i1> %1) {
; CHECK-LABEL: pvfmksnenan_MvMl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    pvfmk.s.up.nenan %vm2, %v0, %vm2
; CHECK-NEXT:    pvfmk.s.lo.nenan %vm3, %v0, %vm3
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call <512 x i1> @llvm.ve.vl.pvfmksnenan.MvMl(<256 x double> %0, <512 x i1> %1, i32 256)
  ret <512 x i1> %3
}

; Function Attrs: nounwind readnone
declare <512 x i1> @llvm.ve.vl.pvfmksnenan.MvMl(<256 x double>, <512 x i1>, i32)

; Function Attrs: nounwind readnone
define fastcc <512 x i1> @pvfmkseqnan_Mvl(<256 x double> %0) {
; CHECK-LABEL: pvfmkseqnan_Mvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    pvfmk.s.up.eqnan %vm2, %v0
; CHECK-NEXT:    pvfmk.s.lo.eqnan %vm3, %v0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call <512 x i1> @llvm.ve.vl.pvfmkseqnan.Mvl(<256 x double> %0, i32 256)
  ret <512 x i1> %2
}

; Function Attrs: nounwind readnone
declare <512 x i1> @llvm.ve.vl.pvfmkseqnan.Mvl(<256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <512 x i1> @pvfmkseqnan_MvMl(<256 x double> %0, <512 x i1> %1) {
; CHECK-LABEL: pvfmkseqnan_MvMl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    pvfmk.s.up.eqnan %vm2, %v0, %vm2
; CHECK-NEXT:    pvfmk.s.lo.eqnan %vm3, %v0, %vm3
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call <512 x i1> @llvm.ve.vl.pvfmkseqnan.MvMl(<256 x double> %0, <512 x i1> %1, i32 256)
  ret <512 x i1> %3
}

; Function Attrs: nounwind readnone
declare <512 x i1> @llvm.ve.vl.pvfmkseqnan.MvMl(<256 x double>, <512 x i1>, i32)

; Function Attrs: nounwind readnone
define fastcc <512 x i1> @pvfmksgenan_Mvl(<256 x double> %0) {
; CHECK-LABEL: pvfmksgenan_Mvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    pvfmk.s.up.genan %vm2, %v0
; CHECK-NEXT:    pvfmk.s.lo.genan %vm3, %v0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call <512 x i1> @llvm.ve.vl.pvfmksgenan.Mvl(<256 x double> %0, i32 256)
  ret <512 x i1> %2
}

; Function Attrs: nounwind readnone
declare <512 x i1> @llvm.ve.vl.pvfmksgenan.Mvl(<256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <512 x i1> @pvfmksgenan_MvMl(<256 x double> %0, <512 x i1> %1) {
; CHECK-LABEL: pvfmksgenan_MvMl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    pvfmk.s.up.genan %vm2, %v0, %vm2
; CHECK-NEXT:    pvfmk.s.lo.genan %vm3, %v0, %vm3
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call <512 x i1> @llvm.ve.vl.pvfmksgenan.MvMl(<256 x double> %0, <512 x i1> %1, i32 256)
  ret <512 x i1> %3
}

; Function Attrs: nounwind readnone
declare <512 x i1> @llvm.ve.vl.pvfmksgenan.MvMl(<256 x double>, <512 x i1>, i32)

; Function Attrs: nounwind readnone
define fastcc <512 x i1> @pvfmkslenan_Mvl(<256 x double> %0) {
; CHECK-LABEL: pvfmkslenan_Mvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    pvfmk.s.up.lenan %vm2, %v0
; CHECK-NEXT:    pvfmk.s.lo.lenan %vm3, %v0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call <512 x i1> @llvm.ve.vl.pvfmkslenan.Mvl(<256 x double> %0, i32 256)
  ret <512 x i1> %2
}

; Function Attrs: nounwind readnone
declare <512 x i1> @llvm.ve.vl.pvfmkslenan.Mvl(<256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <512 x i1> @pvfmkslenan_MvMl(<256 x double> %0, <512 x i1> %1) {
; CHECK-LABEL: pvfmkslenan_MvMl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    pvfmk.s.up.lenan %vm2, %v0, %vm2
; CHECK-NEXT:    pvfmk.s.lo.lenan %vm3, %v0, %vm3
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call <512 x i1> @llvm.ve.vl.pvfmkslenan.MvMl(<256 x double> %0, <512 x i1> %1, i32 256)
  ret <512 x i1> %3
}

; Function Attrs: nounwind readnone
declare <512 x i1> @llvm.ve.vl.pvfmkslenan.MvMl(<256 x double>, <512 x i1>, i32)

; Function Attrs: nounwind readnone
define fastcc <512 x i1> @pvfmkwgt_Mvl(<256 x double> %0) {
; CHECK-LABEL: pvfmkwgt_Mvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    pvfmk.w.up.gt %vm2, %v0
; CHECK-NEXT:    vfmk.w.gt %vm3, %v0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call <512 x i1> @llvm.ve.vl.pvfmkwgt.Mvl(<256 x double> %0, i32 256)
  ret <512 x i1> %2
}

; Function Attrs: nounwind readnone
declare <512 x i1> @llvm.ve.vl.pvfmkwgt.Mvl(<256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <512 x i1> @pvfmkwgt_MvMl(<256 x double> %0, <512 x i1> %1) {
; CHECK-LABEL: pvfmkwgt_MvMl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    pvfmk.w.up.gt %vm2, %v0, %vm2
; CHECK-NEXT:    vfmk.w.gt %vm3, %v0, %vm3
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call <512 x i1> @llvm.ve.vl.pvfmkwgt.MvMl(<256 x double> %0, <512 x i1> %1, i32 256)
  ret <512 x i1> %3
}

; Function Attrs: nounwind readnone
declare <512 x i1> @llvm.ve.vl.pvfmkwgt.MvMl(<256 x double>, <512 x i1>, i32)

; Function Attrs: nounwind readnone
define fastcc <512 x i1> @pvfmkwlt_Mvl(<256 x double> %0) {
; CHECK-LABEL: pvfmkwlt_Mvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    pvfmk.w.up.lt %vm2, %v0
; CHECK-NEXT:    vfmk.w.lt %vm3, %v0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call <512 x i1> @llvm.ve.vl.pvfmkwlt.Mvl(<256 x double> %0, i32 256)
  ret <512 x i1> %2
}

; Function Attrs: nounwind readnone
declare <512 x i1> @llvm.ve.vl.pvfmkwlt.Mvl(<256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <512 x i1> @pvfmkwlt_MvMl(<256 x double> %0, <512 x i1> %1) {
; CHECK-LABEL: pvfmkwlt_MvMl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    pvfmk.w.up.lt %vm2, %v0, %vm2
; CHECK-NEXT:    vfmk.w.lt %vm3, %v0, %vm3
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call <512 x i1> @llvm.ve.vl.pvfmkwlt.MvMl(<256 x double> %0, <512 x i1> %1, i32 256)
  ret <512 x i1> %3
}

; Function Attrs: nounwind readnone
declare <512 x i1> @llvm.ve.vl.pvfmkwlt.MvMl(<256 x double>, <512 x i1>, i32)

; Function Attrs: nounwind readnone
define fastcc <512 x i1> @pvfmkwne_Mvl(<256 x double> %0) {
; CHECK-LABEL: pvfmkwne_Mvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    pvfmk.w.up.ne %vm2, %v0
; CHECK-NEXT:    vfmk.w.ne %vm3, %v0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call <512 x i1> @llvm.ve.vl.pvfmkwne.Mvl(<256 x double> %0, i32 256)
  ret <512 x i1> %2
}

; Function Attrs: nounwind readnone
declare <512 x i1> @llvm.ve.vl.pvfmkwne.Mvl(<256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <512 x i1> @pvfmkwne_MvMl(<256 x double> %0, <512 x i1> %1) {
; CHECK-LABEL: pvfmkwne_MvMl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    pvfmk.w.up.ne %vm2, %v0, %vm2
; CHECK-NEXT:    vfmk.w.ne %vm3, %v0, %vm3
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call <512 x i1> @llvm.ve.vl.pvfmkwne.MvMl(<256 x double> %0, <512 x i1> %1, i32 256)
  ret <512 x i1> %3
}

; Function Attrs: nounwind readnone
declare <512 x i1> @llvm.ve.vl.pvfmkwne.MvMl(<256 x double>, <512 x i1>, i32)

; Function Attrs: nounwind readnone
define fastcc <512 x i1> @pvfmkweq_Mvl(<256 x double> %0) {
; CHECK-LABEL: pvfmkweq_Mvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    pvfmk.w.up.eq %vm2, %v0
; CHECK-NEXT:    vfmk.w.eq %vm3, %v0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call <512 x i1> @llvm.ve.vl.pvfmkweq.Mvl(<256 x double> %0, i32 256)
  ret <512 x i1> %2
}

; Function Attrs: nounwind readnone
declare <512 x i1> @llvm.ve.vl.pvfmkweq.Mvl(<256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <512 x i1> @pvfmkweq_MvMl(<256 x double> %0, <512 x i1> %1) {
; CHECK-LABEL: pvfmkweq_MvMl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    pvfmk.w.up.eq %vm2, %v0, %vm2
; CHECK-NEXT:    vfmk.w.eq %vm3, %v0, %vm3
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call <512 x i1> @llvm.ve.vl.pvfmkweq.MvMl(<256 x double> %0, <512 x i1> %1, i32 256)
  ret <512 x i1> %3
}

; Function Attrs: nounwind readnone
declare <512 x i1> @llvm.ve.vl.pvfmkweq.MvMl(<256 x double>, <512 x i1>, i32)

; Function Attrs: nounwind readnone
define fastcc <512 x i1> @pvfmkwge_Mvl(<256 x double> %0) {
; CHECK-LABEL: pvfmkwge_Mvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    pvfmk.w.up.ge %vm2, %v0
; CHECK-NEXT:    vfmk.w.ge %vm3, %v0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call <512 x i1> @llvm.ve.vl.pvfmkwge.Mvl(<256 x double> %0, i32 256)
  ret <512 x i1> %2
}

; Function Attrs: nounwind readnone
declare <512 x i1> @llvm.ve.vl.pvfmkwge.Mvl(<256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <512 x i1> @pvfmkwge_MvMl(<256 x double> %0, <512 x i1> %1) {
; CHECK-LABEL: pvfmkwge_MvMl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    pvfmk.w.up.ge %vm2, %v0, %vm2
; CHECK-NEXT:    vfmk.w.ge %vm3, %v0, %vm3
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call <512 x i1> @llvm.ve.vl.pvfmkwge.MvMl(<256 x double> %0, <512 x i1> %1, i32 256)
  ret <512 x i1> %3
}

; Function Attrs: nounwind readnone
declare <512 x i1> @llvm.ve.vl.pvfmkwge.MvMl(<256 x double>, <512 x i1>, i32)

; Function Attrs: nounwind readnone
define fastcc <512 x i1> @pvfmkwle_Mvl(<256 x double> %0) {
; CHECK-LABEL: pvfmkwle_Mvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    pvfmk.w.up.le %vm2, %v0
; CHECK-NEXT:    vfmk.w.le %vm3, %v0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call <512 x i1> @llvm.ve.vl.pvfmkwle.Mvl(<256 x double> %0, i32 256)
  ret <512 x i1> %2
}

; Function Attrs: nounwind readnone
declare <512 x i1> @llvm.ve.vl.pvfmkwle.Mvl(<256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <512 x i1> @pvfmkwle_MvMl(<256 x double> %0, <512 x i1> %1) {
; CHECK-LABEL: pvfmkwle_MvMl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    pvfmk.w.up.le %vm2, %v0, %vm2
; CHECK-NEXT:    vfmk.w.le %vm3, %v0, %vm3
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call <512 x i1> @llvm.ve.vl.pvfmkwle.MvMl(<256 x double> %0, <512 x i1> %1, i32 256)
  ret <512 x i1> %3
}

; Function Attrs: nounwind readnone
declare <512 x i1> @llvm.ve.vl.pvfmkwle.MvMl(<256 x double>, <512 x i1>, i32)

; Function Attrs: nounwind readnone
define fastcc <512 x i1> @pvfmkwnum_Mvl(<256 x double> %0) {
; CHECK-LABEL: pvfmkwnum_Mvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    pvfmk.w.up.num %vm2, %v0
; CHECK-NEXT:    vfmk.w.num %vm3, %v0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call <512 x i1> @llvm.ve.vl.pvfmkwnum.Mvl(<256 x double> %0, i32 256)
  ret <512 x i1> %2
}

; Function Attrs: nounwind readnone
declare <512 x i1> @llvm.ve.vl.pvfmkwnum.Mvl(<256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <512 x i1> @pvfmkwnum_MvMl(<256 x double> %0, <512 x i1> %1) {
; CHECK-LABEL: pvfmkwnum_MvMl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    pvfmk.w.up.num %vm2, %v0, %vm2
; CHECK-NEXT:    vfmk.w.num %vm3, %v0, %vm3
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call <512 x i1> @llvm.ve.vl.pvfmkwnum.MvMl(<256 x double> %0, <512 x i1> %1, i32 256)
  ret <512 x i1> %3
}

; Function Attrs: nounwind readnone
declare <512 x i1> @llvm.ve.vl.pvfmkwnum.MvMl(<256 x double>, <512 x i1>, i32)

; Function Attrs: nounwind readnone
define fastcc <512 x i1> @pvfmkwnan_Mvl(<256 x double> %0) {
; CHECK-LABEL: pvfmkwnan_Mvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    pvfmk.w.up.nan %vm2, %v0
; CHECK-NEXT:    vfmk.w.nan %vm3, %v0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call <512 x i1> @llvm.ve.vl.pvfmkwnan.Mvl(<256 x double> %0, i32 256)
  ret <512 x i1> %2
}

; Function Attrs: nounwind readnone
declare <512 x i1> @llvm.ve.vl.pvfmkwnan.Mvl(<256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <512 x i1> @pvfmkwnan_MvMl(<256 x double> %0, <512 x i1> %1) {
; CHECK-LABEL: pvfmkwnan_MvMl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    pvfmk.w.up.nan %vm2, %v0, %vm2
; CHECK-NEXT:    vfmk.w.nan %vm3, %v0, %vm3
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call <512 x i1> @llvm.ve.vl.pvfmkwnan.MvMl(<256 x double> %0, <512 x i1> %1, i32 256)
  ret <512 x i1> %3
}

; Function Attrs: nounwind readnone
declare <512 x i1> @llvm.ve.vl.pvfmkwnan.MvMl(<256 x double>, <512 x i1>, i32)

; Function Attrs: nounwind readnone
define fastcc <512 x i1> @pvfmkwgtnan_Mvl(<256 x double> %0) {
; CHECK-LABEL: pvfmkwgtnan_Mvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    pvfmk.w.up.gtnan %vm2, %v0
; CHECK-NEXT:    vfmk.w.gtnan %vm3, %v0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call <512 x i1> @llvm.ve.vl.pvfmkwgtnan.Mvl(<256 x double> %0, i32 256)
  ret <512 x i1> %2
}

; Function Attrs: nounwind readnone
declare <512 x i1> @llvm.ve.vl.pvfmkwgtnan.Mvl(<256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <512 x i1> @pvfmkwgtnan_MvMl(<256 x double> %0, <512 x i1> %1) {
; CHECK-LABEL: pvfmkwgtnan_MvMl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    pvfmk.w.up.gtnan %vm2, %v0, %vm2
; CHECK-NEXT:    vfmk.w.gtnan %vm3, %v0, %vm3
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call <512 x i1> @llvm.ve.vl.pvfmkwgtnan.MvMl(<256 x double> %0, <512 x i1> %1, i32 256)
  ret <512 x i1> %3
}

; Function Attrs: nounwind readnone
declare <512 x i1> @llvm.ve.vl.pvfmkwgtnan.MvMl(<256 x double>, <512 x i1>, i32)

; Function Attrs: nounwind readnone
define fastcc <512 x i1> @pvfmkwltnan_Mvl(<256 x double> %0) {
; CHECK-LABEL: pvfmkwltnan_Mvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    pvfmk.w.up.ltnan %vm2, %v0
; CHECK-NEXT:    vfmk.w.ltnan %vm3, %v0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call <512 x i1> @llvm.ve.vl.pvfmkwltnan.Mvl(<256 x double> %0, i32 256)
  ret <512 x i1> %2
}

; Function Attrs: nounwind readnone
declare <512 x i1> @llvm.ve.vl.pvfmkwltnan.Mvl(<256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <512 x i1> @pvfmkwltnan_MvMl(<256 x double> %0, <512 x i1> %1) {
; CHECK-LABEL: pvfmkwltnan_MvMl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    pvfmk.w.up.ltnan %vm2, %v0, %vm2
; CHECK-NEXT:    vfmk.w.ltnan %vm3, %v0, %vm3
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call <512 x i1> @llvm.ve.vl.pvfmkwltnan.MvMl(<256 x double> %0, <512 x i1> %1, i32 256)
  ret <512 x i1> %3
}

; Function Attrs: nounwind readnone
declare <512 x i1> @llvm.ve.vl.pvfmkwltnan.MvMl(<256 x double>, <512 x i1>, i32)

; Function Attrs: nounwind readnone
define fastcc <512 x i1> @pvfmkwnenan_Mvl(<256 x double> %0) {
; CHECK-LABEL: pvfmkwnenan_Mvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    pvfmk.w.up.nenan %vm2, %v0
; CHECK-NEXT:    vfmk.w.nenan %vm3, %v0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call <512 x i1> @llvm.ve.vl.pvfmkwnenan.Mvl(<256 x double> %0, i32 256)
  ret <512 x i1> %2
}

; Function Attrs: nounwind readnone
declare <512 x i1> @llvm.ve.vl.pvfmkwnenan.Mvl(<256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <512 x i1> @pvfmkwnenan_MvMl(<256 x double> %0, <512 x i1> %1) {
; CHECK-LABEL: pvfmkwnenan_MvMl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    pvfmk.w.up.nenan %vm2, %v0, %vm2
; CHECK-NEXT:    vfmk.w.nenan %vm3, %v0, %vm3
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call <512 x i1> @llvm.ve.vl.pvfmkwnenan.MvMl(<256 x double> %0, <512 x i1> %1, i32 256)
  ret <512 x i1> %3
}

; Function Attrs: nounwind readnone
declare <512 x i1> @llvm.ve.vl.pvfmkwnenan.MvMl(<256 x double>, <512 x i1>, i32)

; Function Attrs: nounwind readnone
define fastcc <512 x i1> @pvfmkweqnan_Mvl(<256 x double> %0) {
; CHECK-LABEL: pvfmkweqnan_Mvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    pvfmk.w.up.eqnan %vm2, %v0
; CHECK-NEXT:    vfmk.w.eqnan %vm3, %v0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call <512 x i1> @llvm.ve.vl.pvfmkweqnan.Mvl(<256 x double> %0, i32 256)
  ret <512 x i1> %2
}

; Function Attrs: nounwind readnone
declare <512 x i1> @llvm.ve.vl.pvfmkweqnan.Mvl(<256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <512 x i1> @pvfmkweqnan_MvMl(<256 x double> %0, <512 x i1> %1) {
; CHECK-LABEL: pvfmkweqnan_MvMl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    pvfmk.w.up.eqnan %vm2, %v0, %vm2
; CHECK-NEXT:    vfmk.w.eqnan %vm3, %v0, %vm3
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call <512 x i1> @llvm.ve.vl.pvfmkweqnan.MvMl(<256 x double> %0, <512 x i1> %1, i32 256)
  ret <512 x i1> %3
}

; Function Attrs: nounwind readnone
declare <512 x i1> @llvm.ve.vl.pvfmkweqnan.MvMl(<256 x double>, <512 x i1>, i32)

; Function Attrs: nounwind readnone
define fastcc <512 x i1> @pvfmkwgenan_Mvl(<256 x double> %0) {
; CHECK-LABEL: pvfmkwgenan_Mvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    pvfmk.w.up.genan %vm2, %v0
; CHECK-NEXT:    vfmk.w.genan %vm3, %v0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call <512 x i1> @llvm.ve.vl.pvfmkwgenan.Mvl(<256 x double> %0, i32 256)
  ret <512 x i1> %2
}

; Function Attrs: nounwind readnone
declare <512 x i1> @llvm.ve.vl.pvfmkwgenan.Mvl(<256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <512 x i1> @pvfmkwgenan_MvMl(<256 x double> %0, <512 x i1> %1) {
; CHECK-LABEL: pvfmkwgenan_MvMl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    pvfmk.w.up.genan %vm2, %v0, %vm2
; CHECK-NEXT:    vfmk.w.genan %vm3, %v0, %vm3
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call <512 x i1> @llvm.ve.vl.pvfmkwgenan.MvMl(<256 x double> %0, <512 x i1> %1, i32 256)
  ret <512 x i1> %3
}

; Function Attrs: nounwind readnone
declare <512 x i1> @llvm.ve.vl.pvfmkwgenan.MvMl(<256 x double>, <512 x i1>, i32)

; Function Attrs: nounwind readnone
define fastcc <512 x i1> @pvfmkwlenan_Mvl(<256 x double> %0) {
; CHECK-LABEL: pvfmkwlenan_Mvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    pvfmk.w.up.lenan %vm2, %v0
; CHECK-NEXT:    vfmk.w.lenan %vm3, %v0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call <512 x i1> @llvm.ve.vl.pvfmkwlenan.Mvl(<256 x double> %0, i32 256)
  ret <512 x i1> %2
}

; Function Attrs: nounwind readnone
declare <512 x i1> @llvm.ve.vl.pvfmkwlenan.Mvl(<256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <512 x i1> @pvfmkwlenan_MvMl(<256 x double> %0, <512 x i1> %1) {
; CHECK-LABEL: pvfmkwlenan_MvMl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    pvfmk.w.up.lenan %vm2, %v0, %vm2
; CHECK-NEXT:    vfmk.w.lenan %vm3, %v0, %vm3
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call <512 x i1> @llvm.ve.vl.pvfmkwlenan.MvMl(<256 x double> %0, <512 x i1> %1, i32 256)
  ret <512 x i1> %3
}

; Function Attrs: nounwind readnone
declare <512 x i1> @llvm.ve.vl.pvfmkwlenan.MvMl(<256 x double>, <512 x i1>, i32)
