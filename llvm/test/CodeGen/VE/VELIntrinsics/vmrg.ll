; RUN: llc < %s -mtriple=ve -mattr=+vpu | FileCheck %s

;;; Test vector merge intrinsic instructions
;;;
;;; Note:
;;;   We test VMRG*vvml, VMRG*vvml_v, VMRG*rvml, VMRG*rvml_v, VMRG*ivml, and
;;;   VMRG*ivml_v instructions.

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vmrg_vvvml(<256 x double> %0, <256 x double> %1, <256 x i1> %2) {
; CHECK-LABEL: vmrg_vvvml:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vmrg %v0, %v0, %v1, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = tail call fast <256 x double> @llvm.ve.vl.vmrg.vvvml(<256 x double> %0, <256 x double> %1, <256 x i1> %2, i32 256)
  ret <256 x double> %4
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vmrg.vvvml(<256 x double>, <256 x double>, <256 x i1>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vmrg_vvvmvl(<256 x double> %0, <256 x double> %1, <256 x i1> %2, <256 x double> %3) {
; CHECK-LABEL: vmrg_vvvmvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 128
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vmrg %v2, %v0, %v1, %vm1
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v2
; CHECK-NEXT:    b.l.t (, %s10)
  %5 = tail call fast <256 x double> @llvm.ve.vl.vmrg.vvvmvl(<256 x double> %0, <256 x double> %1, <256 x i1> %2, <256 x double> %3, i32 128)
  ret <256 x double> %5
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vmrg.vvvmvl(<256 x double>, <256 x double>, <256 x i1>, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vmrg_vsvml(i64 %0, <256 x double> %1, <256 x i1> %2) {
; CHECK-LABEL: vmrg_vsvml:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vmrg %v0, %s0, %v0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = tail call fast <256 x double> @llvm.ve.vl.vmrg.vsvml(i64 %0, <256 x double> %1, <256 x i1> %2, i32 256)
  ret <256 x double> %4
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vmrg.vsvml(i64, <256 x double>, <256 x i1>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vmrg_vsvmvl(i64 %0, <256 x double> %1, <256 x i1> %2, <256 x double> %3) {
; CHECK-LABEL: vmrg_vsvmvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 128
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vmrg %v1, %s0, %v0, %vm1
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v1
; CHECK-NEXT:    b.l.t (, %s10)
  %5 = tail call fast <256 x double> @llvm.ve.vl.vmrg.vsvmvl(i64 %0, <256 x double> %1, <256 x i1> %2, <256 x double> %3, i32 128)
  ret <256 x double> %5
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vmrg.vsvmvl(i64, <256 x double>, <256 x i1>, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vmrg_vsvml_imm(<256 x double> %0, <256 x i1> %1) {
; CHECK-LABEL: vmrg_vsvml_imm:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vmrg %v0, 8, %v0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.vmrg.vsvml(i64 8, <256 x double> %0, <256 x i1> %1, i32 256)
  ret <256 x double> %3
}

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vmrg_vsvmvl_imm(<256 x double> %0, <256 x i1> %1, <256 x double> %2) {
; CHECK-LABEL: vmrg_vsvmvl_imm:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 128
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vmrg %v1, 8, %v0, %vm1
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v1
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = tail call fast <256 x double> @llvm.ve.vl.vmrg.vsvmvl(i64 8, <256 x double> %0, <256 x i1> %1, <256 x double> %2, i32 128)
  ret <256 x double> %4
}

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vmrgw_vvvMl(<256 x double> %0, <256 x double> %1, <512 x i1> %2) {
; CHECK-LABEL: vmrgw_vvvMl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vmrg.w %v0, %v0, %v1, %vm2
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = tail call fast <256 x double> @llvm.ve.vl.vmrgw.vvvMl(<256 x double> %0, <256 x double> %1, <512 x i1> %2, i32 256)
  ret <256 x double> %4
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vmrgw.vvvMl(<256 x double>, <256 x double>, <512 x i1>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vmrgw_vvvMvl(<256 x double> %0, <256 x double> %1, <512 x i1> %2, <256 x double> %3) {
; CHECK-LABEL: vmrgw_vvvMvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 128
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vmrg.w %v2, %v0, %v1, %vm2
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v2
; CHECK-NEXT:    b.l.t (, %s10)
  %5 = tail call fast <256 x double> @llvm.ve.vl.vmrgw.vvvMvl(<256 x double> %0, <256 x double> %1, <512 x i1> %2, <256 x double> %3, i32 128)
  ret <256 x double> %5
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vmrgw.vvvMvl(<256 x double>, <256 x double>, <512 x i1>, <256 x double>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vmrgw_vsvMl(i32 signext %0, <256 x double> %1, <512 x i1> %2) {
; CHECK-LABEL: vmrgw_vsvMl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vmrg.w %v0, %s0, %v0, %vm2
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = tail call fast <256 x double> @llvm.ve.vl.vmrgw.vsvMl(i32 %0, <256 x double> %1, <512 x i1> %2, i32 256)
  ret <256 x double> %4
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vmrgw.vsvMl(i32, <256 x double>, <512 x i1>, i32)

; Function Attrs: nounwind readnone
define fastcc <256 x double> @vmrgw_vsvMvl(i32 signext %0, <256 x double> %1, <512 x i1> %2, <256 x double> %3) {
; CHECK-LABEL: vmrgw_vsvMvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    lea %s1, 128
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vmrg.w %v1, %s0, %v0, %vm2
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v1
; CHECK-NEXT:    b.l.t (, %s10)
  %5 = tail call fast <256 x double> @llvm.ve.vl.vmrgw.vsvMvl(i32 %0, <256 x double> %1, <512 x i1> %2, <256 x double> %3, i32 128)
  ret <256 x double> %5
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vmrgw.vsvMvl(i32, <256 x double>, <512 x i1>, <256 x double>, i32)
