; RUN: llc < %s -mtriple=ve -mattr=+vpu | FileCheck %s

;;; Test vector move intrinsic instructions
;;;
;;; Note:
;;;   We test VMVivl and VMVivl_v, and VMVivml_v instructions.

; Function Attrs: nounwind
define void @vmv_vsvl(i8* %0, i32 signext %1) {
; CHECK-LABEL: vmv_vsvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s2, 256
; CHECK-NEXT:    lvl %s2
; CHECK-NEXT:    vld %v0, 8, %s0
; CHECK-NEXT:    and %s1, %s1, (32)0
; CHECK-NEXT:    vmv %v0, %s1, %v0
; CHECK-NEXT:    vst %v0, 8, %s0
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.vld.vssl(i64 8, i8* %0, i32 256)
  %4 = tail call fast <256 x double> @llvm.ve.vl.vmv.vsvl(i32 %1, <256 x double> %3, i32 256)
  tail call void @llvm.ve.vl.vst.vssl(<256 x double> %4, i64 8, i8* %0, i32 256)
  ret void
}

; Function Attrs: nounwind readonly
declare <256 x double> @llvm.ve.vl.vld.vssl(i64, i8*, i32)

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vmv.vsvl(i32, <256 x double>, i32)

; Function Attrs: nounwind writeonly
declare void @llvm.ve.vl.vst.vssl(<256 x double>, i64, i8*, i32)

; Function Attrs: nounwind
define void @vmv_vsvl_imm(i8* %0) {
; CHECK-LABEL: vmv_vsvl_imm:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vld %v0, 8, %s0
; CHECK-NEXT:    vmv %v0, 31, %v0
; CHECK-NEXT:    vst %v0, 8, %s0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call fast <256 x double> @llvm.ve.vl.vld.vssl(i64 8, i8* %0, i32 256)
  %3 = tail call fast <256 x double> @llvm.ve.vl.vmv.vsvl(i32 31, <256 x double> %2, i32 256)
  tail call void @llvm.ve.vl.vst.vssl(<256 x double> %3, i64 8, i8* %0, i32 256)
  ret void
}

; Function Attrs: nounwind
define void @vmv_vsvvl(i8* %0, i32 signext %1) {
; CHECK-LABEL: vmv_vsvvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s2, 256
; CHECK-NEXT:    lvl %s2
; CHECK-NEXT:    vld %v0, 8, %s0
; CHECK-NEXT:    and %s1, %s1, (32)0
; CHECK-NEXT:    lea %s3, 128
; CHECK-NEXT:    lvl %s3
; CHECK-NEXT:    vmv %v0, %s1, %v0
; CHECK-NEXT:    lvl %s2
; CHECK-NEXT:    vst %v0, 8, %s0
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.vld.vssl(i64 8, i8* %0, i32 256)
  %4 = tail call fast <256 x double> @llvm.ve.vl.vmv.vsvvl(i32 %1, <256 x double> %3, <256 x double> %3, i32 128)
  tail call void @llvm.ve.vl.vst.vssl(<256 x double> %4, i64 8, i8* %0, i32 256)
  ret void
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vmv.vsvvl(i32, <256 x double>, <256 x double>, i32)

; Function Attrs: nounwind
define void @vmv_vsvvl_imm(i8* %0) {
; CHECK-LABEL: vmv_vsvvl_imm:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vld %v0, 8, %s0
; CHECK-NEXT:    lea %s2, 128
; CHECK-NEXT:    lvl %s2
; CHECK-NEXT:    vmv %v0, 31, %v0
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vst %v0, 8, %s0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call fast <256 x double> @llvm.ve.vl.vld.vssl(i64 8, i8* %0, i32 256)
  %3 = tail call fast <256 x double> @llvm.ve.vl.vmv.vsvvl(i32 31, <256 x double> %2, <256 x double> %2, i32 128)
  tail call void @llvm.ve.vl.vst.vssl(<256 x double> %3, i64 8, i8* %0, i32 256)
  ret void
}

; Function Attrs: nounwind
define void @vmv_vsvmvl(i8* %0, i32 signext %1) {
; CHECK-LABEL: vmv_vsvmvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s2, 256
; CHECK-NEXT:    lvl %s2
; CHECK-NEXT:    vld %v0, 8, %s0
; CHECK-NEXT:    and %s1, %s1, (32)0
; CHECK-NEXT:    lea %s3, 128
; CHECK-NEXT:    lvl %s3
; CHECK-NEXT:    vmv %v0, %s1, %v0, %vm1
; CHECK-NEXT:    lvl %s2
; CHECK-NEXT:    vst %v0, 8, %s0
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.vld.vssl(i64 8, i8* %0, i32 256)
  %4 = tail call fast <256 x double> @llvm.ve.vl.vmv.vsvmvl(i32 %1, <256 x double> %3, <256 x i1> undef, <256 x double> %3, i32 128)
  tail call void @llvm.ve.vl.vst.vssl(<256 x double> %4, i64 8, i8* %0, i32 256)
  ret void
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vmv.vsvmvl(i32, <256 x double>, <256 x i1>, <256 x double>, i32)

; Function Attrs: nounwind
define void @vmv_vsvmvl_imm(i8* %0) {
; CHECK-LABEL: vmv_vsvmvl_imm:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vld %v0, 8, %s0
; CHECK-NEXT:    lea %s2, 128
; CHECK-NEXT:    lvl %s2
; CHECK-NEXT:    vmv %v0, 31, %v0, %vm1
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vst %v0, 8, %s0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call fast <256 x double> @llvm.ve.vl.vld.vssl(i64 8, i8* %0, i32 256)
  %3 = tail call fast <256 x double> @llvm.ve.vl.vmv.vsvmvl(i32 31, <256 x double> %2, <256 x i1> undef, <256 x double> %2, i32 128)
  tail call void @llvm.ve.vl.vst.vssl(<256 x double> %3, i64 8, i8* %0, i32 256)
  ret void
}
