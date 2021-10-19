; RUN: llc < %s -mtriple=ve -mattr=+vpu | FileCheck %s

;;; Test vector load intrinsic instructions
;;;
;;; Note:
;;;   We test VLD*rrl, VLD*irl, VLD*rrl_v, and VLD*irl_v instructions.

; Function Attrs: nounwind
define void @vld_vssl(i8* %0, i64 %1) {
; CHECK-LABEL: vld_vssl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s2, 256
; CHECK-NEXT:    lvl %s2
; CHECK-NEXT:    vld %v0, %s1, %s0
; CHECK-NEXT:    #APP
; CHECK-NEXT:    vst %v0, %s1, %s0
; CHECK-NEXT:    #NO_APP
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.vld.vssl(i64 %1, i8* %0, i32 256)
  tail call void asm sideeffect "vst ${0:v}, $2, $1", "v,r,r"(<256 x double> %3, i8* %0, i64 %1)
  ret void
}

; Function Attrs: nounwind readonly
declare <256 x double> @llvm.ve.vl.vld.vssl(i64, i8*, i32)

; Function Attrs: nounwind
define void @vld_vssvl(i8* %0, i64 %1, i8* %2) {
; CHECK-LABEL: vld_vssvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s3, 256
; CHECK-NEXT:    lvl %s3
; CHECK-NEXT:    vld %v0, %s1, %s2
; CHECK-NEXT:    vld %v0, %s1, %s0
; CHECK-NEXT:    #APP
; CHECK-NEXT:    vst %v0, %s1, %s0
; CHECK-NEXT:    #NO_APP
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = tail call fast <256 x double> @llvm.ve.vl.vld.vssl(i64 %1, i8* %2, i32 256)
  %5 = tail call fast <256 x double> @llvm.ve.vl.vld.vssvl(i64 %1, i8* %0, <256 x double> %4, i32 256)
  tail call void asm sideeffect "vst ${0:v}, $2, $1", "v,r,r"(<256 x double> %5, i8* %0, i64 %1)
  ret void
}

; Function Attrs: nounwind readonly
declare <256 x double> @llvm.ve.vl.vld.vssvl(i64, i8*, <256 x double>, i32)

; Function Attrs: nounwind
define void @vld_vssl_imm(i8* %0) {
; CHECK-LABEL: vld_vssl_imm:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vld %v0, 8, %s0
; CHECK-NEXT:    #APP
; CHECK-NEXT:    vst %v0, 8, %s0
; CHECK-NEXT:    #NO_APP
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call fast <256 x double> @llvm.ve.vl.vld.vssl(i64 8, i8* %0, i32 256)
  tail call void asm sideeffect "vst ${0:v}, 8, $1", "v,r"(<256 x double> %2, i8* %0)
  ret void
}

; Function Attrs: nounwind
define void @vld_vssvl_imm(i8* %0, i8* %1) {
; CHECK-LABEL: vld_vssvl_imm:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s2, 256
; CHECK-NEXT:    lvl %s2
; CHECK-NEXT:    vld %v0, 8, %s1
; CHECK-NEXT:    vld %v0, 8, %s0
; CHECK-NEXT:    #APP
; CHECK-NEXT:    vst %v0, 8, %s0
; CHECK-NEXT:    #NO_APP
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.vld.vssl(i64 8, i8* %1, i32 256)
  %4 = tail call fast <256 x double> @llvm.ve.vl.vld.vssvl(i64 8, i8* %0, <256 x double> %3, i32 256)
  tail call void asm sideeffect "vst ${0:v}, 8, $1", "v,r"(<256 x double> %4, i8* %0)
  ret void
}

; Function Attrs: nounwind
define void @vldnc_vssl(i8* %0, i64 %1) {
; CHECK-LABEL: vldnc_vssl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s2, 256
; CHECK-NEXT:    lvl %s2
; CHECK-NEXT:    vld.nc %v0, %s1, %s0
; CHECK-NEXT:    #APP
; CHECK-NEXT:    vst %v0, %s1, %s0
; CHECK-NEXT:    #NO_APP
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.vldnc.vssl(i64 %1, i8* %0, i32 256)
  tail call void asm sideeffect "vst ${0:v}, $2, $1", "v,r,r"(<256 x double> %3, i8* %0, i64 %1)
  ret void
}

; Function Attrs: nounwind readonly
declare <256 x double> @llvm.ve.vl.vldnc.vssl(i64, i8*, i32)

; Function Attrs: nounwind
define void @vldnc_vssvl(i8* %0, i64 %1, i8* %2) {
; CHECK-LABEL: vldnc_vssvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s3, 256
; CHECK-NEXT:    lvl %s3
; CHECK-NEXT:    vld.nc %v0, %s1, %s2
; CHECK-NEXT:    vld.nc %v0, %s1, %s0
; CHECK-NEXT:    #APP
; CHECK-NEXT:    vst %v0, %s1, %s0
; CHECK-NEXT:    #NO_APP
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = tail call fast <256 x double> @llvm.ve.vl.vldnc.vssl(i64 %1, i8* %2, i32 256)
  %5 = tail call fast <256 x double> @llvm.ve.vl.vldnc.vssvl(i64 %1, i8* %0, <256 x double> %4, i32 256)
  tail call void asm sideeffect "vst ${0:v}, $2, $1", "v,r,r"(<256 x double> %5, i8* %0, i64 %1)
  ret void
}

; Function Attrs: nounwind readonly
declare <256 x double> @llvm.ve.vl.vldnc.vssvl(i64, i8*, <256 x double>, i32)

; Function Attrs: nounwind
define void @vldnc_vssl_imm(i8* %0) {
; CHECK-LABEL: vldnc_vssl_imm:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vld.nc %v0, 8, %s0
; CHECK-NEXT:    #APP
; CHECK-NEXT:    vst %v0, 8, %s0
; CHECK-NEXT:    #NO_APP
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call fast <256 x double> @llvm.ve.vl.vldnc.vssl(i64 8, i8* %0, i32 256)
  tail call void asm sideeffect "vst ${0:v}, 8, $1", "v,r"(<256 x double> %2, i8* %0)
  ret void
}

; Function Attrs: nounwind
define void @vldnc_vssvl_imm(i8* %0, i8* %1) {
; CHECK-LABEL: vldnc_vssvl_imm:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s2, 256
; CHECK-NEXT:    lvl %s2
; CHECK-NEXT:    vld.nc %v0, 8, %s1
; CHECK-NEXT:    vld.nc %v0, 8, %s0
; CHECK-NEXT:    #APP
; CHECK-NEXT:    vst %v0, 8, %s0
; CHECK-NEXT:    #NO_APP
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.vldnc.vssl(i64 8, i8* %1, i32 256)
  %4 = tail call fast <256 x double> @llvm.ve.vl.vldnc.vssvl(i64 8, i8* %0, <256 x double> %3, i32 256)
  tail call void asm sideeffect "vst ${0:v}, 8, $1", "v,r"(<256 x double> %4, i8* %0)
  ret void
}

; Function Attrs: nounwind
define void @vldu_vssl(i8* %0, i64 %1) {
; CHECK-LABEL: vldu_vssl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s2, 256
; CHECK-NEXT:    lvl %s2
; CHECK-NEXT:    vldu %v0, %s1, %s0
; CHECK-NEXT:    #APP
; CHECK-NEXT:    vst %v0, %s1, %s0
; CHECK-NEXT:    #NO_APP
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.vldu.vssl(i64 %1, i8* %0, i32 256)
  tail call void asm sideeffect "vst ${0:v}, $2, $1", "v,r,r"(<256 x double> %3, i8* %0, i64 %1)
  ret void
}

; Function Attrs: nounwind readonly
declare <256 x double> @llvm.ve.vl.vldu.vssl(i64, i8*, i32)

; Function Attrs: nounwind
define void @vldu_vssvl(i8* %0, i64 %1, i8* %2) {
; CHECK-LABEL: vldu_vssvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s3, 256
; CHECK-NEXT:    lvl %s3
; CHECK-NEXT:    vldu %v0, %s1, %s2
; CHECK-NEXT:    vldu %v0, %s1, %s0
; CHECK-NEXT:    #APP
; CHECK-NEXT:    vst %v0, %s1, %s0
; CHECK-NEXT:    #NO_APP
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = tail call fast <256 x double> @llvm.ve.vl.vldu.vssl(i64 %1, i8* %2, i32 256)
  %5 = tail call fast <256 x double> @llvm.ve.vl.vldu.vssvl(i64 %1, i8* %0, <256 x double> %4, i32 256)
  tail call void asm sideeffect "vst ${0:v}, $2, $1", "v,r,r"(<256 x double> %5, i8* %0, i64 %1)
  ret void
}

; Function Attrs: nounwind readonly
declare <256 x double> @llvm.ve.vl.vldu.vssvl(i64, i8*, <256 x double>, i32)

; Function Attrs: nounwind
define void @vldu_vssl_imm(i8* %0) {
; CHECK-LABEL: vldu_vssl_imm:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vldu %v0, 8, %s0
; CHECK-NEXT:    #APP
; CHECK-NEXT:    vst %v0, 8, %s0
; CHECK-NEXT:    #NO_APP
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call fast <256 x double> @llvm.ve.vl.vldu.vssl(i64 8, i8* %0, i32 256)
  tail call void asm sideeffect "vst ${0:v}, 8, $1", "v,r"(<256 x double> %2, i8* %0)
  ret void
}

; Function Attrs: nounwind
define void @vldu_vssvl_imm(i8* %0, i8* %1) {
; CHECK-LABEL: vldu_vssvl_imm:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s2, 256
; CHECK-NEXT:    lvl %s2
; CHECK-NEXT:    vldu %v0, 8, %s1
; CHECK-NEXT:    vldu %v0, 8, %s0
; CHECK-NEXT:    #APP
; CHECK-NEXT:    vst %v0, 8, %s0
; CHECK-NEXT:    #NO_APP
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.vldu.vssl(i64 8, i8* %1, i32 256)
  %4 = tail call fast <256 x double> @llvm.ve.vl.vldu.vssvl(i64 8, i8* %0, <256 x double> %3, i32 256)
  tail call void asm sideeffect "vst ${0:v}, 8, $1", "v,r"(<256 x double> %4, i8* %0)
  ret void
}

; Function Attrs: nounwind
define void @vldunc_vssl(i8* %0, i64 %1) {
; CHECK-LABEL: vldunc_vssl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s2, 256
; CHECK-NEXT:    lvl %s2
; CHECK-NEXT:    vldu.nc %v0, %s1, %s0
; CHECK-NEXT:    #APP
; CHECK-NEXT:    vst %v0, %s1, %s0
; CHECK-NEXT:    #NO_APP
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.vldunc.vssl(i64 %1, i8* %0, i32 256)
  tail call void asm sideeffect "vst ${0:v}, $2, $1", "v,r,r"(<256 x double> %3, i8* %0, i64 %1)
  ret void
}

; Function Attrs: nounwind readonly
declare <256 x double> @llvm.ve.vl.vldunc.vssl(i64, i8*, i32)

; Function Attrs: nounwind
define void @vldunc_vssvl(i8* %0, i64 %1, i8* %2) {
; CHECK-LABEL: vldunc_vssvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s3, 256
; CHECK-NEXT:    lvl %s3
; CHECK-NEXT:    vldu.nc %v0, %s1, %s2
; CHECK-NEXT:    vldu.nc %v0, %s1, %s0
; CHECK-NEXT:    #APP
; CHECK-NEXT:    vst %v0, %s1, %s0
; CHECK-NEXT:    #NO_APP
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = tail call fast <256 x double> @llvm.ve.vl.vldunc.vssl(i64 %1, i8* %2, i32 256)
  %5 = tail call fast <256 x double> @llvm.ve.vl.vldunc.vssvl(i64 %1, i8* %0, <256 x double> %4, i32 256)
  tail call void asm sideeffect "vst ${0:v}, $2, $1", "v,r,r"(<256 x double> %5, i8* %0, i64 %1)
  ret void
}

; Function Attrs: nounwind readonly
declare <256 x double> @llvm.ve.vl.vldunc.vssvl(i64, i8*, <256 x double>, i32)

; Function Attrs: nounwind
define void @vldunc_vssl_imm(i8* %0) {
; CHECK-LABEL: vldunc_vssl_imm:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vldu.nc %v0, 8, %s0
; CHECK-NEXT:    #APP
; CHECK-NEXT:    vst %v0, 8, %s0
; CHECK-NEXT:    #NO_APP
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call fast <256 x double> @llvm.ve.vl.vldunc.vssl(i64 8, i8* %0, i32 256)
  tail call void asm sideeffect "vst ${0:v}, 8, $1", "v,r"(<256 x double> %2, i8* %0)
  ret void
}

; Function Attrs: nounwind
define void @vldunc_vssvl_imm(i8* %0, i8* %1) {
; CHECK-LABEL: vldunc_vssvl_imm:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s2, 256
; CHECK-NEXT:    lvl %s2
; CHECK-NEXT:    vldu.nc %v0, 8, %s1
; CHECK-NEXT:    vldu.nc %v0, 8, %s0
; CHECK-NEXT:    #APP
; CHECK-NEXT:    vst %v0, 8, %s0
; CHECK-NEXT:    #NO_APP
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.vldunc.vssl(i64 8, i8* %1, i32 256)
  %4 = tail call fast <256 x double> @llvm.ve.vl.vldunc.vssvl(i64 8, i8* %0, <256 x double> %3, i32 256)
  tail call void asm sideeffect "vst ${0:v}, 8, $1", "v,r"(<256 x double> %4, i8* %0)
  ret void
}

; Function Attrs: nounwind
define void @vldlsx_vssl(i8* %0, i64 %1) {
; CHECK-LABEL: vldlsx_vssl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s2, 256
; CHECK-NEXT:    lvl %s2
; CHECK-NEXT:    vldl.sx %v0, %s1, %s0
; CHECK-NEXT:    #APP
; CHECK-NEXT:    vst %v0, %s1, %s0
; CHECK-NEXT:    #NO_APP
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.vldlsx.vssl(i64 %1, i8* %0, i32 256)
  tail call void asm sideeffect "vst ${0:v}, $2, $1", "v,r,r"(<256 x double> %3, i8* %0, i64 %1)
  ret void
}

; Function Attrs: nounwind readonly
declare <256 x double> @llvm.ve.vl.vldlsx.vssl(i64, i8*, i32)

; Function Attrs: nounwind
define void @vldlsx_vssvl(i8* %0, i64 %1, i8* %2) {
; CHECK-LABEL: vldlsx_vssvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s3, 256
; CHECK-NEXT:    lvl %s3
; CHECK-NEXT:    vldl.sx %v0, %s1, %s2
; CHECK-NEXT:    vldl.sx %v0, %s1, %s0
; CHECK-NEXT:    #APP
; CHECK-NEXT:    vst %v0, %s1, %s0
; CHECK-NEXT:    #NO_APP
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = tail call fast <256 x double> @llvm.ve.vl.vldlsx.vssl(i64 %1, i8* %2, i32 256)
  %5 = tail call fast <256 x double> @llvm.ve.vl.vldlsx.vssvl(i64 %1, i8* %0, <256 x double> %4, i32 256)
  tail call void asm sideeffect "vst ${0:v}, $2, $1", "v,r,r"(<256 x double> %5, i8* %0, i64 %1)
  ret void
}

; Function Attrs: nounwind readonly
declare <256 x double> @llvm.ve.vl.vldlsx.vssvl(i64, i8*, <256 x double>, i32)

; Function Attrs: nounwind
define void @vldlsx_vssl_imm(i8* %0) {
; CHECK-LABEL: vldlsx_vssl_imm:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vldl.sx %v0, 8, %s0
; CHECK-NEXT:    #APP
; CHECK-NEXT:    vst %v0, 8, %s0
; CHECK-NEXT:    #NO_APP
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call fast <256 x double> @llvm.ve.vl.vldlsx.vssl(i64 8, i8* %0, i32 256)
  tail call void asm sideeffect "vst ${0:v}, 8, $1", "v,r"(<256 x double> %2, i8* %0)
  ret void
}

; Function Attrs: nounwind
define void @vldlsx_vssvl_imm(i8* %0, i8* %1) {
; CHECK-LABEL: vldlsx_vssvl_imm:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s2, 256
; CHECK-NEXT:    lvl %s2
; CHECK-NEXT:    vldl.sx %v0, 8, %s1
; CHECK-NEXT:    vldl.sx %v0, 8, %s0
; CHECK-NEXT:    #APP
; CHECK-NEXT:    vst %v0, 8, %s0
; CHECK-NEXT:    #NO_APP
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.vldlsx.vssl(i64 8, i8* %1, i32 256)
  %4 = tail call fast <256 x double> @llvm.ve.vl.vldlsx.vssvl(i64 8, i8* %0, <256 x double> %3, i32 256)
  tail call void asm sideeffect "vst ${0:v}, 8, $1", "v,r"(<256 x double> %4, i8* %0)
  ret void
}

; Function Attrs: nounwind
define void @vldlsxnc_vssl(i8* %0, i64 %1) {
; CHECK-LABEL: vldlsxnc_vssl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s2, 256
; CHECK-NEXT:    lvl %s2
; CHECK-NEXT:    vldl.sx.nc %v0, %s1, %s0
; CHECK-NEXT:    #APP
; CHECK-NEXT:    vst %v0, %s1, %s0
; CHECK-NEXT:    #NO_APP
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.vldlsxnc.vssl(i64 %1, i8* %0, i32 256)
  tail call void asm sideeffect "vst ${0:v}, $2, $1", "v,r,r"(<256 x double> %3, i8* %0, i64 %1)
  ret void
}

; Function Attrs: nounwind readonly
declare <256 x double> @llvm.ve.vl.vldlsxnc.vssl(i64, i8*, i32)

; Function Attrs: nounwind
define void @vldlsxnc_vssvl(i8* %0, i64 %1, i8* %2) {
; CHECK-LABEL: vldlsxnc_vssvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s3, 256
; CHECK-NEXT:    lvl %s3
; CHECK-NEXT:    vldl.sx.nc %v0, %s1, %s2
; CHECK-NEXT:    vldl.sx.nc %v0, %s1, %s0
; CHECK-NEXT:    #APP
; CHECK-NEXT:    vst %v0, %s1, %s0
; CHECK-NEXT:    #NO_APP
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = tail call fast <256 x double> @llvm.ve.vl.vldlsxnc.vssl(i64 %1, i8* %2, i32 256)
  %5 = tail call fast <256 x double> @llvm.ve.vl.vldlsxnc.vssvl(i64 %1, i8* %0, <256 x double> %4, i32 256)
  tail call void asm sideeffect "vst ${0:v}, $2, $1", "v,r,r"(<256 x double> %5, i8* %0, i64 %1)
  ret void
}

; Function Attrs: nounwind readonly
declare <256 x double> @llvm.ve.vl.vldlsxnc.vssvl(i64, i8*, <256 x double>, i32)

; Function Attrs: nounwind
define void @vldlsxnc_vssl_imm(i8* %0) {
; CHECK-LABEL: vldlsxnc_vssl_imm:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vldl.sx.nc %v0, 8, %s0
; CHECK-NEXT:    #APP
; CHECK-NEXT:    vst %v0, 8, %s0
; CHECK-NEXT:    #NO_APP
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call fast <256 x double> @llvm.ve.vl.vldlsxnc.vssl(i64 8, i8* %0, i32 256)
  tail call void asm sideeffect "vst ${0:v}, 8, $1", "v,r"(<256 x double> %2, i8* %0)
  ret void
}

; Function Attrs: nounwind
define void @vldlsxnc_vssvl_imm(i8* %0, i8* %1) {
; CHECK-LABEL: vldlsxnc_vssvl_imm:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s2, 256
; CHECK-NEXT:    lvl %s2
; CHECK-NEXT:    vldl.sx.nc %v0, 8, %s1
; CHECK-NEXT:    vldl.sx.nc %v0, 8, %s0
; CHECK-NEXT:    #APP
; CHECK-NEXT:    vst %v0, 8, %s0
; CHECK-NEXT:    #NO_APP
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.vldlsxnc.vssl(i64 8, i8* %1, i32 256)
  %4 = tail call fast <256 x double> @llvm.ve.vl.vldlsxnc.vssvl(i64 8, i8* %0, <256 x double> %3, i32 256)
  tail call void asm sideeffect "vst ${0:v}, 8, $1", "v,r"(<256 x double> %4, i8* %0)
  ret void
}

; Function Attrs: nounwind
define void @vldlzx_vssl(i8* %0, i64 %1) {
; CHECK-LABEL: vldlzx_vssl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s2, 256
; CHECK-NEXT:    lvl %s2
; CHECK-NEXT:    vldl.zx %v0, %s1, %s0
; CHECK-NEXT:    #APP
; CHECK-NEXT:    vst %v0, %s1, %s0
; CHECK-NEXT:    #NO_APP
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.vldlzx.vssl(i64 %1, i8* %0, i32 256)
  tail call void asm sideeffect "vst ${0:v}, $2, $1", "v,r,r"(<256 x double> %3, i8* %0, i64 %1)
  ret void
}

; Function Attrs: nounwind readonly
declare <256 x double> @llvm.ve.vl.vldlzx.vssl(i64, i8*, i32)

; Function Attrs: nounwind
define void @vldlzx_vssvl(i8* %0, i64 %1, i8* %2) {
; CHECK-LABEL: vldlzx_vssvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s3, 256
; CHECK-NEXT:    lvl %s3
; CHECK-NEXT:    vldl.zx %v0, %s1, %s2
; CHECK-NEXT:    vldl.zx %v0, %s1, %s0
; CHECK-NEXT:    #APP
; CHECK-NEXT:    vst %v0, %s1, %s0
; CHECK-NEXT:    #NO_APP
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = tail call fast <256 x double> @llvm.ve.vl.vldlzx.vssl(i64 %1, i8* %2, i32 256)
  %5 = tail call fast <256 x double> @llvm.ve.vl.vldlzx.vssvl(i64 %1, i8* %0, <256 x double> %4, i32 256)
  tail call void asm sideeffect "vst ${0:v}, $2, $1", "v,r,r"(<256 x double> %5, i8* %0, i64 %1)
  ret void
}

; Function Attrs: nounwind readonly
declare <256 x double> @llvm.ve.vl.vldlzx.vssvl(i64, i8*, <256 x double>, i32)

; Function Attrs: nounwind
define void @vldlzx_vssl_imm(i8* %0) {
; CHECK-LABEL: vldlzx_vssl_imm:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vldl.zx %v0, 8, %s0
; CHECK-NEXT:    #APP
; CHECK-NEXT:    vst %v0, 8, %s0
; CHECK-NEXT:    #NO_APP
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call fast <256 x double> @llvm.ve.vl.vldlzx.vssl(i64 8, i8* %0, i32 256)
  tail call void asm sideeffect "vst ${0:v}, 8, $1", "v,r"(<256 x double> %2, i8* %0)
  ret void
}

; Function Attrs: nounwind
define void @vldlzx_vssvl_imm(i8* %0, i8* %1) {
; CHECK-LABEL: vldlzx_vssvl_imm:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s2, 256
; CHECK-NEXT:    lvl %s2
; CHECK-NEXT:    vldl.zx %v0, 8, %s1
; CHECK-NEXT:    vldl.zx %v0, 8, %s0
; CHECK-NEXT:    #APP
; CHECK-NEXT:    vst %v0, 8, %s0
; CHECK-NEXT:    #NO_APP
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.vldlzx.vssl(i64 8, i8* %1, i32 256)
  %4 = tail call fast <256 x double> @llvm.ve.vl.vldlzx.vssvl(i64 8, i8* %0, <256 x double> %3, i32 256)
  tail call void asm sideeffect "vst ${0:v}, 8, $1", "v,r"(<256 x double> %4, i8* %0)
  ret void
}

; Function Attrs: nounwind
define void @vldlzxnc_vssl(i8* %0, i64 %1) {
; CHECK-LABEL: vldlzxnc_vssl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s2, 256
; CHECK-NEXT:    lvl %s2
; CHECK-NEXT:    vldl.zx.nc %v0, %s1, %s0
; CHECK-NEXT:    #APP
; CHECK-NEXT:    vst %v0, %s1, %s0
; CHECK-NEXT:    #NO_APP
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.vldlzxnc.vssl(i64 %1, i8* %0, i32 256)
  tail call void asm sideeffect "vst ${0:v}, $2, $1", "v,r,r"(<256 x double> %3, i8* %0, i64 %1)
  ret void
}

; Function Attrs: nounwind readonly
declare <256 x double> @llvm.ve.vl.vldlzxnc.vssl(i64, i8*, i32)

; Function Attrs: nounwind
define void @vldlzxnc_vssvl(i8* %0, i64 %1, i8* %2) {
; CHECK-LABEL: vldlzxnc_vssvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s3, 256
; CHECK-NEXT:    lvl %s3
; CHECK-NEXT:    vldl.zx.nc %v0, %s1, %s2
; CHECK-NEXT:    vldl.zx.nc %v0, %s1, %s0
; CHECK-NEXT:    #APP
; CHECK-NEXT:    vst %v0, %s1, %s0
; CHECK-NEXT:    #NO_APP
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = tail call fast <256 x double> @llvm.ve.vl.vldlzxnc.vssl(i64 %1, i8* %2, i32 256)
  %5 = tail call fast <256 x double> @llvm.ve.vl.vldlzxnc.vssvl(i64 %1, i8* %0, <256 x double> %4, i32 256)
  tail call void asm sideeffect "vst ${0:v}, $2, $1", "v,r,r"(<256 x double> %5, i8* %0, i64 %1)
  ret void
}

; Function Attrs: nounwind readonly
declare <256 x double> @llvm.ve.vl.vldlzxnc.vssvl(i64, i8*, <256 x double>, i32)

; Function Attrs: nounwind
define void @vldlzxnc_vssl_imm(i8* %0) {
; CHECK-LABEL: vldlzxnc_vssl_imm:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vldl.zx.nc %v0, 8, %s0
; CHECK-NEXT:    #APP
; CHECK-NEXT:    vst %v0, 8, %s0
; CHECK-NEXT:    #NO_APP
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call fast <256 x double> @llvm.ve.vl.vldlzxnc.vssl(i64 8, i8* %0, i32 256)
  tail call void asm sideeffect "vst ${0:v}, 8, $1", "v,r"(<256 x double> %2, i8* %0)
  ret void
}

; Function Attrs: nounwind
define void @vldlzxnc_vssvl_imm(i8* %0, i8* %1) {
; CHECK-LABEL: vldlzxnc_vssvl_imm:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s2, 256
; CHECK-NEXT:    lvl %s2
; CHECK-NEXT:    vldl.zx.nc %v0, 8, %s1
; CHECK-NEXT:    vldl.zx.nc %v0, 8, %s0
; CHECK-NEXT:    #APP
; CHECK-NEXT:    vst %v0, 8, %s0
; CHECK-NEXT:    #NO_APP
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.vldlzxnc.vssl(i64 8, i8* %1, i32 256)
  %4 = tail call fast <256 x double> @llvm.ve.vl.vldlzxnc.vssvl(i64 8, i8* %0, <256 x double> %3, i32 256)
  tail call void asm sideeffect "vst ${0:v}, 8, $1", "v,r"(<256 x double> %4, i8* %0)
  ret void
}

; Function Attrs: nounwind
define void @vld2d_vssl(i8* %0, i64 %1) {
; CHECK-LABEL: vld2d_vssl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s2, 256
; CHECK-NEXT:    lvl %s2
; CHECK-NEXT:    vld2d %v0, %s1, %s0
; CHECK-NEXT:    #APP
; CHECK-NEXT:    vst %v0, %s1, %s0
; CHECK-NEXT:    #NO_APP
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.vld2d.vssl(i64 %1, i8* %0, i32 256)
  tail call void asm sideeffect "vst ${0:v}, $2, $1", "v,r,r"(<256 x double> %3, i8* %0, i64 %1)
  ret void
}

; Function Attrs: nounwind readonly
declare <256 x double> @llvm.ve.vl.vld2d.vssl(i64, i8*, i32)

; Function Attrs: nounwind
define void @vld2d_vssvl(i8* %0, i64 %1, i8* %2) {
; CHECK-LABEL: vld2d_vssvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s3, 256
; CHECK-NEXT:    lvl %s3
; CHECK-NEXT:    vld2d %v0, %s1, %s2
; CHECK-NEXT:    vld2d %v0, %s1, %s0
; CHECK-NEXT:    #APP
; CHECK-NEXT:    vst %v0, %s1, %s0
; CHECK-NEXT:    #NO_APP
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = tail call fast <256 x double> @llvm.ve.vl.vld2d.vssl(i64 %1, i8* %2, i32 256)
  %5 = tail call fast <256 x double> @llvm.ve.vl.vld2d.vssvl(i64 %1, i8* %0, <256 x double> %4, i32 256)
  tail call void asm sideeffect "vst ${0:v}, $2, $1", "v,r,r"(<256 x double> %5, i8* %0, i64 %1)
  ret void
}

; Function Attrs: nounwind readonly
declare <256 x double> @llvm.ve.vl.vld2d.vssvl(i64, i8*, <256 x double>, i32)

; Function Attrs: nounwind
define void @vld2d_vssl_imm(i8* %0) {
; CHECK-LABEL: vld2d_vssl_imm:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vld2d %v0, 8, %s0
; CHECK-NEXT:    #APP
; CHECK-NEXT:    vst %v0, 8, %s0
; CHECK-NEXT:    #NO_APP
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call fast <256 x double> @llvm.ve.vl.vld2d.vssl(i64 8, i8* %0, i32 256)
  tail call void asm sideeffect "vst ${0:v}, 8, $1", "v,r"(<256 x double> %2, i8* %0)
  ret void
}

; Function Attrs: nounwind
define void @vld2d_vssvl_imm(i8* %0, i8* %1) {
; CHECK-LABEL: vld2d_vssvl_imm:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s2, 256
; CHECK-NEXT:    lvl %s2
; CHECK-NEXT:    vld2d %v0, 8, %s1
; CHECK-NEXT:    vld2d %v0, 8, %s0
; CHECK-NEXT:    #APP
; CHECK-NEXT:    vst %v0, 8, %s0
; CHECK-NEXT:    #NO_APP
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.vld2d.vssl(i64 8, i8* %1, i32 256)
  %4 = tail call fast <256 x double> @llvm.ve.vl.vld2d.vssvl(i64 8, i8* %0, <256 x double> %3, i32 256)
  tail call void asm sideeffect "vst ${0:v}, 8, $1", "v,r"(<256 x double> %4, i8* %0)
  ret void
}

; Function Attrs: nounwind
define void @vld2dnc_vssl(i8* %0, i64 %1) {
; CHECK-LABEL: vld2dnc_vssl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s2, 256
; CHECK-NEXT:    lvl %s2
; CHECK-NEXT:    vld2d.nc %v0, %s1, %s0
; CHECK-NEXT:    #APP
; CHECK-NEXT:    vst %v0, %s1, %s0
; CHECK-NEXT:    #NO_APP
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.vld2dnc.vssl(i64 %1, i8* %0, i32 256)
  tail call void asm sideeffect "vst ${0:v}, $2, $1", "v,r,r"(<256 x double> %3, i8* %0, i64 %1)
  ret void
}

; Function Attrs: nounwind readonly
declare <256 x double> @llvm.ve.vl.vld2dnc.vssl(i64, i8*, i32)

; Function Attrs: nounwind
define void @vld2dnc_vssvl(i8* %0, i64 %1, i8* %2) {
; CHECK-LABEL: vld2dnc_vssvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s3, 256
; CHECK-NEXT:    lvl %s3
; CHECK-NEXT:    vld2d.nc %v0, %s1, %s2
; CHECK-NEXT:    vld2d.nc %v0, %s1, %s0
; CHECK-NEXT:    #APP
; CHECK-NEXT:    vst %v0, %s1, %s0
; CHECK-NEXT:    #NO_APP
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = tail call fast <256 x double> @llvm.ve.vl.vld2dnc.vssl(i64 %1, i8* %2, i32 256)
  %5 = tail call fast <256 x double> @llvm.ve.vl.vld2dnc.vssvl(i64 %1, i8* %0, <256 x double> %4, i32 256)
  tail call void asm sideeffect "vst ${0:v}, $2, $1", "v,r,r"(<256 x double> %5, i8* %0, i64 %1)
  ret void
}

; Function Attrs: nounwind readonly
declare <256 x double> @llvm.ve.vl.vld2dnc.vssvl(i64, i8*, <256 x double>, i32)

; Function Attrs: nounwind
define void @vld2dnc_vssl_imm(i8* %0) {
; CHECK-LABEL: vld2dnc_vssl_imm:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vld2d.nc %v0, 8, %s0
; CHECK-NEXT:    #APP
; CHECK-NEXT:    vst %v0, 8, %s0
; CHECK-NEXT:    #NO_APP
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call fast <256 x double> @llvm.ve.vl.vld2dnc.vssl(i64 8, i8* %0, i32 256)
  tail call void asm sideeffect "vst ${0:v}, 8, $1", "v,r"(<256 x double> %2, i8* %0)
  ret void
}

; Function Attrs: nounwind
define void @vld2dnc_vssvl_imm(i8* %0, i8* %1) {
; CHECK-LABEL: vld2dnc_vssvl_imm:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s2, 256
; CHECK-NEXT:    lvl %s2
; CHECK-NEXT:    vld2d.nc %v0, 8, %s1
; CHECK-NEXT:    vld2d.nc %v0, 8, %s0
; CHECK-NEXT:    #APP
; CHECK-NEXT:    vst %v0, 8, %s0
; CHECK-NEXT:    #NO_APP
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.vld2dnc.vssl(i64 8, i8* %1, i32 256)
  %4 = tail call fast <256 x double> @llvm.ve.vl.vld2dnc.vssvl(i64 8, i8* %0, <256 x double> %3, i32 256)
  tail call void asm sideeffect "vst ${0:v}, 8, $1", "v,r"(<256 x double> %4, i8* %0)
  ret void
}

; Function Attrs: nounwind
define void @vldu2d_vssl(i8* %0, i64 %1) {
; CHECK-LABEL: vldu2d_vssl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s2, 256
; CHECK-NEXT:    lvl %s2
; CHECK-NEXT:    vldu2d %v0, %s1, %s0
; CHECK-NEXT:    #APP
; CHECK-NEXT:    vst %v0, %s1, %s0
; CHECK-NEXT:    #NO_APP
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.vldu2d.vssl(i64 %1, i8* %0, i32 256)
  tail call void asm sideeffect "vst ${0:v}, $2, $1", "v,r,r"(<256 x double> %3, i8* %0, i64 %1)
  ret void
}

; Function Attrs: nounwind readonly
declare <256 x double> @llvm.ve.vl.vldu2d.vssl(i64, i8*, i32)

; Function Attrs: nounwind
define void @vldu2d_vssvl(i8* %0, i64 %1, i8* %2) {
; CHECK-LABEL: vldu2d_vssvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s3, 256
; CHECK-NEXT:    lvl %s3
; CHECK-NEXT:    vldu2d %v0, %s1, %s2
; CHECK-NEXT:    vldu2d %v0, %s1, %s0
; CHECK-NEXT:    #APP
; CHECK-NEXT:    vst %v0, %s1, %s0
; CHECK-NEXT:    #NO_APP
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = tail call fast <256 x double> @llvm.ve.vl.vldu2d.vssl(i64 %1, i8* %2, i32 256)
  %5 = tail call fast <256 x double> @llvm.ve.vl.vldu2d.vssvl(i64 %1, i8* %0, <256 x double> %4, i32 256)
  tail call void asm sideeffect "vst ${0:v}, $2, $1", "v,r,r"(<256 x double> %5, i8* %0, i64 %1)
  ret void
}

; Function Attrs: nounwind readonly
declare <256 x double> @llvm.ve.vl.vldu2d.vssvl(i64, i8*, <256 x double>, i32)

; Function Attrs: nounwind
define void @vldu2d_vssl_imm(i8* %0) {
; CHECK-LABEL: vldu2d_vssl_imm:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vldu2d %v0, 8, %s0
; CHECK-NEXT:    #APP
; CHECK-NEXT:    vst %v0, 8, %s0
; CHECK-NEXT:    #NO_APP
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call fast <256 x double> @llvm.ve.vl.vldu2d.vssl(i64 8, i8* %0, i32 256)
  tail call void asm sideeffect "vst ${0:v}, 8, $1", "v,r"(<256 x double> %2, i8* %0)
  ret void
}

; Function Attrs: nounwind
define void @vldu2d_vssvl_imm(i8* %0, i8* %1) {
; CHECK-LABEL: vldu2d_vssvl_imm:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s2, 256
; CHECK-NEXT:    lvl %s2
; CHECK-NEXT:    vldu2d %v0, 8, %s1
; CHECK-NEXT:    vldu2d %v0, 8, %s0
; CHECK-NEXT:    #APP
; CHECK-NEXT:    vst %v0, 8, %s0
; CHECK-NEXT:    #NO_APP
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.vldu2d.vssl(i64 8, i8* %1, i32 256)
  %4 = tail call fast <256 x double> @llvm.ve.vl.vldu2d.vssvl(i64 8, i8* %0, <256 x double> %3, i32 256)
  tail call void asm sideeffect "vst ${0:v}, 8, $1", "v,r"(<256 x double> %4, i8* %0)
  ret void
}

; Function Attrs: nounwind
define void @vldu2dnc_vssl(i8* %0, i64 %1) {
; CHECK-LABEL: vldu2dnc_vssl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s2, 256
; CHECK-NEXT:    lvl %s2
; CHECK-NEXT:    vldu2d.nc %v0, %s1, %s0
; CHECK-NEXT:    #APP
; CHECK-NEXT:    vst %v0, %s1, %s0
; CHECK-NEXT:    #NO_APP
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.vldu2dnc.vssl(i64 %1, i8* %0, i32 256)
  tail call void asm sideeffect "vst ${0:v}, $2, $1", "v,r,r"(<256 x double> %3, i8* %0, i64 %1)
  ret void
}

; Function Attrs: nounwind readonly
declare <256 x double> @llvm.ve.vl.vldu2dnc.vssl(i64, i8*, i32)

; Function Attrs: nounwind
define void @vldu2dnc_vssvl(i8* %0, i64 %1, i8* %2) {
; CHECK-LABEL: vldu2dnc_vssvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s3, 256
; CHECK-NEXT:    lvl %s3
; CHECK-NEXT:    vldu2d.nc %v0, %s1, %s2
; CHECK-NEXT:    vldu2d.nc %v0, %s1, %s0
; CHECK-NEXT:    #APP
; CHECK-NEXT:    vst %v0, %s1, %s0
; CHECK-NEXT:    #NO_APP
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = tail call fast <256 x double> @llvm.ve.vl.vldu2dnc.vssl(i64 %1, i8* %2, i32 256)
  %5 = tail call fast <256 x double> @llvm.ve.vl.vldu2dnc.vssvl(i64 %1, i8* %0, <256 x double> %4, i32 256)
  tail call void asm sideeffect "vst ${0:v}, $2, $1", "v,r,r"(<256 x double> %5, i8* %0, i64 %1)
  ret void
}

; Function Attrs: nounwind readonly
declare <256 x double> @llvm.ve.vl.vldu2dnc.vssvl(i64, i8*, <256 x double>, i32)

; Function Attrs: nounwind
define void @vldu2dnc_vssl_imm(i8* %0) {
; CHECK-LABEL: vldu2dnc_vssl_imm:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vldu2d.nc %v0, 8, %s0
; CHECK-NEXT:    #APP
; CHECK-NEXT:    vst %v0, 8, %s0
; CHECK-NEXT:    #NO_APP
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call fast <256 x double> @llvm.ve.vl.vldu2dnc.vssl(i64 8, i8* %0, i32 256)
  tail call void asm sideeffect "vst ${0:v}, 8, $1", "v,r"(<256 x double> %2, i8* %0)
  ret void
}

; Function Attrs: nounwind
define void @vldu2dnc_vssvl_imm(i8* %0, i8* %1) {
; CHECK-LABEL: vldu2dnc_vssvl_imm:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s2, 256
; CHECK-NEXT:    lvl %s2
; CHECK-NEXT:    vldu2d.nc %v0, 8, %s1
; CHECK-NEXT:    vldu2d.nc %v0, 8, %s0
; CHECK-NEXT:    #APP
; CHECK-NEXT:    vst %v0, 8, %s0
; CHECK-NEXT:    #NO_APP
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.vldu2dnc.vssl(i64 8, i8* %1, i32 256)
  %4 = tail call fast <256 x double> @llvm.ve.vl.vldu2dnc.vssvl(i64 8, i8* %0, <256 x double> %3, i32 256)
  tail call void asm sideeffect "vst ${0:v}, 8, $1", "v,r"(<256 x double> %4, i8* %0)
  ret void
}

; Function Attrs: nounwind
define void @vldl2dsx_vssl(i8* %0, i64 %1) {
; CHECK-LABEL: vldl2dsx_vssl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s2, 256
; CHECK-NEXT:    lvl %s2
; CHECK-NEXT:    vldl2d.sx %v0, %s1, %s0
; CHECK-NEXT:    #APP
; CHECK-NEXT:    vst %v0, %s1, %s0
; CHECK-NEXT:    #NO_APP
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.vldl2dsx.vssl(i64 %1, i8* %0, i32 256)
  tail call void asm sideeffect "vst ${0:v}, $2, $1", "v,r,r"(<256 x double> %3, i8* %0, i64 %1)
  ret void
}

; Function Attrs: nounwind readonly
declare <256 x double> @llvm.ve.vl.vldl2dsx.vssl(i64, i8*, i32)

; Function Attrs: nounwind
define void @vldl2dsx_vssvl(i8* %0, i64 %1, i8* %2) {
; CHECK-LABEL: vldl2dsx_vssvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s3, 256
; CHECK-NEXT:    lvl %s3
; CHECK-NEXT:    vldl2d.sx %v0, %s1, %s2
; CHECK-NEXT:    vldl2d.sx %v0, %s1, %s0
; CHECK-NEXT:    #APP
; CHECK-NEXT:    vst %v0, %s1, %s0
; CHECK-NEXT:    #NO_APP
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = tail call fast <256 x double> @llvm.ve.vl.vldl2dsx.vssl(i64 %1, i8* %2, i32 256)
  %5 = tail call fast <256 x double> @llvm.ve.vl.vldl2dsx.vssvl(i64 %1, i8* %0, <256 x double> %4, i32 256)
  tail call void asm sideeffect "vst ${0:v}, $2, $1", "v,r,r"(<256 x double> %5, i8* %0, i64 %1)
  ret void
}

; Function Attrs: nounwind readonly
declare <256 x double> @llvm.ve.vl.vldl2dsx.vssvl(i64, i8*, <256 x double>, i32)

; Function Attrs: nounwind
define void @vldl2dsx_vssl_imm(i8* %0) {
; CHECK-LABEL: vldl2dsx_vssl_imm:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vldl2d.sx %v0, 8, %s0
; CHECK-NEXT:    #APP
; CHECK-NEXT:    vst %v0, 8, %s0
; CHECK-NEXT:    #NO_APP
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call fast <256 x double> @llvm.ve.vl.vldl2dsx.vssl(i64 8, i8* %0, i32 256)
  tail call void asm sideeffect "vst ${0:v}, 8, $1", "v,r"(<256 x double> %2, i8* %0)
  ret void
}

; Function Attrs: nounwind
define void @vldl2dsx_vssvl_imm(i8* %0, i8* %1) {
; CHECK-LABEL: vldl2dsx_vssvl_imm:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s2, 256
; CHECK-NEXT:    lvl %s2
; CHECK-NEXT:    vldl2d.sx %v0, 8, %s1
; CHECK-NEXT:    vldl2d.sx %v0, 8, %s0
; CHECK-NEXT:    #APP
; CHECK-NEXT:    vst %v0, 8, %s0
; CHECK-NEXT:    #NO_APP
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.vldl2dsx.vssl(i64 8, i8* %1, i32 256)
  %4 = tail call fast <256 x double> @llvm.ve.vl.vldl2dsx.vssvl(i64 8, i8* %0, <256 x double> %3, i32 256)
  tail call void asm sideeffect "vst ${0:v}, 8, $1", "v,r"(<256 x double> %4, i8* %0)
  ret void
}

; Function Attrs: nounwind
define void @vldl2dsxnc_vssl(i8* %0, i64 %1) {
; CHECK-LABEL: vldl2dsxnc_vssl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s2, 256
; CHECK-NEXT:    lvl %s2
; CHECK-NEXT:    vldl2d.sx.nc %v0, %s1, %s0
; CHECK-NEXT:    #APP
; CHECK-NEXT:    vst %v0, %s1, %s0
; CHECK-NEXT:    #NO_APP
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.vldl2dsxnc.vssl(i64 %1, i8* %0, i32 256)
  tail call void asm sideeffect "vst ${0:v}, $2, $1", "v,r,r"(<256 x double> %3, i8* %0, i64 %1)
  ret void
}

; Function Attrs: nounwind readonly
declare <256 x double> @llvm.ve.vl.vldl2dsxnc.vssl(i64, i8*, i32)

; Function Attrs: nounwind
define void @vldl2dsxnc_vssvl(i8* %0, i64 %1, i8* %2) {
; CHECK-LABEL: vldl2dsxnc_vssvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s3, 256
; CHECK-NEXT:    lvl %s3
; CHECK-NEXT:    vldl2d.sx.nc %v0, %s1, %s2
; CHECK-NEXT:    vldl2d.sx.nc %v0, %s1, %s0
; CHECK-NEXT:    #APP
; CHECK-NEXT:    vst %v0, %s1, %s0
; CHECK-NEXT:    #NO_APP
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = tail call fast <256 x double> @llvm.ve.vl.vldl2dsxnc.vssl(i64 %1, i8* %2, i32 256)
  %5 = tail call fast <256 x double> @llvm.ve.vl.vldl2dsxnc.vssvl(i64 %1, i8* %0, <256 x double> %4, i32 256)
  tail call void asm sideeffect "vst ${0:v}, $2, $1", "v,r,r"(<256 x double> %5, i8* %0, i64 %1)
  ret void
}

; Function Attrs: nounwind readonly
declare <256 x double> @llvm.ve.vl.vldl2dsxnc.vssvl(i64, i8*, <256 x double>, i32)

; Function Attrs: nounwind
define void @vldl2dsxnc_vssl_imm(i8* %0) {
; CHECK-LABEL: vldl2dsxnc_vssl_imm:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vldl2d.sx.nc %v0, 8, %s0
; CHECK-NEXT:    #APP
; CHECK-NEXT:    vst %v0, 8, %s0
; CHECK-NEXT:    #NO_APP
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call fast <256 x double> @llvm.ve.vl.vldl2dsxnc.vssl(i64 8, i8* %0, i32 256)
  tail call void asm sideeffect "vst ${0:v}, 8, $1", "v,r"(<256 x double> %2, i8* %0)
  ret void
}

; Function Attrs: nounwind
define void @vldl2dsxnc_vssvl_imm(i8* %0, i8* %1) {
; CHECK-LABEL: vldl2dsxnc_vssvl_imm:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s2, 256
; CHECK-NEXT:    lvl %s2
; CHECK-NEXT:    vldl2d.sx.nc %v0, 8, %s1
; CHECK-NEXT:    vldl2d.sx.nc %v0, 8, %s0
; CHECK-NEXT:    #APP
; CHECK-NEXT:    vst %v0, 8, %s0
; CHECK-NEXT:    #NO_APP
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.vldl2dsxnc.vssl(i64 8, i8* %1, i32 256)
  %4 = tail call fast <256 x double> @llvm.ve.vl.vldl2dsxnc.vssvl(i64 8, i8* %0, <256 x double> %3, i32 256)
  tail call void asm sideeffect "vst ${0:v}, 8, $1", "v,r"(<256 x double> %4, i8* %0)
  ret void
}

; Function Attrs: nounwind
define void @vldl2dzx_vssl(i8* %0, i64 %1) {
; CHECK-LABEL: vldl2dzx_vssl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s2, 256
; CHECK-NEXT:    lvl %s2
; CHECK-NEXT:    vldl2d.zx %v0, %s1, %s0
; CHECK-NEXT:    #APP
; CHECK-NEXT:    vst %v0, %s1, %s0
; CHECK-NEXT:    #NO_APP
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.vldl2dzx.vssl(i64 %1, i8* %0, i32 256)
  tail call void asm sideeffect "vst ${0:v}, $2, $1", "v,r,r"(<256 x double> %3, i8* %0, i64 %1)
  ret void
}

; Function Attrs: nounwind readonly
declare <256 x double> @llvm.ve.vl.vldl2dzx.vssl(i64, i8*, i32)

; Function Attrs: nounwind
define void @vldl2dzx_vssvl(i8* %0, i64 %1, i8* %2) {
; CHECK-LABEL: vldl2dzx_vssvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s3, 256
; CHECK-NEXT:    lvl %s3
; CHECK-NEXT:    vldl2d.zx %v0, %s1, %s2
; CHECK-NEXT:    vldl2d.zx %v0, %s1, %s0
; CHECK-NEXT:    #APP
; CHECK-NEXT:    vst %v0, %s1, %s0
; CHECK-NEXT:    #NO_APP
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = tail call fast <256 x double> @llvm.ve.vl.vldl2dzx.vssl(i64 %1, i8* %2, i32 256)
  %5 = tail call fast <256 x double> @llvm.ve.vl.vldl2dzx.vssvl(i64 %1, i8* %0, <256 x double> %4, i32 256)
  tail call void asm sideeffect "vst ${0:v}, $2, $1", "v,r,r"(<256 x double> %5, i8* %0, i64 %1)
  ret void
}

; Function Attrs: nounwind readonly
declare <256 x double> @llvm.ve.vl.vldl2dzx.vssvl(i64, i8*, <256 x double>, i32)

; Function Attrs: nounwind
define void @vldl2dzx_vssl_imm(i8* %0) {
; CHECK-LABEL: vldl2dzx_vssl_imm:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vldl2d.zx %v0, 8, %s0
; CHECK-NEXT:    #APP
; CHECK-NEXT:    vst %v0, 8, %s0
; CHECK-NEXT:    #NO_APP
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call fast <256 x double> @llvm.ve.vl.vldl2dzx.vssl(i64 8, i8* %0, i32 256)
  tail call void asm sideeffect "vst ${0:v}, 8, $1", "v,r"(<256 x double> %2, i8* %0)
  ret void
}

; Function Attrs: nounwind
define void @vldl2dzx_vssvl_imm(i8* %0, i8* %1) {
; CHECK-LABEL: vldl2dzx_vssvl_imm:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s2, 256
; CHECK-NEXT:    lvl %s2
; CHECK-NEXT:    vldl2d.zx %v0, 8, %s1
; CHECK-NEXT:    vldl2d.zx %v0, 8, %s0
; CHECK-NEXT:    #APP
; CHECK-NEXT:    vst %v0, 8, %s0
; CHECK-NEXT:    #NO_APP
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.vldl2dzx.vssl(i64 8, i8* %1, i32 256)
  %4 = tail call fast <256 x double> @llvm.ve.vl.vldl2dzx.vssvl(i64 8, i8* %0, <256 x double> %3, i32 256)
  tail call void asm sideeffect "vst ${0:v}, 8, $1", "v,r"(<256 x double> %4, i8* %0)
  ret void
}

; Function Attrs: nounwind
define void @vldl2dzxnc_vssl(i8* %0, i64 %1) {
; CHECK-LABEL: vldl2dzxnc_vssl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s2, 256
; CHECK-NEXT:    lvl %s2
; CHECK-NEXT:    vldl2d.zx.nc %v0, %s1, %s0
; CHECK-NEXT:    #APP
; CHECK-NEXT:    vst %v0, %s1, %s0
; CHECK-NEXT:    #NO_APP
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.vldl2dzxnc.vssl(i64 %1, i8* %0, i32 256)
  tail call void asm sideeffect "vst ${0:v}, $2, $1", "v,r,r"(<256 x double> %3, i8* %0, i64 %1)
  ret void
}

; Function Attrs: nounwind readonly
declare <256 x double> @llvm.ve.vl.vldl2dzxnc.vssl(i64, i8*, i32)

; Function Attrs: nounwind
define void @vldl2dzxnc_vssvl(i8* %0, i64 %1, i8* %2) {
; CHECK-LABEL: vldl2dzxnc_vssvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s3, 256
; CHECK-NEXT:    lvl %s3
; CHECK-NEXT:    vldl2d.zx.nc %v0, %s1, %s2
; CHECK-NEXT:    vldl2d.zx.nc %v0, %s1, %s0
; CHECK-NEXT:    #APP
; CHECK-NEXT:    vst %v0, %s1, %s0
; CHECK-NEXT:    #NO_APP
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = tail call fast <256 x double> @llvm.ve.vl.vldl2dzxnc.vssl(i64 %1, i8* %2, i32 256)
  %5 = tail call fast <256 x double> @llvm.ve.vl.vldl2dzxnc.vssvl(i64 %1, i8* %0, <256 x double> %4, i32 256)
  tail call void asm sideeffect "vst ${0:v}, $2, $1", "v,r,r"(<256 x double> %5, i8* %0, i64 %1)
  ret void
}

; Function Attrs: nounwind readonly
declare <256 x double> @llvm.ve.vl.vldl2dzxnc.vssvl(i64, i8*, <256 x double>, i32)

; Function Attrs: nounwind
define void @vldl2dzxnc_vssl_imm(i8* %0) {
; CHECK-LABEL: vldl2dzxnc_vssl_imm:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vldl2d.zx.nc %v0, 8, %s0
; CHECK-NEXT:    #APP
; CHECK-NEXT:    vst %v0, 8, %s0
; CHECK-NEXT:    #NO_APP
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call fast <256 x double> @llvm.ve.vl.vldl2dzxnc.vssl(i64 8, i8* %0, i32 256)
  tail call void asm sideeffect "vst ${0:v}, 8, $1", "v,r"(<256 x double> %2, i8* %0)
  ret void
}

; Function Attrs: nounwind
define void @vldl2dzxnc_vssvl_imm(i8* %0, i8* %1) {
; CHECK-LABEL: vldl2dzxnc_vssvl_imm:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s2, 256
; CHECK-NEXT:    lvl %s2
; CHECK-NEXT:    vldl2d.zx.nc %v0, 8, %s1
; CHECK-NEXT:    vldl2d.zx.nc %v0, 8, %s0
; CHECK-NEXT:    #APP
; CHECK-NEXT:    vst %v0, 8, %s0
; CHECK-NEXT:    #NO_APP
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.vldl2dzxnc.vssl(i64 8, i8* %1, i32 256)
  %4 = tail call fast <256 x double> @llvm.ve.vl.vldl2dzxnc.vssvl(i64 8, i8* %0, <256 x double> %3, i32 256)
  tail call void asm sideeffect "vst ${0:v}, 8, $1", "v,r"(<256 x double> %4, i8* %0)
  ret void
}
