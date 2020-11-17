; RUN: llc < %s -mtriple=ve -mattr=+vpu | FileCheck %s

;;; Test vector broadcast intrinsic instructions
;;;
;;; Note:
;;;   We test VLD*rrl, VLD*irl, VLD*rrl_v, and VLD*irl_v instructions.

; Function Attrs: nounwind
define void @vbrdd_vsl(double %0, i8* %1) {
; CHECK-LABEL: vbrdd_vsl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s2, 256
; CHECK-NEXT:    lvl %s2
; CHECK-NEXT:    vbrd %v0, %s0
; CHECK-NEXT:    #APP
; CHECK-NEXT:    vst %v0, 8, %s1
; CHECK-NEXT:    #NO_APP
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.vbrdd.vsl(double %0, i32 256)
  tail call void asm sideeffect "vst ${0:v}, 8, $1", "v,r"(<256 x double> %3, i8* %1)
  ret void
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vbrdd.vsl(double, i32)

; Function Attrs: nounwind
define void @vbrdd_vsvl(double %0, i8* %1) {
; CHECK-LABEL: vbrdd_vsvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s2, 256
; CHECK-NEXT:    lvl %s2
; CHECK-NEXT:    vld %v0, 8, %s1
; CHECK-NEXT:    vbrd %v0, %s0
; CHECK-NEXT:    vst %v0, 8, %s1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.vld.vssl(i64 8, i8* %1, i32 256)
  %4 = tail call fast <256 x double> @llvm.ve.vl.vbrdd.vsvl(double %0, <256 x double> %3, i32 256)
  tail call void @llvm.ve.vl.vst.vssl(<256 x double> %4, i64 8, i8* %1, i32 256)
  ret void
}

; Function Attrs: nounwind readonly
declare <256 x double> @llvm.ve.vl.vld.vssl(i64, i8*, i32)

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vbrdd.vsvl(double, <256 x double>, i32)

; Function Attrs: nounwind writeonly
declare void @llvm.ve.vl.vst.vssl(<256 x double>, i64, i8*, i32)

; Function Attrs: nounwind
define void @vbrdd_vsmvl(double %0, i8* %1) {
; CHECK-LABEL: vbrdd_vsmvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s2, 256
; CHECK-NEXT:    lvl %s2
; CHECK-NEXT:    vld %v0, 8, %s1
; CHECK-NEXT:    lea.sl %s3, 1138753536
; CHECK-NEXT:    fcmp.d %s4, %s0, %s3
; CHECK-NEXT:    fsub.d %s3, %s0, %s3
; CHECK-NEXT:    cvt.l.d.rz %s3, %s3
; CHECK-NEXT:    xor %s3, %s3, (1)1
; CHECK-NEXT:    cvt.l.d.rz %s5, %s0
; CHECK-NEXT:    cmov.d.lt %s3, %s5, %s4
; CHECK-NEXT:    lvm %vm1, 3, %s3
; CHECK-NEXT:    vbrd %v0, %s0, %vm1
; CHECK-NEXT:    vst %v0, 8, %s1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.vld.vssl(i64 8, i8* %1, i32 256)
  %4 = fptoui double %0 to i64
  %5 = tail call <256 x i1> @llvm.ve.vl.lvm.mmss(<256 x i1> undef, i64 3, i64 %4)
  %6 = tail call fast <256 x double> @llvm.ve.vl.vbrdd.vsmvl(double %0, <256 x i1> %5, <256 x double> %3, i32 256)
  tail call void @llvm.ve.vl.vst.vssl(<256 x double> %6, i64 8, i8* %1, i32 256)
  ret void
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.lvm.mmss(<256 x i1>, i64, i64)

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vbrdd.vsmvl(double, <256 x i1>, <256 x double>, i32)

; Function Attrs: nounwind
define void @vbrdl_vsl(i64 %0, i8* %1) {
; CHECK-LABEL: vbrdl_vsl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s2, 256
; CHECK-NEXT:    lvl %s2
; CHECK-NEXT:    vbrd %v0, %s0
; CHECK-NEXT:    #APP
; CHECK-NEXT:    vst %v0, 8, %s1
; CHECK-NEXT:    #NO_APP
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.vbrdl.vsl(i64 %0, i32 256)
  tail call void asm sideeffect "vst ${0:v}, 8, $1", "v,r"(<256 x double> %3, i8* %1)
  ret void
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vbrdl.vsl(i64, i32)

; Function Attrs: nounwind
define void @vbrdl_vsvl(i64 %0, i8* %1) {
; CHECK-LABEL: vbrdl_vsvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s2, 256
; CHECK-NEXT:    lvl %s2
; CHECK-NEXT:    vld %v0, 8, %s1
; CHECK-NEXT:    vbrd %v0, %s0
; CHECK-NEXT:    vst %v0, 8, %s1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.vld.vssl(i64 8, i8* %1, i32 256)
  %4 = tail call fast <256 x double> @llvm.ve.vl.vbrdl.vsvl(i64 %0, <256 x double> %3, i32 256)
  tail call void @llvm.ve.vl.vst.vssl(<256 x double> %4, i64 8, i8* %1, i32 256)
  ret void
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vbrdl.vsvl(i64, <256 x double>, i32)

; Function Attrs: nounwind
define void @vbrdl_vsmvl(i64 %0, i8* %1) {
; CHECK-LABEL: vbrdl_vsmvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s2, 256
; CHECK-NEXT:    lvl %s2
; CHECK-NEXT:    vld %v0, 8, %s1
; CHECK-NEXT:    lvm %vm1, 3, %s0
; CHECK-NEXT:    vbrd %v0, %s0, %vm1
; CHECK-NEXT:    vst %v0, 8, %s1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.vld.vssl(i64 8, i8* %1, i32 256)
  %4 = tail call <256 x i1> @llvm.ve.vl.lvm.mmss(<256 x i1> undef, i64 3, i64 %0)
  %5 = tail call fast <256 x double> @llvm.ve.vl.vbrdl.vsmvl(i64 %0, <256 x i1> %4, <256 x double> %3, i32 256)
  tail call void @llvm.ve.vl.vst.vssl(<256 x double> %5, i64 8, i8* %1, i32 256)
  ret void
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vbrdl.vsmvl(i64, <256 x i1>, <256 x double>, i32)

; Function Attrs: nounwind
define void @vbrdl_imm_vsl(i64 %0, i8* %1) {
; CHECK-LABEL: vbrdl_imm_vsl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vbrd %v0, 31
; CHECK-NEXT:    #APP
; CHECK-NEXT:    vst %v0, 8, %s1
; CHECK-NEXT:    #NO_APP
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.vbrdl.vsl(i64 31, i32 256)
  tail call void asm sideeffect "vst ${0:v}, 8, $1", "v,r"(<256 x double> %3, i8* %1)
  ret void
}

; Function Attrs: nounwind
define void @vbrdl_imm_vsvl(i64 %0, i8* %1) {
; CHECK-LABEL: vbrdl_imm_vsvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vld %v0, 8, %s1
; CHECK-NEXT:    vbrd %v0, 31
; CHECK-NEXT:    vst %v0, 8, %s1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.vld.vssl(i64 8, i8* %1, i32 256)
  %4 = tail call fast <256 x double> @llvm.ve.vl.vbrdl.vsvl(i64 31, <256 x double> %3, i32 256)
  tail call void @llvm.ve.vl.vst.vssl(<256 x double> %4, i64 8, i8* %1, i32 256)
  ret void
}

; Function Attrs: nounwind
define void @vbrdl_imm_vsmvl(i64 %0, i8* %1) {
; CHECK-LABEL: vbrdl_imm_vsmvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s2, 256
; CHECK-NEXT:    lvl %s2
; CHECK-NEXT:    vld %v0, 8, %s1
; CHECK-NEXT:    lvm %vm1, 3, %s0
; CHECK-NEXT:    vbrd %v0, 31, %vm1
; CHECK-NEXT:    vst %v0, 8, %s1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.vld.vssl(i64 8, i8* %1, i32 256)
  %4 = tail call <256 x i1> @llvm.ve.vl.lvm.mmss(<256 x i1> undef, i64 3, i64 %0)
  %5 = tail call fast <256 x double> @llvm.ve.vl.vbrdl.vsmvl(i64 31, <256 x i1> %4, <256 x double> %3, i32 256)
  tail call void @llvm.ve.vl.vst.vssl(<256 x double> %5, i64 8, i8* %1, i32 256)
  ret void
}

; Function Attrs: nounwind
define void @vbrds_vsl(float %0, i8* %1) {
; CHECK-LABEL: vbrds_vsl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s2, 256
; CHECK-NEXT:    lvl %s2
; CHECK-NEXT:    vbrdu %v0, %s0
; CHECK-NEXT:    #APP
; CHECK-NEXT:    vst %v0, 8, %s1
; CHECK-NEXT:    #NO_APP
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.vbrds.vsl(float %0, i32 256)
  tail call void asm sideeffect "vst ${0:v}, 8, $1", "v,r"(<256 x double> %3, i8* %1)
  ret void
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vbrds.vsl(float, i32)

; Function Attrs: nounwind
define void @vbrds_vsvl(float %0, i8* %1) {
; CHECK-LABEL: vbrds_vsvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s2, 256
; CHECK-NEXT:    lvl %s2
; CHECK-NEXT:    vld %v0, 8, %s1
; CHECK-NEXT:    vbrdu %v0, %s0
; CHECK-NEXT:    vst %v0, 8, %s1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.vld.vssl(i64 8, i8* %1, i32 256)
  %4 = tail call fast <256 x double> @llvm.ve.vl.vbrds.vsvl(float %0, <256 x double> %3, i32 256)
  tail call void @llvm.ve.vl.vst.vssl(<256 x double> %4, i64 8, i8* %1, i32 256)
  ret void
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vbrds.vsvl(float, <256 x double>, i32)

; Function Attrs: nounwind
define void @vbrds_vsmvl(float %0, i8* %1) {
; CHECK-LABEL: vbrds_vsmvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s2, 256
; CHECK-NEXT:    lvl %s2
; CHECK-NEXT:    vld %v0, 8, %s1
; CHECK-NEXT:    lea.sl %s3, 1593835520
; CHECK-NEXT:    fcmp.s %s4, %s0, %s3
; CHECK-NEXT:    fsub.s %s3, %s0, %s3
; CHECK-NEXT:    cvt.d.s %s3, %s3
; CHECK-NEXT:    cvt.l.d.rz %s3, %s3
; CHECK-NEXT:    xor %s3, %s3, (1)1
; CHECK-NEXT:    cvt.d.s %s5, %s0
; CHECK-NEXT:    cvt.l.d.rz %s5, %s5
; CHECK-NEXT:    cmov.s.lt %s3, %s5, %s4
; CHECK-NEXT:    lvm %vm1, 3, %s3
; CHECK-NEXT:    vbrdu %v0, %s0, %vm1
; CHECK-NEXT:    vst %v0, 8, %s1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.vld.vssl(i64 8, i8* %1, i32 256)
  %4 = fptoui float %0 to i64
  %5 = tail call <256 x i1> @llvm.ve.vl.lvm.mmss(<256 x i1> undef, i64 3, i64 %4)
  %6 = tail call fast <256 x double> @llvm.ve.vl.vbrds.vsmvl(float %0, <256 x i1> %5, <256 x double> %3, i32 256)
  tail call void @llvm.ve.vl.vst.vssl(<256 x double> %6, i64 8, i8* %1, i32 256)
  ret void
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vbrds.vsmvl(float, <256 x i1>, <256 x double>, i32)

; Function Attrs: nounwind
define void @vbrdw_vsl(i32 signext %0, i8* %1) {
; CHECK-LABEL: vbrdw_vsl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    adds.w.sx %s0, %s0, (0)1
; CHECK-NEXT:    lea %s2, 256
; CHECK-NEXT:    lvl %s2
; CHECK-NEXT:    vbrdl %v0, %s0
; CHECK-NEXT:    #APP
; CHECK-NEXT:    vst %v0, 8, %s1
; CHECK-NEXT:    #NO_APP
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.vbrdw.vsl(i32 %0, i32 256)
  tail call void asm sideeffect "vst ${0:v}, 8, $1", "v,r"(<256 x double> %3, i8* %1)
  ret void
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vbrdw.vsl(i32, i32)

; Function Attrs: nounwind
define void @vbrdw_vsvl(i32 signext %0, i8* %1) {
; CHECK-LABEL: vbrdw_vsvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s2, 256
; CHECK-NEXT:    lvl %s2
; CHECK-NEXT:    vld %v0, 8, %s1
; CHECK-NEXT:    adds.w.sx %s0, %s0, (0)1
; CHECK-NEXT:    vbrdl %v0, %s0
; CHECK-NEXT:    vst %v0, 8, %s1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.vld.vssl(i64 8, i8* %1, i32 256)
  %4 = tail call fast <256 x double> @llvm.ve.vl.vbrdw.vsvl(i32 %0, <256 x double> %3, i32 256)
  tail call void @llvm.ve.vl.vst.vssl(<256 x double> %4, i64 8, i8* %1, i32 256)
  ret void
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vbrdw.vsvl(i32, <256 x double>, i32)

; Function Attrs: nounwind
define void @vbrdw_vsmvl(i32 signext %0, i8* %1) {
; CHECK-LABEL: vbrdw_vsmvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s2, 256
; CHECK-NEXT:    lvl %s2
; CHECK-NEXT:    vld %v0, 8, %s1
; CHECK-NEXT:    adds.w.sx %s3, %s0, (0)1
; CHECK-NEXT:    lvm %vm1, 3, %s0
; CHECK-NEXT:    vbrdl %v0, %s3, %vm1
; CHECK-NEXT:    vst %v0, 8, %s1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.vld.vssl(i64 8, i8* %1, i32 256)
  %4 = sext i32 %0 to i64
  %5 = tail call <256 x i1> @llvm.ve.vl.lvm.mmss(<256 x i1> undef, i64 3, i64 %4)
  %6 = tail call fast <256 x double> @llvm.ve.vl.vbrdw.vsmvl(i32 %0, <256 x i1> %5, <256 x double> %3, i32 256)
  tail call void @llvm.ve.vl.vst.vssl(<256 x double> %6, i64 8, i8* %1, i32 256)
  ret void
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.vbrdw.vsmvl(i32, <256 x i1>, <256 x double>, i32)

; Function Attrs: nounwind
define void @vbrdw_imm_vsl(i32 signext %0, i8* %1) {
; CHECK-LABEL: vbrdw_imm_vsl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vbrdl %v0, 31
; CHECK-NEXT:    #APP
; CHECK-NEXT:    vst %v0, 8, %s1
; CHECK-NEXT:    #NO_APP
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.vbrdw.vsl(i32 31, i32 256)
  tail call void asm sideeffect "vst ${0:v}, 8, $1", "v,r"(<256 x double> %3, i8* %1)
  ret void
}

; Function Attrs: nounwind
define void @vbrdw_imm_vsvl(i32 signext %0, i8* %1) {
; CHECK-LABEL: vbrdw_imm_vsvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vld %v0, 8, %s1
; CHECK-NEXT:    vbrdl %v0, 31
; CHECK-NEXT:    vst %v0, 8, %s1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.vld.vssl(i64 8, i8* %1, i32 256)
  %4 = tail call fast <256 x double> @llvm.ve.vl.vbrdw.vsvl(i32 31, <256 x double> %3, i32 256)
  tail call void @llvm.ve.vl.vst.vssl(<256 x double> %4, i64 8, i8* %1, i32 256)
  ret void
}

; Function Attrs: nounwind
define void @vbrdw_imm_vsmvl(i32 signext %0, i8* %1) {
; CHECK-LABEL: vbrdw_imm_vsmvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s2, 256
; CHECK-NEXT:    lvl %s2
; CHECK-NEXT:    vld %v0, 8, %s1
; CHECK-NEXT:    lvm %vm1, 3, %s0
; CHECK-NEXT:    vbrdl %v0, 31, %vm1
; CHECK-NEXT:    vst %v0, 8, %s1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.vld.vssl(i64 8, i8* %1, i32 256)
  %4 = sext i32 %0 to i64
  %5 = tail call <256 x i1> @llvm.ve.vl.lvm.mmss(<256 x i1> undef, i64 3, i64 %4)
  %6 = tail call fast <256 x double> @llvm.ve.vl.vbrdw.vsmvl(i32 31, <256 x i1> %5, <256 x double> %3, i32 256)
  tail call void @llvm.ve.vl.vst.vssl(<256 x double> %6, i64 8, i8* %1, i32 256)
  ret void
}

; Function Attrs: nounwind
define void @pvbrd_vsl(i64 %0, i8* %1) {
; CHECK-LABEL: pvbrd_vsl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s2, 256
; CHECK-NEXT:    lvl %s2
; CHECK-NEXT:    pvbrd %v0, %s0
; CHECK-NEXT:    #APP
; CHECK-NEXT:    vst %v0, 8, %s1
; CHECK-NEXT:    #NO_APP
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.pvbrd.vsl(i64 %0, i32 256)
  tail call void asm sideeffect "vst ${0:v}, 8, $1", "v,r"(<256 x double> %3, i8* %1)
  ret void
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.pvbrd.vsl(i64, i32)

; Function Attrs: nounwind
define void @pvbrd_vsvl(i64 %0, i8* %1) {
; CHECK-LABEL: pvbrd_vsvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s2, 256
; CHECK-NEXT:    lvl %s2
; CHECK-NEXT:    vld %v0, 8, %s1
; CHECK-NEXT:    pvbrd %v0, %s0
; CHECK-NEXT:    vst %v0, 8, %s1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.vld.vssl(i64 8, i8* %1, i32 256)
  %4 = tail call fast <256 x double> @llvm.ve.vl.pvbrd.vsvl(i64 %0, <256 x double> %3, i32 256)
  tail call void @llvm.ve.vl.vst.vssl(<256 x double> %4, i64 8, i8* %1, i32 256)
  ret void
}

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.pvbrd.vsvl(i64, <256 x double>, i32)

; Function Attrs: nounwind
define void @pvbrd_vsMvl(i64 %0, i8* %1) {
; CHECK-LABEL: pvbrd_vsMvl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s2, 256
; CHECK-NEXT:    lvl %s2
; CHECK-NEXT:    vld %v0, 8, %s1
; CHECK-NEXT:    lvm %vm3, 1, %s0
; CHECK-NEXT:    lvm %vm2, 2, %s0
; CHECK-NEXT:    pvbrd %v0, %s0, %vm2
; CHECK-NEXT:    vst %v0, 8, %s1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.vld.vssl(i64 8, i8* %1, i32 256)
  %4 = tail call <512 x i1> @llvm.ve.vl.lvm.MMss(<512 x i1> undef, i64 1, i64 %0)
  %5 = tail call <512 x i1> @llvm.ve.vl.lvm.MMss(<512 x i1> %4, i64 6, i64 %0)
  %6 = tail call fast <256 x double> @llvm.ve.vl.pvbrd.vsMvl(i64 %0, <512 x i1> %5, <256 x double> %3, i32 256)
  tail call void @llvm.ve.vl.vst.vssl(<256 x double> %6, i64 8, i8* %1, i32 256)
  ret void
}

; Function Attrs: nounwind readnone
declare <512 x i1> @llvm.ve.vl.lvm.MMss(<512 x i1>, i64, i64)

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.pvbrd.vsMvl(i64, <512 x i1>, <256 x double>, i32)
