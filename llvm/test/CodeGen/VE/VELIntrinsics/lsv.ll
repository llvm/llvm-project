; RUN: llc < %s -mtriple=ve -mattr=+vpu | FileCheck %s

;;; Test prefetch vector intrinsic instructions
;;;
;;; Note:
;;;   We test LSVrr_v and LVSvr instructions.

; Function Attrs: nounwind
define void @lsv_vvss(i8* %0, i64 %1, i32 signext %2) {
; CHECK-LABEL: lsv_vvss:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s3, 256
; CHECK-NEXT:    lvl %s3
; CHECK-NEXT:    vld %v0, 8, %s0
; CHECK-NEXT:    and %s2, %s2, (32)0
; CHECK-NEXT:    lsv %v0(%s2), %s1
; CHECK-NEXT:    vst %v0, 8, %s0
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = tail call fast <256 x double> @llvm.ve.vl.vld.vssl(i64 8, i8* %0, i32 256)
  %5 = tail call fast <256 x double> @llvm.ve.vl.lsv.vvss(<256 x double> %4, i32 %2, i64 %1)
  tail call void @llvm.ve.vl.vst.vssl(<256 x double> %5, i64 8, i8* %0, i32 256)
  ret void
}

; Function Attrs: nounwind readonly
declare <256 x double> @llvm.ve.vl.vld.vssl(i64, i8*, i32)

; Function Attrs: nounwind readnone
declare <256 x double> @llvm.ve.vl.lsv.vvss(<256 x double>, i32, i64)

; Function Attrs: nounwind writeonly
declare void @llvm.ve.vl.vst.vssl(<256 x double>, i64, i8*, i32)

; Function Attrs: nounwind readonly
define i64 @lvsl_vssl_imm(i8* readonly %0, i32 signext %1) {
; CHECK-LABEL: lvsl_vssl_imm:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s2, 256
; CHECK-NEXT:    lvl %s2
; CHECK-NEXT:    vld %v0, 8, %s0
; CHECK-NEXT:    and %s0, %s1, (32)0
; CHECK-NEXT:    lvs %s0, %v0(%s0)
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.vld.vssl(i64 8, i8* %0, i32 256)
  %4 = tail call i64 @llvm.ve.vl.lvsl.svs(<256 x double> %3, i32 %1)
  ret i64 %4
}

; Function Attrs: nounwind readnone
declare i64 @llvm.ve.vl.lvsl.svs(<256 x double>, i32)

; Function Attrs: nounwind readonly
define double @lvsd_vssl_imm(i8* readonly %0, i32 signext %1) {
; CHECK-LABEL: lvsd_vssl_imm:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s2, 256
; CHECK-NEXT:    lvl %s2
; CHECK-NEXT:    vld %v0, 8, %s0
; CHECK-NEXT:    and %s0, %s1, (32)0
; CHECK-NEXT:    lvs %s0, %v0(%s0)
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.vld.vssl(i64 8, i8* %0, i32 256)
  %4 = tail call fast double @llvm.ve.vl.lvsd.svs(<256 x double> %3, i32 %1)
  ret double %4
}

; Function Attrs: nounwind readnone
declare double @llvm.ve.vl.lvsd.svs(<256 x double>, i32)

; Function Attrs: nounwind readonly
define float @lvss_vssl_imm(i8* readonly %0, i32 signext %1) {
; CHECK-LABEL: lvss_vssl_imm:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s2, 256
; CHECK-NEXT:    lvl %s2
; CHECK-NEXT:    vld %v0, 8, %s0
; CHECK-NEXT:    and %s0, %s1, (32)0
; CHECK-NEXT:    lvs %s0, %v0(%s0)
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast <256 x double> @llvm.ve.vl.vld.vssl(i64 8, i8* %0, i32 256)
  %4 = tail call fast float @llvm.ve.vl.lvss.svs(<256 x double> %3, i32 %1)
  ret float %4
}

; Function Attrs: nounwind readnone
declare float @llvm.ve.vl.lvss.svs(<256 x double>, i32)
