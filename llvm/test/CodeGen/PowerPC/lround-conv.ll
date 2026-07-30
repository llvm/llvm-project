; RUN: llc < %s -mtriple=powerpc64le | FileCheck %s

; CHECK-LABEL: testmsws:
; CHECK:       bl      lroundf
define signext i32 @testmsws(float %x) {
entry:
  %0 = tail call i64 @llvm.lround.i64.f32(float %x)
  %conv = trunc i64 %0 to i32
  ret i32 %conv
}

; CHECK-LABEL: testmsxs:
; CHECK:       bl      lroundf
define i64 @testmsxs(float %x) {
entry:
  %0 = tail call i64 @llvm.lround.i64.f32(float %x)
  ret i64 %0
}

; CHECK-LABEL: testmswd:
; CHECK:       bl      lround
define signext i32 @testmswd(double %x) {
entry:
  %0 = tail call i64 @llvm.lround.i64.f64(double %x)
  %conv = trunc i64 %0 to i32
  ret i32 %conv
}

; CHECK-LABEL: testmsxd:
; CHECK:       bl      lround
define i64 @testmsxd(double %x) {
entry:
  %0 = tail call i64 @llvm.lround.i64.f64(double %x)
  ret i64 %0
}

; CHECK-LABEL: testmswl:
; CHECK:       bl      lroundl
define signext i32 @testmswl(ppc_fp128 %x) {
entry:
  %0 = tail call i64 @llvm.lround.i64.ppcf128(ppc_fp128 %x)
  %conv = trunc i64 %0 to i32
  ret i32 %conv
}

; CHECK-LABEL: testmsll:
; CHECK:       bl      lroundl
define i64 @testmsll(ppc_fp128 %x) {
entry:
  %0 = tail call i64 @llvm.lround.i64.ppcf128(ppc_fp128 %x)
  ret i64 %0
}

declare i64 @llvm.lround.i64.f32(float) nounwind readnone
declare i64 @llvm.lround.i64.f64(double) nounwind readnone
declare i64 @llvm.lround.i64.ppcf128(ppc_fp128) nounwind readnone
