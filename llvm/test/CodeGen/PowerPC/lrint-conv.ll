; RUN: llc < %s -mtriple=powerpc64le | FileCheck %s
; RUN: llc < %s -mtriple=powerpc | FileCheck %s

; FIXME: crash "Input type needs to be promoted!"
; define signext i32 @testmswh(half %x) {
; entry:
;   %0 = tail call i64 @llvm.lrint.i64.f16(half %x)
;   %conv = trunc i64 %0 to i32
;   ret i32 %conv
; }

; define i64 @testmsxh(half %x) {
; entry:
;   %0 = tail call i64 @llvm.lrint.i64.f16(half %x)
;   ret i64 %0
; }

; CHECK-LABEL: testmsws:
; CHECK:       bl      lrintf
define signext i32 @testmsws(float %x) {
entry:
  %0 = tail call i64 @llvm.lrint.i64.f32(float %x)
  %conv = trunc i64 %0 to i32
  ret i32 %conv
}

; CHECK-LABEL: testmsxs:
; CHECK:       bl      lrintf
define i64 @testmsxs(float %x) {
entry:
  %0 = tail call i64 @llvm.lrint.i64.f32(float %x)
  ret i64 %0
}

; CHECK-LABEL: testmswd:
; CHECK:       bl      lrint
define signext i32 @testmswd(double %x) {
entry:
  %0 = tail call i64 @llvm.lrint.i64.f64(double %x)
  %conv = trunc i64 %0 to i32
  ret i32 %conv
}

; CHECK-LABEL: testmsxd:
; CHECK:       bl      lrint
define i64 @testmsxd(double %x) {
entry:
  %0 = tail call i64 @llvm.lrint.i64.f64(double %x)
  ret i64 %0
}

; CHECK-LABEL: testmswl:
; CHECK:       bl      lrintl
define signext i32 @testmswl(ppc_fp128 %x) {
entry:
  %0 = tail call i64 @llvm.lrint.i64.ppcf128(ppc_fp128 %x)
  %conv = trunc i64 %0 to i32
  ret i32 %conv
}

; CHECK-LABEL: testmsll:
; CHECK:       bl      lrintl
define i64 @testmsll(ppc_fp128 %x) {
entry:
  %0 = tail call i64 @llvm.lrint.i64.ppcf128(ppc_fp128 %x)
  ret i64 %0
}

; CHECK-LABEL: testmswq:
; CHECK:       bl      lrintf128
define signext i32 @testmswq(fp128 %x) {
entry:
  %0 = tail call i64 @llvm.lrint.i64.f128(fp128 %x)
  %conv = trunc i64 %0 to i32
  ret i32 %conv
}

; CHECK-LABEL: testmslq:
; CHECK:       bl      lrintf128
define i64 @testmslq(fp128 %x) {
entry:
  %0 = tail call i64 @llvm.lrint.i64.f128(fp128 %x)
  ret i64 %0
}

declare i64 @llvm.lrint.i64.f32(float) nounwind readnone
declare i64 @llvm.lrint.i64.f64(double) nounwind readnone
declare i64 @llvm.lrint.i64.ppcf128(ppc_fp128) nounwind readnone
