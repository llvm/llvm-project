; RUN: llc < %s -mtriple=mips64el -mattr=+soft-float | FileCheck %s
; RUN: llc < %s -mtriple=mips -mattr=+soft-float     | FileCheck %s

; FIXME: crash
; define signext i32 @testmswh(half %x) {
; entry:
;   %0 = tail call i64 @llvm.llrint.i64.f16(half %x)
;   %conv = trunc i64 %0 to i32
;   ret i32 %conv
; }

; define i64 @testmsxh(half %x) {
; entry:
;   %0 = tail call i64 @llvm.llrint.i64.f16(half %x)
;   ret i64 %0
; }

define signext i32 @testmsws(float %x) {
; CHECK-LABEL: testmsws:
; CHECK:       jal     llrintf
entry:
  %0 = tail call i64 @llvm.llrint.i64.f32(float %x)
  %conv = trunc i64 %0 to i32
  ret i32 %conv
}

define i64 @testmsxs(float %x) {
; CHECK-LABEL: testmsxs:
; CHECK:       jal     llrintf
entry:
  %0 = tail call i64 @llvm.llrint.i64.f32(float %x)
  ret i64 %0
}

define signext i32 @testmswd(double %x) {
; CHECK-LABEL: testmswd:
; CHECK:       jal     llrint
entry:
  %0 = tail call i64 @llvm.llrint.i64.f64(double %x)
  %conv = trunc i64 %0 to i32
  ret i32 %conv
}

define i64 @testmsxd(double %x) {
; CHECK-LABEL: testmsxd:
; CHECK:       jal     llrint
entry:
  %0 = tail call i64 @llvm.llrint.i64.f64(double %x)
  ret i64 %0
}

define signext i32 @testmswl(fp128 %x) {
; CHECK-LABEL: testmswl:
; CHECK:       jal     llrintl
entry:
  %0 = tail call i64 @llvm.llrint.i64.f128(fp128 %x)
  %conv = trunc i64 %0 to i32
  ret i32 %conv
}

define i64 @testmsll(fp128 %x) {
; CHECK-LABEL: testmsll:
; CHECK:       jal     llrintl
entry:
  %0 = tail call i64 @llvm.llrint.i64.f128(fp128 %x)
  ret i64 %0
}

declare i64 @llvm.llrint.i64.f32(float) nounwind readnone
declare i64 @llvm.llrint.i64.f64(double) nounwind readnone
declare i64 @llvm.llrint.i64.f128(fp128) nounwind readnone
