; RUN: llc < %s -mtriple=mips64el -mattr=+soft-float | FileCheck %s
; RUN: llc < %s -mtriple=mips -mattr=+soft-float     | FileCheck %s

define signext i32 @testmswh(half %x) {
; CHECK-LABEL: testmswh:
; CHECK:       jal     lrintf
entry:
  %0 = tail call i64 @llvm.lrint.i64.f16(half %x)
  %conv = trunc i64 %0 to i32
  ret i32 %conv
}

define i64 @testmsxh(half %x) {
; CHECK-LABEL: testmsxh:
; CHECK:       jal     lrintf
entry:
  %0 = tail call i64 @llvm.lrint.i64.f16(half %x)
  ret i64 %0
}

define signext i32 @testmsws(float %x) {
; CHECK-LABEL: testmsws:
; CHECK:       jal     lrintf
entry:
  %0 = tail call i64 @llvm.lrint.i64.f32(float %x)
  %conv = trunc i64 %0 to i32
  ret i32 %conv
}

define i64 @testmsxs(float %x) {
; CHECK-LABEL: testmsxs:
; CHECK:       jal     lrintf
entry:
  %0 = tail call i64 @llvm.lrint.i64.f32(float %x)
  ret i64 %0
}

define signext i32 @testmswd(double %x) {
; CHECK-LABEL: testmswd:
; CHECK:       jal     lrint
entry:
  %0 = tail call i64 @llvm.lrint.i64.f64(double %x)
  %conv = trunc i64 %0 to i32
  ret i32 %conv
}

define i64 @testmsxd(double %x) {
; CHECK-LABEL: testmsxd:
; CHECK:       jal     lrint
entry:
  %0 = tail call i64 @llvm.lrint.i64.f64(double %x)
  ret i64 %0
}

; fp128 lrint has no libcall on mips (o32); tested on mips64 in
; lrint-conv-fp128.ll.

declare i64 @llvm.lrint.i64.f32(float) nounwind readnone
declare i64 @llvm.lrint.i64.f64(double) nounwind readnone
