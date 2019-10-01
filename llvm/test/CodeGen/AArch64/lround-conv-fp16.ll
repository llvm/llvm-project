; RUN: llc < %s -mtriple=aarch64 -mattr=+fullfp16 | FileCheck %s

; CHECK-LABEL: testmhhs:
; CHECK:       fcvtas  x0, h0
; CHECK:       ret
define i16 @testmhhs(half %x) {
entry:
  %0 = tail call i64 @llvm.lround.i64.f16(half %x)
  %conv = trunc i64 %0 to i16
  ret i16 %conv
}

; CHECK-LABEL: testmhws:
; CHECK:       fcvtas  x0, h0
; CHECK:       ret
define i32 @testmhws(half %x) {
entry:
  %0 = tail call i64 @llvm.lround.i64.f16(half %x)
  %conv = trunc i64 %0 to i32
  ret i32 %conv
}

; CHECK-LABEL: testmhxs:
; CHECK:       fcvtas  x0, h0
; CHECK-NEXT:  ret
define i64 @testmhxs(half %x) {
entry:
  %0 = tail call i64 @llvm.lround.i64.f16(half %x)
  ret i64 %0
}

declare i64 @llvm.lround.i64.f16(half) nounwind readnone
