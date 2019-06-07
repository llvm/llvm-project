; RUN: llc < %s -mtriple=aarch64-windows -mattr=+fullfp16 | FileCheck %s

; CHECK-LABEL: testmhhs:
; CHECK:       fcvtas  w0, h0
; CHECK:       ret
define i16 @testmhhs(half %x) {
entry:
  %0 = tail call i32 @llvm.lround.i32.f16(half %x)
  %conv = trunc i32 %0 to i16
  ret i16 %conv
}

; CHECK-LABEL: testmhws:
; CHECK:       fcvtas  w0, h0
; CHECK:       ret
define i32 @testmhws(half %x) {
entry:
  %0 = tail call i32 @llvm.lround.i32.f16(half %x)
  ret i32 %0
}

; CHECK-LABEL: testmhxs:
; CHECK:       fcvtas  w8, h0
; CHECK-NEXT:  sxtw    x0, w8
; CHECK-NEXT:  ret
define i64 @testmhxs(half %x) {
entry:
  %0 = tail call i32 @llvm.lround.i32.f16(half %x)
  %conv = sext i32 %0 to i64
  ret i64 %conv
}

declare i32 @llvm.lround.i32.f16(half) nounwind readnone
