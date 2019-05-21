; RUN: llc < %s -mtriple=aarch64-windows -mattr=+neon | FileCheck %s

; CHECK-LABEL: testmsxs:
; CHECK:       fcvtas  w8, s0
; CHECK-NEXT:  sxtw    x0, w8
; CHECK-NEXT:  ret
define i64 @testmsxs(float %x) {
entry:
  %0 = tail call i32 @llvm.lround.i32.f32(float %x)
  %conv = sext i32 %0 to i64
  ret i64 %conv
}

; CHECK-LABEL: testmsws:
; CHECK:       fcvtas  w0, s0
; CHECK-NEXT:  ret
define i32 @testmsws(float %x) {
entry:
  %0 = tail call i32 @llvm.lround.i32.f32(float %x)
  ret i32 %0
}

; CHECK-LABEL: testmsxd:
; CHECK:       fcvtas  w8, d0
; CHECK-NEXT:  sxtw    x0, w8
; CHECK-NEXT:  ret
define i64 @testmsxd(double %x) {
entry:
  %0 = tail call i32 @llvm.lround.i32.f64(double %x)
  %conv = sext i32 %0 to i64
  ret i64 %conv
}

; CHECK-LABEL: testmswd:
; CHECK:       fcvtas  w0, d0
; CHECK-NEXT:  ret
define i32 @testmswd(double %x) {
entry:
  %0 = tail call i32 @llvm.lround.i32.f64(double %x)
  ret i32 %0
}

declare i32 @llvm.lround.i32.f32(float) nounwind readnone
declare i32 @llvm.lround.i32.f64(double) nounwind readnone
