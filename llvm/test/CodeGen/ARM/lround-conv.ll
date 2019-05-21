; RUN: llc < %s -mtriple=arm-eabi -float-abi=soft | FileCheck %s --check-prefix=SOFTFP
; RUN: llc < %s -mtriple=arm-eabi -float-abi=hard | FileCheck %s --check-prefix=HARDFP

; SOFTFP-LABEL: testmsws_builtin:
; SOFTFP:       bl      lroundf
; HARDFP-LABEL: testmsws_builtin:
; HARDFP:       bl      lroundf
define i32 @testmsws_builtin(float %x) {
entry:
  %0 = tail call i32 @llvm.lround.i32.f32(float %x)
  ret i32 %0
}

; SOFTFP-LABEL: testmswd_builtin:
; SOFTFP:       bl      lround
; HARDFP-LABEL: testmswd_builtin:
; HARDFP:       bl      lround
define i32 @testmswd_builtin(double %x) {
entry:
  %0 = tail call i32 @llvm.lround.i32.f64(double %x)
  ret i32 %0
}

declare i32 @llvm.lround.i32.f32(float) nounwind readnone
declare i32 @llvm.lround.i32.f64(double) nounwind readnone
