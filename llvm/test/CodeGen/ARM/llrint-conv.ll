; RUN: llc < %s -mtriple=arm-eabi -float-abi=soft | FileCheck %s --check-prefix=SOFTFP
; RUN: llc < %s -mtriple=arm-eabi -float-abi=hard | FileCheck %s --check-prefix=HARDFP

; SOFTFP-LABEL: testmsxs_builtin:
; SOFTFP:       bl      llrintf
; HARDFP-LABEL: testmsxs_builtin:
; HARDFP:       bl      llrintf
define i64 @testmsxs_builtin(float %x) {
entry:
  %0 = tail call i64 @llvm.llrint.f32(float %x)
  ret i64 %0
}

; SOFTFP-LABEL: testmsxd_builtin:
; SOFTFP:       bl      llrint
; HARDFP-LABEL: testmsxd_builtin:
; HARDFP:       bl      llrint
define i64 @testmsxd_builtin(double %x) {
entry:
  %0 = tail call i64 @llvm.llrint.f64(double %x)
  ret i64 %0
}

declare i64 @llvm.llrint.f32(float) nounwind readnone
declare i64 @llvm.llrint.f64(double) nounwind readnone
