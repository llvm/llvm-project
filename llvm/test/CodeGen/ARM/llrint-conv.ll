; RUN: llc < %s -mtriple=arm-eabi -float-abi=soft | FileCheck %s --check-prefix=SOFTFP
; RUN: llc < %s -mtriple=arm-eabi -float-abi=hard | FileCheck %s --check-prefix=HARDFP

; SOFTFP-LABEL: testmsxh_builtin:
; SOFTFP:       bl      llrintf
; HARDFP-LABEL: testmsxh_builtin:
; HARDFP:       bl      llrintf
define i64 @testmsxh_builtin(half %x) {
entry:
  %0 = tail call i64 @llvm.llrint.i64.f16(half %x)
  ret i64 %0
}

; SOFTFP-LABEL: testmsxs_builtin:
; SOFTFP:       bl      llrintf
; HARDFP-LABEL: testmsxs_builtin:
; HARDFP:       bl      llrintf
define i64 @testmsxs_builtin(float %x) {
entry:
  %0 = tail call i64 @llvm.llrint.i64.f32(float %x)
  ret i64 %0
}

; SOFTFP-LABEL: testmsxd_builtin:
; SOFTFP:       bl      llrint
; HARDFP-LABEL: testmsxd_builtin:
; HARDFP:       bl      llrint
define i64 @testmsxd_builtin(double %x) {
entry:
  %0 = tail call i64 @llvm.llrint.i64.f64(double %x)
  ret i64 %0
}

; FIXME(#44744): incorrect libcall
; SOFTFP-LABEL: testmsxq_builtin:
; SOFTFP:       bl      llrintl
; HARDFP-LABEL: testmsxq_builtin:
; HARDFP:       bl      llrintl
define i64 @testmsxq_builtin(fp128 %x) {
entry:
  %0 = tail call i64 @llvm.llrint.i64.f128(fp128 %x)
  ret i64 %0
}

declare i64 @llvm.llrint.i64.f32(float) nounwind readnone
declare i64 @llvm.llrint.i64.f64(double) nounwind readnone
