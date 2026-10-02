; RUN: llc -mtriple=aarch64                      -global-isel=0 %s -o - | FileCheck %s --check-prefix=CHECK,FP
; RUN: llc -mtriple=aarch64                  -O0 -global-isel=1 %s -o - | FileCheck %s --check-prefix=CHECK,FP
; RUN: llc -mtriple=aarch64 -mattr=-fp-armv8     -global-isel=0 %s -o - | FileCheck %s --check-prefix=CHECK,NOFP
; RUN: llc -mtriple=aarch64 -mattr=-fp-armv8 -O0 -global-isel=1 %s -o - | FileCheck %s --check-prefix=CHECK,NOFP

; Test that @llvm.fmuladd, with the semantics "multiply and add can be
; fused or separate at the compiler's option", fuses them in the
; normal case of hardware FP, but doesn't fuse them in AArch64 softfp,
; on the theory that the libc fma function is likely slower than the
; combination.
;
; However, @llvm.fma must fuse them regardless, so that _should_ emit
; a call to fma in softfp modes.

define float @whichever_float(float %0, float %1, float %2) {
  %result = call float @llvm.fmuladd.f32(float %0, float %1, float %2)
  ret float %result
; CHECK-LABEL: {{^}}whichever_float:
; FP: fmadd s0, s0, s1, s2
; NOFP: bl __mulsf3
; NOFP: bl __addsf3
}

define double @whichever_double(double %0, double %1, double %2) {
  %result = call double @llvm.fmuladd.f64(double %0, double %1, double %2)
  ret double %result
; CHECK-LABEL: {{^}}whichever_double:
; FP: fmadd d0, d0, d1, d2
; NOFP: bl __muldf3
; NOFP: bl __adddf3
}

define float @force_fma_float(float %0, float %1, float %2) {
  %result = call float @llvm.fma.f32(float %0, float %1, float %2)
  ret float %result
; CHECK-LABEL: {{^}}force_fma_float:
; FP: fmadd s0, s0, s1, s2
; NOFP: bl fmaf
}

define double @force_fma_double(double %0, double %1, double %2) {
  %result = call double @llvm.fma.f64(double %0, double %1, double %2)
  ret double %result
; CHECK-LABEL: {{^}}force_fma_double:
; FP: fmadd d0, d0, d1, d2
; NOFP: bl fma
}
