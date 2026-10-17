; RUN: llc < %s -mtriple=mipsel -mcpu=mips32r6 -mattr=+fp64,+single-float -verify-machineinstrs -stop-after=finalize-isel | FileCheck %s --check-prefix=SINGLE
; RUN: llc < %s -mtriple=mipsel -mcpu=mips32r2 -mattr=-fp64 -verify-machineinstrs | FileCheck %s --check-prefixes=CHECK,LIB32,LIB64
; RUN: llc < %s -mtriple=mipsel -mcpu=mips32r2 -mattr=+fpxx,-fp64 -verify-machineinstrs | FileCheck %s --check-prefixes=CHECK,LIB32,LIB64
; RUN: llc < %s -mtriple=mipsel -mcpu=mips32r6 -mattr=+fp64,+nooddspreg -verify-machineinstrs | FileCheck %s --check-prefixes=CHECK,NATIVE32,NATIVE64

; R6 requires FP64. Test FP32 and FPXX fallback on R2, where those modes
; are supported, rather than pairing them with R6. Single-float limits
; operations to f32 without changing R6's 64-bit FPR requirement.
; nooddspreg must not disable FMA with FP64. Check ordinary and strict nodes.

define float @fma_f32(float %a, float %b, float %c) {
; CHECK-LABEL: fma_f32:
; SINGLE-LABEL: name: fma_f32
; SINGLE: nofpexcept MADDF_S
; NATIVE32: maddf.s
; LIB32-NOT: maddf.s
; LIB32: jal fmaf
; LIB32-NOT: maddf.s
; CHECK: .end fma_f32
  %r = call float @llvm.fma.f32(float %a, float %b, float %c)
  ret float %r
}

define double @fma_f64(double %a, double %b, double %c) {
; CHECK-LABEL: fma_f64:
; SINGLE-LABEL: name: fma_f64
; SINGLE: JAL &fma,
; NATIVE64: maddf.d
; LIB64-NOT: maddf.d
; LIB64: jal fma
; LIB64-NOT: maddf.d
; CHECK: .end fma_f64
  %r = call double @llvm.fma.f64(double %a, double %b, double %c)
  ret double %r
}

define float @strict_fma_f32(float %a, float %b, float %c) strictfp {
; CHECK-LABEL: strict_fma_f32:
; SINGLE-LABEL: name: strict_fma_f32
; SINGLE: = MADDF_S
; NATIVE32: maddf.s
; LIB32-NOT: maddf.s
; LIB32: jal fmaf
; LIB32-NOT: maddf.s
; CHECK: .end strict_fma_f32
  %r = call float @llvm.experimental.constrained.fma.f32(float %a, float %b, float %c, metadata !"round.dynamic", metadata !"fpexcept.strict") strictfp
  ret float %r
}

define double @strict_fma_f64(double %a, double %b, double %c) strictfp {
; CHECK-LABEL: strict_fma_f64:
; SINGLE-LABEL: name: strict_fma_f64
; SINGLE: JAL &fma,
; NATIVE64: maddf.d
; LIB64-NOT: maddf.d
; LIB64: jal fma
; LIB64-NOT: maddf.d
; CHECK: .end strict_fma_f64
  %r = call double @llvm.experimental.constrained.fma.f64(double %a, double %b, double %c, metadata !"round.dynamic", metadata !"fpexcept.strict") strictfp
  ret double %r
}

declare float @llvm.fma.f32(float, float, float)
declare double @llvm.fma.f64(double, double, double)
declare float @llvm.experimental.constrained.fma.f32(float, float, float, metadata, metadata)
declare double @llvm.experimental.constrained.fma.f64(double, double, double, metadata, metadata)
