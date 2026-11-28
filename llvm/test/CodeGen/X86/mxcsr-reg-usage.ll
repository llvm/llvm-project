; RUN: llc -mtriple=x86_64-unknown-unknown -mattr=+mmx,+fma,+f16c,+avx512f -stop-after finalize-isel -o - %s | FileCheck %s

define half @mxcsr_f16c(float %a) {
; CHECK: VCVTPS2PH{{.*}}mxcsr
  %res = fptrunc float %a to half
  ret half %res
}

define <4 x float> @mxcsr_fma_ss(<4 x float> %a, <4 x float> %b) {
; CHECK: VFMADD{{.*}}mxcsr
  %res = call <4 x float> @llvm.x86.fma.vfmadd.ss(<4 x float> %b, <4 x float> %a, <4 x float>
%a)
  ret <4 x float> %res
}

define <4 x float> @mxcsr_fma_ps(<4 x float> %a, <4 x float> %b) {
; CHECK: VFMADD{{.*}}mxcsr
  %res = call <4 x float> @llvm.x86.fma.vfmadd.ps(<4 x float> %b, <4 x float> %a, <4 x float>
%a)
  ret <4 x float> %res
}

define <8 x double> @mxcsr_fma_sae(<8 x double> %a, <8 x double> %b, <8 x double> %c) {
; CHECK: VFMADD{{.*}}mxcsr
  %res = call <8 x double> @llvm.x86.avx512.mask.vfmadd.pd.512(<8 x double> %a, <8 x double> %b, <8 x double> %c, i8 -1, i32 10)
  ret <8 x double> %res
}

declare <4 x float> @llvm.x86.fma.vfmadd.ss(<4 x float>, <4 x float>, <4 x float>)
declare <4 x float> @llvm.x86.fma.vfmadd.ps(<4 x float>, <4 x float>, <4 x float>)
declare <8 x double> @llvm.x86.avx512.mask.vfmadd.pd.512(<8 x double>, <8 x double>, <8 x double>, i8, i32)
