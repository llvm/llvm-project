; RUN: llc -mtriple=x86_64-unknown-unknown -mattr=+mmx,+fma,+f16c,+avx512f -stop-after finalize-isel -o - %s | FileCheck %s
; This test ensures that the MXCSR is implicitly used by MMX FP instructions.

define x86_mmx @mxcsr_mmx(<4 x float> %a0) {
; CHECK: MMX_CVTPS2PIrr %{{[0-9]}}, implicit $mxcsr
; CHECK: MMX_CVTPI2PSrr %{{[0-9]}}, killed %{{[0-9]}}, implicit $mxcsr
; CHECK: MMX_CVTTPS2PIrr killed %{{[0-9]}}, implicit $mxcsr
; CHECK: MMX_CVTPI2PDrr killed %{{[0-9]$}}
; CHECK: MMX_CVTPD2PIrr killed %{{[0-9]}}, implicit $mxcsr
  %1 = call x86_mmx @llvm.x86.sse.cvtps2pi(<4 x float> %a0)
  %2 = call <4 x float> @llvm.x86.sse.cvtpi2ps(<4 x float> %a0, x86_mmx %1)
  %3 = call x86_mmx @llvm.x86.sse.cvttps2pi(<4 x float> %2)
  %4 = call <2 x double> @llvm.x86.sse.cvtpi2pd(x86_mmx %3)
  %5 = call x86_mmx @llvm.x86.sse.cvtpd2pi(<2 x double> %4)
  ret x86_mmx %5
}

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

declare x86_mmx @llvm.x86.sse.cvtps2pi(<4 x float>)
declare<4 x float> @llvm.x86.sse.cvtpi2ps(<4 x float>, x86_mmx)
declare x86_mmx @llvm.x86.sse.cvttps2pi(<4 x float>)
declare <2 x double> @llvm.x86.sse.cvtpi2pd(x86_mmx)
declare x86_mmx @llvm.x86.sse.cvtpd2pi(<2 x double>)
declare <4 x float> @llvm.x86.fma.vfmadd.ss(<4 x float>, <4 x float>, <4 x float>)
declare <4 x float> @llvm.x86.fma.vfmadd.ps(<4 x float>, <4 x float>, <4 x float>)
declare <8 x double> @llvm.x86.avx512.mask.vfmadd.pd.512(<8 x double>, <8 x double>, <8 x double>, i8, i32)
