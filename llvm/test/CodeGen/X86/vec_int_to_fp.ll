; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mattr=+sse2 | FileCheck %s --check-prefix=ALL --check-prefix=SSE --check-prefix=SSE2
; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mattr=+avx | FileCheck %s --check-prefix=ALL --check-prefix=AVX --check-prefix=AVX1
; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mattr=+avx2 | FileCheck %s --check-prefix=ALL --check-prefix=AVX --check-prefix=AVX2
;
; 32-bit tests to make sure we're not doing anything stupid.
; RUN: llc < %s -mtriple=i686-unknown-unknown
; RUN: llc < %s -mtriple=i686-unknown-unknown -mattr=+sse
; RUN: llc < %s -mtriple=i686-unknown-unknown -mattr=+sse2

;
; Signed Integer to Double
;

define <2 x double> @sitofp_2i64_to_2f64(<2 x i64> %a) {
; SSE-LABEL: sitofp_2i64_to_2f64:
; SSE:       # BB#0:
; SSE-NEXT:    movd %xmm0, %rax
; SSE-NEXT:    cvtsi2sdq %rax, %xmm1
; SSE-NEXT:    pshufd {{.*#+}} xmm0 = xmm0[2,3,0,1]
; SSE-NEXT:    movd %xmm0, %rax
; SSE-NEXT:    xorps %xmm0, %xmm0
; SSE-NEXT:    cvtsi2sdq %rax, %xmm0
; SSE-NEXT:    unpcklpd {{.*#+}} xmm1 = xmm1[0],xmm0[0]
; SSE-NEXT:    movapd %xmm1, %xmm0
; SSE-NEXT:    retq
;
; AVX-LABEL: sitofp_2i64_to_2f64:
; AVX:       # BB#0:
; AVX-NEXT:    vpextrq $1, %xmm0, %rax
; AVX-NEXT:    vcvtsi2sdq %rax, %xmm0, %xmm1
; AVX-NEXT:    vmovq %xmm0, %rax
; AVX-NEXT:    vxorps %xmm0, %xmm0, %xmm0
; AVX-NEXT:    vcvtsi2sdq %rax, %xmm0, %xmm0
; AVX-NEXT:    vunpcklpd {{.*#+}} xmm0 = xmm0[0],xmm1[0]
; AVX-NEXT:    retq
  %cvt = sitofp <2 x i64> %a to <2 x double>
  ret <2 x double> %cvt
}

define <2 x double> @sitofp_2i32_to_2f64(<4 x i32> %a) {
; SSE-LABEL: sitofp_2i32_to_2f64:
; SSE:       # BB#0:
; SSE-NEXT:    cvtdq2pd %xmm0, %xmm0
; SSE-NEXT:    retq
;
; AVX-LABEL: sitofp_2i32_to_2f64:
; AVX:       # BB#0:
; AVX-NEXT:    vcvtdq2pd %xmm0, %xmm0
; AVX-NEXT:    retq
  %shuf = shufflevector <4 x i32> %a, <4 x i32> undef, <2 x i32> <i32 0, i32 1>
  %cvt = sitofp <2 x i32> %shuf to <2 x double>
  ret <2 x double> %cvt
}

define <2 x double> @sitofp_4i32_to_2f64(<4 x i32> %a) {
; SSE-LABEL: sitofp_4i32_to_2f64:
; SSE:       # BB#0:
; SSE-NEXT:    cvtdq2pd %xmm0, %xmm0
; SSE-NEXT:    retq
;
; AVX-LABEL: sitofp_4i32_to_2f64:
; AVX:       # BB#0:
; AVX-NEXT:    vcvtdq2pd %xmm0, %ymm0
; AVX-NEXT:    vzeroupper
; AVX-NEXT:    retq
  %cvt = sitofp <4 x i32> %a to <4 x double>
  %shuf = shufflevector <4 x double> %cvt, <4 x double> undef, <2 x i32> <i32 0, i32 1>
  ret <2 x double> %shuf
}

define <2 x double> @sitofp_2i16_to_2f64(<8 x i16> %a) {
; SSE-LABEL: sitofp_2i16_to_2f64:
; SSE:       # BB#0:
; SSE-NEXT:    punpcklwd {{.*#+}} xmm0 = xmm0[0,0,1,1,2,2,3,3]
; SSE-NEXT:    psrad $16, %xmm0
; SSE-NEXT:    cvtdq2pd %xmm0, %xmm0
; SSE-NEXT:    retq
;
; AVX-LABEL: sitofp_2i16_to_2f64:
; AVX:       # BB#0:
; AVX-NEXT:    vpmovsxwd %xmm0, %xmm0
; AVX-NEXT:    vcvtdq2pd %xmm0, %xmm0
; AVX-NEXT:    retq
  %shuf = shufflevector <8 x i16> %a, <8 x i16> undef, <2 x i32> <i32 0, i32 1>
  %cvt = sitofp <2 x i16> %shuf to <2 x double>
  ret <2 x double> %cvt
}

define <2 x double> @sitofp_8i16_to_2f64(<8 x i16> %a) {
; SSE-LABEL: sitofp_8i16_to_2f64:
; SSE:       # BB#0:
; SSE-NEXT:    punpcklwd {{.*#+}} xmm0 = xmm0[0,0,1,1,2,2,3,3]
; SSE-NEXT:    psrad $16, %xmm0
; SSE-NEXT:    cvtdq2pd %xmm0, %xmm0
; SSE-NEXT:    retq
;
; AVX1-LABEL: sitofp_8i16_to_2f64:
; AVX1:       # BB#0:
; AVX1-NEXT:    vpmovsxwd %xmm0, %xmm0
; AVX1-NEXT:    vcvtdq2pd %xmm0, %ymm0
; AVX1-NEXT:    vzeroupper
; AVX1-NEXT:    retq
;
; AVX2-LABEL: sitofp_8i16_to_2f64:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpmovsxwd %xmm0, %ymm0
; AVX2-NEXT:    vcvtdq2pd %xmm0, %ymm0
; AVX2-NEXT:    vzeroupper
; AVX2-NEXT:    retq
  %cvt = sitofp <8 x i16> %a to <8 x double>
  %shuf = shufflevector <8 x double> %cvt, <8 x double> undef, <2 x i32> <i32 0, i32 1>
  ret <2 x double> %shuf
}

define <2 x double> @sitofp_2i8_to_2f64(<16 x i8> %a) {
; SSE-LABEL: sitofp_2i8_to_2f64:
; SSE:       # BB#0:
; SSE-NEXT:    punpcklbw {{.*#+}} xmm0 = xmm0[0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7]
; SSE-NEXT:    punpcklwd {{.*#+}} xmm0 = xmm0[0,0,1,1,2,2,3,3]
; SSE-NEXT:    psrad $24, %xmm0
; SSE-NEXT:    cvtdq2pd %xmm0, %xmm0
; SSE-NEXT:    retq
;
; AVX-LABEL: sitofp_2i8_to_2f64:
; AVX:       # BB#0:
; AVX-NEXT:    vpmovsxbd %xmm0, %xmm0
; AVX-NEXT:    vcvtdq2pd %xmm0, %xmm0
; AVX-NEXT:    retq
  %shuf = shufflevector <16 x i8> %a, <16 x i8> undef, <2 x i32> <i32 0, i32 1>
  %cvt = sitofp <2 x i8> %shuf to <2 x double>
  ret <2 x double> %cvt
}

define <2 x double> @sitofp_16i8_to_2f64(<16 x i8> %a) {
; SSE-LABEL: sitofp_16i8_to_2f64:
; SSE:       # BB#0:
; SSE-NEXT:    punpcklbw {{.*#+}} xmm0 = xmm0[0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7]
; SSE-NEXT:    punpcklwd {{.*#+}} xmm0 = xmm0[0,0,1,1,2,2,3,3]
; SSE-NEXT:    psrad $24, %xmm0
; SSE-NEXT:    cvtdq2pd %xmm0, %xmm0
; SSE-NEXT:    retq
;
; AVX1-LABEL: sitofp_16i8_to_2f64:
; AVX1:       # BB#0:
; AVX1-NEXT:    vpmovsxbd %xmm0, %xmm0
; AVX1-NEXT:    vcvtdq2pd %xmm0, %ymm0
; AVX1-NEXT:    vzeroupper
; AVX1-NEXT:    retq
;
; AVX2-LABEL: sitofp_16i8_to_2f64:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpmovsxbw %xmm0, %ymm0
; AVX2-NEXT:    vpmovsxwd %xmm0, %ymm0
; AVX2-NEXT:    vcvtdq2pd %xmm0, %ymm0
; AVX2-NEXT:    vzeroupper
; AVX2-NEXT:    retq
  %cvt = sitofp <16 x i8> %a to <16 x double>
  %shuf = shufflevector <16 x double> %cvt, <16 x double> undef, <2 x i32> <i32 0, i32 1>
  ret <2 x double> %shuf
}

define <4 x double> @sitofp_4i64_to_4f64(<4 x i64> %a) {
; SSE-LABEL: sitofp_4i64_to_4f64:
; SSE:       # BB#0:
; SSE-NEXT:    movd %xmm0, %rax
; SSE-NEXT:    cvtsi2sdq %rax, %xmm2
; SSE-NEXT:    pshufd {{.*#+}} xmm0 = xmm0[2,3,0,1]
; SSE-NEXT:    movd %xmm0, %rax
; SSE-NEXT:    xorps %xmm0, %xmm0
; SSE-NEXT:    cvtsi2sdq %rax, %xmm0
; SSE-NEXT:    unpcklpd {{.*#+}} xmm2 = xmm2[0],xmm0[0]
; SSE-NEXT:    movd %xmm1, %rax
; SSE-NEXT:    cvtsi2sdq %rax, %xmm3
; SSE-NEXT:    pshufd {{.*#+}} xmm0 = xmm1[2,3,0,1]
; SSE-NEXT:    movd %xmm0, %rax
; SSE-NEXT:    xorps %xmm0, %xmm0
; SSE-NEXT:    cvtsi2sdq %rax, %xmm0
; SSE-NEXT:    unpcklpd {{.*#+}} xmm3 = xmm3[0],xmm0[0]
; SSE-NEXT:    movapd %xmm2, %xmm0
; SSE-NEXT:    movapd %xmm3, %xmm1
; SSE-NEXT:    retq
;
; AVX1-LABEL: sitofp_4i64_to_4f64:
; AVX1:       # BB#0:
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm1
; AVX1-NEXT:    vpextrq $1, %xmm1, %rax
; AVX1-NEXT:    vcvtsi2sdq %rax, %xmm0, %xmm2
; AVX1-NEXT:    vmovq %xmm1, %rax
; AVX1-NEXT:    vcvtsi2sdq %rax, %xmm0, %xmm1
; AVX1-NEXT:    vunpcklpd {{.*#+}} xmm1 = xmm1[0],xmm2[0]
; AVX1-NEXT:    vpextrq $1, %xmm0, %rax
; AVX1-NEXT:    vcvtsi2sdq %rax, %xmm0, %xmm2
; AVX1-NEXT:    vmovq %xmm0, %rax
; AVX1-NEXT:    vxorps %xmm0, %xmm0, %xmm0
; AVX1-NEXT:    vcvtsi2sdq %rax, %xmm0, %xmm0
; AVX1-NEXT:    vunpcklpd {{.*#+}} xmm0 = xmm0[0],xmm2[0]
; AVX1-NEXT:    vinsertf128 $1, %xmm1, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: sitofp_4i64_to_4f64:
; AVX2:       # BB#0:
; AVX2-NEXT:    vextracti128 $1, %ymm0, %xmm1
; AVX2-NEXT:    vpextrq $1, %xmm1, %rax
; AVX2-NEXT:    vcvtsi2sdq %rax, %xmm0, %xmm2
; AVX2-NEXT:    vmovq %xmm1, %rax
; AVX2-NEXT:    vcvtsi2sdq %rax, %xmm0, %xmm1
; AVX2-NEXT:    vunpcklpd {{.*#+}} xmm1 = xmm1[0],xmm2[0]
; AVX2-NEXT:    vpextrq $1, %xmm0, %rax
; AVX2-NEXT:    vcvtsi2sdq %rax, %xmm0, %xmm2
; AVX2-NEXT:    vmovq %xmm0, %rax
; AVX2-NEXT:    vxorps %xmm0, %xmm0, %xmm0
; AVX2-NEXT:    vcvtsi2sdq %rax, %xmm0, %xmm0
; AVX2-NEXT:    vunpcklpd {{.*#+}} xmm0 = xmm0[0],xmm2[0]
; AVX2-NEXT:    vinsertf128 $1, %xmm1, %ymm0, %ymm0
; AVX2-NEXT:    retq
  %cvt = sitofp <4 x i64> %a to <4 x double>
  ret <4 x double> %cvt
}

define <4 x double> @sitofp_4i32_to_4f64(<4 x i32> %a) {
; SSE-LABEL: sitofp_4i32_to_4f64:
; SSE:       # BB#0:
; SSE-NEXT:    cvtdq2pd %xmm0, %xmm2
; SSE-NEXT:    pshufd {{.*#+}} xmm0 = xmm0[2,3,0,1]
; SSE-NEXT:    cvtdq2pd %xmm0, %xmm1
; SSE-NEXT:    movaps %xmm2, %xmm0
; SSE-NEXT:    retq
;
; AVX-LABEL: sitofp_4i32_to_4f64:
; AVX:       # BB#0:
; AVX-NEXT:    vcvtdq2pd %xmm0, %ymm0
; AVX-NEXT:    retq
  %cvt = sitofp <4 x i32> %a to <4 x double>
  ret <4 x double> %cvt
}

define <4 x double> @sitofp_4i16_to_4f64(<8 x i16> %a) {
; SSE-LABEL: sitofp_4i16_to_4f64:
; SSE:       # BB#0:
; SSE-NEXT:    punpcklwd {{.*#+}} xmm1 = xmm1[0],xmm0[0],xmm1[1],xmm0[1],xmm1[2],xmm0[2],xmm1[3],xmm0[3]
; SSE-NEXT:    psrad $16, %xmm1
; SSE-NEXT:    cvtdq2pd %xmm1, %xmm0
; SSE-NEXT:    pshufd {{.*#+}} xmm1 = xmm1[2,3,0,1]
; SSE-NEXT:    cvtdq2pd %xmm1, %xmm1
; SSE-NEXT:    retq
;
; AVX-LABEL: sitofp_4i16_to_4f64:
; AVX:       # BB#0:
; AVX-NEXT:    vpmovsxwd %xmm0, %xmm0
; AVX-NEXT:    vcvtdq2pd %xmm0, %ymm0
; AVX-NEXT:    retq
  %shuf = shufflevector <8 x i16> %a, <8 x i16> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %cvt = sitofp <4 x i16> %shuf to <4 x double>
  ret <4 x double> %cvt
}

define <4 x double> @sitofp_8i16_to_4f64(<8 x i16> %a) {
; SSE-LABEL: sitofp_8i16_to_4f64:
; SSE:       # BB#0:
; SSE-NEXT:    punpcklwd {{.*#+}} xmm1 = xmm1[0],xmm0[0],xmm1[1],xmm0[1],xmm1[2],xmm0[2],xmm1[3],xmm0[3]
; SSE-NEXT:    psrad $16, %xmm1
; SSE-NEXT:    cvtdq2pd %xmm1, %xmm0
; SSE-NEXT:    pshufd {{.*#+}} xmm1 = xmm1[2,3,0,1]
; SSE-NEXT:    cvtdq2pd %xmm1, %xmm1
; SSE-NEXT:    retq
;
; AVX1-LABEL: sitofp_8i16_to_4f64:
; AVX1:       # BB#0:
; AVX1-NEXT:    vpmovsxwd %xmm0, %xmm0
; AVX1-NEXT:    vcvtdq2pd %xmm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: sitofp_8i16_to_4f64:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpmovsxwd %xmm0, %ymm0
; AVX2-NEXT:    vcvtdq2pd %xmm0, %ymm0
; AVX2-NEXT:    retq
  %cvt = sitofp <8 x i16> %a to <8 x double>
  %shuf = shufflevector <8 x double> %cvt, <8 x double> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  ret <4 x double> %shuf
}

define <4 x double> @sitofp_4i8_to_4f64(<16 x i8> %a) {
; SSE-LABEL: sitofp_4i8_to_4f64:
; SSE:       # BB#0:
; SSE-NEXT:    punpcklbw {{.*#+}} xmm0 = xmm0[0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7]
; SSE-NEXT:    punpcklwd {{.*#+}} xmm1 = xmm1[0],xmm0[0],xmm1[1],xmm0[1],xmm1[2],xmm0[2],xmm1[3],xmm0[3]
; SSE-NEXT:    psrad $24, %xmm1
; SSE-NEXT:    cvtdq2pd %xmm1, %xmm0
; SSE-NEXT:    pshufd {{.*#+}} xmm1 = xmm1[2,3,0,1]
; SSE-NEXT:    cvtdq2pd %xmm1, %xmm1
; SSE-NEXT:    retq
;
; AVX-LABEL: sitofp_4i8_to_4f64:
; AVX:       # BB#0:
; AVX-NEXT:    vpmovsxbd %xmm0, %xmm0
; AVX-NEXT:    vcvtdq2pd %xmm0, %ymm0
; AVX-NEXT:    retq
  %shuf = shufflevector <16 x i8> %a, <16 x i8> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %cvt = sitofp <4 x i8> %shuf to <4 x double>
  ret <4 x double> %cvt
}

define <4 x double> @sitofp_16i8_to_4f64(<16 x i8> %a) {
; SSE-LABEL: sitofp_16i8_to_4f64:
; SSE:       # BB#0:
; SSE-NEXT:    punpcklbw {{.*#+}} xmm0 = xmm0[0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7]
; SSE-NEXT:    punpcklwd {{.*#+}} xmm1 = xmm1[0],xmm0[0],xmm1[1],xmm0[1],xmm1[2],xmm0[2],xmm1[3],xmm0[3]
; SSE-NEXT:    psrad $24, %xmm1
; SSE-NEXT:    cvtdq2pd %xmm1, %xmm0
; SSE-NEXT:    pshufd {{.*#+}} xmm1 = xmm1[2,3,0,1]
; SSE-NEXT:    cvtdq2pd %xmm1, %xmm1
; SSE-NEXT:    retq
;
; AVX1-LABEL: sitofp_16i8_to_4f64:
; AVX1:       # BB#0:
; AVX1-NEXT:    vpmovsxbd %xmm0, %xmm0
; AVX1-NEXT:    vcvtdq2pd %xmm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: sitofp_16i8_to_4f64:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpmovsxbw %xmm0, %ymm0
; AVX2-NEXT:    vpmovsxwd %xmm0, %ymm0
; AVX2-NEXT:    vcvtdq2pd %xmm0, %ymm0
; AVX2-NEXT:    retq
  %cvt = sitofp <16 x i8> %a to <16 x double>
  %shuf = shufflevector <16 x double> %cvt, <16 x double> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  ret <4 x double> %shuf
}

;
; Unsigned Integer to Double
;

define <2 x double> @uitofp_2i64_to_2f64(<2 x i64> %a) {
; SSE-LABEL: uitofp_2i64_to_2f64:
; SSE:       # BB#0:
; SSE-NEXT:    movdqa {{.*#+}} xmm1 = [1127219200,1160773632,0,0]
; SSE-NEXT:    pshufd {{.*#+}} xmm2 = xmm0[2,3,0,1]
; SSE-NEXT:    punpckldq {{.*#+}} xmm0 = xmm0[0],xmm1[0],xmm0[1],xmm1[1]
; SSE-NEXT:    movapd {{.*#+}} xmm3 = [4.503600e+15,1.934281e+25]
; SSE-NEXT:    subpd %xmm3, %xmm0
; SSE-NEXT:    pshufd {{.*#+}} xmm4 = xmm0[2,3,0,1]
; SSE-NEXT:    addpd %xmm4, %xmm0
; SSE-NEXT:    punpckldq {{.*#+}} xmm2 = xmm2[0],xmm1[0],xmm2[1],xmm1[1]
; SSE-NEXT:    subpd %xmm3, %xmm2
; SSE-NEXT:    pshufd {{.*#+}} xmm1 = xmm2[2,3,0,1]
; SSE-NEXT:    addpd %xmm2, %xmm1
; SSE-NEXT:    unpcklpd {{.*#+}} xmm0 = xmm0[0],xmm1[0]
; SSE-NEXT:    retq
;
; AVX-LABEL: uitofp_2i64_to_2f64:
; AVX:       # BB#0:
; AVX-NEXT:    vmovdqa {{.*#+}} xmm1 = [1127219200,1160773632,0,0]
; AVX-NEXT:    vpunpckldq {{.*#+}} xmm2 = xmm0[0],xmm1[0],xmm0[1],xmm1[1]
; AVX-NEXT:    vmovapd {{.*#+}} xmm3 = [4.503600e+15,1.934281e+25]
; AVX-NEXT:    vsubpd %xmm3, %xmm2, %xmm2
; AVX-NEXT:    vhaddpd %xmm2, %xmm2, %xmm2
; AVX-NEXT:    vpshufd {{.*#+}} xmm0 = xmm0[2,3,0,1]
; AVX-NEXT:    vpunpckldq {{.*#+}} xmm0 = xmm0[0],xmm1[0],xmm0[1],xmm1[1]
; AVX-NEXT:    vsubpd %xmm3, %xmm0, %xmm0
; AVX-NEXT:    vhaddpd %xmm0, %xmm0, %xmm0
; AVX-NEXT:    vunpcklpd {{.*#+}} xmm0 = xmm2[0],xmm0[0]
; AVX-NEXT:    retq
  %cvt = uitofp <2 x i64> %a to <2 x double>
  ret <2 x double> %cvt
}

define <2 x double> @uitofp_2i32_to_2f64(<4 x i32> %a) {
; SSE-LABEL: uitofp_2i32_to_2f64:
; SSE:       # BB#0:
; SSE-NEXT:    pxor %xmm1, %xmm1
; SSE-NEXT:    punpckldq {{.*#+}} xmm0 = xmm0[0],xmm1[0],xmm0[1],xmm1[1]
; SSE-NEXT:    movdqa {{.*#+}} xmm1 = [1127219200,1160773632,0,0]
; SSE-NEXT:    pshufd {{.*#+}} xmm2 = xmm0[2,3,0,1]
; SSE-NEXT:    punpckldq {{.*#+}} xmm0 = xmm0[0],xmm1[0],xmm0[1],xmm1[1]
; SSE-NEXT:    movapd {{.*#+}} xmm3 = [4.503600e+15,1.934281e+25]
; SSE-NEXT:    subpd %xmm3, %xmm0
; SSE-NEXT:    pshufd {{.*#+}} xmm4 = xmm0[2,3,0,1]
; SSE-NEXT:    addpd %xmm4, %xmm0
; SSE-NEXT:    punpckldq {{.*#+}} xmm2 = xmm2[0],xmm1[0],xmm2[1],xmm1[1]
; SSE-NEXT:    subpd %xmm3, %xmm2
; SSE-NEXT:    pshufd {{.*#+}} xmm1 = xmm2[2,3,0,1]
; SSE-NEXT:    addpd %xmm2, %xmm1
; SSE-NEXT:    unpcklpd {{.*#+}} xmm0 = xmm0[0],xmm1[0]
; SSE-NEXT:    retq
;
; AVX-LABEL: uitofp_2i32_to_2f64:
; AVX:       # BB#0:
; AVX-NEXT:    vpmovzxdq {{.*#+}} xmm0 = xmm0[0],zero,xmm0[1],zero
; AVX-NEXT:    vmovdqa {{.*#+}} xmm1 = [1127219200,1160773632,0,0]
; AVX-NEXT:    vpunpckldq {{.*#+}} xmm2 = xmm0[0],xmm1[0],xmm0[1],xmm1[1]
; AVX-NEXT:    vmovapd {{.*#+}} xmm3 = [4.503600e+15,1.934281e+25]
; AVX-NEXT:    vsubpd %xmm3, %xmm2, %xmm2
; AVX-NEXT:    vhaddpd %xmm2, %xmm2, %xmm2
; AVX-NEXT:    vpshufd {{.*#+}} xmm0 = xmm0[2,3,0,1]
; AVX-NEXT:    vpunpckldq {{.*#+}} xmm0 = xmm0[0],xmm1[0],xmm0[1],xmm1[1]
; AVX-NEXT:    vsubpd %xmm3, %xmm0, %xmm0
; AVX-NEXT:    vhaddpd %xmm0, %xmm0, %xmm0
; AVX-NEXT:    vunpcklpd {{.*#+}} xmm0 = xmm2[0],xmm0[0]
; AVX-NEXT:    retq
  %shuf = shufflevector <4 x i32> %a, <4 x i32> undef, <2 x i32> <i32 0, i32 1>
  %cvt = uitofp <2 x i32> %shuf to <2 x double>
  ret <2 x double> %cvt
}

define <2 x double> @uitofp_4i32_to_2f64(<4 x i32> %a) {
; SSE-LABEL: uitofp_4i32_to_2f64:
; SSE:       # BB#0:
; SSE-NEXT:    pxor %xmm1, %xmm1
; SSE-NEXT:    punpckldq {{.*#+}} xmm0 = xmm0[0],xmm1[0],xmm0[1],xmm1[1]
; SSE-NEXT:    movdqa {{.*#+}} xmm1 = [1127219200,1160773632,0,0]
; SSE-NEXT:    pshufd {{.*#+}} xmm2 = xmm0[2,3,0,1]
; SSE-NEXT:    punpckldq {{.*#+}} xmm0 = xmm0[0],xmm1[0],xmm0[1],xmm1[1]
; SSE-NEXT:    movapd {{.*#+}} xmm3 = [4.503600e+15,1.934281e+25]
; SSE-NEXT:    subpd %xmm3, %xmm0
; SSE-NEXT:    pshufd {{.*#+}} xmm4 = xmm0[2,3,0,1]
; SSE-NEXT:    addpd %xmm4, %xmm0
; SSE-NEXT:    punpckldq {{.*#+}} xmm2 = xmm2[0],xmm1[0],xmm2[1],xmm1[1]
; SSE-NEXT:    subpd %xmm3, %xmm2
; SSE-NEXT:    pshufd {{.*#+}} xmm1 = xmm2[2,3,0,1]
; SSE-NEXT:    addpd %xmm2, %xmm1
; SSE-NEXT:    unpcklpd {{.*#+}} xmm0 = xmm0[0],xmm1[0]
; SSE-NEXT:    retq
;
; AVX1-LABEL: uitofp_4i32_to_2f64:
; AVX1:       # BB#0:
; AVX1-NEXT:    vpand {{.*}}(%rip), %xmm0, %xmm1
; AVX1-NEXT:    vcvtdq2pd %xmm1, %ymm1
; AVX1-NEXT:    vpsrld $16, %xmm0, %xmm0
; AVX1-NEXT:    vcvtdq2pd %xmm0, %ymm0
; AVX1-NEXT:    vmulpd {{.*}}(%rip), %ymm0, %ymm0
; AVX1-NEXT:    vaddpd %ymm1, %ymm0, %ymm0
; AVX1-NEXT:    vzeroupper
; AVX1-NEXT:    retq
;
; AVX2-LABEL: uitofp_4i32_to_2f64:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpsrld $16, %xmm0, %xmm1
; AVX2-NEXT:    vcvtdq2pd %xmm1, %ymm1
; AVX2-NEXT:    vbroadcastsd {{.*}}(%rip), %ymm2
; AVX2-NEXT:    vmulpd %ymm2, %ymm1, %ymm1
; AVX2-NEXT:    vpbroadcastd {{.*}}(%rip), %xmm2
; AVX2-NEXT:    vpand %xmm2, %xmm0, %xmm0
; AVX2-NEXT:    vcvtdq2pd %xmm0, %ymm0
; AVX2-NEXT:    vaddpd %ymm0, %ymm1, %ymm0
; AVX2-NEXT:    vzeroupper
; AVX2-NEXT:    retq
  %cvt = uitofp <4 x i32> %a to <4 x double>
  %shuf = shufflevector <4 x double> %cvt, <4 x double> undef, <2 x i32> <i32 0, i32 1>
  ret <2 x double> %shuf
}

define <2 x double> @uitofp_2i16_to_2f64(<8 x i16> %a) {
; SSE-LABEL: uitofp_2i16_to_2f64:
; SSE:       # BB#0:
; SSE-NEXT:    pxor %xmm1, %xmm1
; SSE-NEXT:    punpcklwd {{.*#+}} xmm0 = xmm0[0],xmm1[0],xmm0[1],xmm1[1],xmm0[2],xmm1[2],xmm0[3],xmm1[3]
; SSE-NEXT:    cvtdq2pd %xmm0, %xmm0
; SSE-NEXT:    retq
;
; AVX-LABEL: uitofp_2i16_to_2f64:
; AVX:       # BB#0:
; AVX-NEXT:    vpmovzxwd {{.*#+}} xmm0 = xmm0[0],zero,xmm0[1],zero,xmm0[2],zero,xmm0[3],zero
; AVX-NEXT:    vcvtdq2pd %xmm0, %xmm0
; AVX-NEXT:    retq
  %shuf = shufflevector <8 x i16> %a, <8 x i16> undef, <2 x i32> <i32 0, i32 1>
  %cvt = uitofp <2 x i16> %shuf to <2 x double>
  ret <2 x double> %cvt
}

define <2 x double> @uitofp_8i16_to_2f64(<8 x i16> %a) {
; SSE-LABEL: uitofp_8i16_to_2f64:
; SSE:       # BB#0:
; SSE-NEXT:    pxor %xmm1, %xmm1
; SSE-NEXT:    punpcklwd {{.*#+}} xmm0 = xmm0[0],xmm1[0],xmm0[1],xmm1[1],xmm0[2],xmm1[2],xmm0[3],xmm1[3]
; SSE-NEXT:    cvtdq2pd %xmm0, %xmm0
; SSE-NEXT:    retq
;
; AVX1-LABEL: uitofp_8i16_to_2f64:
; AVX1:       # BB#0:
; AVX1-NEXT:    vpmovzxwd {{.*#+}} xmm0 = xmm0[0],zero,xmm0[1],zero,xmm0[2],zero,xmm0[3],zero
; AVX1-NEXT:    vcvtdq2pd %xmm0, %ymm0
; AVX1-NEXT:    vzeroupper
; AVX1-NEXT:    retq
;
; AVX2-LABEL: uitofp_8i16_to_2f64:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpmovzxwd {{.*#+}} ymm0 = xmm0[0],zero,xmm0[1],zero,xmm0[2],zero,xmm0[3],zero,xmm0[4],zero,xmm0[5],zero,xmm0[6],zero,xmm0[7],zero
; AVX2-NEXT:    vcvtdq2pd %xmm0, %ymm0
; AVX2-NEXT:    vzeroupper
; AVX2-NEXT:    retq
  %cvt = uitofp <8 x i16> %a to <8 x double>
  %shuf = shufflevector <8 x double> %cvt, <8 x double> undef, <2 x i32> <i32 0, i32 1>
  ret <2 x double> %shuf
}

define <2 x double> @uitofp_2i8_to_2f64(<16 x i8> %a) {
; SSE-LABEL: uitofp_2i8_to_2f64:
; SSE:       # BB#0:
; SSE-NEXT:    pxor %xmm1, %xmm1
; SSE-NEXT:    punpcklbw {{.*#+}} xmm0 = xmm0[0],xmm1[0],xmm0[1],xmm1[1],xmm0[2],xmm1[2],xmm0[3],xmm1[3],xmm0[4],xmm1[4],xmm0[5],xmm1[5],xmm0[6],xmm1[6],xmm0[7],xmm1[7]
; SSE-NEXT:    punpcklwd {{.*#+}} xmm0 = xmm0[0],xmm1[0],xmm0[1],xmm1[1],xmm0[2],xmm1[2],xmm0[3],xmm1[3]
; SSE-NEXT:    cvtdq2pd %xmm0, %xmm0
; SSE-NEXT:    retq
;
; AVX-LABEL: uitofp_2i8_to_2f64:
; AVX:       # BB#0:
; AVX-NEXT:    vpmovzxbd {{.*#+}} xmm0 = xmm0[0],zero,zero,zero,xmm0[1],zero,zero,zero,xmm0[2],zero,zero,zero,xmm0[3],zero,zero,zero
; AVX-NEXT:    vcvtdq2pd %xmm0, %xmm0
; AVX-NEXT:    retq
  %shuf = shufflevector <16 x i8> %a, <16 x i8> undef, <2 x i32> <i32 0, i32 1>
  %cvt = uitofp <2 x i8> %shuf to <2 x double>
  ret <2 x double> %cvt
}

define <2 x double> @uitofp_16i8_to_2f64(<16 x i8> %a) {
; SSE-LABEL: uitofp_16i8_to_2f64:
; SSE:       # BB#0:
; SSE-NEXT:    pxor %xmm1, %xmm1
; SSE-NEXT:    punpcklbw {{.*#+}} xmm0 = xmm0[0],xmm1[0],xmm0[1],xmm1[1],xmm0[2],xmm1[2],xmm0[3],xmm1[3],xmm0[4],xmm1[4],xmm0[5],xmm1[5],xmm0[6],xmm1[6],xmm0[7],xmm1[7]
; SSE-NEXT:    punpcklwd {{.*#+}} xmm0 = xmm0[0],xmm1[0],xmm0[1],xmm1[1],xmm0[2],xmm1[2],xmm0[3],xmm1[3]
; SSE-NEXT:    cvtdq2pd %xmm0, %xmm0
; SSE-NEXT:    retq
;
; AVX1-LABEL: uitofp_16i8_to_2f64:
; AVX1:       # BB#0:
; AVX1-NEXT:    vpmovzxbd {{.*#+}} xmm0 = xmm0[0],zero,zero,zero,xmm0[1],zero,zero,zero,xmm0[2],zero,zero,zero,xmm0[3],zero,zero,zero
; AVX1-NEXT:    vcvtdq2pd %xmm0, %ymm0
; AVX1-NEXT:    vzeroupper
; AVX1-NEXT:    retq
;
; AVX2-LABEL: uitofp_16i8_to_2f64:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpmovzxbw {{.*#+}} ymm0 = xmm0[0],zero,xmm0[1],zero,xmm0[2],zero,xmm0[3],zero,xmm0[4],zero,xmm0[5],zero,xmm0[6],zero,xmm0[7],zero,xmm0[8],zero,xmm0[9],zero,xmm0[10],zero,xmm0[11],zero,xmm0[12],zero,xmm0[13],zero,xmm0[14],zero,xmm0[15],zero
; AVX2-NEXT:    vpmovzxwd {{.*#+}} ymm0 = xmm0[0],zero,xmm0[1],zero,xmm0[2],zero,xmm0[3],zero,xmm0[4],zero,xmm0[5],zero,xmm0[6],zero,xmm0[7],zero
; AVX2-NEXT:    vcvtdq2pd %xmm0, %ymm0
; AVX2-NEXT:    vzeroupper
; AVX2-NEXT:    retq
  %cvt = uitofp <16 x i8> %a to <16 x double>
  %shuf = shufflevector <16 x double> %cvt, <16 x double> undef, <2 x i32> <i32 0, i32 1>
  ret <2 x double> %shuf
}

define <4 x double> @uitofp_4i64_to_4f64(<4 x i64> %a) {
; SSE-LABEL: uitofp_4i64_to_4f64:
; SSE:       # BB#0:
; SSE-NEXT:    movdqa {{.*#+}} xmm2 = [1127219200,1160773632,0,0]
; SSE-NEXT:    pshufd {{.*#+}} xmm3 = xmm0[2,3,0,1]
; SSE-NEXT:    punpckldq {{.*#+}} xmm0 = xmm0[0],xmm2[0],xmm0[1],xmm2[1]
; SSE-NEXT:    movapd {{.*#+}} xmm4 = [4.503600e+15,1.934281e+25]
; SSE-NEXT:    subpd %xmm4, %xmm0
; SSE-NEXT:    pshufd {{.*#+}} xmm5 = xmm0[2,3,0,1]
; SSE-NEXT:    addpd %xmm5, %xmm0
; SSE-NEXT:    punpckldq {{.*#+}} xmm3 = xmm3[0],xmm2[0],xmm3[1],xmm2[1]
; SSE-NEXT:    subpd %xmm4, %xmm3
; SSE-NEXT:    pshufd {{.*#+}} xmm5 = xmm3[2,3,0,1]
; SSE-NEXT:    addpd %xmm3, %xmm5
; SSE-NEXT:    unpcklpd {{.*#+}} xmm0 = xmm0[0],xmm5[0]
; SSE-NEXT:    pshufd {{.*#+}} xmm3 = xmm1[2,3,0,1]
; SSE-NEXT:    punpckldq {{.*#+}} xmm1 = xmm1[0],xmm2[0],xmm1[1],xmm2[1]
; SSE-NEXT:    subpd %xmm4, %xmm1
; SSE-NEXT:    pshufd {{.*#+}} xmm5 = xmm1[2,3,0,1]
; SSE-NEXT:    addpd %xmm5, %xmm1
; SSE-NEXT:    punpckldq {{.*#+}} xmm3 = xmm3[0],xmm2[0],xmm3[1],xmm2[1]
; SSE-NEXT:    subpd %xmm4, %xmm3
; SSE-NEXT:    pshufd {{.*#+}} xmm2 = xmm3[2,3,0,1]
; SSE-NEXT:    addpd %xmm3, %xmm2
; SSE-NEXT:    unpcklpd {{.*#+}} xmm1 = xmm1[0],xmm2[0]
; SSE-NEXT:    retq
;
; AVX1-LABEL: uitofp_4i64_to_4f64:
; AVX1:       # BB#0:
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm1
; AVX1-NEXT:    vmovdqa {{.*#+}} xmm2 = [1127219200,1160773632,0,0]
; AVX1-NEXT:    vpunpckldq {{.*#+}} xmm3 = xmm1[0],xmm2[0],xmm1[1],xmm2[1]
; AVX1-NEXT:    vmovapd {{.*#+}} xmm4 = [4.503600e+15,1.934281e+25]
; AVX1-NEXT:    vsubpd %xmm4, %xmm3, %xmm3
; AVX1-NEXT:    vhaddpd %xmm3, %xmm3, %xmm3
; AVX1-NEXT:    vpshufd {{.*#+}} xmm1 = xmm1[2,3,0,1]
; AVX1-NEXT:    vpunpckldq {{.*#+}} xmm1 = xmm1[0],xmm2[0],xmm1[1],xmm2[1]
; AVX1-NEXT:    vsubpd %xmm4, %xmm1, %xmm1
; AVX1-NEXT:    vhaddpd %xmm1, %xmm1, %xmm1
; AVX1-NEXT:    vunpcklpd {{.*#+}} xmm1 = xmm3[0],xmm1[0]
; AVX1-NEXT:    vpunpckldq {{.*#+}} xmm3 = xmm0[0],xmm2[0],xmm0[1],xmm2[1]
; AVX1-NEXT:    vsubpd %xmm4, %xmm3, %xmm3
; AVX1-NEXT:    vhaddpd %xmm3, %xmm3, %xmm3
; AVX1-NEXT:    vpshufd {{.*#+}} xmm0 = xmm0[2,3,0,1]
; AVX1-NEXT:    vpunpckldq {{.*#+}} xmm0 = xmm0[0],xmm2[0],xmm0[1],xmm2[1]
; AVX1-NEXT:    vsubpd %xmm4, %xmm0, %xmm0
; AVX1-NEXT:    vhaddpd %xmm0, %xmm0, %xmm0
; AVX1-NEXT:    vunpcklpd {{.*#+}} xmm0 = xmm3[0],xmm0[0]
; AVX1-NEXT:    vinsertf128 $1, %xmm1, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: uitofp_4i64_to_4f64:
; AVX2:       # BB#0:
; AVX2-NEXT:    vextracti128 $1, %ymm0, %xmm1
; AVX2-NEXT:    vmovdqa {{.*#+}} xmm2 = [1127219200,1160773632,0,0]
; AVX2-NEXT:    vpunpckldq {{.*#+}} xmm3 = xmm1[0],xmm2[0],xmm1[1],xmm2[1]
; AVX2-NEXT:    vmovapd {{.*#+}} xmm4 = [4.503600e+15,1.934281e+25]
; AVX2-NEXT:    vsubpd %xmm4, %xmm3, %xmm3
; AVX2-NEXT:    vhaddpd %xmm3, %xmm3, %xmm3
; AVX2-NEXT:    vpshufd {{.*#+}} xmm1 = xmm1[2,3,0,1]
; AVX2-NEXT:    vpunpckldq {{.*#+}} xmm1 = xmm1[0],xmm2[0],xmm1[1],xmm2[1]
; AVX2-NEXT:    vsubpd %xmm4, %xmm1, %xmm1
; AVX2-NEXT:    vhaddpd %xmm1, %xmm1, %xmm1
; AVX2-NEXT:    vunpcklpd {{.*#+}} xmm1 = xmm3[0],xmm1[0]
; AVX2-NEXT:    vpunpckldq {{.*#+}} xmm3 = xmm0[0],xmm2[0],xmm0[1],xmm2[1]
; AVX2-NEXT:    vsubpd %xmm4, %xmm3, %xmm3
; AVX2-NEXT:    vhaddpd %xmm3, %xmm3, %xmm3
; AVX2-NEXT:    vpshufd {{.*#+}} xmm0 = xmm0[2,3,0,1]
; AVX2-NEXT:    vpunpckldq {{.*#+}} xmm0 = xmm0[0],xmm2[0],xmm0[1],xmm2[1]
; AVX2-NEXT:    vsubpd %xmm4, %xmm0, %xmm0
; AVX2-NEXT:    vhaddpd %xmm0, %xmm0, %xmm0
; AVX2-NEXT:    vunpcklpd {{.*#+}} xmm0 = xmm3[0],xmm0[0]
; AVX2-NEXT:    vinsertf128 $1, %xmm1, %ymm0, %ymm0
; AVX2-NEXT:    retq
  %cvt = uitofp <4 x i64> %a to <4 x double>
  ret <4 x double> %cvt
}

define <4 x double> @uitofp_4i32_to_4f64(<4 x i32> %a) {
; SSE-LABEL: uitofp_4i32_to_4f64:
; SSE:       # BB#0:
; SSE-NEXT:    movdqa %xmm0, %xmm2
; SSE-NEXT:    pxor %xmm1, %xmm1
; SSE-NEXT:    punpckldq {{.*#+}} xmm0 = xmm0[0],xmm1[0],xmm0[1],xmm1[1]
; SSE-NEXT:    movdqa {{.*#+}} xmm3 = [1127219200,1160773632,0,0]
; SSE-NEXT:    pshufd {{.*#+}} xmm4 = xmm0[2,3,0,1]
; SSE-NEXT:    punpckldq {{.*#+}} xmm0 = xmm0[0],xmm3[0],xmm0[1],xmm3[1]
; SSE-NEXT:    movapd {{.*#+}} xmm5 = [4.503600e+15,1.934281e+25]
; SSE-NEXT:    subpd %xmm5, %xmm0
; SSE-NEXT:    pshufd {{.*#+}} xmm6 = xmm0[2,3,0,1]
; SSE-NEXT:    addpd %xmm6, %xmm0
; SSE-NEXT:    punpckldq {{.*#+}} xmm4 = xmm4[0],xmm3[0],xmm4[1],xmm3[1]
; SSE-NEXT:    subpd %xmm5, %xmm4
; SSE-NEXT:    pshufd {{.*#+}} xmm6 = xmm4[2,3,0,1]
; SSE-NEXT:    addpd %xmm4, %xmm6
; SSE-NEXT:    unpcklpd {{.*#+}} xmm0 = xmm0[0],xmm6[0]
; SSE-NEXT:    punpckhdq {{.*#+}} xmm2 = xmm2[2],xmm1[2],xmm2[3],xmm1[3]
; SSE-NEXT:    pshufd {{.*#+}} xmm4 = xmm2[2,3,0,1]
; SSE-NEXT:    punpckldq {{.*#+}} xmm2 = xmm2[0],xmm3[0],xmm2[1],xmm3[1]
; SSE-NEXT:    subpd %xmm5, %xmm2
; SSE-NEXT:    pshufd {{.*#+}} xmm1 = xmm2[2,3,0,1]
; SSE-NEXT:    addpd %xmm2, %xmm1
; SSE-NEXT:    punpckldq {{.*#+}} xmm4 = xmm4[0],xmm3[0],xmm4[1],xmm3[1]
; SSE-NEXT:    subpd %xmm5, %xmm4
; SSE-NEXT:    pshufd {{.*#+}} xmm2 = xmm4[2,3,0,1]
; SSE-NEXT:    addpd %xmm4, %xmm2
; SSE-NEXT:    unpcklpd {{.*#+}} xmm1 = xmm1[0],xmm2[0]
; SSE-NEXT:    retq
;
; AVX1-LABEL: uitofp_4i32_to_4f64:
; AVX1:       # BB#0:
; AVX1-NEXT:    vpand {{.*}}(%rip), %xmm0, %xmm1
; AVX1-NEXT:    vcvtdq2pd %xmm1, %ymm1
; AVX1-NEXT:    vpsrld $16, %xmm0, %xmm0
; AVX1-NEXT:    vcvtdq2pd %xmm0, %ymm0
; AVX1-NEXT:    vmulpd {{.*}}(%rip), %ymm0, %ymm0
; AVX1-NEXT:    vaddpd %ymm1, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: uitofp_4i32_to_4f64:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpsrld $16, %xmm0, %xmm1
; AVX2-NEXT:    vcvtdq2pd %xmm1, %ymm1
; AVX2-NEXT:    vbroadcastsd {{.*}}(%rip), %ymm2
; AVX2-NEXT:    vmulpd %ymm2, %ymm1, %ymm1
; AVX2-NEXT:    vpbroadcastd {{.*}}(%rip), %xmm2
; AVX2-NEXT:    vpand %xmm2, %xmm0, %xmm0
; AVX2-NEXT:    vcvtdq2pd %xmm0, %ymm0
; AVX2-NEXT:    vaddpd %ymm0, %ymm1, %ymm0
; AVX2-NEXT:    retq
  %cvt = uitofp <4 x i32> %a to <4 x double>
  ret <4 x double> %cvt
}

define <4 x double> @uitofp_4i16_to_4f64(<8 x i16> %a) {
; SSE-LABEL: uitofp_4i16_to_4f64:
; SSE:       # BB#0:
; SSE-NEXT:    pxor %xmm1, %xmm1
; SSE-NEXT:    punpcklwd {{.*#+}} xmm0 = xmm0[0],xmm1[0],xmm0[1],xmm1[1],xmm0[2],xmm1[2],xmm0[3],xmm1[3]
; SSE-NEXT:    cvtdq2pd %xmm0, %xmm2
; SSE-NEXT:    pshufd {{.*#+}} xmm0 = xmm0[2,3,0,1]
; SSE-NEXT:    cvtdq2pd %xmm0, %xmm1
; SSE-NEXT:    movaps %xmm2, %xmm0
; SSE-NEXT:    retq
;
; AVX-LABEL: uitofp_4i16_to_4f64:
; AVX:       # BB#0:
; AVX-NEXT:    vpmovzxwd {{.*#+}} xmm0 = xmm0[0],zero,xmm0[1],zero,xmm0[2],zero,xmm0[3],zero
; AVX-NEXT:    vcvtdq2pd %xmm0, %ymm0
; AVX-NEXT:    retq
  %shuf = shufflevector <8 x i16> %a, <8 x i16> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %cvt = uitofp <4 x i16> %shuf to <4 x double>
  ret <4 x double> %cvt
}

define <4 x double> @uitofp_8i16_to_4f64(<8 x i16> %a) {
; SSE-LABEL: uitofp_8i16_to_4f64:
; SSE:       # BB#0:
; SSE-NEXT:    pxor %xmm1, %xmm1
; SSE-NEXT:    punpcklwd {{.*#+}} xmm0 = xmm0[0],xmm1[0],xmm0[1],xmm1[1],xmm0[2],xmm1[2],xmm0[3],xmm1[3]
; SSE-NEXT:    cvtdq2pd %xmm0, %xmm2
; SSE-NEXT:    pshufd {{.*#+}} xmm0 = xmm0[2,3,0,1]
; SSE-NEXT:    cvtdq2pd %xmm0, %xmm1
; SSE-NEXT:    movaps %xmm2, %xmm0
; SSE-NEXT:    retq
;
; AVX1-LABEL: uitofp_8i16_to_4f64:
; AVX1:       # BB#0:
; AVX1-NEXT:    vpmovzxwd {{.*#+}} xmm0 = xmm0[0],zero,xmm0[1],zero,xmm0[2],zero,xmm0[3],zero
; AVX1-NEXT:    vcvtdq2pd %xmm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: uitofp_8i16_to_4f64:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpmovzxwd {{.*#+}} ymm0 = xmm0[0],zero,xmm0[1],zero,xmm0[2],zero,xmm0[3],zero,xmm0[4],zero,xmm0[5],zero,xmm0[6],zero,xmm0[7],zero
; AVX2-NEXT:    vcvtdq2pd %xmm0, %ymm0
; AVX2-NEXT:    retq
  %cvt = uitofp <8 x i16> %a to <8 x double>
  %shuf = shufflevector <8 x double> %cvt, <8 x double> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  ret <4 x double> %shuf
}

define <4 x double> @uitofp_4i8_to_4f64(<16 x i8> %a) {
; SSE-LABEL: uitofp_4i8_to_4f64:
; SSE:       # BB#0:
; SSE-NEXT:    pxor %xmm1, %xmm1
; SSE-NEXT:    punpcklbw {{.*#+}} xmm0 = xmm0[0],xmm1[0],xmm0[1],xmm1[1],xmm0[2],xmm1[2],xmm0[3],xmm1[3],xmm0[4],xmm1[4],xmm0[5],xmm1[5],xmm0[6],xmm1[6],xmm0[7],xmm1[7]
; SSE-NEXT:    punpcklwd {{.*#+}} xmm0 = xmm0[0],xmm1[0],xmm0[1],xmm1[1],xmm0[2],xmm1[2],xmm0[3],xmm1[3]
; SSE-NEXT:    cvtdq2pd %xmm0, %xmm2
; SSE-NEXT:    pshufd {{.*#+}} xmm0 = xmm0[2,3,0,1]
; SSE-NEXT:    cvtdq2pd %xmm0, %xmm1
; SSE-NEXT:    movaps %xmm2, %xmm0
; SSE-NEXT:    retq
;
; AVX-LABEL: uitofp_4i8_to_4f64:
; AVX:       # BB#0:
; AVX-NEXT:    vpmovzxbd {{.*#+}} xmm0 = xmm0[0],zero,zero,zero,xmm0[1],zero,zero,zero,xmm0[2],zero,zero,zero,xmm0[3],zero,zero,zero
; AVX-NEXT:    vcvtdq2pd %xmm0, %ymm0
; AVX-NEXT:    retq
  %shuf = shufflevector <16 x i8> %a, <16 x i8> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %cvt = uitofp <4 x i8> %shuf to <4 x double>
  ret <4 x double> %cvt
}

define <4 x double> @uitofp_16i8_to_4f64(<16 x i8> %a) {
; SSE-LABEL: uitofp_16i8_to_4f64:
; SSE:       # BB#0:
; SSE-NEXT:    pxor %xmm1, %xmm1
; SSE-NEXT:    punpcklbw {{.*#+}} xmm0 = xmm0[0],xmm1[0],xmm0[1],xmm1[1],xmm0[2],xmm1[2],xmm0[3],xmm1[3],xmm0[4],xmm1[4],xmm0[5],xmm1[5],xmm0[6],xmm1[6],xmm0[7],xmm1[7]
; SSE-NEXT:    punpcklwd {{.*#+}} xmm0 = xmm0[0],xmm1[0],xmm0[1],xmm1[1],xmm0[2],xmm1[2],xmm0[3],xmm1[3]
; SSE-NEXT:    cvtdq2pd %xmm0, %xmm2
; SSE-NEXT:    pshufd {{.*#+}} xmm0 = xmm0[2,3,0,1]
; SSE-NEXT:    cvtdq2pd %xmm0, %xmm1
; SSE-NEXT:    movaps %xmm2, %xmm0
; SSE-NEXT:    retq
;
; AVX1-LABEL: uitofp_16i8_to_4f64:
; AVX1:       # BB#0:
; AVX1-NEXT:    vpmovzxbd {{.*#+}} xmm0 = xmm0[0],zero,zero,zero,xmm0[1],zero,zero,zero,xmm0[2],zero,zero,zero,xmm0[3],zero,zero,zero
; AVX1-NEXT:    vcvtdq2pd %xmm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: uitofp_16i8_to_4f64:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpmovzxbw {{.*#+}} ymm0 = xmm0[0],zero,xmm0[1],zero,xmm0[2],zero,xmm0[3],zero,xmm0[4],zero,xmm0[5],zero,xmm0[6],zero,xmm0[7],zero,xmm0[8],zero,xmm0[9],zero,xmm0[10],zero,xmm0[11],zero,xmm0[12],zero,xmm0[13],zero,xmm0[14],zero,xmm0[15],zero
; AVX2-NEXT:    vpmovzxwd {{.*#+}} ymm0 = xmm0[0],zero,xmm0[1],zero,xmm0[2],zero,xmm0[3],zero,xmm0[4],zero,xmm0[5],zero,xmm0[6],zero,xmm0[7],zero
; AVX2-NEXT:    vcvtdq2pd %xmm0, %ymm0
; AVX2-NEXT:    retq
  %cvt = uitofp <16 x i8> %a to <16 x double>
  %shuf = shufflevector <16 x double> %cvt, <16 x double> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  ret <4 x double> %shuf
}

;
; Signed Integer to Float
;

define <4 x float> @sitofp_2i64_to_4f32(<2 x i64> %a) {
; SSE-LABEL: sitofp_2i64_to_4f32:
; SSE:       # BB#0:
; SSE-NEXT:    movd %xmm0, %rax
; SSE-NEXT:    cvtsi2ssq %rax, %xmm1
; SSE-NEXT:    pshufd {{.*#+}} xmm0 = xmm0[2,3,0,1]
; SSE-NEXT:    movd %xmm0, %rax
; SSE-NEXT:    xorps %xmm0, %xmm0
; SSE-NEXT:    cvtsi2ssq %rax, %xmm0
; SSE-NEXT:    unpcklps {{.*#+}} xmm1 = xmm1[0],xmm0[0],xmm1[1],xmm0[1]
; SSE-NEXT:    movaps %xmm1, %xmm0
; SSE-NEXT:    retq
;
; AVX-LABEL: sitofp_2i64_to_4f32:
; AVX:       # BB#0:
; AVX-NEXT:    vpextrq $1, %xmm0, %rax
; AVX-NEXT:    vcvtsi2ssq %rax, %xmm0, %xmm1
; AVX-NEXT:    vmovq %xmm0, %rax
; AVX-NEXT:    vxorps %xmm0, %xmm0, %xmm0
; AVX-NEXT:    vcvtsi2ssq %rax, %xmm0, %xmm0
; AVX-NEXT:    vinsertps {{.*#+}} xmm0 = xmm0[0],xmm1[0],xmm0[2,3]
; AVX-NEXT:    vcvtsi2ssq %rax, %xmm0, %xmm1
; AVX-NEXT:    vinsertps {{.*#+}} xmm0 = xmm0[0,1],xmm1[0],xmm0[3]
; AVX-NEXT:    vinsertps {{.*#+}} xmm0 = xmm0[0,1,2],xmm1[0]
; AVX-NEXT:    retq
  %cvt = sitofp <2 x i64> %a to <2 x float>
  %ext = shufflevector <2 x float> %cvt, <2 x float> undef, <4 x i32> <i32 0, i32 1, i32 undef, i32 undef>
  ret <4 x float> %ext
}

define <4 x float> @sitofp_4i64_to_4f32_undef(<2 x i64> %a) {
; SSE-LABEL: sitofp_4i64_to_4f32_undef:
; SSE:       # BB#0:
; SSE-NEXT:    cvtsi2ssq %rax, %xmm2
; SSE-NEXT:    movd %xmm0, %rax
; SSE-NEXT:    cvtsi2ssq %rax, %xmm1
; SSE-NEXT:    unpcklps {{.*#+}} xmm1 = xmm1[0],xmm2[0],xmm1[1],xmm2[1]
; SSE-NEXT:    pshufd {{.*#+}} xmm0 = xmm0[2,3,0,1]
; SSE-NEXT:    movd %xmm0, %rax
; SSE-NEXT:    xorps %xmm0, %xmm0
; SSE-NEXT:    cvtsi2ssq %rax, %xmm0
; SSE-NEXT:    unpcklps {{.*#+}} xmm0 = xmm0[0],xmm2[0],xmm0[1],xmm2[1]
; SSE-NEXT:    unpcklps {{.*#+}} xmm1 = xmm1[0],xmm0[0],xmm1[1],xmm0[1]
; SSE-NEXT:    movaps %xmm1, %xmm0
; SSE-NEXT:    retq
;
; AVX-LABEL: sitofp_4i64_to_4f32_undef:
; AVX:       # BB#0:
; AVX-NEXT:    vpextrq $1, %xmm0, %rax
; AVX-NEXT:    vcvtsi2ssq %rax, %xmm0, %xmm1
; AVX-NEXT:    vmovq %xmm0, %rax
; AVX-NEXT:    vxorps %xmm0, %xmm0, %xmm0
; AVX-NEXT:    vcvtsi2ssq %rax, %xmm0, %xmm0
; AVX-NEXT:    vinsertps {{.*#+}} xmm0 = xmm0[0],xmm1[0],xmm0[2,3]
; AVX-NEXT:    vcvtsi2ssq %rax, %xmm0, %xmm1
; AVX-NEXT:    vinsertps {{.*#+}} xmm0 = xmm0[0,1],xmm1[0],xmm0[3]
; AVX-NEXT:    vinsertps {{.*#+}} xmm0 = xmm0[0,1,2],xmm1[0]
; AVX-NEXT:    retq
  %ext = shufflevector <2 x i64> %a, <2 x i64> undef, <4 x i32> <i32 0, i32 1, i32 undef, i32 undef>
  %cvt = sitofp <4 x i64> %ext to <4 x float>
  ret <4 x float> %cvt
}

define <4 x float> @sitofp_4i32_to_4f32(<4 x i32> %a) {
; SSE-LABEL: sitofp_4i32_to_4f32:
; SSE:       # BB#0:
; SSE-NEXT:    cvtdq2ps %xmm0, %xmm0
; SSE-NEXT:    retq
;
; AVX-LABEL: sitofp_4i32_to_4f32:
; AVX:       # BB#0:
; AVX-NEXT:    vcvtdq2ps %xmm0, %xmm0
; AVX-NEXT:    retq
  %cvt = sitofp <4 x i32> %a to <4 x float>
  ret <4 x float> %cvt
}

define <4 x float> @sitofp_4i16_to_4f32(<8 x i16> %a) {
; SSE-LABEL: sitofp_4i16_to_4f32:
; SSE:       # BB#0:
; SSE-NEXT:    punpcklwd {{.*#+}} xmm0 = xmm0[0,0,1,1,2,2,3,3]
; SSE-NEXT:    psrad $16, %xmm0
; SSE-NEXT:    cvtdq2ps %xmm0, %xmm0
; SSE-NEXT:    retq
;
; AVX-LABEL: sitofp_4i16_to_4f32:
; AVX:       # BB#0:
; AVX-NEXT:    vpmovsxwd %xmm0, %xmm0
; AVX-NEXT:    vcvtdq2ps %xmm0, %xmm0
; AVX-NEXT:    retq
  %shuf = shufflevector <8 x i16> %a, <8 x i16> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %cvt = sitofp <4 x i16> %shuf to <4 x float>
  ret <4 x float> %cvt
}

define <4 x float> @sitofp_8i16_to_4f32(<8 x i16> %a) {
; SSE-LABEL: sitofp_8i16_to_4f32:
; SSE:       # BB#0:
; SSE-NEXT:    punpcklwd {{.*#+}} xmm0 = xmm0[0,0,1,1,2,2,3,3]
; SSE-NEXT:    psrad $16, %xmm0
; SSE-NEXT:    cvtdq2ps %xmm0, %xmm0
; SSE-NEXT:    retq
;
; AVX1-LABEL: sitofp_8i16_to_4f32:
; AVX1:       # BB#0:
; AVX1-NEXT:    vpmovsxwd %xmm0, %xmm1
; AVX1-NEXT:    vpshufd {{.*#+}} xmm0 = xmm0[2,3,0,1]
; AVX1-NEXT:    vpmovsxwd %xmm0, %xmm0
; AVX1-NEXT:    vinsertf128 $1, %xmm0, %ymm1, %ymm0
; AVX1-NEXT:    vcvtdq2ps %ymm0, %ymm0
; AVX1-NEXT:    vzeroupper
; AVX1-NEXT:    retq
;
; AVX2-LABEL: sitofp_8i16_to_4f32:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpmovsxwd %xmm0, %ymm0
; AVX2-NEXT:    vcvtdq2ps %ymm0, %ymm0
; AVX2-NEXT:    vzeroupper
; AVX2-NEXT:    retq
  %cvt = sitofp <8 x i16> %a to <8 x float>
  %shuf = shufflevector <8 x float> %cvt, <8 x float> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  ret <4 x float> %shuf
}

define <4 x float> @sitofp_4i8_to_4f32(<16 x i8> %a) {
; SSE-LABEL: sitofp_4i8_to_4f32:
; SSE:       # BB#0:
; SSE-NEXT:    punpcklbw {{.*#+}} xmm0 = xmm0[0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7]
; SSE-NEXT:    punpcklwd {{.*#+}} xmm0 = xmm0[0,0,1,1,2,2,3,3]
; SSE-NEXT:    psrad $24, %xmm0
; SSE-NEXT:    cvtdq2ps %xmm0, %xmm0
; SSE-NEXT:    retq
;
; AVX-LABEL: sitofp_4i8_to_4f32:
; AVX:       # BB#0:
; AVX-NEXT:    vpmovsxbd %xmm0, %xmm0
; AVX-NEXT:    vcvtdq2ps %xmm0, %xmm0
; AVX-NEXT:    retq
  %shuf = shufflevector <16 x i8> %a, <16 x i8> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %cvt = sitofp <4 x i8> %shuf to <4 x float>
  ret <4 x float> %cvt
}

define <4 x float> @sitofp_16i8_to_4f32(<16 x i8> %a) {
; SSE-LABEL: sitofp_16i8_to_4f32:
; SSE:       # BB#0:
; SSE-NEXT:    punpcklbw {{.*#+}} xmm0 = xmm0[0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7]
; SSE-NEXT:    punpcklwd {{.*#+}} xmm0 = xmm0[0,0,1,1,2,2,3,3]
; SSE-NEXT:    psrad $24, %xmm0
; SSE-NEXT:    cvtdq2ps %xmm0, %xmm0
; SSE-NEXT:    retq
;
; AVX1-LABEL: sitofp_16i8_to_4f32:
; AVX1:       # BB#0:
; AVX1-NEXT:    vpmovsxbd %xmm0, %xmm1
; AVX1-NEXT:    vpshufd {{.*#+}} xmm0 = xmm0[1,1,2,3]
; AVX1-NEXT:    vpmovsxbd %xmm0, %xmm0
; AVX1-NEXT:    vinsertf128 $1, %xmm0, %ymm1, %ymm0
; AVX1-NEXT:    vcvtdq2ps %ymm0, %ymm0
; AVX1-NEXT:    vzeroupper
; AVX1-NEXT:    retq
;
; AVX2-LABEL: sitofp_16i8_to_4f32:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpmovsxbw %xmm0, %ymm0
; AVX2-NEXT:    vpmovsxwd %xmm0, %ymm0
; AVX2-NEXT:    vcvtdq2ps %ymm0, %ymm0
; AVX2-NEXT:    vzeroupper
; AVX2-NEXT:    retq
  %cvt = sitofp <16 x i8> %a to <16 x float>
  %shuf = shufflevector <16 x float> %cvt, <16 x float> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  ret <4 x float> %shuf
}

define <4 x float> @sitofp_4i64_to_4f32(<4 x i64> %a) {
; SSE-LABEL: sitofp_4i64_to_4f32:
; SSE:       # BB#0:
; SSE-NEXT:    movd %xmm1, %rax
; SSE-NEXT:    cvtsi2ssq %rax, %xmm3
; SSE-NEXT:    movd %xmm0, %rax
; SSE-NEXT:    cvtsi2ssq %rax, %xmm2
; SSE-NEXT:    unpcklps {{.*#+}} xmm2 = xmm2[0],xmm3[0],xmm2[1],xmm3[1]
; SSE-NEXT:    pshufd {{.*#+}} xmm1 = xmm1[2,3,0,1]
; SSE-NEXT:    movd %xmm1, %rax
; SSE-NEXT:    xorps %xmm1, %xmm1
; SSE-NEXT:    cvtsi2ssq %rax, %xmm1
; SSE-NEXT:    pshufd {{.*#+}} xmm0 = xmm0[2,3,0,1]
; SSE-NEXT:    movd %xmm0, %rax
; SSE-NEXT:    xorps %xmm0, %xmm0
; SSE-NEXT:    cvtsi2ssq %rax, %xmm0
; SSE-NEXT:    unpcklps {{.*#+}} xmm0 = xmm0[0],xmm1[0],xmm0[1],xmm1[1]
; SSE-NEXT:    unpcklps {{.*#+}} xmm2 = xmm2[0],xmm0[0],xmm2[1],xmm0[1]
; SSE-NEXT:    movaps %xmm2, %xmm0
; SSE-NEXT:    retq
;
; AVX1-LABEL: sitofp_4i64_to_4f32:
; AVX1:       # BB#0:
; AVX1-NEXT:    vpextrq $1, %xmm0, %rax
; AVX1-NEXT:    vcvtsi2ssq %rax, %xmm0, %xmm1
; AVX1-NEXT:    vmovq %xmm0, %rax
; AVX1-NEXT:    vcvtsi2ssq %rax, %xmm0, %xmm2
; AVX1-NEXT:    vinsertps {{.*#+}} xmm1 = xmm2[0],xmm1[0],xmm2[2,3]
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm0
; AVX1-NEXT:    vmovq %xmm0, %rax
; AVX1-NEXT:    vcvtsi2ssq %rax, %xmm0, %xmm2
; AVX1-NEXT:    vinsertps {{.*#+}} xmm1 = xmm1[0,1],xmm2[0],xmm1[3]
; AVX1-NEXT:    vpextrq $1, %xmm0, %rax
; AVX1-NEXT:    vxorps %xmm0, %xmm0, %xmm0
; AVX1-NEXT:    vcvtsi2ssq %rax, %xmm0, %xmm0
; AVX1-NEXT:    vinsertps {{.*#+}} xmm0 = xmm1[0,1,2],xmm0[0]
; AVX1-NEXT:    vzeroupper
; AVX1-NEXT:    retq
;
; AVX2-LABEL: sitofp_4i64_to_4f32:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpextrq $1, %xmm0, %rax
; AVX2-NEXT:    vcvtsi2ssq %rax, %xmm0, %xmm1
; AVX2-NEXT:    vmovq %xmm0, %rax
; AVX2-NEXT:    vcvtsi2ssq %rax, %xmm0, %xmm2
; AVX2-NEXT:    vinsertps {{.*#+}} xmm1 = xmm2[0],xmm1[0],xmm2[2,3]
; AVX2-NEXT:    vextracti128 $1, %ymm0, %xmm0
; AVX2-NEXT:    vmovq %xmm0, %rax
; AVX2-NEXT:    vcvtsi2ssq %rax, %xmm0, %xmm2
; AVX2-NEXT:    vinsertps {{.*#+}} xmm1 = xmm1[0,1],xmm2[0],xmm1[3]
; AVX2-NEXT:    vpextrq $1, %xmm0, %rax
; AVX2-NEXT:    vxorps %xmm0, %xmm0, %xmm0
; AVX2-NEXT:    vcvtsi2ssq %rax, %xmm0, %xmm0
; AVX2-NEXT:    vinsertps {{.*#+}} xmm0 = xmm1[0,1,2],xmm0[0]
; AVX2-NEXT:    vzeroupper
; AVX2-NEXT:    retq
  %cvt = sitofp <4 x i64> %a to <4 x float>
  ret <4 x float> %cvt
}

define <8 x float> @sitofp_8i32_to_8f32(<8 x i32> %a) {
; SSE-LABEL: sitofp_8i32_to_8f32:
; SSE:       # BB#0:
; SSE-NEXT:    cvtdq2ps %xmm0, %xmm0
; SSE-NEXT:    cvtdq2ps %xmm1, %xmm1
; SSE-NEXT:    retq
;
; AVX-LABEL: sitofp_8i32_to_8f32:
; AVX:       # BB#0:
; AVX-NEXT:    vcvtdq2ps %ymm0, %ymm0
; AVX-NEXT:    retq
  %cvt = sitofp <8 x i32> %a to <8 x float>
  ret <8 x float> %cvt
}

define <8 x float> @sitofp_8i16_to_8f32(<8 x i16> %a) {
; SSE-LABEL: sitofp_8i16_to_8f32:
; SSE:       # BB#0:
; SSE-NEXT:    punpcklwd {{.*#+}} xmm1 = xmm1[0],xmm0[0],xmm1[1],xmm0[1],xmm1[2],xmm0[2],xmm1[3],xmm0[3]
; SSE-NEXT:    psrad $16, %xmm1
; SSE-NEXT:    cvtdq2ps %xmm1, %xmm2
; SSE-NEXT:    pshufd {{.*#+}} xmm0 = xmm0[2,3,0,1]
; SSE-NEXT:    punpcklwd {{.*#+}} xmm0 = xmm0[0,0,1,1,2,2,3,3]
; SSE-NEXT:    psrad $16, %xmm0
; SSE-NEXT:    cvtdq2ps %xmm0, %xmm1
; SSE-NEXT:    movaps %xmm2, %xmm0
; SSE-NEXT:    retq
;
; AVX1-LABEL: sitofp_8i16_to_8f32:
; AVX1:       # BB#0:
; AVX1-NEXT:    vpmovsxwd %xmm0, %xmm1
; AVX1-NEXT:    vpshufd {{.*#+}} xmm0 = xmm0[2,3,0,1]
; AVX1-NEXT:    vpmovsxwd %xmm0, %xmm0
; AVX1-NEXT:    vinsertf128 $1, %xmm0, %ymm1, %ymm0
; AVX1-NEXT:    vcvtdq2ps %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: sitofp_8i16_to_8f32:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpmovsxwd %xmm0, %ymm0
; AVX2-NEXT:    vcvtdq2ps %ymm0, %ymm0
; AVX2-NEXT:    retq
  %cvt = sitofp <8 x i16> %a to <8 x float>
  ret <8 x float> %cvt
}

define <8 x float> @sitofp_8i8_to_8f32(<16 x i8> %a) {
; SSE-LABEL: sitofp_8i8_to_8f32:
; SSE:       # BB#0:
; SSE-NEXT:    punpcklbw {{.*#+}} xmm1 = xmm1[0],xmm0[0],xmm1[1],xmm0[1],xmm1[2],xmm0[2],xmm1[3],xmm0[3],xmm1[4],xmm0[4],xmm1[5],xmm0[5],xmm1[6],xmm0[6],xmm1[7],xmm0[7]
; SSE-NEXT:    punpcklwd {{.*#+}} xmm1 = xmm1[0,0,1,1,2,2,3,3]
; SSE-NEXT:    psrad $24, %xmm1
; SSE-NEXT:    cvtdq2ps %xmm1, %xmm2
; SSE-NEXT:    pshufd {{.*#+}} xmm0 = xmm0[1,1,2,3]
; SSE-NEXT:    punpcklbw {{.*#+}} xmm0 = xmm0[0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7]
; SSE-NEXT:    punpcklwd {{.*#+}} xmm0 = xmm0[0,0,1,1,2,2,3,3]
; SSE-NEXT:    psrad $24, %xmm0
; SSE-NEXT:    cvtdq2ps %xmm0, %xmm1
; SSE-NEXT:    movaps %xmm2, %xmm0
; SSE-NEXT:    retq
;
; AVX1-LABEL: sitofp_8i8_to_8f32:
; AVX1:       # BB#0:
; AVX1-NEXT:    vpmovsxbd %xmm0, %xmm1
; AVX1-NEXT:    vpshufd {{.*#+}} xmm0 = xmm0[1,1,2,3]
; AVX1-NEXT:    vpmovsxbd %xmm0, %xmm0
; AVX1-NEXT:    vinsertf128 $1, %xmm0, %ymm1, %ymm0
; AVX1-NEXT:    vcvtdq2ps %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: sitofp_8i8_to_8f32:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpmovzxbd {{.*#+}} ymm0 = xmm0[0],zero,zero,zero,xmm0[1],zero,zero,zero,xmm0[2],zero,zero,zero,xmm0[3],zero,zero,zero,xmm0[4],zero,zero,zero,xmm0[5],zero,zero,zero,xmm0[6],zero,zero,zero,xmm0[7],zero,zero,zero
; AVX2-NEXT:    vpslld $24, %ymm0, %ymm0
; AVX2-NEXT:    vpsrad $24, %ymm0, %ymm0
; AVX2-NEXT:    vcvtdq2ps %ymm0, %ymm0
; AVX2-NEXT:    retq
  %shuf = shufflevector <16 x i8> %a, <16 x i8> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %cvt = sitofp <8 x i8> %shuf to <8 x float>
  ret <8 x float> %cvt
}

define <8 x float> @sitofp_16i8_to_8f32(<16 x i8> %a) {
; SSE-LABEL: sitofp_16i8_to_8f32:
; SSE:       # BB#0:
; SSE-NEXT:    punpcklbw {{.*#+}} xmm1 = xmm1[0],xmm0[0],xmm1[1],xmm0[1],xmm1[2],xmm0[2],xmm1[3],xmm0[3],xmm1[4],xmm0[4],xmm1[5],xmm0[5],xmm1[6],xmm0[6],xmm1[7],xmm0[7]
; SSE-NEXT:    punpcklwd {{.*#+}} xmm1 = xmm1[0,0,1,1,2,2,3,3]
; SSE-NEXT:    psrad $24, %xmm1
; SSE-NEXT:    cvtdq2ps %xmm1, %xmm2
; SSE-NEXT:    pshufd {{.*#+}} xmm0 = xmm0[1,1,2,3]
; SSE-NEXT:    punpcklbw {{.*#+}} xmm0 = xmm0[0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7]
; SSE-NEXT:    punpcklwd {{.*#+}} xmm0 = xmm0[0,0,1,1,2,2,3,3]
; SSE-NEXT:    psrad $24, %xmm0
; SSE-NEXT:    cvtdq2ps %xmm0, %xmm1
; SSE-NEXT:    movaps %xmm2, %xmm0
; SSE-NEXT:    retq
;
; AVX1-LABEL: sitofp_16i8_to_8f32:
; AVX1:       # BB#0:
; AVX1-NEXT:    vpmovsxbd %xmm0, %xmm1
; AVX1-NEXT:    vpshufd {{.*#+}} xmm0 = xmm0[1,1,2,3]
; AVX1-NEXT:    vpmovsxbd %xmm0, %xmm0
; AVX1-NEXT:    vinsertf128 $1, %xmm0, %ymm1, %ymm0
; AVX1-NEXT:    vcvtdq2ps %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: sitofp_16i8_to_8f32:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpmovsxbw %xmm0, %ymm0
; AVX2-NEXT:    vpmovsxwd %xmm0, %ymm0
; AVX2-NEXT:    vcvtdq2ps %ymm0, %ymm0
; AVX2-NEXT:    retq
  %cvt = sitofp <16 x i8> %a to <16 x float>
  %shuf = shufflevector <16 x float> %cvt, <16 x float> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  ret <8 x float> %shuf
}

;
; Unsigned Integer to Float
;

define <4 x float> @uitofp_2i64_to_4f32(<2 x i64> %a) {
; SSE-LABEL: uitofp_2i64_to_4f32:
; SSE:       # BB#0:
; SSE-NEXT:    movdqa %xmm0, %xmm1
; SSE-NEXT:    movd %xmm1, %rax
; SSE-NEXT:    movl %eax, %ecx
; SSE-NEXT:    andl $1, %ecx
; SSE-NEXT:    testq %rax, %rax
; SSE-NEXT:    js .LBB38_1
; SSE-NEXT:  # BB#2:
; SSE-NEXT:    xorps %xmm0, %xmm0
; SSE-NEXT:    cvtsi2ssq %rax, %xmm0
; SSE-NEXT:    jmp .LBB38_3
; SSE-NEXT:  .LBB38_1:
; SSE-NEXT:    shrq %rax
; SSE-NEXT:    orq %rax, %rcx
; SSE-NEXT:    xorps %xmm0, %xmm0
; SSE-NEXT:    cvtsi2ssq %rcx, %xmm0
; SSE-NEXT:    addss %xmm0, %xmm0
; SSE-NEXT:  .LBB38_3:
; SSE-NEXT:    pshufd {{.*#+}} xmm1 = xmm1[2,3,0,1]
; SSE-NEXT:    movd %xmm1, %rax
; SSE-NEXT:    movl %eax, %ecx
; SSE-NEXT:    andl $1, %ecx
; SSE-NEXT:    testq %rax, %rax
; SSE-NEXT:    js .LBB38_4
; SSE-NEXT:  # BB#5:
; SSE-NEXT:    xorps %xmm1, %xmm1
; SSE-NEXT:    cvtsi2ssq %rax, %xmm1
; SSE-NEXT:    unpcklps {{.*#+}} xmm0 = xmm0[0],xmm1[0],xmm0[1],xmm1[1]
; SSE-NEXT:    retq
; SSE-NEXT:  .LBB38_4:
; SSE-NEXT:    shrq %rax
; SSE-NEXT:    orq %rax, %rcx
; SSE-NEXT:    xorps %xmm1, %xmm1
; SSE-NEXT:    cvtsi2ssq %rcx, %xmm1
; SSE-NEXT:    addss %xmm1, %xmm1
; SSE-NEXT:    unpcklps {{.*#+}} xmm0 = xmm0[0],xmm1[0],xmm0[1],xmm1[1]
; SSE-NEXT:    retq
;
; AVX-LABEL: uitofp_2i64_to_4f32:
; AVX:       # BB#0:
; AVX-NEXT:    vpextrq $1, %xmm0, %rax
; AVX-NEXT:    movl %eax, %ecx
; AVX-NEXT:    andl $1, %ecx
; AVX-NEXT:    testq %rax, %rax
; AVX-NEXT:    js .LBB38_1
; AVX-NEXT:  # BB#2:
; AVX-NEXT:    vcvtsi2ssq %rax, %xmm0, %xmm1
; AVX-NEXT:    jmp .LBB38_3
; AVX-NEXT:  .LBB38_1:
; AVX-NEXT:    shrq %rax
; AVX-NEXT:    orq %rax, %rcx
; AVX-NEXT:    vcvtsi2ssq %rcx, %xmm0, %xmm1
; AVX-NEXT:    vaddss %xmm1, %xmm1, %xmm1
; AVX-NEXT:  .LBB38_3:
; AVX-NEXT:    vmovq %xmm0, %rax
; AVX-NEXT:    movl %eax, %ecx
; AVX-NEXT:    andl $1, %ecx
; AVX-NEXT:    testq %rax, %rax
; AVX-NEXT:    js .LBB38_4
; AVX-NEXT:  # BB#5:
; AVX-NEXT:    vxorps %xmm0, %xmm0, %xmm0
; AVX-NEXT:    vcvtsi2ssq %rax, %xmm0, %xmm0
; AVX-NEXT:    jmp .LBB38_6
; AVX-NEXT:  .LBB38_4:
; AVX-NEXT:    shrq %rax
; AVX-NEXT:    orq %rax, %rcx
; AVX-NEXT:    vxorps %xmm0, %xmm0, %xmm0
; AVX-NEXT:    vcvtsi2ssq %rcx, %xmm0, %xmm0
; AVX-NEXT:    vaddss %xmm0, %xmm0, %xmm0
; AVX-NEXT:  .LBB38_6:
; AVX-NEXT:    vinsertps {{.*#+}} xmm0 = xmm0[0],xmm1[0],xmm0[2,3]
; AVX-NEXT:    vxorps %xmm1, %xmm1, %xmm1
; AVX-NEXT:    testq %rax, %rax
; AVX-NEXT:    js .LBB38_8
; AVX-NEXT:  # BB#7:
; AVX-NEXT:    vcvtsi2ssq %rax, %xmm0, %xmm1
; AVX-NEXT:  .LBB38_8:
; AVX-NEXT:    vinsertps {{.*#+}} xmm0 = xmm0[0,1],xmm1[0],xmm0[3]
; AVX-NEXT:    vinsertps {{.*#+}} xmm0 = xmm0[0,1,2],xmm1[0]
; AVX-NEXT:    retq
  %cvt = uitofp <2 x i64> %a to <2 x float>
  %ext = shufflevector <2 x float> %cvt, <2 x float> undef, <4 x i32> <i32 0, i32 1, i32 undef, i32 undef>
  ret <4 x float> %ext
}

define <4 x float> @uitofp_4i64_to_4f32_undef(<2 x i64> %a) {
; SSE-LABEL: uitofp_4i64_to_4f32_undef:
; SSE:       # BB#0:
; SSE-NEXT:    movdqa %xmm0, %xmm1
; SSE-NEXT:    testq %rax, %rax
; SSE-NEXT:    xorps %xmm2, %xmm2
; SSE-NEXT:    js .LBB39_2
; SSE-NEXT:  # BB#1:
; SSE-NEXT:    xorps %xmm2, %xmm2
; SSE-NEXT:    cvtsi2ssq %rax, %xmm2
; SSE-NEXT:  .LBB39_2:
; SSE-NEXT:    movd %xmm1, %rax
; SSE-NEXT:    movl %eax, %ecx
; SSE-NEXT:    andl $1, %ecx
; SSE-NEXT:    testq %rax, %rax
; SSE-NEXT:    js .LBB39_3
; SSE-NEXT:  # BB#4:
; SSE-NEXT:    xorps %xmm0, %xmm0
; SSE-NEXT:    cvtsi2ssq %rax, %xmm0
; SSE-NEXT:    jmp .LBB39_5
; SSE-NEXT:  .LBB39_3:
; SSE-NEXT:    shrq %rax
; SSE-NEXT:    orq %rax, %rcx
; SSE-NEXT:    xorps %xmm0, %xmm0
; SSE-NEXT:    cvtsi2ssq %rcx, %xmm0
; SSE-NEXT:    addss %xmm0, %xmm0
; SSE-NEXT:  .LBB39_5:
; SSE-NEXT:    unpcklps {{.*#+}} xmm0 = xmm0[0],xmm2[0],xmm0[1],xmm2[1]
; SSE-NEXT:    pshufd {{.*#+}} xmm1 = xmm1[2,3,0,1]
; SSE-NEXT:    movd %xmm1, %rax
; SSE-NEXT:    movl %eax, %ecx
; SSE-NEXT:    andl $1, %ecx
; SSE-NEXT:    testq %rax, %rax
; SSE-NEXT:    js .LBB39_6
; SSE-NEXT:  # BB#7:
; SSE-NEXT:    xorps %xmm1, %xmm1
; SSE-NEXT:    cvtsi2ssq %rax, %xmm1
; SSE-NEXT:    jmp .LBB39_8
; SSE-NEXT:  .LBB39_6:
; SSE-NEXT:    shrq %rax
; SSE-NEXT:    orq %rax, %rcx
; SSE-NEXT:    xorps %xmm1, %xmm1
; SSE-NEXT:    cvtsi2ssq %rcx, %xmm1
; SSE-NEXT:    addss %xmm1, %xmm1
; SSE-NEXT:  .LBB39_8:
; SSE-NEXT:    unpcklps {{.*#+}} xmm1 = xmm1[0],xmm2[0],xmm1[1],xmm2[1]
; SSE-NEXT:    unpcklps {{.*#+}} xmm0 = xmm0[0],xmm1[0],xmm0[1],xmm1[1]
; SSE-NEXT:    retq
;
; AVX-LABEL: uitofp_4i64_to_4f32_undef:
; AVX:       # BB#0:
; AVX-NEXT:    vpextrq $1, %xmm0, %rax
; AVX-NEXT:    movl %eax, %ecx
; AVX-NEXT:    andl $1, %ecx
; AVX-NEXT:    testq %rax, %rax
; AVX-NEXT:    js .LBB39_1
; AVX-NEXT:  # BB#2:
; AVX-NEXT:    vcvtsi2ssq %rax, %xmm0, %xmm1
; AVX-NEXT:    jmp .LBB39_3
; AVX-NEXT:  .LBB39_1:
; AVX-NEXT:    shrq %rax
; AVX-NEXT:    orq %rax, %rcx
; AVX-NEXT:    vcvtsi2ssq %rcx, %xmm0, %xmm1
; AVX-NEXT:    vaddss %xmm1, %xmm1, %xmm1
; AVX-NEXT:  .LBB39_3:
; AVX-NEXT:    vmovq %xmm0, %rax
; AVX-NEXT:    movl %eax, %ecx
; AVX-NEXT:    andl $1, %ecx
; AVX-NEXT:    testq %rax, %rax
; AVX-NEXT:    js .LBB39_4
; AVX-NEXT:  # BB#5:
; AVX-NEXT:    vxorps %xmm0, %xmm0, %xmm0
; AVX-NEXT:    vcvtsi2ssq %rax, %xmm0, %xmm0
; AVX-NEXT:    jmp .LBB39_6
; AVX-NEXT:  .LBB39_4:
; AVX-NEXT:    shrq %rax
; AVX-NEXT:    orq %rax, %rcx
; AVX-NEXT:    vxorps %xmm0, %xmm0, %xmm0
; AVX-NEXT:    vcvtsi2ssq %rcx, %xmm0, %xmm0
; AVX-NEXT:    vaddss %xmm0, %xmm0, %xmm0
; AVX-NEXT:  .LBB39_6:
; AVX-NEXT:    vinsertps {{.*#+}} xmm0 = xmm0[0],xmm1[0],xmm0[2,3]
; AVX-NEXT:    vxorps %xmm1, %xmm1, %xmm1
; AVX-NEXT:    testq %rax, %rax
; AVX-NEXT:    js .LBB39_8
; AVX-NEXT:  # BB#7:
; AVX-NEXT:    vcvtsi2ssq %rax, %xmm0, %xmm1
; AVX-NEXT:  .LBB39_8:
; AVX-NEXT:    vinsertps {{.*#+}} xmm0 = xmm0[0,1],xmm1[0],xmm0[3]
; AVX-NEXT:    vinsertps {{.*#+}} xmm0 = xmm0[0,1,2],xmm1[0]
; AVX-NEXT:    retq
  %ext = shufflevector <2 x i64> %a, <2 x i64> undef, <4 x i32> <i32 0, i32 1, i32 undef, i32 undef>
  %cvt = uitofp <4 x i64> %ext to <4 x float>
  ret <4 x float> %cvt
}

define <4 x float> @uitofp_4i32_to_4f32(<4 x i32> %a) {
; SSE-LABEL: uitofp_4i32_to_4f32:
; SSE:       # BB#0:
; SSE-NEXT:    movdqa {{.*#+}} xmm1 = [65535,65535,65535,65535]
; SSE-NEXT:    pand %xmm0, %xmm1
; SSE-NEXT:    por {{.*}}(%rip), %xmm1
; SSE-NEXT:    psrld $16, %xmm0
; SSE-NEXT:    por {{.*}}(%rip), %xmm0
; SSE-NEXT:    addps {{.*}}(%rip), %xmm0
; SSE-NEXT:    addps %xmm1, %xmm0
; SSE-NEXT:    retq
;
; AVX1-LABEL: uitofp_4i32_to_4f32:
; AVX1:       # BB#0:
; AVX1-NEXT:    vpblendw {{.*#+}} xmm1 = xmm0[0],mem[1],xmm0[2],mem[3],xmm0[4],mem[5],xmm0[6],mem[7]
; AVX1-NEXT:    vpsrld $16, %xmm0, %xmm0
; AVX1-NEXT:    vpblendw {{.*#+}} xmm0 = xmm0[0],mem[1],xmm0[2],mem[3],xmm0[4],mem[5],xmm0[6],mem[7]
; AVX1-NEXT:    vaddps {{.*}}(%rip), %xmm0, %xmm0
; AVX1-NEXT:    vaddps %xmm0, %xmm1, %xmm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: uitofp_4i32_to_4f32:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpbroadcastd {{.*}}(%rip), %xmm1
; AVX2-NEXT:    vpblendw {{.*#+}} xmm1 = xmm0[0],xmm1[1],xmm0[2],xmm1[3],xmm0[4],xmm1[5],xmm0[6],xmm1[7]
; AVX2-NEXT:    vpsrld $16, %xmm0, %xmm0
; AVX2-NEXT:    vpbroadcastd {{.*}}(%rip), %xmm2
; AVX2-NEXT:    vpblendw {{.*#+}} xmm0 = xmm0[0],xmm2[1],xmm0[2],xmm2[3],xmm0[4],xmm2[5],xmm0[6],xmm2[7]
; AVX2-NEXT:    vbroadcastss {{.*}}(%rip), %xmm2
; AVX2-NEXT:    vaddps %xmm2, %xmm0, %xmm0
; AVX2-NEXT:    vaddps %xmm0, %xmm1, %xmm0
; AVX2-NEXT:    retq
  %cvt = uitofp <4 x i32> %a to <4 x float>
  ret <4 x float> %cvt
}

define <4 x float> @uitofp_4i16_to_4f32(<8 x i16> %a) {
; SSE-LABEL: uitofp_4i16_to_4f32:
; SSE:       # BB#0:
; SSE-NEXT:    pxor %xmm1, %xmm1
; SSE-NEXT:    punpcklwd {{.*#+}} xmm0 = xmm0[0],xmm1[0],xmm0[1],xmm1[1],xmm0[2],xmm1[2],xmm0[3],xmm1[3]
; SSE-NEXT:    cvtdq2ps %xmm0, %xmm0
; SSE-NEXT:    retq
;
; AVX-LABEL: uitofp_4i16_to_4f32:
; AVX:       # BB#0:
; AVX-NEXT:    vpmovzxwd {{.*#+}} xmm0 = xmm0[0],zero,xmm0[1],zero,xmm0[2],zero,xmm0[3],zero
; AVX-NEXT:    vcvtdq2ps %xmm0, %xmm0
; AVX-NEXT:    retq
  %shuf = shufflevector <8 x i16> %a, <8 x i16> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %cvt = uitofp <4 x i16> %shuf to <4 x float>
  ret <4 x float> %cvt
}

define <4 x float> @uitofp_8i16_to_4f32(<8 x i16> %a) {
; SSE-LABEL: uitofp_8i16_to_4f32:
; SSE:       # BB#0:
; SSE-NEXT:    pxor %xmm1, %xmm1
; SSE-NEXT:    punpcklwd {{.*#+}} xmm0 = xmm0[0],xmm1[0],xmm0[1],xmm1[1],xmm0[2],xmm1[2],xmm0[3],xmm1[3]
; SSE-NEXT:    cvtdq2ps %xmm0, %xmm0
; SSE-NEXT:    retq
;
; AVX1-LABEL: uitofp_8i16_to_4f32:
; AVX1:       # BB#0:
; AVX1-NEXT:    vpxor %xmm1, %xmm1, %xmm1
; AVX1-NEXT:    vpunpckhwd {{.*#+}} xmm1 = xmm0[4],xmm1[4],xmm0[5],xmm1[5],xmm0[6],xmm1[6],xmm0[7],xmm1[7]
; AVX1-NEXT:    vpmovzxwd {{.*#+}} xmm0 = xmm0[0],zero,xmm0[1],zero,xmm0[2],zero,xmm0[3],zero
; AVX1-NEXT:    vinsertf128 $1, %xmm1, %ymm0, %ymm0
; AVX1-NEXT:    vcvtdq2ps %ymm0, %ymm0
; AVX1-NEXT:    vzeroupper
; AVX1-NEXT:    retq
;
; AVX2-LABEL: uitofp_8i16_to_4f32:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpmovzxwd {{.*#+}} ymm0 = xmm0[0],zero,xmm0[1],zero,xmm0[2],zero,xmm0[3],zero,xmm0[4],zero,xmm0[5],zero,xmm0[6],zero,xmm0[7],zero
; AVX2-NEXT:    vcvtdq2ps %ymm0, %ymm0
; AVX2-NEXT:    vzeroupper
; AVX2-NEXT:    retq
  %cvt = uitofp <8 x i16> %a to <8 x float>
  %shuf = shufflevector <8 x float> %cvt, <8 x float> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  ret <4 x float> %shuf
}

define <4 x float> @uitofp_4i8_to_4f32(<16 x i8> %a) {
; SSE-LABEL: uitofp_4i8_to_4f32:
; SSE:       # BB#0:
; SSE-NEXT:    pxor %xmm1, %xmm1
; SSE-NEXT:    punpcklbw {{.*#+}} xmm0 = xmm0[0],xmm1[0],xmm0[1],xmm1[1],xmm0[2],xmm1[2],xmm0[3],xmm1[3],xmm0[4],xmm1[4],xmm0[5],xmm1[5],xmm0[6],xmm1[6],xmm0[7],xmm1[7]
; SSE-NEXT:    punpcklwd {{.*#+}} xmm0 = xmm0[0],xmm1[0],xmm0[1],xmm1[1],xmm0[2],xmm1[2],xmm0[3],xmm1[3]
; SSE-NEXT:    cvtdq2ps %xmm0, %xmm0
; SSE-NEXT:    retq
;
; AVX-LABEL: uitofp_4i8_to_4f32:
; AVX:       # BB#0:
; AVX-NEXT:    vpmovzxbd {{.*#+}} xmm0 = xmm0[0],zero,zero,zero,xmm0[1],zero,zero,zero,xmm0[2],zero,zero,zero,xmm0[3],zero,zero,zero
; AVX-NEXT:    vcvtdq2ps %xmm0, %xmm0
; AVX-NEXT:    retq
  %shuf = shufflevector <16 x i8> %a, <16 x i8> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %cvt = uitofp <4 x i8> %shuf to <4 x float>
  ret <4 x float> %cvt
}

define <4 x float> @uitofp_16i8_to_4f32(<16 x i8> %a) {
; SSE-LABEL: uitofp_16i8_to_4f32:
; SSE:       # BB#0:
; SSE-NEXT:    pxor %xmm1, %xmm1
; SSE-NEXT:    punpcklbw {{.*#+}} xmm0 = xmm0[0],xmm1[0],xmm0[1],xmm1[1],xmm0[2],xmm1[2],xmm0[3],xmm1[3],xmm0[4],xmm1[4],xmm0[5],xmm1[5],xmm0[6],xmm1[6],xmm0[7],xmm1[7]
; SSE-NEXT:    punpcklwd {{.*#+}} xmm0 = xmm0[0],xmm1[0],xmm0[1],xmm1[1],xmm0[2],xmm1[2],xmm0[3],xmm1[3]
; SSE-NEXT:    cvtdq2ps %xmm0, %xmm0
; SSE-NEXT:    retq
;
; AVX1-LABEL: uitofp_16i8_to_4f32:
; AVX1:       # BB#0:
; AVX1-NEXT:    vpmovzxbw {{.*#+}} xmm1 = xmm0[0],zero,xmm0[1],zero,xmm0[2],zero,xmm0[3],zero,xmm0[4],zero,xmm0[5],zero,xmm0[6],zero,xmm0[7],zero
; AVX1-NEXT:    vpxor %xmm2, %xmm2, %xmm2
; AVX1-NEXT:    vpunpckhwd {{.*#+}} xmm1 = xmm1[4],xmm2[4],xmm1[5],xmm2[5],xmm1[6],xmm2[6],xmm1[7],xmm2[7]
; AVX1-NEXT:    vpmovzxbd {{.*#+}} xmm0 = xmm0[0],zero,zero,zero,xmm0[1],zero,zero,zero,xmm0[2],zero,zero,zero,xmm0[3],zero,zero,zero
; AVX1-NEXT:    vinsertf128 $1, %xmm1, %ymm0, %ymm0
; AVX1-NEXT:    vcvtdq2ps %ymm0, %ymm0
; AVX1-NEXT:    vzeroupper
; AVX1-NEXT:    retq
;
; AVX2-LABEL: uitofp_16i8_to_4f32:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpmovzxbw {{.*#+}} ymm0 = xmm0[0],zero,xmm0[1],zero,xmm0[2],zero,xmm0[3],zero,xmm0[4],zero,xmm0[5],zero,xmm0[6],zero,xmm0[7],zero,xmm0[8],zero,xmm0[9],zero,xmm0[10],zero,xmm0[11],zero,xmm0[12],zero,xmm0[13],zero,xmm0[14],zero,xmm0[15],zero
; AVX2-NEXT:    vpmovzxwd {{.*#+}} ymm0 = xmm0[0],zero,xmm0[1],zero,xmm0[2],zero,xmm0[3],zero,xmm0[4],zero,xmm0[5],zero,xmm0[6],zero,xmm0[7],zero
; AVX2-NEXT:    vcvtdq2ps %ymm0, %ymm0
; AVX2-NEXT:    vzeroupper
; AVX2-NEXT:    retq
  %cvt = uitofp <16 x i8> %a to <16 x float>
  %shuf = shufflevector <16 x float> %cvt, <16 x float> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  ret <4 x float> %shuf
}

define <4 x float> @uitofp_4i64_to_4f32(<4 x i64> %a) {
; SSE-LABEL: uitofp_4i64_to_4f32:
; SSE:       # BB#0:
; SSE-NEXT:    movd %xmm1, %rax
; SSE-NEXT:    movl %eax, %ecx
; SSE-NEXT:    andl $1, %ecx
; SSE-NEXT:    testq %rax, %rax
; SSE-NEXT:    js .LBB45_1
; SSE-NEXT:  # BB#2:
; SSE-NEXT:    cvtsi2ssq %rax, %xmm3
; SSE-NEXT:    jmp .LBB45_3
; SSE-NEXT:  .LBB45_1:
; SSE-NEXT:    shrq %rax
; SSE-NEXT:    orq %rax, %rcx
; SSE-NEXT:    cvtsi2ssq %rcx, %xmm3
; SSE-NEXT:    addss %xmm3, %xmm3
; SSE-NEXT:  .LBB45_3:
; SSE-NEXT:    movd %xmm0, %rax
; SSE-NEXT:    movl %eax, %ecx
; SSE-NEXT:    andl $1, %ecx
; SSE-NEXT:    testq %rax, %rax
; SSE-NEXT:    js .LBB45_4
; SSE-NEXT:  # BB#5:
; SSE-NEXT:    cvtsi2ssq %rax, %xmm2
; SSE-NEXT:    jmp .LBB45_6
; SSE-NEXT:  .LBB45_4:
; SSE-NEXT:    shrq %rax
; SSE-NEXT:    orq %rax, %rcx
; SSE-NEXT:    cvtsi2ssq %rcx, %xmm2
; SSE-NEXT:    addss %xmm2, %xmm2
; SSE-NEXT:  .LBB45_6:
; SSE-NEXT:    pshufd {{.*#+}} xmm1 = xmm1[2,3,0,1]
; SSE-NEXT:    movd %xmm1, %rax
; SSE-NEXT:    movl %eax, %ecx
; SSE-NEXT:    andl $1, %ecx
; SSE-NEXT:    testq %rax, %rax
; SSE-NEXT:    js .LBB45_7
; SSE-NEXT:  # BB#8:
; SSE-NEXT:    xorps %xmm1, %xmm1
; SSE-NEXT:    cvtsi2ssq %rax, %xmm1
; SSE-NEXT:    jmp .LBB45_9
; SSE-NEXT:  .LBB45_7:
; SSE-NEXT:    shrq %rax
; SSE-NEXT:    orq %rax, %rcx
; SSE-NEXT:    xorps %xmm1, %xmm1
; SSE-NEXT:    cvtsi2ssq %rcx, %xmm1
; SSE-NEXT:    addss %xmm1, %xmm1
; SSE-NEXT:  .LBB45_9:
; SSE-NEXT:    unpcklps {{.*#+}} xmm2 = xmm2[0],xmm3[0],xmm2[1],xmm3[1]
; SSE-NEXT:    pshufd {{.*#+}} xmm0 = xmm0[2,3,0,1]
; SSE-NEXT:    movd %xmm0, %rax
; SSE-NEXT:    movl %eax, %ecx
; SSE-NEXT:    andl $1, %ecx
; SSE-NEXT:    testq %rax, %rax
; SSE-NEXT:    js .LBB45_10
; SSE-NEXT:  # BB#11:
; SSE-NEXT:    xorps %xmm0, %xmm0
; SSE-NEXT:    cvtsi2ssq %rax, %xmm0
; SSE-NEXT:    jmp .LBB45_12
; SSE-NEXT:  .LBB45_10:
; SSE-NEXT:    shrq %rax
; SSE-NEXT:    orq %rax, %rcx
; SSE-NEXT:    xorps %xmm0, %xmm0
; SSE-NEXT:    cvtsi2ssq %rcx, %xmm0
; SSE-NEXT:    addss %xmm0, %xmm0
; SSE-NEXT:  .LBB45_12:
; SSE-NEXT:    unpcklps {{.*#+}} xmm0 = xmm0[0],xmm1[0],xmm0[1],xmm1[1]
; SSE-NEXT:    unpcklps {{.*#+}} xmm2 = xmm2[0],xmm0[0],xmm2[1],xmm0[1]
; SSE-NEXT:    movaps %xmm2, %xmm0
; SSE-NEXT:    retq
;
; AVX1-LABEL: uitofp_4i64_to_4f32:
; AVX1:       # BB#0:
; AVX1-NEXT:    vpextrq $1, %xmm0, %rax
; AVX1-NEXT:    movl %eax, %ecx
; AVX1-NEXT:    andl $1, %ecx
; AVX1-NEXT:    testq %rax, %rax
; AVX1-NEXT:    js .LBB45_1
; AVX1-NEXT:  # BB#2:
; AVX1-NEXT:    vcvtsi2ssq %rax, %xmm0, %xmm1
; AVX1-NEXT:    jmp .LBB45_3
; AVX1-NEXT:  .LBB45_1:
; AVX1-NEXT:    shrq %rax
; AVX1-NEXT:    orq %rax, %rcx
; AVX1-NEXT:    vcvtsi2ssq %rcx, %xmm0, %xmm1
; AVX1-NEXT:    vaddss %xmm1, %xmm1, %xmm1
; AVX1-NEXT:  .LBB45_3:
; AVX1-NEXT:    vmovq %xmm0, %rax
; AVX1-NEXT:    movl %eax, %ecx
; AVX1-NEXT:    andl $1, %ecx
; AVX1-NEXT:    testq %rax, %rax
; AVX1-NEXT:    js .LBB45_4
; AVX1-NEXT:  # BB#5:
; AVX1-NEXT:    vcvtsi2ssq %rax, %xmm0, %xmm2
; AVX1-NEXT:    jmp .LBB45_6
; AVX1-NEXT:  .LBB45_4:
; AVX1-NEXT:    shrq %rax
; AVX1-NEXT:    orq %rax, %rcx
; AVX1-NEXT:    vcvtsi2ssq %rcx, %xmm0, %xmm2
; AVX1-NEXT:    vaddss %xmm2, %xmm2, %xmm2
; AVX1-NEXT:  .LBB45_6:
; AVX1-NEXT:    vinsertps {{.*#+}} xmm1 = xmm2[0],xmm1[0],xmm2[2,3]
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm0
; AVX1-NEXT:    vmovq %xmm0, %rax
; AVX1-NEXT:    movl %eax, %ecx
; AVX1-NEXT:    andl $1, %ecx
; AVX1-NEXT:    testq %rax, %rax
; AVX1-NEXT:    js .LBB45_7
; AVX1-NEXT:  # BB#8:
; AVX1-NEXT:    vcvtsi2ssq %rax, %xmm0, %xmm2
; AVX1-NEXT:    jmp .LBB45_9
; AVX1-NEXT:  .LBB45_7:
; AVX1-NEXT:    shrq %rax
; AVX1-NEXT:    orq %rax, %rcx
; AVX1-NEXT:    vcvtsi2ssq %rcx, %xmm0, %xmm2
; AVX1-NEXT:    vaddss %xmm2, %xmm2, %xmm2
; AVX1-NEXT:  .LBB45_9:
; AVX1-NEXT:    vinsertps {{.*#+}} xmm1 = xmm1[0,1],xmm2[0],xmm1[3]
; AVX1-NEXT:    vpextrq $1, %xmm0, %rax
; AVX1-NEXT:    movl %eax, %ecx
; AVX1-NEXT:    andl $1, %ecx
; AVX1-NEXT:    testq %rax, %rax
; AVX1-NEXT:    js .LBB45_10
; AVX1-NEXT:  # BB#11:
; AVX1-NEXT:    vxorps %xmm0, %xmm0, %xmm0
; AVX1-NEXT:    vcvtsi2ssq %rax, %xmm0, %xmm0
; AVX1-NEXT:    vinsertps {{.*#+}} xmm0 = xmm1[0,1,2],xmm0[0]
; AVX1-NEXT:    vzeroupper
; AVX1-NEXT:    retq
; AVX1-NEXT:  .LBB45_10:
; AVX1-NEXT:    shrq %rax
; AVX1-NEXT:    orq %rax, %rcx
; AVX1-NEXT:    vcvtsi2ssq %rcx, %xmm0, %xmm0
; AVX1-NEXT:    vaddss %xmm0, %xmm0, %xmm0
; AVX1-NEXT:    vinsertps {{.*#+}} xmm0 = xmm1[0,1,2],xmm0[0]
; AVX1-NEXT:    vzeroupper
; AVX1-NEXT:    retq
;
; AVX2-LABEL: uitofp_4i64_to_4f32:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpextrq $1, %xmm0, %rax
; AVX2-NEXT:    movl %eax, %ecx
; AVX2-NEXT:    andl $1, %ecx
; AVX2-NEXT:    testq %rax, %rax
; AVX2-NEXT:    js .LBB45_1
; AVX2-NEXT:  # BB#2:
; AVX2-NEXT:    vcvtsi2ssq %rax, %xmm0, %xmm1
; AVX2-NEXT:    jmp .LBB45_3
; AVX2-NEXT:  .LBB45_1:
; AVX2-NEXT:    shrq %rax
; AVX2-NEXT:    orq %rax, %rcx
; AVX2-NEXT:    vcvtsi2ssq %rcx, %xmm0, %xmm1
; AVX2-NEXT:    vaddss %xmm1, %xmm1, %xmm1
; AVX2-NEXT:  .LBB45_3:
; AVX2-NEXT:    vmovq %xmm0, %rax
; AVX2-NEXT:    movl %eax, %ecx
; AVX2-NEXT:    andl $1, %ecx
; AVX2-NEXT:    testq %rax, %rax
; AVX2-NEXT:    js .LBB45_4
; AVX2-NEXT:  # BB#5:
; AVX2-NEXT:    vcvtsi2ssq %rax, %xmm0, %xmm2
; AVX2-NEXT:    jmp .LBB45_6
; AVX2-NEXT:  .LBB45_4:
; AVX2-NEXT:    shrq %rax
; AVX2-NEXT:    orq %rax, %rcx
; AVX2-NEXT:    vcvtsi2ssq %rcx, %xmm0, %xmm2
; AVX2-NEXT:    vaddss %xmm2, %xmm2, %xmm2
; AVX2-NEXT:  .LBB45_6:
; AVX2-NEXT:    vinsertps {{.*#+}} xmm1 = xmm2[0],xmm1[0],xmm2[2,3]
; AVX2-NEXT:    vextracti128 $1, %ymm0, %xmm0
; AVX2-NEXT:    vmovq %xmm0, %rax
; AVX2-NEXT:    movl %eax, %ecx
; AVX2-NEXT:    andl $1, %ecx
; AVX2-NEXT:    testq %rax, %rax
; AVX2-NEXT:    js .LBB45_7
; AVX2-NEXT:  # BB#8:
; AVX2-NEXT:    vcvtsi2ssq %rax, %xmm0, %xmm2
; AVX2-NEXT:    jmp .LBB45_9
; AVX2-NEXT:  .LBB45_7:
; AVX2-NEXT:    shrq %rax
; AVX2-NEXT:    orq %rax, %rcx
; AVX2-NEXT:    vcvtsi2ssq %rcx, %xmm0, %xmm2
; AVX2-NEXT:    vaddss %xmm2, %xmm2, %xmm2
; AVX2-NEXT:  .LBB45_9:
; AVX2-NEXT:    vinsertps {{.*#+}} xmm1 = xmm1[0,1],xmm2[0],xmm1[3]
; AVX2-NEXT:    vpextrq $1, %xmm0, %rax
; AVX2-NEXT:    movl %eax, %ecx
; AVX2-NEXT:    andl $1, %ecx
; AVX2-NEXT:    testq %rax, %rax
; AVX2-NEXT:    js .LBB45_10
; AVX2-NEXT:  # BB#11:
; AVX2-NEXT:    vxorps %xmm0, %xmm0, %xmm0
; AVX2-NEXT:    vcvtsi2ssq %rax, %xmm0, %xmm0
; AVX2-NEXT:    vinsertps {{.*#+}} xmm0 = xmm1[0,1,2],xmm0[0]
; AVX2-NEXT:    vzeroupper
; AVX2-NEXT:    retq
; AVX2-NEXT:  .LBB45_10:
; AVX2-NEXT:    shrq %rax
; AVX2-NEXT:    orq %rax, %rcx
; AVX2-NEXT:    vcvtsi2ssq %rcx, %xmm0, %xmm0
; AVX2-NEXT:    vaddss %xmm0, %xmm0, %xmm0
; AVX2-NEXT:    vinsertps {{.*#+}} xmm0 = xmm1[0,1,2],xmm0[0]
; AVX2-NEXT:    vzeroupper
; AVX2-NEXT:    retq
  %cvt = uitofp <4 x i64> %a to <4 x float>
  ret <4 x float> %cvt
}

define <8 x float> @uitofp_8i32_to_8f32(<8 x i32> %a) {
; SSE-LABEL: uitofp_8i32_to_8f32:
; SSE:       # BB#0:
; SSE-NEXT:    movdqa {{.*#+}} xmm2 = [65535,65535,65535,65535]
; SSE-NEXT:    movdqa %xmm0, %xmm3
; SSE-NEXT:    pand %xmm2, %xmm3
; SSE-NEXT:    movdqa {{.*#+}} xmm4 = [1258291200,1258291200,1258291200,1258291200]
; SSE-NEXT:    por %xmm4, %xmm3
; SSE-NEXT:    psrld $16, %xmm0
; SSE-NEXT:    movdqa {{.*#+}} xmm5 = [1392508928,1392508928,1392508928,1392508928]
; SSE-NEXT:    por %xmm5, %xmm0
; SSE-NEXT:    movaps {{.*#+}} xmm6 = [-5.497642e+11,-5.497642e+11,-5.497642e+11,-5.497642e+11]
; SSE-NEXT:    addps %xmm6, %xmm0
; SSE-NEXT:    addps %xmm3, %xmm0
; SSE-NEXT:    pand %xmm1, %xmm2
; SSE-NEXT:    por %xmm4, %xmm2
; SSE-NEXT:    psrld $16, %xmm1
; SSE-NEXT:    por %xmm5, %xmm1
; SSE-NEXT:    addps %xmm6, %xmm1
; SSE-NEXT:    addps %xmm2, %xmm1
; SSE-NEXT:    retq
;
; AVX1-LABEL: uitofp_8i32_to_8f32:
; AVX1:       # BB#0:
; AVX1-NEXT:    vandps {{.*}}(%rip), %ymm0, %ymm1
; AVX1-NEXT:    vcvtdq2ps %ymm1, %ymm1
; AVX1-NEXT:    vpsrld $16, %xmm0, %xmm2
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm0
; AVX1-NEXT:    vpsrld $16, %xmm0, %xmm0
; AVX1-NEXT:    vinsertf128 $1, %xmm0, %ymm2, %ymm0
; AVX1-NEXT:    vcvtdq2ps %ymm0, %ymm0
; AVX1-NEXT:    vmulps {{.*}}(%rip), %ymm0, %ymm0
; AVX1-NEXT:    vaddps %ymm1, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: uitofp_8i32_to_8f32:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpbroadcastd {{.*}}(%rip), %ymm1
; AVX2-NEXT:    vpblendw {{.*#+}} ymm1 = ymm0[0],ymm1[1],ymm0[2],ymm1[3],ymm0[4],ymm1[5],ymm0[6],ymm1[7],ymm0[8],ymm1[9],ymm0[10],ymm1[11],ymm0[12],ymm1[13],ymm0[14],ymm1[15]
; AVX2-NEXT:    vpsrld $16, %ymm0, %ymm0
; AVX2-NEXT:    vpbroadcastd {{.*}}(%rip), %ymm2
; AVX2-NEXT:    vpblendw {{.*#+}} ymm0 = ymm0[0],ymm2[1],ymm0[2],ymm2[3],ymm0[4],ymm2[5],ymm0[6],ymm2[7],ymm0[8],ymm2[9],ymm0[10],ymm2[11],ymm0[12],ymm2[13],ymm0[14],ymm2[15]
; AVX2-NEXT:    vbroadcastss {{.*}}(%rip), %ymm2
; AVX2-NEXT:    vaddps %ymm2, %ymm0, %ymm0
; AVX2-NEXT:    vaddps %ymm0, %ymm1, %ymm0
; AVX2-NEXT:    retq
  %cvt = uitofp <8 x i32> %a to <8 x float>
  ret <8 x float> %cvt
}

define <8 x float> @uitofp_8i16_to_8f32(<8 x i16> %a) {
; SSE-LABEL: uitofp_8i16_to_8f32:
; SSE:       # BB#0:
; SSE-NEXT:    pxor %xmm1, %xmm1
; SSE-NEXT:    movdqa %xmm0, %xmm2
; SSE-NEXT:    punpcklwd {{.*#+}} xmm2 = xmm2[0],xmm1[0],xmm2[1],xmm1[1],xmm2[2],xmm1[2],xmm2[3],xmm1[3]
; SSE-NEXT:    cvtdq2ps %xmm2, %xmm2
; SSE-NEXT:    punpckhwd {{.*#+}} xmm0 = xmm0[4],xmm1[4],xmm0[5],xmm1[5],xmm0[6],xmm1[6],xmm0[7],xmm1[7]
; SSE-NEXT:    cvtdq2ps %xmm0, %xmm1
; SSE-NEXT:    movaps %xmm2, %xmm0
; SSE-NEXT:    retq
;
; AVX1-LABEL: uitofp_8i16_to_8f32:
; AVX1:       # BB#0:
; AVX1-NEXT:    vpxor %xmm1, %xmm1, %xmm1
; AVX1-NEXT:    vpunpckhwd {{.*#+}} xmm1 = xmm0[4],xmm1[4],xmm0[5],xmm1[5],xmm0[6],xmm1[6],xmm0[7],xmm1[7]
; AVX1-NEXT:    vpmovzxwd {{.*#+}} xmm0 = xmm0[0],zero,xmm0[1],zero,xmm0[2],zero,xmm0[3],zero
; AVX1-NEXT:    vinsertf128 $1, %xmm1, %ymm0, %ymm0
; AVX1-NEXT:    vcvtdq2ps %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: uitofp_8i16_to_8f32:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpmovzxwd {{.*#+}} ymm0 = xmm0[0],zero,xmm0[1],zero,xmm0[2],zero,xmm0[3],zero,xmm0[4],zero,xmm0[5],zero,xmm0[6],zero,xmm0[7],zero
; AVX2-NEXT:    vcvtdq2ps %ymm0, %ymm0
; AVX2-NEXT:    retq
  %cvt = uitofp <8 x i16> %a to <8 x float>
  ret <8 x float> %cvt
}

define <8 x float> @uitofp_8i8_to_8f32(<16 x i8> %a) {
; SSE-LABEL: uitofp_8i8_to_8f32:
; SSE:       # BB#0:
; SSE-NEXT:    pxor %xmm1, %xmm1
; SSE-NEXT:    punpcklbw {{.*#+}} xmm0 = xmm0[0],xmm1[0],xmm0[1],xmm1[1],xmm0[2],xmm1[2],xmm0[3],xmm1[3],xmm0[4],xmm1[4],xmm0[5],xmm1[5],xmm0[6],xmm1[6],xmm0[7],xmm1[7]
; SSE-NEXT:    movdqa %xmm0, %xmm2
; SSE-NEXT:    punpcklwd {{.*#+}} xmm2 = xmm2[0],xmm1[0],xmm2[1],xmm1[1],xmm2[2],xmm1[2],xmm2[3],xmm1[3]
; SSE-NEXT:    cvtdq2ps %xmm2, %xmm2
; SSE-NEXT:    punpckhwd {{.*#+}} xmm0 = xmm0[4],xmm1[4],xmm0[5],xmm1[5],xmm0[6],xmm1[6],xmm0[7],xmm1[7]
; SSE-NEXT:    cvtdq2ps %xmm0, %xmm1
; SSE-NEXT:    movaps %xmm2, %xmm0
; SSE-NEXT:    retq
;
; AVX1-LABEL: uitofp_8i8_to_8f32:
; AVX1:       # BB#0:
; AVX1-NEXT:    vpmovzxbw {{.*#+}} xmm1 = xmm0[0],zero,xmm0[1],zero,xmm0[2],zero,xmm0[3],zero,xmm0[4],zero,xmm0[5],zero,xmm0[6],zero,xmm0[7],zero
; AVX1-NEXT:    vpunpckhwd {{.*#+}} xmm1 = xmm1[4],xmm0[4],xmm1[5],xmm0[5],xmm1[6],xmm0[6],xmm1[7],xmm0[7]
; AVX1-NEXT:    vmovdqa {{.*#+}} xmm2 = [255,0,0,0,255,0,0,0,255,0,0,0,255,0,0,0]
; AVX1-NEXT:    vpand %xmm2, %xmm1, %xmm1
; AVX1-NEXT:    vpmovzxbd {{.*#+}} xmm0 = xmm0[0],zero,zero,zero,xmm0[1],zero,zero,zero,xmm0[2],zero,zero,zero,xmm0[3],zero,zero,zero
; AVX1-NEXT:    vpand %xmm2, %xmm0, %xmm0
; AVX1-NEXT:    vinsertf128 $1, %xmm1, %ymm0, %ymm0
; AVX1-NEXT:    vcvtdq2ps %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: uitofp_8i8_to_8f32:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpmovzxbd {{.*#+}} ymm0 = xmm0[0],zero,zero,zero,xmm0[1],zero,zero,zero,xmm0[2],zero,zero,zero,xmm0[3],zero,zero,zero,xmm0[4],zero,zero,zero,xmm0[5],zero,zero,zero,xmm0[6],zero,zero,zero,xmm0[7],zero,zero,zero
; AVX2-NEXT:    vpand {{.*}}(%rip), %ymm0, %ymm0
; AVX2-NEXT:    vcvtdq2ps %ymm0, %ymm0
; AVX2-NEXT:    retq
  %shuf = shufflevector <16 x i8> %a, <16 x i8> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %cvt = uitofp <8 x i8> %shuf to <8 x float>
  ret <8 x float> %cvt
}

define <8 x float> @uitofp_16i8_to_8f32(<16 x i8> %a) {
; SSE-LABEL: uitofp_16i8_to_8f32:
; SSE:       # BB#0:
; SSE-NEXT:    pxor %xmm1, %xmm1
; SSE-NEXT:    punpcklbw {{.*#+}} xmm0 = xmm0[0],xmm1[0],xmm0[1],xmm1[1],xmm0[2],xmm1[2],xmm0[3],xmm1[3],xmm0[4],xmm1[4],xmm0[5],xmm1[5],xmm0[6],xmm1[6],xmm0[7],xmm1[7]
; SSE-NEXT:    movdqa %xmm0, %xmm2
; SSE-NEXT:    punpcklwd {{.*#+}} xmm2 = xmm2[0],xmm1[0],xmm2[1],xmm1[1],xmm2[2],xmm1[2],xmm2[3],xmm1[3]
; SSE-NEXT:    cvtdq2ps %xmm2, %xmm2
; SSE-NEXT:    punpckhwd {{.*#+}} xmm0 = xmm0[4],xmm1[4],xmm0[5],xmm1[5],xmm0[6],xmm1[6],xmm0[7],xmm1[7]
; SSE-NEXT:    cvtdq2ps %xmm0, %xmm1
; SSE-NEXT:    movaps %xmm2, %xmm0
; SSE-NEXT:    retq
;
; AVX1-LABEL: uitofp_16i8_to_8f32:
; AVX1:       # BB#0:
; AVX1-NEXT:    vpmovzxbw {{.*#+}} xmm1 = xmm0[0],zero,xmm0[1],zero,xmm0[2],zero,xmm0[3],zero,xmm0[4],zero,xmm0[5],zero,xmm0[6],zero,xmm0[7],zero
; AVX1-NEXT:    vpxor %xmm2, %xmm2, %xmm2
; AVX1-NEXT:    vpunpckhwd {{.*#+}} xmm1 = xmm1[4],xmm2[4],xmm1[5],xmm2[5],xmm1[6],xmm2[6],xmm1[7],xmm2[7]
; AVX1-NEXT:    vpmovzxbd {{.*#+}} xmm0 = xmm0[0],zero,zero,zero,xmm0[1],zero,zero,zero,xmm0[2],zero,zero,zero,xmm0[3],zero,zero,zero
; AVX1-NEXT:    vinsertf128 $1, %xmm1, %ymm0, %ymm0
; AVX1-NEXT:    vcvtdq2ps %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: uitofp_16i8_to_8f32:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpmovzxbw {{.*#+}} ymm0 = xmm0[0],zero,xmm0[1],zero,xmm0[2],zero,xmm0[3],zero,xmm0[4],zero,xmm0[5],zero,xmm0[6],zero,xmm0[7],zero,xmm0[8],zero,xmm0[9],zero,xmm0[10],zero,xmm0[11],zero,xmm0[12],zero,xmm0[13],zero,xmm0[14],zero,xmm0[15],zero
; AVX2-NEXT:    vpmovzxwd {{.*#+}} ymm0 = xmm0[0],zero,xmm0[1],zero,xmm0[2],zero,xmm0[3],zero,xmm0[4],zero,xmm0[5],zero,xmm0[6],zero,xmm0[7],zero
; AVX2-NEXT:    vcvtdq2ps %ymm0, %ymm0
; AVX2-NEXT:    retq
  %cvt = uitofp <16 x i8> %a to <16 x float>
  %shuf = shufflevector <16 x float> %cvt, <16 x float> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  ret <8 x float> %shuf
}

;
; Aggregates
;

%Arguments = type <{ <8 x i8>, <8 x i16>, <8 x float>* }>
define void @aggregate_sitofp_8i16_to_8f32(%Arguments* nocapture readonly %a0) {
; SSE-LABEL: aggregate_sitofp_8i16_to_8f32:
; SSE:       # BB#0:
; SSE-NEXT:    movq 24(%rdi), %rax
; SSE-NEXT:    movdqu 8(%rdi), %xmm0
; SSE-NEXT:    pshufd {{.*#+}} xmm1 = xmm0[2,3,0,1]
; SSE-NEXT:    punpcklwd {{.*#+}} xmm1 = xmm1[0,0,1,1,2,2,3,3]
; SSE-NEXT:    psrad $16, %xmm1
; SSE-NEXT:    cvtdq2ps %xmm1, %xmm1
; SSE-NEXT:    punpcklwd {{.*#+}} xmm0 = xmm0[0,0,1,1,2,2,3,3]
; SSE-NEXT:    psrad $16, %xmm0
; SSE-NEXT:    cvtdq2ps %xmm0, %xmm0
; SSE-NEXT:    movaps %xmm0, (%rax)
; SSE-NEXT:    movaps %xmm1, 16(%rax)
; SSE-NEXT:    retq
;
; AVX1-LABEL: aggregate_sitofp_8i16_to_8f32:
; AVX1:       # BB#0:
; AVX1-NEXT:    movq 24(%rdi), %rax
; AVX1-NEXT:    vmovdqu 8(%rdi), %xmm0
; AVX1-NEXT:    vpmovsxwd %xmm0, %xmm1
; AVX1-NEXT:    vpshufd {{.*#+}} xmm0 = xmm0[2,3,0,1]
; AVX1-NEXT:    vpmovsxwd %xmm0, %xmm0
; AVX1-NEXT:    vinsertf128 $1, %xmm0, %ymm1, %ymm0
; AVX1-NEXT:    vcvtdq2ps %ymm0, %ymm0
; AVX1-NEXT:    vmovaps %ymm0, (%rax)
; AVX1-NEXT:    vzeroupper
; AVX1-NEXT:    retq
;
; AVX2-LABEL: aggregate_sitofp_8i16_to_8f32:
; AVX2:       # BB#0:
; AVX2-NEXT:    movq 24(%rdi), %rax
; AVX2-NEXT:    vpmovsxwd 8(%rdi), %ymm0
; AVX2-NEXT:    vcvtdq2ps %ymm0, %ymm0
; AVX2-NEXT:    vmovaps %ymm0, (%rax)
; AVX2-NEXT:    vzeroupper
; AVX2-NEXT:    retq
 %1 = load %Arguments, %Arguments* %a0, align 1
 %2 = extractvalue %Arguments %1, 1
 %3 = extractvalue %Arguments %1, 2
 %4 = sitofp <8 x i16> %2 to <8 x float>
 store <8 x float> %4, <8 x float>* %3, align 32
 ret void
}
