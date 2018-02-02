; RUN: llc -mtriple=x86_64-unknown-unknown -mattr=sse2  < %s | FileCheck %s --check-prefix=CHECK --check-prefix=SSE
; RUN: llc -mtriple=x86_64-unknown-unknown -mattr=avx  < %s | FileCheck %s --check-prefix=CHECK --check-prefix=AVX

declare float @fmaxf(float, float)
declare double @fmax(double, double)
declare x86_fp80 @fmaxl(x86_fp80, x86_fp80)
declare float @llvm.maxnum.f32(float, float)
declare double @llvm.maxnum.f64(double, double)
declare x86_fp80 @llvm.maxnum.f80(x86_fp80, x86_fp80)

declare <2 x float> @llvm.maxnum.v2f32(<2 x float>, <2 x float>)
declare <4 x float> @llvm.maxnum.v4f32(<4 x float>, <4 x float>)
declare <2 x double> @llvm.maxnum.v2f64(<2 x double>, <2 x double>)
declare <4 x double> @llvm.maxnum.v4f64(<4 x double>, <4 x double>)
declare <8 x double> @llvm.maxnum.v8f64(<8 x double>, <8 x double>)

; FIXME: As the vector tests show, the SSE run shouldn't need this many moves.

; CHECK-LABEL: @test_fmaxf
; SSE:         movaps %xmm0, %xmm2
; SSE-NEXT:    cmpunordss %xmm0, %xmm2
; SSE-NEXT:    movaps %xmm2, %xmm3
; SSE-NEXT:    andps %xmm1, %xmm3
; SSE-NEXT:    maxss %xmm0, %xmm1
; SSE-NEXT:    andnps %xmm1, %xmm2
; SSE-NEXT:    orps %xmm3, %xmm2
; SSE-NEXT:    movaps %xmm2, %xmm0
; SSE-NEXT:    retq
;
; AVX:         vmaxss %xmm0, %xmm1, %xmm2
; AVX-NEXT:    vcmpunordss %xmm0, %xmm0, %xmm0
; AVX-NEXT:    vblendvps %xmm0, %xmm1, %xmm2, %xmm0
; AVX-NEXT:    retq
define float @test_fmaxf(float %x, float %y) {
  %z = call float @fmaxf(float %x, float %y) readnone
  ret float %z
}

; CHECK-LABEL: @test_fmaxf_minsize
; CHECK:       jmp fmaxf
define float @test_fmaxf_minsize(float %x, float %y) minsize {
  %z = call float @fmaxf(float %x, float %y) readnone
  ret float %z
}

; FIXME: As the vector tests show, the SSE run shouldn't need this many moves.

; CHECK-LABEL: @test_fmax
; SSE:         movapd %xmm0, %xmm2
; SSE-NEXT:    cmpunordsd %xmm0, %xmm2
; SSE-NEXT:    movapd %xmm2, %xmm3
; SSE-NEXT:    andpd %xmm1, %xmm3
; SSE-NEXT:    maxsd %xmm0, %xmm1
; SSE-NEXT:    andnpd %xmm1, %xmm2
; SSE-NEXT:    orpd %xmm3, %xmm2
; SSE-NEXT:    movapd %xmm2, %xmm0
; SSE-NEXT:    retq
;
; AVX:         vmaxsd %xmm0, %xmm1, %xmm2
; AVX-NEXT:    vcmpunordsd %xmm0, %xmm0, %xmm0
; AVX-NEXT:    vblendvpd %xmm0, %xmm1, %xmm2, %xmm0
; AVX-NEXT:    retq
define double @test_fmax(double %x, double %y) {
  %z = call double @fmax(double %x, double %y) readnone
  ret double %z
}

; CHECK-LABEL: @test_fmaxl
; CHECK: callq fmaxl
define x86_fp80 @test_fmaxl(x86_fp80 %x, x86_fp80 %y) {
  %z = call x86_fp80 @fmaxl(x86_fp80 %x, x86_fp80 %y) readnone
  ret x86_fp80 %z
}

; CHECK-LABEL: @test_intrinsic_fmaxf
; SSE:         movaps %xmm0, %xmm2
; SSE-NEXT:    cmpunordss %xmm0, %xmm2
; SSE-NEXT:    movaps %xmm2, %xmm3
; SSE-NEXT:    andps %xmm1, %xmm3
; SSE-NEXT:    maxss %xmm0, %xmm1
; SSE-NEXT:    andnps %xmm1, %xmm2
; SSE-NEXT:    orps %xmm3, %xmm2
; SSE-NEXT:    movaps %xmm2, %xmm0
; SSE-NEXT:    retq
;
; AVX:         vmaxss %xmm0, %xmm1, %xmm2
; AVX-NEXT:    vcmpunordss %xmm0, %xmm0, %xmm0
; AVX-NEXT:    vblendvps %xmm0, %xmm1, %xmm2, %xmm0
; AVX-NEXT:    retq
define float @test_intrinsic_fmaxf(float %x, float %y) {
  %z = call float @llvm.maxnum.f32(float %x, float %y) readnone
  ret float %z
}


; CHECK-LABEL: @test_intrinsic_fmax
; SSE:         movapd %xmm0, %xmm2
; SSE-NEXT:    cmpunordsd %xmm0, %xmm2
; SSE-NEXT:    movapd %xmm2, %xmm3
; SSE-NEXT:    andpd %xmm1, %xmm3
; SSE-NEXT:    maxsd %xmm0, %xmm1
; SSE-NEXT:    andnpd %xmm1, %xmm2
; SSE-NEXT:    orpd %xmm3, %xmm2
; SSE-NEXT:    movapd %xmm2, %xmm0
; SSE-NEXT:    retq
;
; AVX:         vmaxsd %xmm0, %xmm1, %xmm2
; AVX-NEXT:    vcmpunordsd %xmm0, %xmm0, %xmm0
; AVX-NEXT:    vblendvpd %xmm0, %xmm1, %xmm2, %xmm0
; AVX-NEXT:    retq
define double @test_intrinsic_fmax(double %x, double %y) {
  %z = call double @llvm.maxnum.f64(double %x, double %y) readnone
  ret double %z
}

; CHECK-LABEL: @test_intrinsic_fmaxl
; CHECK: callq fmaxl
define x86_fp80 @test_intrinsic_fmaxl(x86_fp80 %x, x86_fp80 %y) {
  %z = call x86_fp80 @llvm.maxnum.f80(x86_fp80 %x, x86_fp80 %y) readnone
  ret x86_fp80 %z
}

; CHECK-LABEL: @test_intrinsic_fmax_v2f32
; SSE:         movaps %xmm1, %xmm2
; SSE-NEXT:    maxps %xmm0, %xmm2
; SSE-NEXT:    cmpunordps %xmm0, %xmm0
; SSE-NEXT:    andps %xmm0, %xmm1
; SSE-NEXT:    andnps %xmm2, %xmm0
; SSE-NEXT:    orps %xmm1, %xmm0
; SSE-NEXT:    retq
;
; AVX:         vmaxps %xmm0, %xmm1, %xmm2
; AVX-NEXT:    vcmpunordps %xmm0, %xmm0, %xmm0
; AVX-NEXT:    vblendvps %xmm0, %xmm1, %xmm2, %xmm0
; AVX-NEXT:    retq
define <2 x float> @test_intrinsic_fmax_v2f32(<2 x float> %x, <2 x float> %y) {
  %z = call <2 x float> @llvm.maxnum.v2f32(<2 x float> %x, <2 x float> %y) readnone
  ret <2 x float> %z
}

; CHECK-LABEL: @test_intrinsic_fmax_v4f32
; SSE:         movaps %xmm1, %xmm2
; SSE-NEXT:    maxps %xmm0, %xmm2
; SSE-NEXT:    cmpunordps %xmm0, %xmm0
; SSE-NEXT:    andps %xmm0, %xmm1
; SSE-NEXT:    andnps %xmm2, %xmm0
; SSE-NEXT:    orps %xmm1, %xmm0
; SSE-NEXT:    retq
;
; AVX:         vmaxps %xmm0, %xmm1, %xmm2
; AVX-NEXT:    vcmpunordps %xmm0, %xmm0, %xmm0
; AVX-NEXT:    vblendvps %xmm0, %xmm1, %xmm2, %xmm0
; AVX-NEXT:    retq
define <4 x float> @test_intrinsic_fmax_v4f32(<4 x float> %x, <4 x float> %y) {
  %z = call <4 x float> @llvm.maxnum.v4f32(<4 x float> %x, <4 x float> %y) readnone
  ret <4 x float> %z
}

; CHECK-LABEL: @test_intrinsic_fmax_v2f64
; SSE:         movapd %xmm1, %xmm2
; SSE-NEXT:    maxpd %xmm0, %xmm2
; SSE-NEXT:    cmpunordpd %xmm0, %xmm0
; SSE-NEXT:    andpd %xmm0, %xmm1
; SSE-NEXT:    andnpd %xmm2, %xmm0
; SSE-NEXT:    orpd %xmm1, %xmm0
; SSE-NEXT:    retq
;
; AVX:         vmaxpd %xmm0, %xmm1, %xmm2
; AVX-NEXT:    vcmpunordpd %xmm0, %xmm0, %xmm0
; AVX-NEXT:    vblendvpd %xmm0, %xmm1, %xmm2, %xmm0
; AVX-NEXT:    retq
define <2 x double> @test_intrinsic_fmax_v2f64(<2 x double> %x, <2 x double> %y) {
  %z = call <2 x double> @llvm.maxnum.v2f64(<2 x double> %x, <2 x double> %y) readnone
  ret <2 x double> %z
}

; CHECK-LABEL: @test_intrinsic_fmax_v4f64
; SSE:         movapd  %xmm2, %xmm4
; SSE-NEXT:    maxpd %xmm0, %xmm4
; SSE-NEXT:    cmpunordpd  %xmm0, %xmm0
; SSE-NEXT:    andpd %xmm0, %xmm2
; SSE-NEXT:    andnpd  %xmm4, %xmm0
; SSE-NEXT:    orpd  %xmm2, %xmm0
; SSE-NEXT:    movapd  %xmm3, %xmm2
; SSE-NEXT:    maxpd %xmm1, %xmm2
; SSE-NEXT:    cmpunordpd  %xmm1, %xmm1
; SSE-NEXT:    andpd %xmm1, %xmm3
; SSE-NEXT:    andnpd  %xmm2, %xmm1
; SSE-NEXT:    orpd  %xmm3, %xmm1
; SSE-NEXT:    retq
;
; AVX:         vmaxpd  %ymm0, %ymm1, %ymm2
; AVX-NEXT:    vcmpunordpd %ymm0, %ymm0, %ymm0
; AVX-NEXT:    vblendvpd %ymm0, %ymm1, %ymm2, %ymm0
; AVX-NEXT:    retq
define <4 x double> @test_intrinsic_fmax_v4f64(<4 x double> %x, <4 x double> %y) {
  %z = call <4 x double> @llvm.maxnum.v4f64(<4 x double> %x, <4 x double> %y) readnone
  ret <4 x double> %z
}

; CHECK-LABEL: @test_intrinsic_fmax_v8f64
; SSE:         movapd  %xmm4, %xmm8
; SSE-NEXT:    maxpd %xmm0, %xmm8
; SSE-NEXT:    cmpunordpd  %xmm0, %xmm0
; SSE-NEXT:    andpd %xmm0, %xmm4
; SSE-NEXT:    andnpd  %xmm8, %xmm0
; SSE-NEXT:    orpd  %xmm4, %xmm0
; SSE-NEXT:    movapd  %xmm5, %xmm4
; SSE-NEXT:    maxpd %xmm1, %xmm4
; SSE-NEXT:    cmpunordpd  %xmm1, %xmm1
; SSE-NEXT:    andpd %xmm1, %xmm5
; SSE-NEXT:    andnpd  %xmm4, %xmm1
; SSE-NEXT:    orpd  %xmm5, %xmm1
; SSE-NEXT:    movapd  %xmm6, %xmm4
; SSE-NEXT:    maxpd %xmm2, %xmm4
; SSE-NEXT:    cmpunordpd  %xmm2, %xmm2
; SSE-NEXT:    andpd %xmm2, %xmm6
; SSE-NEXT:    andnpd  %xmm4, %xmm2
; SSE-NEXT:    orpd  %xmm6, %xmm2
; SSE-NEXT:    movapd  %xmm7, %xmm4
; SSE-NEXT:    maxpd %xmm3, %xmm4
; SSE-NEXT:    cmpunordpd  %xmm3, %xmm3
; SSE-NEXT:    andpd %xmm3, %xmm7
; SSE-NEXT:    andnpd  %xmm4, %xmm3
; SSE-NEXT:    orpd  %xmm7, %xmm3
; SSE-NEXT:    retq
;
; AVX:         vmaxpd  %ymm0, %ymm2, %ymm4
; AVX-NEXT:    vcmpunordpd %ymm0, %ymm0, %ymm0
; AVX-NEXT:    vblendvpd %ymm0, %ymm2, %ymm4, %ymm0
; AVX-NEXT:    vmaxpd  %ymm1, %ymm3, %ymm2
; AVX-NEXT:    vcmpunordpd %ymm1, %ymm1, %ymm1
; AVX-NEXT:    vblendvpd %ymm1, %ymm3, %ymm2, %ymm1
; AVX-NEXT:    retq
define <8 x double> @test_intrinsic_fmax_v8f64(<8 x double> %x, <8 x double> %y) {
  %z = call <8 x double> @llvm.maxnum.v8f64(<8 x double> %x, <8 x double> %y) readnone
  ret <8 x double> %z
}

