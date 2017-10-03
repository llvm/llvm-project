; RUN: llc -mtriple=x86_64-unknown-unknown -mattr=sse2  < %s | FileCheck %s --check-prefix=CHECK --check-prefix=SSE
; RUN: llc -mtriple=x86_64-unknown-unknown -mattr=avx  < %s | FileCheck %s --check-prefix=CHECK --check-prefix=AVX

declare float @fminf(float, float)
declare double @fmin(double, double)
declare x86_fp80 @fminl(x86_fp80, x86_fp80)
declare float @llvm.minnum.f32(float, float)
declare double @llvm.minnum.f64(double, double)
declare x86_fp80 @llvm.minnum.f80(x86_fp80, x86_fp80)

declare <2 x float> @llvm.minnum.v2f32(<2 x float>, <2 x float>)
declare <4 x float> @llvm.minnum.v4f32(<4 x float>, <4 x float>)
declare <2 x double> @llvm.minnum.v2f64(<2 x double>, <2 x double>)
declare <4 x double> @llvm.minnum.v4f64(<4 x double>, <4 x double>)
declare <8 x double> @llvm.minnum.v8f64(<8 x double>, <8 x double>)

; FIXME: As the vector tests show, the SSE run shouldn't need this many moves.

; CHECK-LABEL: @test_fminf
; SSE:         movaps %xmm0, %xmm2
; SSE-NEXT:    cmpunordss %xmm2, %xmm2
; SSE-NEXT:    movaps %xmm2, %xmm3
; SSE-NEXT:    andps %xmm1, %xmm3
; SSE-NEXT:    minss %xmm0, %xmm1
; SSE-NEXT:    andnps %xmm1, %xmm2
; SSE-NEXT:    orps %xmm3, %xmm2
; SSE-NEXT:    movaps %xmm2, %xmm0
; SSE-NEXT:    retq
;
; AVX:         vminss %xmm0, %xmm1, %xmm2
; AVX-NEXT:    vcmpunordss %xmm0, %xmm0, %xmm0
; AVX-NEXT:    vblendvps %xmm0, %xmm1, %xmm2, %xmm0
; AVX-NEXT:    retq
define float @test_fminf(float %x, float %y) {
  %z = call float @fminf(float %x, float %y) readnone
  ret float %z
}

; FIXME: As the vector tests show, the SSE run shouldn't need this many moves.

; CHECK-LABEL: @test_fmin
; SSE:         movapd %xmm0, %xmm2
; SSE-NEXT:    cmpunordsd %xmm2, %xmm2
; SSE-NEXT:    movapd %xmm2, %xmm3
; SSE-NEXT:    andpd %xmm1, %xmm3
; SSE-NEXT:    minsd %xmm0, %xmm1
; SSE-NEXT:    andnpd %xmm1, %xmm2
; SSE-NEXT:    orpd %xmm3, %xmm2
; SSE-NEXT:    movapd %xmm2, %xmm0
; SSE-NEXT:    retq
;
; AVX:         vminsd %xmm0, %xmm1, %xmm2
; AVX-NEXT:    vcmpunordsd %xmm0, %xmm0, %xmm0
; AVX-NEXT:    vblendvpd %xmm0, %xmm1, %xmm2, %xmm0
; AVX-NEXT:    retq
define double @test_fmin(double %x, double %y) {
  %z = call double @fmin(double %x, double %y) readnone
  ret double %z
}

; CHECK-LABEL: @test_fminl
; CHECK: callq fminl
define x86_fp80 @test_fminl(x86_fp80 %x, x86_fp80 %y) {
  %z = call x86_fp80 @fminl(x86_fp80 %x, x86_fp80 %y) readnone
  ret x86_fp80 %z
}

; CHECK-LABEL: @test_intrinsic_fminf
; SSE:         movaps %xmm0, %xmm2
; SSE-NEXT:    cmpunordss %xmm2, %xmm2
; SSE-NEXT:    movaps %xmm2, %xmm3
; SSE-NEXT:    andps %xmm1, %xmm3
; SSE-NEXT:    minss %xmm0, %xmm1
; SSE-NEXT:    andnps %xmm1, %xmm2
; SSE-NEXT:    orps %xmm3, %xmm2
; SSE-NEXT:    movaps %xmm2, %xmm0
; SSE-NEXT:    retq
;
; AVX:         vminss %xmm0, %xmm1, %xmm2
; AVX-NEXT:    vcmpunordss %xmm0, %xmm0, %xmm0
; AVX-NEXT:    vblendvps %xmm0, %xmm1, %xmm2, %xmm0
; AVX-NEXT:    retq
define float @test_intrinsic_fminf(float %x, float %y) {
  %z = call float @llvm.minnum.f32(float %x, float %y) readnone
  ret float %z
}

; CHECK-LABEL: @test_intrinsic_fmin
; SSE:         movapd %xmm0, %xmm2
; SSE-NEXT:    cmpunordsd %xmm2, %xmm2
; SSE-NEXT:    movapd %xmm2, %xmm3
; SSE-NEXT:    andpd %xmm1, %xmm3
; SSE-NEXT:    minsd %xmm0, %xmm1
; SSE-NEXT:    andnpd %xmm1, %xmm2
; SSE-NEXT:    orpd %xmm3, %xmm2
; SSE-NEXT:    movapd %xmm2, %xmm0
; SSE-NEXT:    retq
;
; AVX:         vminsd %xmm0, %xmm1, %xmm2
; AVX-NEXT:    vcmpunordsd %xmm0, %xmm0, %xmm0
; AVX-NEXT:    vblendvpd %xmm0, %xmm1, %xmm2, %xmm0
; AVX-NEXT:    retq
define double @test_intrinsic_fmin(double %x, double %y) {
  %z = call double @llvm.minnum.f64(double %x, double %y) readnone
  ret double %z
}

; CHECK-LABEL: @test_intrinsic_fminl
; CHECK: callq fminl
define x86_fp80 @test_intrinsic_fminl(x86_fp80 %x, x86_fp80 %y) {
  %z = call x86_fp80 @llvm.minnum.f80(x86_fp80 %x, x86_fp80 %y) readnone
  ret x86_fp80 %z
}

; CHECK-LABEL: @test_intrinsic_fmin_v2f32
; SSE:         movaps %xmm1, %xmm2
; SSE-NEXT:    minps %xmm0, %xmm2
; SSE-NEXT:    cmpunordps %xmm0, %xmm0
; SSE-NEXT:    andps %xmm0, %xmm1
; SSE-NEXT:    andnps %xmm2, %xmm0
; SSE-NEXT:    orps %xmm1, %xmm0
; SSE-NEXT:    retq
;
; AVX:         vminps %xmm0, %xmm1, %xmm2
; AVX-NEXT:    vcmpunordps %xmm0, %xmm0, %xmm0
; AVX-NEXT:    vblendvps %xmm0, %xmm1, %xmm2, %xmm0
; AVX-NEXT:    retq
define <2 x float> @test_intrinsic_fmin_v2f32(<2 x float> %x, <2 x float> %y) {
  %z = call <2 x float> @llvm.minnum.v2f32(<2 x float> %x, <2 x float> %y) readnone
  ret <2 x float> %z
}

; CHECK-LABEL: @test_intrinsic_fmin_v4f32
; SSE:         movaps %xmm1, %xmm2
; SSE-NEXT:    minps %xmm0, %xmm2
; SSE-NEXT:    cmpunordps %xmm0, %xmm0
; SSE-NEXT:    andps %xmm0, %xmm1
; SSE-NEXT:    andnps %xmm2, %xmm0
; SSE-NEXT:    orps %xmm1, %xmm0
; SSE-NEXT:    retq
;
; AVX:         vminps %xmm0, %xmm1, %xmm2
; AVX-NEXT:    vcmpunordps %xmm0, %xmm0, %xmm0
; AVX-NEXT:    vblendvps %xmm0, %xmm1, %xmm2, %xmm0
; AVX-NEXT:    retq
define <4 x float> @test_intrinsic_fmin_v4f32(<4 x float> %x, <4 x float> %y) {
  %z = call <4 x float> @llvm.minnum.v4f32(<4 x float> %x, <4 x float> %y) readnone
  ret <4 x float> %z
}

; CHECK-LABEL: @test_intrinsic_fmin_v2f64
; SSE:         movapd %xmm1, %xmm2
; SSE-NEXT:    minpd %xmm0, %xmm2
; SSE-NEXT:    cmpunordpd %xmm0, %xmm0
; SSE-NEXT:    andpd %xmm0, %xmm1
; SSE-NEXT:    andnpd %xmm2, %xmm0
; SSE-NEXT:    orpd %xmm1, %xmm0
; SSE-NEXT:    retq
;
; AVX:         vminpd %xmm0, %xmm1, %xmm2
; AVX-NEXT:    vcmpunordpd %xmm0, %xmm0, %xmm0
; AVX-NEXT:    vblendvpd %xmm0, %xmm1, %xmm2, %xmm0
; AVX-NEXT:    retq
define <2 x double> @test_intrinsic_fmin_v2f64(<2 x double> %x, <2 x double> %y) {
  %z = call <2 x double> @llvm.minnum.v2f64(<2 x double> %x, <2 x double> %y) readnone
  ret <2 x double> %z
}

; CHECK-LABEL: @test_intrinsic_fmin_v4f64
; SSE:         movapd  %xmm2, %xmm4
; SSE-NEXT:    minpd %xmm0, %xmm4
; SSE-NEXT:    cmpunordpd  %xmm0, %xmm0
; SSE-NEXT:    andpd %xmm0, %xmm2
; SSE-NEXT:    andnpd  %xmm4, %xmm0
; SSE-NEXT:    orpd  %xmm2, %xmm0
; SSE-NEXT:    movapd  %xmm3, %xmm2
; SSE-NEXT:    minpd %xmm1, %xmm2
; SSE-NEXT:    cmpunordpd  %xmm1, %xmm1
; SSE-NEXT:    andpd %xmm1, %xmm3
; SSE-NEXT:    andnpd  %xmm2, %xmm1
; SSE-NEXT:    orpd  %xmm3, %xmm1
; SSE-NEXT:    retq
;
; AVX:         vminpd  %ymm0, %ymm1, %ymm2
; AVX-NEXT:    vcmpunordpd %ymm0, %ymm0, %ymm0
; AVX-NEXT:    vblendvpd %ymm0, %ymm1, %ymm2, %ymm0
; AVX-NEXT:    retq
define <4 x double> @test_intrinsic_fmin_v4f64(<4 x double> %x, <4 x double> %y) {
  %z = call <4 x double> @llvm.minnum.v4f64(<4 x double> %x, <4 x double> %y) readnone
  ret <4 x double> %z
}

; CHECK-LABEL: @test_intrinsic_fmin_v8f64
; SSE:         movapd  %xmm4, %xmm8
; SSE-NEXT:    minpd %xmm0, %xmm8
; SSE-NEXT:    cmpunordpd  %xmm0, %xmm0
; SSE-NEXT:    andpd %xmm0, %xmm4
; SSE-NEXT:    andnpd  %xmm8, %xmm0
; SSE-NEXT:    orpd  %xmm4, %xmm0
; SSE-NEXT:    movapd  %xmm5, %xmm4
; SSE-NEXT:    minpd %xmm1, %xmm4
; SSE-NEXT:    cmpunordpd  %xmm1, %xmm1
; SSE-NEXT:    andpd %xmm1, %xmm5
; SSE-NEXT:    andnpd  %xmm4, %xmm1
; SSE-NEXT:    orpd  %xmm5, %xmm1
; SSE-NEXT:    movapd  %xmm6, %xmm4
; SSE-NEXT:    minpd %xmm2, %xmm4
; SSE-NEXT:    cmpunordpd  %xmm2, %xmm2
; SSE-NEXT:    andpd %xmm2, %xmm6
; SSE-NEXT:    andnpd  %xmm4, %xmm2
; SSE-NEXT:    orpd  %xmm6, %xmm2
; SSE-NEXT:    movapd  %xmm7, %xmm4
; SSE-NEXT:    minpd %xmm3, %xmm4
; SSE-NEXT:    cmpunordpd  %xmm3, %xmm3
; SSE-NEXT:    andpd %xmm3, %xmm7
; SSE-NEXT:    andnpd  %xmm4, %xmm3
; SSE-NEXT:    orpd  %xmm7, %xmm3
; SSE-NEXT:    retq
;
; AVX:         vminpd  %ymm0, %ymm2, %ymm4
; AVX-NEXT:    vcmpunordpd %ymm0, %ymm0, %ymm0
; AVX-NEXT:    vblendvpd %ymm0, %ymm2, %ymm4, %ymm0
; AVX-NEXT:    vminpd  %ymm1, %ymm3, %ymm2
; AVX-NEXT:    vcmpunordpd %ymm1, %ymm1, %ymm1
; AVX-NEXT:    vblendvpd %ymm1, %ymm3, %ymm2, %ymm1
; AVX-NEXT:    retq
define <8 x double> @test_intrinsic_fmin_v8f64(<8 x double> %x, <8 x double> %y) {
  %z = call <8 x double> @llvm.minnum.v8f64(<8 x double> %x, <8 x double> %y) readnone
  ret <8 x double> %z
}
