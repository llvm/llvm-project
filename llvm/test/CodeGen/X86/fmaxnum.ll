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


; CHECK-LABEL: @test_fmaxf
; SSE:         movaps %xmm0, %xmm2
; SSE-NEXT:    cmpunordss %xmm2, %xmm2
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

; FIXME: Doubles should be inlined similarly to floats.

; CHECK-LABEL: @test_fmax
; CHECK: jmp fmax
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
; SSE-NEXT:    cmpunordss %xmm2, %xmm2
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

; FIXME: Doubles should be inlined similarly to floats.

; CHECK-LABEL: @test_intrinsic_fmax
; CHECK: jmp fmax
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

; FIXME: This should not be doing 4 scalar ops on a 2 element vector.
; FIXME: This should use vector ops (maxps / cmpps).

; CHECK-LABEL: @test_intrinsic_fmax_v2f32
; SSE:         movaps %xmm1, %xmm2
; SSE-NEXT:    shufps {{.*#+}} xmm2 = xmm2[3,1,2,3]
; SSE-NEXT:    movaps %xmm0, %xmm3
; SSE-NEXT:    shufps {{.*#+}} xmm3 = xmm3[3,1,2,3]
; SSE-NEXT:    movaps %xmm3, %xmm4
; SSE-NEXT:    cmpunordss %xmm4, %xmm4
; SSE-NEXT:    movaps %xmm4, %xmm5
; SSE-NEXT:    andps %xmm2, %xmm5
; SSE-NEXT:    maxss %xmm3, %xmm2
; SSE-NEXT:    andnps %xmm2, %xmm4
; SSE-NEXT:    orps %xmm5, %xmm4
; SSE-NEXT:    movaps %xmm1, %xmm2
; SSE-NEXT:    shufps {{.*#+}} xmm2 = xmm2[1,1,2,3]
; SSE-NEXT:    movaps %xmm0, %xmm5
; SSE-NEXT:    shufps {{.*#+}} xmm5 = xmm5[1,1,2,3]
; SSE-NEXT:    movaps %xmm5, %xmm3
; SSE-NEXT:    cmpunordss %xmm3, %xmm3
; SSE-NEXT:    movaps %xmm3, %xmm6
; SSE-NEXT:    andps %xmm2, %xmm6
; SSE-NEXT:    maxss %xmm5, %xmm2
; SSE-NEXT:    andnps %xmm2, %xmm3
; SSE-NEXT:    orps %xmm6, %xmm3
; SSE-NEXT:    unpcklps {{.*#+}} xmm3 = xmm3[0],xmm4[0],xmm3[1],xmm4[1]
; SSE-NEXT:    movaps %xmm0, %xmm2
; SSE-NEXT:    cmpunordss %xmm2, %xmm2
; SSE-NEXT:    movaps %xmm2, %xmm4
; SSE-NEXT:    andps %xmm1, %xmm4
; SSE-NEXT:    movaps %xmm1, %xmm5
; SSE-NEXT:    maxss %xmm0, %xmm5
; SSE-NEXT:    andnps %xmm5, %xmm2
; SSE-NEXT:    orps %xmm4, %xmm2
; SSE-NEXT:    shufpd {{.*#+}} xmm1 = xmm1[1,0]
; SSE-NEXT:    shufpd {{.*#+}} xmm0 = xmm0[1,0]
; SSE-NEXT:    movapd %xmm0, %xmm4
; SSE-NEXT:    cmpunordss %xmm4, %xmm4
; SSE-NEXT:    movaps %xmm4, %xmm5
; SSE-NEXT:    andps %xmm1, %xmm5
; SSE-NEXT:    maxss %xmm0, %xmm1
; SSE-NEXT:    andnps %xmm1, %xmm4
; SSE-NEXT:    orps %xmm5, %xmm4
; SSE-NEXT:    unpcklps {{.*#+}} xmm2 = xmm2[0],xmm4[0],xmm2[1],xmm4[1]
; SSE-NEXT:    unpcklps {{.*#+}} xmm2 = xmm2[0],xmm3[0],xmm2[1],xmm3[1]
; SSE-NEXT:    movaps %xmm2, %xmm0
; SSE-NEXT:    retq
;
; AVX:         vmaxss %xmm0, %xmm1, %xmm2
; AVX-NEXT:    vcmpunordss %xmm0, %xmm0, %xmm3
; AVX-NEXT:    vblendvps %xmm3, %xmm1, %xmm2, %xmm2
; AVX-NEXT:    vmovshdup {{.*#+}} xmm3 = xmm0[1,1,3,3]
; AVX-NEXT:    vmovshdup {{.*#+}} xmm4 = xmm1[1,1,3,3]
; AVX-NEXT:    vmaxss %xmm3, %xmm4, %xmm5
; AVX-NEXT:    vcmpunordss %xmm3, %xmm3, %xmm3
; AVX-NEXT:    vblendvps %xmm3, %xmm4, %xmm5, %xmm3
; AVX-NEXT:    vinsertps {{.*#+}} xmm2 = xmm2[0],xmm3[0],xmm2[2,3]
; AVX-NEXT:    vpermilpd {{.*#+}} xmm3 = xmm0[1,0]
; AVX-NEXT:    vpermilpd {{.*#+}} xmm4 = xmm1[1,0]
; AVX-NEXT:    vmaxss %xmm3, %xmm4, %xmm5
; AVX-NEXT:    vcmpunordss %xmm3, %xmm3, %xmm3
; AVX-NEXT:    vblendvps %xmm3, %xmm4, %xmm5, %xmm3
; AVX-NEXT:    vinsertps {{.*#+}} xmm2 = xmm2[0,1],xmm3[0],xmm2[3]
; AVX-NEXT:    vpermilps {{.*#+}} xmm0 = xmm0[3,1,2,3]
; AVX-NEXT:    vpermilps {{.*#+}} xmm1 = xmm1[3,1,2,3]
; AVX-NEXT:    vmaxss %xmm0, %xmm1, %xmm3
; AVX-NEXT:    vcmpunordss %xmm0, %xmm0, %xmm0
; AVX-NEXT:    vblendvps %xmm0, %xmm1, %xmm3, %xmm0
; AVX-NEXT:    vinsertps {{.*#+}} xmm0 = xmm2[0,1,2],xmm0[0]
; AVX-NEXT:    retq
define <2 x float> @test_intrinsic_fmax_v2f32(<2 x float> %x, <2 x float> %y) {
  %z = call <2 x float> @llvm.maxnum.v2f32(<2 x float> %x, <2 x float> %y) readnone
  ret <2 x float> %z
}

; FIXME: This should use vector ops (maxps / cmpps).

; CHECK-LABEL: @test_intrinsic_fmax_v4f32
; SSE:         movaps %xmm1, %xmm2
; SSE-NEXT:    shufps {{.*#+}} xmm2 = xmm2[3,1,2,3]
; SSE-NEXT:    movaps %xmm0, %xmm3
; SSE-NEXT:    shufps {{.*#+}} xmm3 = xmm3[3,1,2,3]
; SSE-NEXT:    movaps %xmm3, %xmm4
; SSE-NEXT:    cmpunordss %xmm4, %xmm4
; SSE-NEXT:    movaps %xmm4, %xmm5
; SSE-NEXT:    andps %xmm2, %xmm5
; SSE-NEXT:    maxss %xmm3, %xmm2
; SSE-NEXT:    andnps %xmm2, %xmm4
; SSE-NEXT:    orps %xmm5, %xmm4
; SSE-NEXT:    movaps %xmm1, %xmm2
; SSE-NEXT:    shufps {{.*#+}} xmm2 = xmm2[1,1,2,3]
; SSE-NEXT:    movaps %xmm0, %xmm5
; SSE-NEXT:    shufps {{.*#+}} xmm5 = xmm5[1,1,2,3]
; SSE-NEXT:    movaps %xmm5, %xmm3
; SSE-NEXT:    cmpunordss %xmm3, %xmm3
; SSE-NEXT:    movaps %xmm3, %xmm6
; SSE-NEXT:    andps %xmm2, %xmm6
; SSE-NEXT:    maxss %xmm5, %xmm2
; SSE-NEXT:    andnps %xmm2, %xmm3
; SSE-NEXT:    orps %xmm6, %xmm3
; SSE-NEXT:    unpcklps {{.*#+}} xmm3 = xmm3[0],xmm4[0],xmm3[1],xmm4[1]
; SSE-NEXT:    movaps %xmm0, %xmm2
; SSE-NEXT:    cmpunordss %xmm2, %xmm2
; SSE-NEXT:    movaps %xmm2, %xmm4
; SSE-NEXT:    andps %xmm1, %xmm4
; SSE-NEXT:    movaps %xmm1, %xmm5
; SSE-NEXT:    maxss %xmm0, %xmm5
; SSE-NEXT:    andnps %xmm5, %xmm2
; SSE-NEXT:    orps %xmm4, %xmm2
; SSE-NEXT:    shufpd {{.*#+}} xmm1 = xmm1[1,0]
; SSE-NEXT:    shufpd {{.*#+}} xmm0 = xmm0[1,0]
; SSE-NEXT:    movapd %xmm0, %xmm4
; SSE-NEXT:    cmpunordss %xmm4, %xmm4
; SSE-NEXT:    movaps %xmm4, %xmm5
; SSE-NEXT:    andps %xmm1, %xmm5
; SSE-NEXT:    maxss %xmm0, %xmm1
; SSE-NEXT:    andnps %xmm1, %xmm4
; SSE-NEXT:    orps %xmm5, %xmm4
; SSE-NEXT:    unpcklps {{.*#+}} xmm2 = xmm2[0],xmm4[0],xmm2[1],xmm4[1]
; SSE-NEXT:    unpcklps {{.*#+}} xmm2 = xmm2[0],xmm3[0],xmm2[1],xmm3[1]
; SSE-NEXT:    movaps %xmm2, %xmm0
; SSE-NEXT:    retq
;
; AVX:         vmaxss %xmm0, %xmm1, %xmm2
; AVX-NEXT:    vcmpunordss %xmm0, %xmm0, %xmm3
; AVX-NEXT:    vblendvps %xmm3, %xmm1, %xmm2, %xmm2
; AVX-NEXT:    vmovshdup {{.*#+}} xmm3 = xmm0[1,1,3,3]
; AVX-NEXT:    vmovshdup {{.*#+}} xmm4 = xmm1[1,1,3,3]
; AVX-NEXT:    vmaxss %xmm3, %xmm4, %xmm5
; AVX-NEXT:    vcmpunordss %xmm3, %xmm3, %xmm3
; AVX-NEXT:    vblendvps %xmm3, %xmm4, %xmm5, %xmm3
; AVX-NEXT:    vinsertps {{.*#+}} xmm2 = xmm2[0],xmm3[0],xmm2[2,3]
; AVX-NEXT:    vpermilpd {{.*#+}} xmm3 = xmm0[1,0]
; AVX-NEXT:    vpermilpd {{.*#+}} xmm4 = xmm1[1,0]
; AVX-NEXT:    vmaxss %xmm3, %xmm4, %xmm5
; AVX-NEXT:    vcmpunordss %xmm3, %xmm3, %xmm3
; AVX-NEXT:    vblendvps %xmm3, %xmm4, %xmm5, %xmm3
; AVX-NEXT:    vinsertps {{.*#+}} xmm2 = xmm2[0,1],xmm3[0],xmm2[3]
; AVX-NEXT:    vpermilps {{.*#+}} xmm0 = xmm0[3,1,2,3]
; AVX-NEXT:    vpermilps {{.*#+}} xmm1 = xmm1[3,1,2,3]
; AVX-NEXT:    vmaxss %xmm0, %xmm1, %xmm3
; AVX-NEXT:    vcmpunordss %xmm0, %xmm0, %xmm0
; AVX-NEXT:    vblendvps %xmm0, %xmm1, %xmm3, %xmm0
; AVX-NEXT:    vinsertps {{.*#+}} xmm0 = xmm2[0,1,2],xmm0[0]
; AVX-NEXT:    retq
define <4 x float> @test_intrinsic_fmax_v4f32(<4 x float> %x, <4 x float> %y) {
  %z = call <4 x float> @llvm.maxnum.v4f32(<4 x float> %x, <4 x float> %y) readnone
  ret <4 x float> %z
}

; FIXME: Vector of doubles should be inlined similarly to vector of floats.

; CHECK-LABEL: @test_intrinsic_fmax_v2f64
; CHECK: callq fmax
; CHECK: callq fmax
define <2 x double> @test_intrinsic_fmax_v2f64(<2 x double> %x, <2 x double> %y) {
  %z = call <2 x double> @llvm.maxnum.v2f64(<2 x double> %x, <2 x double> %y) readnone
  ret <2 x double> %z
}

; FIXME: Vector of doubles should be inlined similarly to vector of floats.

; CHECK-LABEL: @test_intrinsic_fmax_v4f64
; CHECK: callq fmax
; CHECK: callq fmax
; CHECK: callq fmax
; CHECK: callq fmax
define <4 x double> @test_intrinsic_fmax_v4f64(<4 x double> %x, <4 x double> %y) {
  %z = call <4 x double> @llvm.maxnum.v4f64(<4 x double> %x, <4 x double> %y) readnone
  ret <4 x double> %z
}

; FIXME: Vector of doubles should be inlined similarly to vector of floats.

; CHECK-LABEL: @test_intrinsic_fmax_v8f64
; CHECK: callq fmax
; CHECK: callq fmax
; CHECK: callq fmax
; CHECK: callq fmax
; CHECK: callq fmax
; CHECK: callq fmax
; CHECK: callq fmax
; CHECK: callq fmax
define <8 x double> @test_intrinsic_fmax_v8f64(<8 x double> %x, <8 x double> %y) {
  %z = call <8 x double> @llvm.maxnum.v8f64(<8 x double> %x, <8 x double> %y) readnone
  ret <8 x double> %z
}

