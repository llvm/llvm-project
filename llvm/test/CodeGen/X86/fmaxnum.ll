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
; CHECK: jmp fmaxf
define float @test_fmaxf(float %x, float %y) {
  %z = call float @fmaxf(float %x, float %y) readnone
  ret float %z
}

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
; CHECK: jmp fmaxf
define float @test_intrinsic_fmaxf(float %x, float %y) {
  %z = call float @llvm.maxnum.f32(float %x, float %y) readnone
  ret float %z
}

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

; CHECK-LABEL: @test_intrinsic_fmax_v2f32
; SSE:         movaps %xmm1, {{[0-9]+}}(%rsp) # 16-byte Spill
; SSE-NEXT:    movaps %xmm0, {{[0-9]+}}(%rsp) # 16-byte Spill
; SSE-NEXT:    shufps {{.*#+}} xmm0 = xmm0[3,1,2,3]
; SSE-NEXT:    shufps {{.*#+}} xmm1 = xmm1[3,1,2,3]
; SSE-NEXT:    callq fmaxf
; SSE-NEXT:    movaps %xmm0, {{[0-9]+}}(%rsp) # 16-byte Spill
; SSE-NEXT:    movaps {{[0-9]+}}(%rsp), %xmm0 # 16-byte Reload
; SSE-NEXT:    shufps {{.*#+}} xmm0 = xmm0[1,1,2,3]
; SSE-NEXT:    movaps {{[0-9]+}}(%rsp), %xmm1 # 16-byte Reload
; SSE-NEXT:    shufps {{.*#+}} xmm1 = xmm1[1,1,2,3]
; SSE-NEXT:    callq fmaxf
; SSE-NEXT:    unpcklps {{[0-9]+}}(%rsp), %xmm0 # 16-byte Folded Reload
; SSE:         movaps %xmm0, {{[0-9]+}}(%rsp) # 16-byte Spill
; SSE-NEXT:    movaps {{[0-9]+}}(%rsp), %xmm0 # 16-byte Reload
; SSE-NEXT:    movaps {{[0-9]+}}(%rsp), %xmm1 # 16-byte Reload
; SSE-NEXT:    callq fmaxf
; SSE-NEXT:    movaps %xmm0, (%rsp) # 16-byte Spill
; SSE-NEXT:    movapd {{[0-9]+}}(%rsp), %xmm0 # 16-byte Reload
; SSE-NEXT:    shufpd {{.*#+}} xmm0 = xmm0[1,0]
; SSE-NEXT:    movapd {{[0-9]+}}(%rsp), %xmm1 # 16-byte Reload
; SSE-NEXT:    shufpd {{.*#+}} xmm1 = xmm1[1,0]
; SSE-NEXT:    callq fmaxf
; SSE-NEXT:    movaps (%rsp), %xmm1 # 16-byte Reload
; SSE-NEXT:    unpcklps {{.*#+}} xmm1 = xmm1[0],xmm0[0],xmm1[1],xmm0[1]
; SSE-NEXT:    unpcklps {{[0-9]+}}(%rsp), %xmm1 # 16-byte Folded Reload
; SSE:         movaps %xmm1, %xmm0
; SSE-NEXT:    addq $72, %rsp
; SSE-NEXT:    retq
;
; AVX:         vmovaps %xmm1, {{[0-9]+}}(%rsp) # 16-byte Spill
; AVX-NEXT:    vmovaps %xmm0, {{[0-9]+}}(%rsp) # 16-byte Spill
; AVX-NEXT:    callq fmaxf
; AVX-NEXT:    vmovaps %xmm0, (%rsp) # 16-byte Spill
; AVX-NEXT:    vmovshdup {{[0-9]+}}(%rsp), %xmm0 # 16-byte Folded Reload
; AVX:         vmovshdup {{[0-9]+}}(%rsp), %xmm1 # 16-byte Folded Reload
; AVX:         callq fmaxf
; AVX-NEXT:    vmovaps (%rsp), %xmm1 # 16-byte Reload
; AVX-NEXT:    vinsertps {{.*#+}} xmm0 = xmm1[0],xmm0[0],xmm1[2,3]
; AVX-NEXT:    vmovaps %xmm0, (%rsp) # 16-byte Spill
; AVX-NEXT:    vpermilpd $1, {{[0-9]+}}(%rsp), %xmm0 # 16-byte Folded Reload
; AVX:         vpermilpd $1, {{[0-9]+}}(%rsp), %xmm1 # 16-byte Folded Reload
; AVX:         callq fmaxf
; AVX-NEXT:    vmovaps (%rsp), %xmm1 # 16-byte Reload
; AVX-NEXT:    vinsertps {{.*#+}} xmm0 = xmm1[0,1],xmm0[0],xmm1[3]
; AVX-NEXT:    vmovaps %xmm0, (%rsp) # 16-byte Spill
; AVX-NEXT:    vpermilps $231, {{[0-9]+}}(%rsp), %xmm0 # 16-byte Folded Reload
; AVX:         vpermilps $231, {{[0-9]+}}(%rsp), %xmm1 # 16-byte Folded Reload
; AVX:         callq fmaxf
; AVX-NEXT:    vmovaps (%rsp), %xmm1 # 16-byte Reload
; AVX-NEXT:    vinsertps {{.*#+}} xmm0 = xmm1[0,1,2],xmm0[0]
; AVX-NEXT:    addq $56, %rsp
; AVX-NEXT:    retq
define <2 x float> @test_intrinsic_fmax_v2f32(<2 x float> %x, <2 x float> %y) {
  %z = call <2 x float> @llvm.maxnum.v2f32(<2 x float> %x, <2 x float> %y) readnone
  ret <2 x float> %z
}

; CHECK-LABEL: @test_intrinsic_fmax_v4f32
; SSE:         movaps %xmm1, {{[0-9]+}}(%rsp) # 16-byte Spill
; SSE-NEXT:    movaps %xmm0, {{[0-9]+}}(%rsp) # 16-byte Spill
; SSE-NEXT:    shufps {{.*#+}} xmm0 = xmm0[3,1,2,3]
; SSE-NEXT:    shufps {{.*#+}} xmm1 = xmm1[3,1,2,3]
; SSE-NEXT:    callq fmaxf
; SSE-NEXT:    movaps %xmm0, {{[0-9]+}}(%rsp) # 16-byte Spill
; SSE-NEXT:    movaps {{[0-9]+}}(%rsp), %xmm0 # 16-byte Reload
; SSE-NEXT:    shufps {{.*#+}} xmm0 = xmm0[1,1,2,3]
; SSE-NEXT:    movaps {{[0-9]+}}(%rsp), %xmm1 # 16-byte Reload
; SSE-NEXT:    shufps {{.*#+}} xmm1 = xmm1[1,1,2,3]
; SSE-NEXT:    callq fmaxf
; SSE-NEXT:    unpcklps {{[0-9]+}}(%rsp), %xmm0 # 16-byte Folded Reload
; SSE:         movaps %xmm0, {{[0-9]+}}(%rsp) # 16-byte Spill
; SSE-NEXT:    movaps {{[0-9]+}}(%rsp), %xmm0 # 16-byte Reload
; SSE-NEXT:    movaps {{[0-9]+}}(%rsp), %xmm1 # 16-byte Reload
; SSE-NEXT:    callq fmaxf
; SSE-NEXT:    movaps %xmm0, (%rsp) # 16-byte Spill
; SSE-NEXT:    movapd {{[0-9]+}}(%rsp), %xmm0 # 16-byte Reload
; SSE-NEXT:    shufpd {{.*#+}} xmm0 = xmm0[1,0]
; SSE-NEXT:    movapd {{[0-9]+}}(%rsp), %xmm1 # 16-byte Reload
; SSE-NEXT:    shufpd {{.*#+}} xmm1 = xmm1[1,0]
; SSE-NEXT:    callq fmaxf
; SSE-NEXT:    movaps (%rsp), %xmm1 # 16-byte Reload
; SSE-NEXT:    unpcklps {{.*#+}} xmm1 = xmm1[0],xmm0[0],xmm1[1],xmm0[1]
; SSE-NEXT:    unpcklps {{[0-9]+}}(%rsp), %xmm1 # 16-byte Folded Reload
; SSE:         movaps %xmm1, %xmm0
; SSE-NEXT:    addq $72, %rsp
; SSE-NEXT:    retq
;
; AVX:         vmovaps %xmm1, {{[0-9]+}}(%rsp) # 16-byte Spill
; AVX-NEXT:    vmovaps %xmm0, {{[0-9]+}}(%rsp) # 16-byte Spill
; AVX-NEXT:    callq fmaxf
; AVX-NEXT:    vmovaps %xmm0, (%rsp) # 16-byte Spill
; AVX-NEXT:    vmovshdup {{[0-9]+}}(%rsp), %xmm0 # 16-byte Folded Reload
; AVX:         vmovshdup {{[0-9]+}}(%rsp), %xmm1 # 16-byte Folded Reload
; AVX:         callq fmaxf
; AVX-NEXT:    vmovaps (%rsp), %xmm1 # 16-byte Reload
; AVX-NEXT:    vinsertps {{.*#+}} xmm0 = xmm1[0],xmm0[0],xmm1[2,3]
; AVX-NEXT:    vmovaps %xmm0, (%rsp) # 16-byte Spill
; AVX-NEXT:    vpermilpd $1, {{[0-9]+}}(%rsp), %xmm0 # 16-byte Folded Reload
; AVX:         vpermilpd $1, {{[0-9]+}}(%rsp), %xmm1 # 16-byte Folded Reload
; AVX:         callq fmaxf
; AVX-NEXT:    vmovaps (%rsp), %xmm1 # 16-byte Reload
; AVX-NEXT:    vinsertps {{.*#+}} xmm0 = xmm1[0,1],xmm0[0],xmm1[3]
; AVX-NEXT:    vmovaps %xmm0, (%rsp) # 16-byte Spill
; AVX-NEXT:    vpermilps $231, {{[0-9]+}}(%rsp), %xmm0 # 16-byte Folded Reload
; AVX:         vpermilps $231, {{[0-9]+}}(%rsp), %xmm1 # 16-byte Folded Reload
; AVX:         callq fmaxf
; AVX-NEXT:    vmovaps (%rsp), %xmm1 # 16-byte Reload
; AVX-NEXT:    vinsertps {{.*#+}} xmm0 = xmm1[0,1,2],xmm0[0]
; AVX-NEXT:    addq $56, %rsp
; AVX-NEXT:    retq
define <4 x float> @test_intrinsic_fmax_v4f32(<4 x float> %x, <4 x float> %y) {
  %z = call <4 x float> @llvm.maxnum.v4f32(<4 x float> %x, <4 x float> %y) readnone
  ret <4 x float> %z
}

; CHECK-LABEL: @test_intrinsic_fmax_v2f64
; CHECK: callq fmax
; CHECK: callq fmax
define <2 x double> @test_intrinsic_fmax_v2f64(<2 x double> %x, <2 x double> %y) {
  %z = call <2 x double> @llvm.maxnum.v2f64(<2 x double> %x, <2 x double> %y) readnone
  ret <2 x double> %z
}

; CHECK-LABEL: @test_intrinsic_fmax_v4f64
; CHECK: callq fmax
; CHECK: callq fmax
; CHECK: callq fmax
; CHECK: callq fmax
define <4 x double> @test_intrinsic_fmax_v4f64(<4 x double> %x, <4 x double> %y) {
  %z = call <4 x double> @llvm.maxnum.v4f64(<4 x double> %x, <4 x double> %y) readnone
  ret <4 x double> %z
}

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

