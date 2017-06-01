; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=knl --show-mc-encoding | FileCheck %s --check-prefix AVX512 --check-prefix AVX512-KNL
; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=skx --show-mc-encoding | FileCheck %s --check-prefix AVX512 --check-prefix AVX512-SKX
; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=corei7-avx --show-mc-encoding | FileCheck %s --check-prefix AVX

; AVX512-LABEL: @test_fdiv
; AVX512: vdivss %xmm{{.*}} ## EVEX TO VEX Compression encoding: [0xc5
; AVX-LABEL: @test_fdiv
; AVX:    vdivss %xmm{{.*}} ## encoding: [0xc5

define float @test_fdiv(float %a, float %b) {
  %c = fdiv float %a, %b
  ret float %c
}

; AVX512-LABEL: @test_fsub
; AVX512: vsubss %xmm{{.*}} ## EVEX TO VEX Compression encoding: [0xc5
; AVX-LABEL: @test_fsub
; AVX:    vsubss %xmm{{.*}} ## encoding: [0xc5

define float @test_fsub(float %a, float %b) {
  %c = fsub float %a, %b
  ret float %c
}

; AVX512-LABEL: @test_fadd
; AVX512: vaddsd %xmm{{.*}} ## EVEX TO VEX Compression encoding: [0xc5 
; AVX-LABEL: @test_fadd
; AVX:    vaddsd %xmm{{.*}} ## encoding: [0xc5

define double @test_fadd(double %a, double %b) {
  %c = fadd double %a, %b
  ret double %c
}

declare float     @llvm.trunc.f32(float  %Val)
declare double    @llvm.trunc.f64(double %Val)
declare float     @llvm.rint.f32(float  %Val)
declare double    @llvm.rint.f64(double %Val)
declare double    @llvm.sqrt.f64(double %Val)
declare float     @llvm.sqrt.f32(float  %Val)

; AVX512-LABEL: @test_trunc
; AVX512: vrndscaless
; AVX-LABEL: @test_trunc
; AVX:    vroundss

define float @test_trunc(float %a) {
  %c = call float @llvm.trunc.f32(float %a)
  ret float %c
}

; AVX512-LABEL: @test_sqrt
; AVX512: vsqrtsd %xmm{{.*}} ## EVEX TO VEX Compression encoding: [0xc5
; AVX-LABEL: @test_sqrt
; AVX:    vsqrtsd %xmm{{.*}} ## encoding: [0xc5

define double @test_sqrt(double %a) {
  %c = call double @llvm.sqrt.f64(double %a)
  ret double %c
}

; AVX512-LABEL: @test_rint
; AVX512: vrndscaless
; AVX-LABEL: @test_rint
; AVX:    vroundss

define float @test_rint(float %a) {
  %c = call float @llvm.rint.f32(float %a)
  ret float %c
}

; AVX512-LABEL: @test_vmax
; AVX512: vmaxss %xmm{{.*}} ## EVEX TO VEX Compression encoding: [0xc5
; AVX-LABEL: @test_vmax
; AVX:    vmaxss %xmm{{.*}} ## encoding: [0xc5

define float @test_vmax(float %i, float %j) {
  %cmp_res = fcmp ogt float %i, %j
  %max = select i1 %cmp_res, float %i, float %j
  ret float %max
}

; AVX512-LABEL: @test_mov
; AVX512: vcmpltss %xmm{{.*}} ## encoding: [0x62
; AVX-LABEL: @test_mov
; AVX:    vcmpltss %xmm{{.*}} ## encoding: [0xc5

define float @test_mov(float %a, float %b, float %i, float %j) {
  %cmp_res = fcmp ogt float %i, %j
  %max = select i1 %cmp_res, float %b, float %a
  ret float %max
}

; AVX512-SKX-LABEL: @zero_float
; AVX512-SKX: vxorps %xmm{{.*}}, %xmm{{.*}}, %xmm{{.*}} ## EVEX TO VEX Compression encoding: [0xc5
; AVX512-KNL-LABEL: @zero_float
; AVX512-KNL: vxorps %xmm{{.*}}, %xmm{{.*}}, %xmm{{.*}} ## encoding: [0xc5,
; AVX-LABEL: @zero_float
; AVX: vxorps %xmm{{.*}}, %xmm{{.*}}, %xmm{{.*}} ## encoding: [0xc5,

define float @zero_float(float %a) {
  %b = fadd float %a, 0.0
  ret float %b
}

; AVX512-SKX-LABEL: @zero_double
; AVX512-SKX: vxorpd %xmm{{.*}}, %xmm{{.*}}, %xmm{{.*}} ## EVEX TO VEX Compression encoding: [0xc5 
; AVX512-KNL-LABEL: @zero_double
; AVX512-KNL: vxorpd %xmm{{.*}}, %xmm{{.*}}, %xmm{{.*}} ## encoding: [0xc5,
; AVX-LABEL: @zero_double
; AVX: vxorpd %xmm{{.*}}, %xmm{{.*}}, %xmm{{.*}} ## encoding: [0xc5,

define double @zero_double(double %a) {
  %b = fadd double %a, 0.0
  ret double %b
}
