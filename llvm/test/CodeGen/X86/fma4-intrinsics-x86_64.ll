; RUN: llc < %s -mtriple=x86_64-unknown-unknown -march=x86-64 -mattr=+avx,+fma4 | FileCheck %s
; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mcpu=bdver2 -mattr=+avx,-fma | FileCheck %s

; VFMADD
define < 4 x float > @test_x86_fma_vfmadd_ss(< 4 x float > %a0, < 4 x float > %a1, < 4 x float > %a2) {
  ; CHECK: vfmaddss
  %res = call < 4 x float > @llvm.x86.fma.vfmadd.ss(< 4 x float > %a0, < 4 x float > %a1, < 4 x float > %a2) ; <i64> [#uses=1]
  ret < 4 x float > %res
}
define < 4 x float > @test_x86_fma_vfmadd_ss_load(< 4 x float > %a0, < 4 x float > %a1, float* %a2) {
  ; CHECK: vfmaddss (%{{.*}})
  %x = load float *%a2
  %y = insertelement <4 x float> undef, float %x, i32 0
  %res = call < 4 x float > @llvm.x86.fma.vfmadd.ss(< 4 x float > %a0, < 4 x float > %a1, < 4 x float > %y) ; <i64> [#uses=1]
  ret < 4 x float > %res
}
define < 4 x float > @test_x86_fma_vfmadd_ss_load2(< 4 x float > %a0, float* %a1, < 4 x float > %a2) {
  ; CHECK: vfmaddss %{{.*}}, (%{{.*}})
  %x = load float *%a1
  %y = insertelement <4 x float> undef, float %x, i32 0
  %res = call < 4 x float > @llvm.x86.fma.vfmadd.ss(< 4 x float > %a0, < 4 x float > %y, < 4 x float > %a2) ; <i64> [#uses=1]
  ret < 4 x float > %res
}
declare < 4 x float > @llvm.x86.fma.vfmadd.ss(< 4 x float >, < 4 x float >, < 4 x float >) nounwind readnone

define < 2 x double > @test_x86_fma_vfmadd_sd(< 2 x double > %a0, < 2 x double > %a1, < 2 x double > %a2) {
  ; CHECK: vfmaddsd
  %res = call < 2 x double > @llvm.x86.fma.vfmadd.sd(< 2 x double > %a0, < 2 x double > %a1, < 2 x double > %a2) ; <i64> [#uses=1]
  ret < 2 x double > %res
}
define < 2 x double > @test_x86_fma_vfmadd_sd_load(< 2 x double > %a0, < 2 x double > %a1, double* %a2) {
  ; CHECK: vfmaddsd (%{{.*}})
  %x = load double *%a2
  %y = insertelement <2 x double> undef, double %x, i32 0
  %res = call < 2 x double > @llvm.x86.fma.vfmadd.sd(< 2 x double > %a0, < 2 x double > %a1, < 2 x double > %y) ; <i64> [#uses=1]
  ret < 2 x double > %res
}
define < 2 x double > @test_x86_fma_vfmadd_sd_load2(< 2 x double > %a0, double* %a1, < 2 x double > %a2) {
  ; CHECK: vfmaddsd %{{.*}}, (%{{.*}})
  %x = load double *%a1
  %y = insertelement <2 x double> undef, double %x, i32 0
  %res = call < 2 x double > @llvm.x86.fma.vfmadd.sd(< 2 x double > %a0, < 2 x double > %y, < 2 x double > %a2) ; <i64> [#uses=1]
  ret < 2 x double > %res
}
declare < 2 x double > @llvm.x86.fma.vfmadd.sd(< 2 x double >, < 2 x double >, < 2 x double >) nounwind readnone

define < 4 x float > @test_x86_fma_vfmadd_ps(< 4 x float > %a0, < 4 x float > %a1, < 4 x float > %a2) {
  ; CHECK: vfmaddps
  %res = call < 4 x float > @llvm.x86.fma.vfmadd.ps(< 4 x float > %a0, < 4 x float > %a1, < 4 x float > %a2) ; <i64> [#uses=1]
  ret < 4 x float > %res
}
define < 4 x float > @test_x86_fma_vfmadd_ps_load(< 4 x float > %a0, < 4 x float > %a1, < 4 x float >* %a2) {
  ; CHECK: vfmaddps (%{{.*}})
  %x = load <4 x float>* %a2
  %res = call < 4 x float > @llvm.x86.fma.vfmadd.ps(< 4 x float > %a0, < 4 x float > %a1, < 4 x float > %x) ; <i64> [#uses=1]
  ret < 4 x float > %res
}
define < 4 x float > @test_x86_fma_vfmadd_ps_load2(< 4 x float > %a0, < 4 x float >* %a1, < 4 x float > %a2) {
  ; CHECK: vfmaddps %{{.*}}, (%{{.*}})
  %x = load <4 x float>* %a1
  %res = call < 4 x float > @llvm.x86.fma.vfmadd.ps(< 4 x float > %a0, < 4 x float > %x, < 4 x float > %a2) ; <i64> [#uses=1]
  ret < 4 x float > %res
}
declare < 4 x float > @llvm.x86.fma.vfmadd.ps(< 4 x float >, < 4 x float >, < 4 x float >) nounwind readnone

define < 2 x double > @test_x86_fma_vfmadd_pd(< 2 x double > %a0, < 2 x double > %a1, < 2 x double > %a2) {
  ; CHECK: vfmaddpd
  %res = call < 2 x double > @llvm.x86.fma.vfmadd.pd(< 2 x double > %a0, < 2 x double > %a1, < 2 x double > %a2) ; <i64> [#uses=1]
  ret < 2 x double > %res
}
define < 2 x double > @test_x86_fma_vfmadd_pd_load(< 2 x double > %a0, < 2 x double > %a1, < 2 x double >* %a2) {
  ; CHECK: vfmaddpd (%{{.*}})
  %x = load <2 x double>* %a2
  %res = call < 2 x double > @llvm.x86.fma.vfmadd.pd(< 2 x double > %a0, < 2 x double > %a1, < 2 x double > %x) ; <i64> [#uses=1]
  ret < 2 x double > %res
}
define < 2 x double > @test_x86_fma_vfmadd_pd_load2(< 2 x double > %a0, < 2 x double >* %a1, < 2 x double > %a2) {
  ; CHECK: vfmaddpd %{{.*}}, (%{{.*}})
  %x = load <2 x double>* %a1
  %res = call < 2 x double > @llvm.x86.fma.vfmadd.pd(< 2 x double > %a0, < 2 x double > %x, < 2 x double > %a2) ; <i64> [#uses=1]
  ret < 2 x double > %res
}
declare < 2 x double > @llvm.x86.fma.vfmadd.pd(< 2 x double >, < 2 x double >, < 2 x double >) nounwind readnone

define < 8 x float > @test_x86_fma_vfmadd_ps_256(< 8 x float > %a0, < 8 x float > %a1, < 8 x float > %a2) {
  ; CHECK: vfmaddps
  ; CHECK: ymm
  %res = call < 8 x float > @llvm.x86.fma.vfmadd.ps.256(< 8 x float > %a0, < 8 x float > %a1, < 8 x float > %a2) ; <i64> [#uses=1]
  ret < 8 x float > %res
}
declare < 8 x float > @llvm.x86.fma.vfmadd.ps.256(< 8 x float >, < 8 x float >, < 8 x float >) nounwind readnone

define < 4 x double > @test_x86_fma_vfmadd_pd_256(< 4 x double > %a0, < 4 x double > %a1, < 4 x double > %a2) {
  ; CHECK: vfmaddpd
  ; CHECK: ymm
  %res = call < 4 x double > @llvm.x86.fma.vfmadd.pd.256(< 4 x double > %a0, < 4 x double > %a1, < 4 x double > %a2) ; <i64> [#uses=1]
  ret < 4 x double > %res
}
declare < 4 x double > @llvm.x86.fma.vfmadd.pd.256(< 4 x double >, < 4 x double >, < 4 x double >) nounwind readnone

; VFMSUB
define < 4 x float > @test_x86_fma_vfmsub_ss(< 4 x float > %a0, < 4 x float > %a1, < 4 x float > %a2) {
  ; CHECK: vfmsubss
  %res = call < 4 x float > @llvm.x86.fma.vfmsub.ss(< 4 x float > %a0, < 4 x float > %a1, < 4 x float > %a2) ; <i64> [#uses=1]
  ret < 4 x float > %res
}
declare < 4 x float > @llvm.x86.fma.vfmsub.ss(< 4 x float >, < 4 x float >, < 4 x float >) nounwind readnone

define < 2 x double > @test_x86_fma_vfmsub_sd(< 2 x double > %a0, < 2 x double > %a1, < 2 x double > %a2) {
  ; CHECK: vfmsubsd
  %res = call < 2 x double > @llvm.x86.fma.vfmsub.sd(< 2 x double > %a0, < 2 x double > %a1, < 2 x double > %a2) ; <i64> [#uses=1]
  ret < 2 x double > %res
}
declare < 2 x double > @llvm.x86.fma.vfmsub.sd(< 2 x double >, < 2 x double >, < 2 x double >) nounwind readnone

define < 4 x float > @test_x86_fma_vfmsub_ps(< 4 x float > %a0, < 4 x float > %a1, < 4 x float > %a2) {
  ; CHECK: vfmsubps
  %res = call < 4 x float > @llvm.x86.fma.vfmsub.ps(< 4 x float > %a0, < 4 x float > %a1, < 4 x float > %a2) ; <i64> [#uses=1]
  ret < 4 x float > %res
}
declare < 4 x float > @llvm.x86.fma.vfmsub.ps(< 4 x float >, < 4 x float >, < 4 x float >) nounwind readnone

define < 2 x double > @test_x86_fma_vfmsub_pd(< 2 x double > %a0, < 2 x double > %a1, < 2 x double > %a2) {
  ; CHECK: vfmsubpd
  %res = call < 2 x double > @llvm.x86.fma.vfmsub.pd(< 2 x double > %a0, < 2 x double > %a1, < 2 x double > %a2) ; <i64> [#uses=1]
  ret < 2 x double > %res
}
declare < 2 x double > @llvm.x86.fma.vfmsub.pd(< 2 x double >, < 2 x double >, < 2 x double >) nounwind readnone

define < 8 x float > @test_x86_fma_vfmsub_ps_256(< 8 x float > %a0, < 8 x float > %a1, < 8 x float > %a2) {
  ; CHECK: vfmsubps
  ; CHECK: ymm
  %res = call < 8 x float > @llvm.x86.fma.vfmsub.ps.256(< 8 x float > %a0, < 8 x float > %a1, < 8 x float > %a2) ; <i64> [#uses=1]
  ret < 8 x float > %res
}
declare < 8 x float > @llvm.x86.fma.vfmsub.ps.256(< 8 x float >, < 8 x float >, < 8 x float >) nounwind readnone

define < 4 x double > @test_x86_fma_vfmsub_pd_256(< 4 x double > %a0, < 4 x double > %a1, < 4 x double > %a2) {
  ; CHECK: vfmsubpd
  ; CHECK: ymm
  %res = call < 4 x double > @llvm.x86.fma.vfmsub.pd.256(< 4 x double > %a0, < 4 x double > %a1, < 4 x double > %a2) ; <i64> [#uses=1]
  ret < 4 x double > %res
}
declare < 4 x double > @llvm.x86.fma.vfmsub.pd.256(< 4 x double >, < 4 x double >, < 4 x double >) nounwind readnone

; VFNMADD
define < 4 x float > @test_x86_fma_vfnmadd_ss(< 4 x float > %a0, < 4 x float > %a1, < 4 x float > %a2) {
  ; CHECK: vfnmaddss
  %res = call < 4 x float > @llvm.x86.fma.vfnmadd.ss(< 4 x float > %a0, < 4 x float > %a1, < 4 x float > %a2) ; <i64> [#uses=1]
  ret < 4 x float > %res
}
declare < 4 x float > @llvm.x86.fma.vfnmadd.ss(< 4 x float >, < 4 x float >, < 4 x float >) nounwind readnone

define < 2 x double > @test_x86_fma_vfnmadd_sd(< 2 x double > %a0, < 2 x double > %a1, < 2 x double > %a2) {
  ; CHECK: vfnmaddsd
  %res = call < 2 x double > @llvm.x86.fma.vfnmadd.sd(< 2 x double > %a0, < 2 x double > %a1, < 2 x double > %a2) ; <i64> [#uses=1]
  ret < 2 x double > %res
}
declare < 2 x double > @llvm.x86.fma.vfnmadd.sd(< 2 x double >, < 2 x double >, < 2 x double >) nounwind readnone

define < 4 x float > @test_x86_fma_vfnmadd_ps(< 4 x float > %a0, < 4 x float > %a1, < 4 x float > %a2) {
  ; CHECK: vfnmaddps
  %res = call < 4 x float > @llvm.x86.fma.vfnmadd.ps(< 4 x float > %a0, < 4 x float > %a1, < 4 x float > %a2) ; <i64> [#uses=1]
  ret < 4 x float > %res
}
declare < 4 x float > @llvm.x86.fma.vfnmadd.ps(< 4 x float >, < 4 x float >, < 4 x float >) nounwind readnone

define < 2 x double > @test_x86_fma_vfnmadd_pd(< 2 x double > %a0, < 2 x double > %a1, < 2 x double > %a2) {
  ; CHECK: vfnmaddpd
  %res = call < 2 x double > @llvm.x86.fma.vfnmadd.pd(< 2 x double > %a0, < 2 x double > %a1, < 2 x double > %a2) ; <i64> [#uses=1]
  ret < 2 x double > %res
}
declare < 2 x double > @llvm.x86.fma.vfnmadd.pd(< 2 x double >, < 2 x double >, < 2 x double >) nounwind readnone

define < 8 x float > @test_x86_fma_vfnmadd_ps_256(< 8 x float > %a0, < 8 x float > %a1, < 8 x float > %a2) {
  ; CHECK: vfnmaddps
  ; CHECK: ymm
  %res = call < 8 x float > @llvm.x86.fma.vfnmadd.ps.256(< 8 x float > %a0, < 8 x float > %a1, < 8 x float > %a2) ; <i64> [#uses=1]
  ret < 8 x float > %res
}
declare < 8 x float > @llvm.x86.fma.vfnmadd.ps.256(< 8 x float >, < 8 x float >, < 8 x float >) nounwind readnone

define < 4 x double > @test_x86_fma_vfnmadd_pd_256(< 4 x double > %a0, < 4 x double > %a1, < 4 x double > %a2) {
  ; CHECK: vfnmaddpd
  ; CHECK: ymm
  %res = call < 4 x double > @llvm.x86.fma.vfnmadd.pd.256(< 4 x double > %a0, < 4 x double > %a1, < 4 x double > %a2) ; <i64> [#uses=1]
  ret < 4 x double > %res
}
declare < 4 x double > @llvm.x86.fma.vfnmadd.pd.256(< 4 x double >, < 4 x double >, < 4 x double >) nounwind readnone

; VFNMSUB
define < 4 x float > @test_x86_fma_vfnmsub_ss(< 4 x float > %a0, < 4 x float > %a1, < 4 x float > %a2) {
  ; CHECK: vfnmsubss
  %res = call < 4 x float > @llvm.x86.fma.vfnmsub.ss(< 4 x float > %a0, < 4 x float > %a1, < 4 x float > %a2) ; <i64> [#uses=1]
  ret < 4 x float > %res
}
declare < 4 x float > @llvm.x86.fma.vfnmsub.ss(< 4 x float >, < 4 x float >, < 4 x float >) nounwind readnone

define < 2 x double > @test_x86_fma_vfnmsub_sd(< 2 x double > %a0, < 2 x double > %a1, < 2 x double > %a2) {
  ; CHECK: vfnmsubsd
  %res = call < 2 x double > @llvm.x86.fma.vfnmsub.sd(< 2 x double > %a0, < 2 x double > %a1, < 2 x double > %a2) ; <i64> [#uses=1]
  ret < 2 x double > %res
}
declare < 2 x double > @llvm.x86.fma.vfnmsub.sd(< 2 x double >, < 2 x double >, < 2 x double >) nounwind readnone

define < 4 x float > @test_x86_fma_vfnmsub_ps(< 4 x float > %a0, < 4 x float > %a1, < 4 x float > %a2) {
  ; CHECK: vfnmsubps
  %res = call < 4 x float > @llvm.x86.fma.vfnmsub.ps(< 4 x float > %a0, < 4 x float > %a1, < 4 x float > %a2) ; <i64> [#uses=1]
  ret < 4 x float > %res
}
declare < 4 x float > @llvm.x86.fma.vfnmsub.ps(< 4 x float >, < 4 x float >, < 4 x float >) nounwind readnone

define < 2 x double > @test_x86_fma_vfnmsub_pd(< 2 x double > %a0, < 2 x double > %a1, < 2 x double > %a2) {
  ; CHECK: vfnmsubpd
  %res = call < 2 x double > @llvm.x86.fma.vfnmsub.pd(< 2 x double > %a0, < 2 x double > %a1, < 2 x double > %a2) ; <i64> [#uses=1]
  ret < 2 x double > %res
}
declare < 2 x double > @llvm.x86.fma.vfnmsub.pd(< 2 x double >, < 2 x double >, < 2 x double >) nounwind readnone

define < 8 x float > @test_x86_fma_vfnmsub_ps_256(< 8 x float > %a0, < 8 x float > %a1, < 8 x float > %a2) {
  ; CHECK: vfnmsubps
  ; CHECK: ymm
  %res = call < 8 x float > @llvm.x86.fma.vfnmsub.ps.256(< 8 x float > %a0, < 8 x float > %a1, < 8 x float > %a2) ; <i64> [#uses=1]
  ret < 8 x float > %res
}
declare < 8 x float > @llvm.x86.fma.vfnmsub.ps.256(< 8 x float >, < 8 x float >, < 8 x float >) nounwind readnone

define < 4 x double > @test_x86_fma_vfnmsub_pd_256(< 4 x double > %a0, < 4 x double > %a1, < 4 x double > %a2) {
  ; CHECK: vfnmsubpd
  ; CHECK: ymm
  %res = call < 4 x double > @llvm.x86.fma.vfnmsub.pd.256(< 4 x double > %a0, < 4 x double > %a1, < 4 x double > %a2) ; <i64> [#uses=1]
  ret < 4 x double > %res
}
declare < 4 x double > @llvm.x86.fma.vfnmsub.pd.256(< 4 x double >, < 4 x double >, < 4 x double >) nounwind readnone

; VFMADDSUB
define < 4 x float > @test_x86_fma_vfmaddsub_ps(< 4 x float > %a0, < 4 x float > %a1, < 4 x float > %a2) {
  ; CHECK: vfmaddsubps
  %res = call < 4 x float > @llvm.x86.fma.vfmaddsub.ps(< 4 x float > %a0, < 4 x float > %a1, < 4 x float > %a2) ; <i64> [#uses=1]
  ret < 4 x float > %res
}
declare < 4 x float > @llvm.x86.fma.vfmaddsub.ps(< 4 x float >, < 4 x float >, < 4 x float >) nounwind readnone

define < 2 x double > @test_x86_fma_vfmaddsub_pd(< 2 x double > %a0, < 2 x double > %a1, < 2 x double > %a2) {
  ; CHECK: vfmaddsubpd
  %res = call < 2 x double > @llvm.x86.fma.vfmaddsub.pd(< 2 x double > %a0, < 2 x double > %a1, < 2 x double > %a2) ; <i64> [#uses=1]
  ret < 2 x double > %res
}
declare < 2 x double > @llvm.x86.fma.vfmaddsub.pd(< 2 x double >, < 2 x double >, < 2 x double >) nounwind readnone

define < 8 x float > @test_x86_fma_vfmaddsub_ps_256(< 8 x float > %a0, < 8 x float > %a1, < 8 x float > %a2) {
  ; CHECK: vfmaddsubps
  ; CHECK: ymm
  %res = call < 8 x float > @llvm.x86.fma.vfmaddsub.ps.256(< 8 x float > %a0, < 8 x float > %a1, < 8 x float > %a2) ; <i64> [#uses=1]
  ret < 8 x float > %res
}
declare < 8 x float > @llvm.x86.fma.vfmaddsub.ps.256(< 8 x float >, < 8 x float >, < 8 x float >) nounwind readnone

define < 4 x double > @test_x86_fma_vfmaddsub_pd_256(< 4 x double > %a0, < 4 x double > %a1, < 4 x double > %a2) {
  ; CHECK: vfmaddsubpd
  ; CHECK: ymm
  %res = call < 4 x double > @llvm.x86.fma.vfmaddsub.pd.256(< 4 x double > %a0, < 4 x double > %a1, < 4 x double > %a2) ; <i64> [#uses=1]
  ret < 4 x double > %res
}
declare < 4 x double > @llvm.x86.fma.vfmaddsub.pd.256(< 4 x double >, < 4 x double >, < 4 x double >) nounwind readnone

; VFMSUBADD
define < 4 x float > @test_x86_fma_vfmsubadd_ps(< 4 x float > %a0, < 4 x float > %a1, < 4 x float > %a2) {
  ; CHECK: vfmsubaddps
  %res = call < 4 x float > @llvm.x86.fma.vfmsubadd.ps(< 4 x float > %a0, < 4 x float > %a1, < 4 x float > %a2) ; <i64> [#uses=1]
  ret < 4 x float > %res
}
declare < 4 x float > @llvm.x86.fma.vfmsubadd.ps(< 4 x float >, < 4 x float >, < 4 x float >) nounwind readnone

define < 2 x double > @test_x86_fma_vfmsubadd_pd(< 2 x double > %a0, < 2 x double > %a1, < 2 x double > %a2) {
  ; CHECK: vfmsubaddpd
  %res = call < 2 x double > @llvm.x86.fma.vfmsubadd.pd(< 2 x double > %a0, < 2 x double > %a1, < 2 x double > %a2) ; <i64> [#uses=1]
  ret < 2 x double > %res
}
declare < 2 x double > @llvm.x86.fma.vfmsubadd.pd(< 2 x double >, < 2 x double >, < 2 x double >) nounwind readnone

define < 8 x float > @test_x86_fma_vfmsubadd_ps_256(< 8 x float > %a0, < 8 x float > %a1, < 8 x float > %a2) {
  ; CHECK: vfmsubaddps
  ; CHECK: ymm
  %res = call < 8 x float > @llvm.x86.fma.vfmsubadd.ps.256(< 8 x float > %a0, < 8 x float > %a1, < 8 x float > %a2) ; <i64> [#uses=1]
  ret < 8 x float > %res
}
declare < 8 x float > @llvm.x86.fma.vfmsubadd.ps.256(< 8 x float >, < 8 x float >, < 8 x float >) nounwind readnone

define < 4 x double > @test_x86_fma_vfmsubadd_pd_256(< 4 x double > %a0, < 4 x double > %a1, < 4 x double > %a2) {
  ; CHECK: vfmsubaddpd
  ; CHECK: ymm
  %res = call < 4 x double > @llvm.x86.fma.vfmsubadd.pd.256(< 4 x double > %a0, < 4 x double > %a1, < 4 x double > %a2) ; <i64> [#uses=1]
  ret < 4 x double > %res
}
declare < 4 x double > @llvm.x86.fma.vfmsubadd.pd.256(< 4 x double >, < 4 x double >, < 4 x double >) nounwind readnone
