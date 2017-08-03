; RUN: llc < %s -mtriple=x86_64-unknown-unknown -march=x86-64 -mcpu=corei7-avx -mattr=+fma4 | FileCheck %s
; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mcpu=bdver2 -mattr=+avx,-fma | FileCheck %s

; VFMADD
define < 4 x float > @test_x86_fma_vfmadd_ss_load(< 4 x float > %a0, < 4 x float > %a1, float* %a2) {
  ; CHECK: vfmaddss (%{{.*}})
  %x = load float , float *%a2
  %y = insertelement <4 x float> undef, float %x, i32 0
  %res = call < 4 x float > @llvm.x86.fma.vfmadd.ss(< 4 x float > %a0, < 4 x float > %a1, < 4 x float > %y)
  ret < 4 x float > %res
}
define < 4 x float > @test_x86_fma_vfmadd_ss_load2(< 4 x float > %a0, float* %a1, < 4 x float > %a2) {
  ; CHECK: vfmaddss %{{.*}}, (%{{.*}})
  %x = load float , float *%a1
  %y = insertelement <4 x float> undef, float %x, i32 0
  %res = call < 4 x float > @llvm.x86.fma.vfmadd.ss(< 4 x float > %a0, < 4 x float > %y, < 4 x float > %a2)
  ret < 4 x float > %res
}

declare < 4 x float > @llvm.x86.fma.vfmadd.ss(< 4 x float >, < 4 x float >, < 4 x float >) nounwind readnone

define < 2 x double > @test_x86_fma_vfmadd_sd_load(< 2 x double > %a0, < 2 x double > %a1, double* %a2) {
  ; CHECK: vfmaddsd (%{{.*}})
  %x = load double , double *%a2
  %y = insertelement <2 x double> undef, double %x, i32 0
  %res = call < 2 x double > @llvm.x86.fma.vfmadd.sd(< 2 x double > %a0, < 2 x double > %a1, < 2 x double > %y)
  ret < 2 x double > %res
}
define < 2 x double > @test_x86_fma_vfmadd_sd_load2(< 2 x double > %a0, double* %a1, < 2 x double > %a2) {
  ; CHECK: vfmaddsd %{{.*}}, (%{{.*}})
  %x = load double , double *%a1
  %y = insertelement <2 x double> undef, double %x, i32 0
  %res = call < 2 x double > @llvm.x86.fma.vfmadd.sd(< 2 x double > %a0, < 2 x double > %y, < 2 x double > %a2)
  ret < 2 x double > %res
}
declare < 2 x double > @llvm.x86.fma.vfmadd.sd(< 2 x double >, < 2 x double >, < 2 x double >) nounwind readnone
define < 4 x float > @test_x86_fma_vfmadd_ps_load(< 4 x float > %a0, < 4 x float > %a1, < 4 x float >* %a2) {
  ; CHECK: vfmaddps (%{{.*}})
  %x = load <4 x float>, <4 x float>* %a2
  %res = call < 4 x float > @llvm.x86.fma.vfmadd.ps(< 4 x float > %a0, < 4 x float > %a1, < 4 x float > %x)
  ret < 4 x float > %res
}
define < 4 x float > @test_x86_fma_vfmadd_ps_load2(< 4 x float > %a0, < 4 x float >* %a1, < 4 x float > %a2) {
  ; CHECK: vfmaddps %{{.*}}, (%{{.*}})
  %x = load <4 x float>, <4 x float>* %a1
  %res = call < 4 x float > @llvm.x86.fma.vfmadd.ps(< 4 x float > %a0, < 4 x float > %x, < 4 x float > %a2)
  ret < 4 x float > %res
}
declare < 4 x float > @llvm.x86.fma.vfmadd.ps(< 4 x float >, < 4 x float >, < 4 x float >) nounwind readnone

; To test execution dependency
define < 4 x float > @test_x86_fma_vfmadd_ps_load3(< 4 x float >* %a0, < 4 x float >* %a1, < 4 x float > %a2) {
  ; CHECK: vmovaps
  ; CHECK: vfmaddps %{{.*}}, (%{{.*}})
  %x = load <4 x float>, <4 x float>* %a0
  %y = load <4 x float>, <4 x float>* %a1
  %res = call < 4 x float > @llvm.x86.fma.vfmadd.ps(< 4 x float > %x, < 4 x float > %y, < 4 x float > %a2)
  ret < 4 x float > %res
}

define < 2 x double > @test_x86_fma_vfmadd_pd_load(< 2 x double > %a0, < 2 x double > %a1, < 2 x double >* %a2) {
  ; CHECK: vfmaddpd (%{{.*}})
  %x = load <2 x double>, <2 x double>* %a2
  %res = call < 2 x double > @llvm.x86.fma.vfmadd.pd(< 2 x double > %a0, < 2 x double > %a1, < 2 x double > %x)
  ret < 2 x double > %res
}
define < 2 x double > @test_x86_fma_vfmadd_pd_load2(< 2 x double > %a0, < 2 x double >* %a1, < 2 x double > %a2) {
  ; CHECK: vfmaddpd %{{.*}}, (%{{.*}})
  %x = load <2 x double>, <2 x double>* %a1
  %res = call < 2 x double > @llvm.x86.fma.vfmadd.pd(< 2 x double > %a0, < 2 x double > %x, < 2 x double > %a2)
  ret < 2 x double > %res
}
declare < 2 x double > @llvm.x86.fma.vfmadd.pd(< 2 x double >, < 2 x double >, < 2 x double >) nounwind readnone

; To test execution dependency
define < 2 x double > @test_x86_fma_vfmadd_pd_load3(< 2 x double >* %a0, < 2 x double >* %a1, < 2 x double > %a2) {
  ; CHECK: vmovapd
  ; CHECK: vfmaddpd %{{.*}}, (%{{.*}})
  %x = load <2 x double>, <2 x double>* %a0
  %y = load <2 x double>, <2 x double>* %a1
  %res = call < 2 x double > @llvm.x86.fma.vfmadd.pd(< 2 x double > %x, < 2 x double > %y, < 2 x double > %a2)
  ret < 2 x double > %res
}

