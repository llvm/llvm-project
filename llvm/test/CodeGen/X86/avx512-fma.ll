; RUN: llc < %s -mtriple=x86_64-apple-darwin -mattr=+avx512f -fp-contract=fast | FileCheck %s
; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=skx -fp-contract=fast | FileCheck %s --check-prefix=SKX

; CHECK-LABEL: test_x86_fmadd_ps_z
; CHECK: vfmadd213ps     %zmm2, %zmm1, %zmm0
; CHECK: ret
define <16 x float> @test_x86_fmadd_ps_z(<16 x float> %a0, <16 x float> %a1, <16 x float> %a2) {
  %x = fmul <16 x float> %a0, %a1
  %res = fadd <16 x float> %x, %a2
  ret <16 x float> %res
}

; CHECK-LABEL: test_x86_fmsub_ps_z
; CHECK: vfmsub213ps     %zmm2, %zmm1, %zmm0
; CHECK: ret
define <16 x float> @test_x86_fmsub_ps_z(<16 x float> %a0, <16 x float> %a1, <16 x float> %a2) {
  %x = fmul <16 x float> %a0, %a1
  %res = fsub <16 x float> %x, %a2
  ret <16 x float> %res
}

; CHECK-LABEL: test_x86_fnmadd_ps_z
; CHECK: vfnmadd213ps     %zmm2, %zmm1, %zmm0
; CHECK: ret
define <16 x float> @test_x86_fnmadd_ps_z(<16 x float> %a0, <16 x float> %a1, <16 x float> %a2) {
  %x = fmul <16 x float> %a0, %a1
  %res = fsub <16 x float> %a2, %x
  ret <16 x float> %res
}

; CHECK-LABEL: test_x86_fnmsub_ps_z
; CHECK: vfnmsub213ps     %zmm2, %zmm1, %zmm0
; CHECK: ret
define <16 x float> @test_x86_fnmsub_ps_z(<16 x float> %a0, <16 x float> %a1, <16 x float> %a2) {
  %x = fmul <16 x float> %a0, %a1
  %y = fsub <16 x float> <float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, 
                          float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00,
						  float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, 
						  float -0.000000e+00>, %x
  %res = fsub <16 x float> %y, %a2
  ret <16 x float> %res
}

; CHECK-LABEL: test_x86_fmadd_pd_z
; CHECK: vfmadd213pd     %zmm2, %zmm1, %zmm0
; CHECK: ret
define <8 x double> @test_x86_fmadd_pd_z(<8 x double> %a0, <8 x double> %a1, <8 x double> %a2) {
  %x = fmul <8 x double> %a0, %a1
  %res = fadd <8 x double> %x, %a2
  ret <8 x double> %res
}

; CHECK-LABEL: test_x86_fmsub_pd_z
; CHECK: vfmsub213pd     %zmm2, %zmm1, %zmm0
; CHECK: ret
define <8 x double> @test_x86_fmsub_pd_z(<8 x double> %a0, <8 x double> %a1, <8 x double> %a2) {
  %x = fmul <8 x double> %a0, %a1
  %res = fsub <8 x double> %x, %a2
  ret <8 x double> %res
}

define double @test_x86_fmsub_213(double %a0, double %a1, double %a2) {
; CHECK-LABEL: test_x86_fmsub_213:
; CHECK:       ## BB#0:
; CHECK-NEXT:    vfmsub213sd %xmm2, %xmm0, %xmm1
; CHECK-NEXT:    vmovaps %zmm1, %zmm0
; CHECK-NEXT:    retq
  %x = fmul double %a0, %a1
  %res = fsub double %x, %a2
  ret double %res
}

define double @test_x86_fmsub_213_m(double %a0, double %a1, double * %a2_ptr) {
; CHECK-LABEL: test_x86_fmsub_213_m:
; CHECK:       ## BB#0:
; CHECK-NEXT:    vfmsub213sd (%rdi), %xmm0, %xmm1
; CHECK-NEXT:    vmovaps %zmm1, %zmm0
; CHECK-NEXT:    retq
  %a2 = load double , double *%a2_ptr
  %x = fmul double %a0, %a1
  %res = fsub double %x, %a2
  ret double %res
}

define double @test_x86_fmsub_231_m(double %a0, double %a1, double * %a2_ptr) {
; CHECK-LABEL: test_x86_fmsub_231_m:
; CHECK:       ## BB#0:
; CHECK-NEXT:    vfmsub231sd (%rdi), %xmm0, %xmm1
; CHECK-NEXT:    vmovaps %zmm1, %zmm0
; CHECK-NEXT:    retq
  %a2 = load double , double *%a2_ptr
  %x = fmul double %a0, %a2
  %res = fsub double %x, %a1
  ret double %res
}

define <16 x float> @test231_br(<16 x float> %a1, <16 x float> %a2) nounwind {
; CHECK-LABEL: test231_br:
; CHECK:       ## BB#0:
; CHECK-NEXT:    vfmadd231ps {{.*}}(%rip){1to16}, %zmm0, %zmm1
; CHECK-NEXT:    vmovaps %zmm1, %zmm0
; CHECK-NEXT:    retq
  %b1 = fmul <16 x float> %a1, <float 0x3FB99999A0000000, float 0x3FB99999A0000000, float 0x3FB99999A0000000, float 0x3FB99999A0000000, float 0x3FB99999A0000000, float 0x3FB99999A0000000, float 0x3FB99999A0000000, float 0x3FB99999A0000000, float 0x3FB99999A0000000, float 0x3FB99999A0000000, float 0x3FB99999A0000000, float 0x3FB99999A0000000, float 0x3FB99999A0000000, float 0x3FB99999A0000000, float 0x3FB99999A0000000, float 0x3FB99999A0000000>
  %b2 = fadd <16 x float> %b1, %a2
  ret <16 x float> %b2
}

define <16 x float> @test213_br(<16 x float> %a1, <16 x float> %a2) nounwind {
; CHECK-LABEL: test213_br:
; CHECK:       ## BB#0:
; CHECK-NEXT:    vfmadd213ps {{.*}}(%rip){1to16}, %zmm1, %zmm0
; CHECK-NEXT:    retq
  %b1 = fmul <16 x float> %a1, %a2
  %b2 = fadd <16 x float> %b1, <float 0x3FB99999A0000000, float 0x3FB99999A0000000, float 0x3FB99999A0000000, float 0x3FB99999A0000000, float 0x3FB99999A0000000, float 0x3FB99999A0000000, float 0x3FB99999A0000000, float 0x3FB99999A0000000, float 0x3FB99999A0000000, float 0x3FB99999A0000000, float 0x3FB99999A0000000, float 0x3FB99999A0000000, float 0x3FB99999A0000000, float 0x3FB99999A0000000, float 0x3FB99999A0000000, float 0x3FB99999A0000000>
  ret <16 x float> %b2
}

;mask (a*c+b , a)
define <16 x float> @test_x86_fmadd132_ps(<16 x float> %a0, <16 x float> %a1, <16 x float> *%a2_ptrt, <16 x i1> %mask) {
; CHECK-LABEL: test_x86_fmadd132_ps:
; CHECK:       ## BB#0:
; CHECK-NEXT:    vpmovsxbd %xmm2, %zmm2
; CHECK-NEXT:    vpandd {{.*}}(%rip){1to16}, %zmm2, %zmm2
; CHECK-NEXT:    vptestmd %zmm2, %zmm2, %k1
; CHECK-NEXT:    vfmadd132ps (%rdi), %zmm1, %zmm0 {%k1}
; CHECK-NEXT:    retq
;
; SKX-LABEL: test_x86_fmadd132_ps:
; SKX:       ## BB#0:
; SKX-NEXT:    vpmovb2m %xmm2, %k1
; SKX-NEXT:    vfmadd132ps (%rdi), %zmm1, %zmm0 {%k1}
; SKX-NEXT:    retq
  %a2   = load <16 x float>,<16 x float> *%a2_ptrt,align 1
  %x = fmul <16 x float> %a0, %a2
  %y = fadd <16 x float> %x, %a1
  %res = select <16 x i1> %mask, <16 x float> %y, <16 x float> %a0
  ret <16 x float> %res
}

;mask (a*c+b , b)
define <16 x float> @test_x86_fmadd231_ps(<16 x float> %a0, <16 x float> %a1, <16 x float> *%a2_ptrt, <16 x i1> %mask) {
; CHECK-LABEL: test_x86_fmadd231_ps:
; CHECK:       ## BB#0:
; CHECK-NEXT:    vpmovsxbd %xmm2, %zmm2
; CHECK-NEXT:    vpandd {{.*}}(%rip){1to16}, %zmm2, %zmm2
; CHECK-NEXT:    vptestmd %zmm2, %zmm2, %k1
; CHECK-NEXT:    vfmadd231ps (%rdi), %zmm0, %zmm1 {%k1}
; CHECK-NEXT:    vmovaps %zmm1, %zmm0
; CHECK-NEXT:    retq
;
; SKX-LABEL: test_x86_fmadd231_ps:
; SKX:       ## BB#0:
; SKX-NEXT:    vpmovb2m %xmm2, %k1
; SKX-NEXT:    vfmadd231ps (%rdi), %zmm0, %zmm1 {%k1}
; SKX-NEXT:    vmovaps %zmm1, %zmm0
; SKX-NEXT:    retq
  %a2   = load <16 x float>,<16 x float> *%a2_ptrt,align 1
  %x = fmul <16 x float> %a0, %a2
  %y = fadd <16 x float> %x, %a1
  %res = select <16 x i1> %mask, <16 x float> %y, <16 x float> %a1
  ret <16 x float> %res
}

;mask (b*a+c , b)
define <16 x float> @test_x86_fmadd213_ps(<16 x float> %a0, <16 x float> %a1, <16 x float> *%a2_ptrt, <16 x i1> %mask) {
; CHECK-LABEL: test_x86_fmadd213_ps:
; CHECK:       ## BB#0:
; CHECK-NEXT:    vpmovsxbd %xmm2, %zmm2
; CHECK-NEXT:    vpandd {{.*}}(%rip){1to16}, %zmm2, %zmm2
; CHECK-NEXT:    vptestmd %zmm2, %zmm2, %k1
; CHECK-NEXT:    vfmadd213ps (%rdi), %zmm0, %zmm1 {%k1}
; CHECK-NEXT:    vmovaps %zmm1, %zmm0
; CHECK-NEXT:    retq
;
; SKX-LABEL: test_x86_fmadd213_ps:
; SKX:       ## BB#0:
; SKX-NEXT:    vpmovb2m %xmm2, %k1
; SKX-NEXT:    vfmadd213ps (%rdi), %zmm0, %zmm1 {%k1}
; SKX-NEXT:    vmovaps %zmm1, %zmm0
; SKX-NEXT:    retq
  %a2   = load <16 x float>,<16 x float> *%a2_ptrt,align 1
  %x = fmul <16 x float> %a1, %a0
  %y = fadd <16 x float> %x, %a2
  %res = select <16 x i1> %mask, <16 x float> %y, <16 x float> %a1
  ret <16 x float> %res
}

