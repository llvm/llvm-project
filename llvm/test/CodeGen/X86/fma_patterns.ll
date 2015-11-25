; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mattr=+avx,+fma -fp-contract=fast | FileCheck %s --check-prefix=ALL --check-prefix=CHECK_FMA
; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mattr=+avx,+fma4,+fma -fp-contract=fast | FileCheck %s --check-prefix=ALL --check-prefix=CHECK_FMA
; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mattr=+avx,+fma4 -fp-contract=fast | FileCheck %s --check-prefix=ALL --check-prefix=CHECK_FMA4

;
; Patterns (+ fneg variants): add(mul(x,y),z), sub(mul(x,y),z)
;

define <4 x float> @test_x86_fmadd_ps(<4 x float> %a0, <4 x float> %a1, <4 x float> %a2) {
; CHECK_FMA-LABEL: test_x86_fmadd_ps:
; CHECK_FMA:       # BB#0:
; CHECK_FMA-NEXT:    vfmadd213ps %xmm2, %xmm1, %xmm0
; CHECK_FMA-NEXT:    retq
;
; CHECK_FMA4-LABEL: test_x86_fmadd_ps:
; CHECK_FMA4:       # BB#0:
; CHECK_FMA4-NEXT:    vfmaddps %xmm2, %xmm1, %xmm0, %xmm0
; CHECK_FMA4-NEXT:    retq
  %x = fmul <4 x float> %a0, %a1
  %res = fadd <4 x float> %x, %a2
  ret <4 x float> %res
}

define <4 x float> @test_x86_fmsub_ps(<4 x float> %a0, <4 x float> %a1, <4 x float> %a2) {
; CHECK_FMA-LABEL: test_x86_fmsub_ps:
; CHECK_FMA:       # BB#0:
; CHECK_FMA-NEXT:    vfmsub213ps %xmm2, %xmm1, %xmm0
; CHECK_FMA-NEXT:    retq
;
; CHECK_FMA4-LABEL: test_x86_fmsub_ps:
; CHECK_FMA4:       # BB#0:
; CHECK_FMA4-NEXT:    vfmsubps %xmm2, %xmm1, %xmm0, %xmm0
; CHECK_FMA4-NEXT:    retq
  %x = fmul <4 x float> %a0, %a1
  %res = fsub <4 x float> %x, %a2
  ret <4 x float> %res
}

define <4 x float> @test_x86_fnmadd_ps(<4 x float> %a0, <4 x float> %a1, <4 x float> %a2) {
; CHECK_FMA-LABEL: test_x86_fnmadd_ps:
; CHECK_FMA:       # BB#0:
; CHECK_FMA-NEXT:    vfnmadd213ps %xmm2, %xmm1, %xmm0
; CHECK_FMA-NEXT:    retq
;
; CHECK_FMA4-LABEL: test_x86_fnmadd_ps:
; CHECK_FMA4:       # BB#0:
; CHECK_FMA4-NEXT:    vfnmaddps %xmm2, %xmm1, %xmm0, %xmm0
; CHECK_FMA4-NEXT:    retq
  %x = fmul <4 x float> %a0, %a1
  %res = fsub <4 x float> %a2, %x
  ret <4 x float> %res
}

define <4 x float> @test_x86_fnmsub_ps(<4 x float> %a0, <4 x float> %a1, <4 x float> %a2) {
; CHECK_FMA-LABEL: test_x86_fnmsub_ps:
; CHECK_FMA:       # BB#0:
; CHECK_FMA-NEXT:    vfnmsub213ps %xmm2, %xmm1, %xmm0
; CHECK_FMA-NEXT:    retq
;
; CHECK_FMA4-LABEL: test_x86_fnmsub_ps:
; CHECK_FMA4:       # BB#0:
; CHECK_FMA4-NEXT:    vfnmsubps %xmm2, %xmm1, %xmm0, %xmm0
; CHECK_FMA4-NEXT:    retq
  %x = fmul <4 x float> %a0, %a1
  %y = fsub <4 x float> <float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00>, %x
  %res = fsub <4 x float> %y, %a2
  ret <4 x float> %res
}

define <8 x float> @test_x86_fmadd_ps_y(<8 x float> %a0, <8 x float> %a1, <8 x float> %a2) {
; CHECK_FMA-LABEL: test_x86_fmadd_ps_y:
; CHECK_FMA:       # BB#0:
; CHECK_FMA-NEXT:    vfmadd213ps %ymm2, %ymm1, %ymm0
; CHECK_FMA-NEXT:    retq
;
; CHECK_FMA4-LABEL: test_x86_fmadd_ps_y:
; CHECK_FMA4:       # BB#0:
; CHECK_FMA4-NEXT:    vfmaddps %ymm2, %ymm1, %ymm0, %ymm0
; CHECK_FMA4-NEXT:    retq
  %x = fmul <8 x float> %a0, %a1
  %res = fadd <8 x float> %x, %a2
  ret <8 x float> %res
}

define <8 x float> @test_x86_fmsub_ps_y(<8 x float> %a0, <8 x float> %a1, <8 x float> %a2) {
; CHECK_FMA-LABEL: test_x86_fmsub_ps_y:
; CHECK_FMA:       # BB#0:
; CHECK_FMA-NEXT:    vfmsub213ps %ymm2, %ymm1, %ymm0
; CHECK_FMA-NEXT:    retq
;
; CHECK_FMA4-LABEL: test_x86_fmsub_ps_y:
; CHECK_FMA4:       # BB#0:
; CHECK_FMA4-NEXT:    vfmsubps %ymm2, %ymm1, %ymm0, %ymm0
; CHECK_FMA4-NEXT:    retq
  %x = fmul <8 x float> %a0, %a1
  %res = fsub <8 x float> %x, %a2
  ret <8 x float> %res
}

define <8 x float> @test_x86_fnmadd_ps_y(<8 x float> %a0, <8 x float> %a1, <8 x float> %a2) {
; CHECK_FMA-LABEL: test_x86_fnmadd_ps_y:
; CHECK_FMA:       # BB#0:
; CHECK_FMA-NEXT:    vfnmadd213ps %ymm2, %ymm1, %ymm0
; CHECK_FMA-NEXT:    retq
;
; CHECK_FMA4-LABEL: test_x86_fnmadd_ps_y:
; CHECK_FMA4:       # BB#0:
; CHECK_FMA4-NEXT:    vfnmaddps %ymm2, %ymm1, %ymm0, %ymm0
; CHECK_FMA4-NEXT:    retq
  %x = fmul <8 x float> %a0, %a1
  %res = fsub <8 x float> %a2, %x
  ret <8 x float> %res
}

define <8 x float> @test_x86_fnmsub_ps_y(<8 x float> %a0, <8 x float> %a1, <8 x float> %a2) {
; CHECK_FMA-LABEL: test_x86_fnmsub_ps_y:
; CHECK_FMA:       # BB#0:
; CHECK_FMA-NEXT:    vfnmsub213ps %ymm2, %ymm1, %ymm0
; CHECK_FMA-NEXT:    retq
;
; CHECK_FMA4-LABEL: test_x86_fnmsub_ps_y:
; CHECK_FMA4:       # BB#0:
; CHECK_FMA4-NEXT:    vfnmsubps %ymm2, %ymm1, %ymm0, %ymm0
; CHECK_FMA4-NEXT:    retq
  %x = fmul <8 x float> %a0, %a1
  %y = fsub <8 x float> <float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00>, %x
  %res = fsub <8 x float> %y, %a2
  ret <8 x float> %res
}

define <4 x double> @test_x86_fmadd_pd_y(<4 x double> %a0, <4 x double> %a1, <4 x double> %a2) {
; CHECK_FMA-LABEL: test_x86_fmadd_pd_y:
; CHECK_FMA:       # BB#0:
; CHECK_FMA-NEXT:    vfmadd213pd %ymm2, %ymm1, %ymm0
; CHECK_FMA-NEXT:    retq
;
; CHECK_FMA4-LABEL: test_x86_fmadd_pd_y:
; CHECK_FMA4:       # BB#0:
; CHECK_FMA4-NEXT:    vfmaddpd %ymm2, %ymm1, %ymm0, %ymm0
; CHECK_FMA4-NEXT:    retq
  %x = fmul <4 x double> %a0, %a1
  %res = fadd <4 x double> %x, %a2
  ret <4 x double> %res
}

define <4 x double> @test_x86_fmsub_pd_y(<4 x double> %a0, <4 x double> %a1, <4 x double> %a2) {
; CHECK_FMA-LABEL: test_x86_fmsub_pd_y:
; CHECK_FMA:       # BB#0:
; CHECK_FMA-NEXT:    vfmsub213pd %ymm2, %ymm1, %ymm0
; CHECK_FMA-NEXT:    retq
;
; CHECK_FMA4-LABEL: test_x86_fmsub_pd_y:
; CHECK_FMA4:       # BB#0:
; CHECK_FMA4-NEXT:    vfmsubpd %ymm2, %ymm1, %ymm0, %ymm0
; CHECK_FMA4-NEXT:    retq
  %x = fmul <4 x double> %a0, %a1
  %res = fsub <4 x double> %x, %a2
  ret <4 x double> %res
}

define <2 x double> @test_x86_fmsub_pd(<2 x double> %a0, <2 x double> %a1, <2 x double> %a2) {
; CHECK_FMA-LABEL: test_x86_fmsub_pd:
; CHECK_FMA:       # BB#0:
; CHECK_FMA-NEXT:    vfmsub213pd %xmm2, %xmm1, %xmm0
; CHECK_FMA-NEXT:    retq
;
; CHECK_FMA4-LABEL: test_x86_fmsub_pd:
; CHECK_FMA4:       # BB#0:
; CHECK_FMA4-NEXT:    vfmsubpd %xmm2, %xmm1, %xmm0, %xmm0
; CHECK_FMA4-NEXT:    retq
  %x = fmul <2 x double> %a0, %a1
  %res = fsub <2 x double> %x, %a2
  ret <2 x double> %res
}

define float @test_x86_fnmadd_ss(float %a0, float %a1, float %a2) {
; CHECK_FMA-LABEL: test_x86_fnmadd_ss:
; CHECK_FMA:       # BB#0:
; CHECK_FMA-NEXT:    vfnmadd213ss %xmm2, %xmm1, %xmm0
; CHECK_FMA-NEXT:    retq
;
; CHECK_FMA4-LABEL: test_x86_fnmadd_ss:
; CHECK_FMA4:       # BB#0:
; CHECK_FMA4-NEXT:    vfnmaddss %xmm2, %xmm1, %xmm0, %xmm0
; CHECK_FMA4-NEXT:    retq
  %x = fmul float %a0, %a1
  %res = fsub float %a2, %x
  ret float %res
}

define double @test_x86_fnmadd_sd(double %a0, double %a1, double %a2) {
; CHECK_FMA-LABEL: test_x86_fnmadd_sd:
; CHECK_FMA:       # BB#0:
; CHECK_FMA-NEXT:    vfnmadd213sd %xmm2, %xmm1, %xmm0
; CHECK_FMA-NEXT:    retq
;
; CHECK_FMA4-LABEL: test_x86_fnmadd_sd:
; CHECK_FMA4:       # BB#0:
; CHECK_FMA4-NEXT:    vfnmaddsd %xmm2, %xmm1, %xmm0, %xmm0
; CHECK_FMA4-NEXT:    retq
  %x = fmul double %a0, %a1
  %res = fsub double %a2, %x
  ret double %res
}

define double @test_x86_fmsub_sd(double %a0, double %a1, double %a2) {
; CHECK_FMA-LABEL: test_x86_fmsub_sd:
; CHECK_FMA:       # BB#0:
; CHECK_FMA-NEXT:    vfmsub213sd %xmm2, %xmm1, %xmm0
; CHECK_FMA-NEXT:    retq
;
; CHECK_FMA4-LABEL: test_x86_fmsub_sd:
; CHECK_FMA4:       # BB#0:
; CHECK_FMA4-NEXT:    vfmsubsd %xmm2, %xmm1, %xmm0, %xmm0
; CHECK_FMA4-NEXT:    retq
  %x = fmul double %a0, %a1
  %res = fsub double %x, %a2
  ret double %res
}

define float @test_x86_fnmsub_ss(float %a0, float %a1, float %a2) {
; CHECK_FMA-LABEL: test_x86_fnmsub_ss:
; CHECK_FMA:       # BB#0:
; CHECK_FMA-NEXT:    vfnmsub213ss %xmm2, %xmm1, %xmm0
; CHECK_FMA-NEXT:    retq
;
; CHECK_FMA4-LABEL: test_x86_fnmsub_ss:
; CHECK_FMA4:       # BB#0:
; CHECK_FMA4-NEXT:    vfnmsubss %xmm2, %xmm1, %xmm0, %xmm0
; CHECK_FMA4-NEXT:    retq
  %x = fsub float -0.000000e+00, %a0
  %y = fmul float %x, %a1
  %res = fsub float %y, %a2
  ret float %res
}

define <4 x float> @test_x86_fmadd_ps_load(<4 x float>* %a0, <4 x float> %a1, <4 x float> %a2) {
; CHECK_FMA-LABEL: test_x86_fmadd_ps_load:
; CHECK_FMA:       # BB#0:
; CHECK_FMA-NEXT:    vfmadd132ps (%rdi), %xmm1, %xmm0
; CHECK_FMA-NEXT:    retq
;
; CHECK_FMA4-LABEL: test_x86_fmadd_ps_load:
; CHECK_FMA4:       # BB#0:
; CHECK_FMA4-NEXT:    vfmaddps %xmm1, (%rdi), %xmm0, %xmm0
; CHECK_FMA4-NEXT:    retq
  %x = load <4 x float>, <4 x float>* %a0
  %y = fmul <4 x float> %x, %a1
  %res = fadd <4 x float> %y, %a2
  ret <4 x float> %res
}

define <4 x float> @test_x86_fmsub_ps_load(<4 x float>* %a0, <4 x float> %a1, <4 x float> %a2) {
; CHECK_FMA-LABEL: test_x86_fmsub_ps_load:
; CHECK_FMA:       # BB#0:
; CHECK_FMA-NEXT:    vfmsub132ps (%rdi), %xmm1, %xmm0
; CHECK_FMA-NEXT:    retq
;
; CHECK_FMA4-LABEL: test_x86_fmsub_ps_load:
; CHECK_FMA4:       # BB#0:
; CHECK_FMA4-NEXT:    vfmsubps %xmm1, (%rdi), %xmm0, %xmm0
; CHECK_FMA4-NEXT:    retq
  %x = load <4 x float>, <4 x float>* %a0
  %y = fmul <4 x float> %x, %a1
  %res = fsub <4 x float> %y, %a2
  ret <4 x float> %res
}

;
; Patterns (+ fneg variants): mul(add(1.0,x),y), mul(sub(1.0,x),y), mul(sub(x,1.0),y)
;

define <4 x float> @test_v4f32_mul_add_x_one_y(<4 x float> %x, <4 x float> %y) {
; CHECK_FMA-LABEL: test_v4f32_mul_add_x_one_y:
; CHECK_FMA:       # BB#0:
; CHECK_FMA-NEXT:    vfmadd213ps %xmm1, %xmm1, %xmm0
; CHECK_FMA-NEXT:    retq
;
; CHECK_FMA4-LABEL: test_v4f32_mul_add_x_one_y:
; CHECK_FMA4:       # BB#0:
; CHECK_FMA4-NEXT:    vfmaddps %xmm1, %xmm1, %xmm0, %xmm0
; CHECK_FMA4-NEXT:    retq
  %a = fadd <4 x float> %x, <float 1.0, float 1.0, float 1.0, float 1.0>
  %m = fmul <4 x float> %a, %y
  ret <4 x float> %m
}

define <4 x float> @test_v4f32_mul_y_add_x_one(<4 x float> %x, <4 x float> %y) {
; CHECK_FMA-LABEL: test_v4f32_mul_y_add_x_one:
; CHECK_FMA:       # BB#0:
; CHECK_FMA-NEXT:    vfmadd213ps %xmm1, %xmm1, %xmm0
; CHECK_FMA-NEXT:    retq
;
; CHECK_FMA4-LABEL: test_v4f32_mul_y_add_x_one:
; CHECK_FMA4:       # BB#0:
; CHECK_FMA4-NEXT:    vfmaddps %xmm1, %xmm1, %xmm0, %xmm0
; CHECK_FMA4-NEXT:    retq
  %a = fadd <4 x float> %x, <float 1.0, float 1.0, float 1.0, float 1.0>
  %m = fmul <4 x float> %y, %a
  ret <4 x float> %m
}

define <4 x float> @test_v4f32_mul_add_x_negone_y(<4 x float> %x, <4 x float> %y) {
; CHECK_FMA-LABEL: test_v4f32_mul_add_x_negone_y:
; CHECK_FMA:       # BB#0:
; CHECK_FMA-NEXT:    vfmsub213ps %xmm1, %xmm1, %xmm0
; CHECK_FMA-NEXT:    retq
;
; CHECK_FMA4-LABEL: test_v4f32_mul_add_x_negone_y:
; CHECK_FMA4:       # BB#0:
; CHECK_FMA4-NEXT:    vfmsubps %xmm1, %xmm1, %xmm0, %xmm0
; CHECK_FMA4-NEXT:    retq
  %a = fadd <4 x float> %x, <float -1.0, float -1.0, float -1.0, float -1.0>
  %m = fmul <4 x float> %a, %y
  ret <4 x float> %m
}

define <4 x float> @test_v4f32_mul_y_add_x_negone(<4 x float> %x, <4 x float> %y) {
; CHECK_FMA-LABEL: test_v4f32_mul_y_add_x_negone:
; CHECK_FMA:       # BB#0:
; CHECK_FMA-NEXT:    vfmsub213ps %xmm1, %xmm1, %xmm0
; CHECK_FMA-NEXT:    retq
;
; CHECK_FMA4-LABEL: test_v4f32_mul_y_add_x_negone:
; CHECK_FMA4:       # BB#0:
; CHECK_FMA4-NEXT:    vfmsubps %xmm1, %xmm1, %xmm0, %xmm0
; CHECK_FMA4-NEXT:    retq
  %a = fadd <4 x float> %x, <float -1.0, float -1.0, float -1.0, float -1.0>
  %m = fmul <4 x float> %y, %a
  ret <4 x float> %m
}

define <4 x float> @test_v4f32_mul_sub_one_x_y(<4 x float> %x, <4 x float> %y) {
; CHECK_FMA-LABEL: test_v4f32_mul_sub_one_x_y:
; CHECK_FMA:       # BB#0:
; CHECK_FMA-NEXT:    vfnmadd213ps %xmm1, %xmm1, %xmm0
; CHECK_FMA-NEXT:    retq
;
; CHECK_FMA4-LABEL: test_v4f32_mul_sub_one_x_y:
; CHECK_FMA4:       # BB#0:
; CHECK_FMA4-NEXT:    vfnmaddps %xmm1, %xmm1, %xmm0, %xmm0
; CHECK_FMA4-NEXT:    retq
  %s = fsub <4 x float> <float 1.0, float 1.0, float 1.0, float 1.0>, %x
  %m = fmul <4 x float> %s, %y
  ret <4 x float> %m
}

define <4 x float> @test_v4f32_mul_y_sub_one_x(<4 x float> %x, <4 x float> %y) {
; CHECK_FMA-LABEL: test_v4f32_mul_y_sub_one_x:
; CHECK_FMA:       # BB#0:
; CHECK_FMA-NEXT:    vfnmadd213ps %xmm1, %xmm1, %xmm0
; CHECK_FMA-NEXT:    retq
;
; CHECK_FMA4-LABEL: test_v4f32_mul_y_sub_one_x:
; CHECK_FMA4:       # BB#0:
; CHECK_FMA4-NEXT:    vfnmaddps %xmm1, %xmm1, %xmm0, %xmm0
; CHECK_FMA4-NEXT:    retq
  %s = fsub <4 x float> <float 1.0, float 1.0, float 1.0, float 1.0>, %x
  %m = fmul <4 x float> %y, %s
  ret <4 x float> %m
}

define <4 x float> @test_v4f32_mul_sub_negone_x_y(<4 x float> %x, <4 x float> %y) {
; CHECK_FMA-LABEL: test_v4f32_mul_sub_negone_x_y:
; CHECK_FMA:       # BB#0:
; CHECK_FMA-NEXT:    vfnmsub213ps %xmm1, %xmm1, %xmm0
; CHECK_FMA-NEXT:    retq
;
; CHECK_FMA4-LABEL: test_v4f32_mul_sub_negone_x_y:
; CHECK_FMA4:       # BB#0:
; CHECK_FMA4-NEXT:    vfnmsubps %xmm1, %xmm1, %xmm0, %xmm0
; CHECK_FMA4-NEXT:    retq
  %s = fsub <4 x float> <float -1.0, float -1.0, float -1.0, float -1.0>, %x
  %m = fmul <4 x float> %s, %y
  ret <4 x float> %m
}

define <4 x float> @test_v4f32_mul_y_sub_negone_x(<4 x float> %x, <4 x float> %y) {
; CHECK_FMA-LABEL: test_v4f32_mul_y_sub_negone_x:
; CHECK_FMA:       # BB#0:
; CHECK_FMA-NEXT:    vfnmsub213ps %xmm1, %xmm1, %xmm0
; CHECK_FMA-NEXT:    retq
;
; CHECK_FMA4-LABEL: test_v4f32_mul_y_sub_negone_x:
; CHECK_FMA4:       # BB#0:
; CHECK_FMA4-NEXT:    vfnmsubps %xmm1, %xmm1, %xmm0, %xmm0
; CHECK_FMA4-NEXT:    retq
  %s = fsub <4 x float> <float -1.0, float -1.0, float -1.0, float -1.0>, %x
  %m = fmul <4 x float> %y, %s
  ret <4 x float> %m
}

define <4 x float> @test_v4f32_mul_sub_x_one_y(<4 x float> %x, <4 x float> %y) {
; CHECK_FMA-LABEL: test_v4f32_mul_sub_x_one_y:
; CHECK_FMA:       # BB#0:
; CHECK_FMA-NEXT:    vfmsub213ps %xmm1, %xmm1, %xmm0
; CHECK_FMA-NEXT:    retq
;
; CHECK_FMA4-LABEL: test_v4f32_mul_sub_x_one_y:
; CHECK_FMA4:       # BB#0:
; CHECK_FMA4-NEXT:    vfmsubps %xmm1, %xmm1, %xmm0, %xmm0
; CHECK_FMA4-NEXT:    retq
  %s = fsub <4 x float> %x, <float 1.0, float 1.0, float 1.0, float 1.0>
  %m = fmul <4 x float> %s, %y
  ret <4 x float> %m
}

define <4 x float> @test_v4f32_mul_y_sub_x_one(<4 x float> %x, <4 x float> %y) {
; CHECK_FMA-LABEL: test_v4f32_mul_y_sub_x_one:
; CHECK_FMA:       # BB#0:
; CHECK_FMA-NEXT:    vfmsub213ps %xmm1, %xmm1, %xmm0
; CHECK_FMA-NEXT:    retq
;
; CHECK_FMA4-LABEL: test_v4f32_mul_y_sub_x_one:
; CHECK_FMA4:       # BB#0:
; CHECK_FMA4-NEXT:    vfmsubps %xmm1, %xmm1, %xmm0, %xmm0
; CHECK_FMA4-NEXT:    retq
  %s = fsub <4 x float> %x, <float 1.0, float 1.0, float 1.0, float 1.0>
  %m = fmul <4 x float> %y, %s
  ret <4 x float> %m
}

define <4 x float> @test_v4f32_mul_sub_x_negone_y(<4 x float> %x, <4 x float> %y) {
; CHECK_FMA-LABEL: test_v4f32_mul_sub_x_negone_y:
; CHECK_FMA:       # BB#0:
; CHECK_FMA-NEXT:    vfmadd213ps %xmm1, %xmm1, %xmm0
; CHECK_FMA-NEXT:    retq
;
; CHECK_FMA4-LABEL: test_v4f32_mul_sub_x_negone_y:
; CHECK_FMA4:       # BB#0:
; CHECK_FMA4-NEXT:    vfmaddps %xmm1, %xmm1, %xmm0, %xmm0
; CHECK_FMA4-NEXT:    retq
  %s = fsub <4 x float> %x, <float -1.0, float -1.0, float -1.0, float -1.0>
  %m = fmul <4 x float> %s, %y
  ret <4 x float> %m
}

define <4 x float> @test_v4f32_mul_y_sub_x_negone(<4 x float> %x, <4 x float> %y) {
; CHECK_FMA-LABEL: test_v4f32_mul_y_sub_x_negone:
; CHECK_FMA:       # BB#0:
; CHECK_FMA-NEXT:    vfmadd213ps %xmm1, %xmm1, %xmm0
; CHECK_FMA-NEXT:    retq
;
; CHECK_FMA4-LABEL: test_v4f32_mul_y_sub_x_negone:
; CHECK_FMA4:       # BB#0:
; CHECK_FMA4-NEXT:    vfmaddps %xmm1, %xmm1, %xmm0, %xmm0
; CHECK_FMA4-NEXT:    retq
  %s = fsub <4 x float> %x, <float -1.0, float -1.0, float -1.0, float -1.0>
  %m = fmul <4 x float> %y, %s
  ret <4 x float> %m
}

;
; Interpolation Patterns: add(mul(x,t),mul(sub(1.0,t),y))
;

define float @test_f32_interp(float %x, float %y, float %t) {
; CHECK_FMA-LABEL: test_f32_interp:
; CHECK_FMA:       # BB#0:
; CHECK_FMA-NEXT:    vfnmadd213ss %xmm1, %xmm2, %xmm1
; CHECK_FMA-NEXT:    vfmadd213ss %xmm1, %xmm2, %xmm0
; CHECK_FMA-NEXT:    retq
;
; CHECK_FMA4-LABEL: test_f32_interp:
; CHECK_FMA4:       # BB#0:
; CHECK_FMA4-NEXT:    vfnmaddss %xmm1, %xmm1, %xmm2, %xmm1
; CHECK_FMA4-NEXT:    vfmaddss %xmm1, %xmm2, %xmm0, %xmm0
; CHECK_FMA4-NEXT:    retq
  %t1 = fsub float 1.0, %t
  %tx = fmul float %x, %t
  %ty = fmul float %y, %t1
  %r = fadd float %tx, %ty
  ret float %r
}

define <4 x float> @test_v4f32_interp(<4 x float> %x, <4 x float> %y, <4 x float> %t) {
; CHECK_FMA-LABEL: test_v4f32_interp:
; CHECK_FMA:       # BB#0:
; CHECK_FMA-NEXT:    vfnmadd213ps %xmm1, %xmm2, %xmm1
; CHECK_FMA-NEXT:    vfmadd213ps %xmm1, %xmm2, %xmm0
; CHECK_FMA-NEXT:    retq
;
; CHECK_FMA4-LABEL: test_v4f32_interp:
; CHECK_FMA4:       # BB#0:
; CHECK_FMA4-NEXT:    vfnmaddps %xmm1, %xmm1, %xmm2, %xmm1
; CHECK_FMA4-NEXT:    vfmaddps %xmm1, %xmm2, %xmm0, %xmm0
; CHECK_FMA4-NEXT:    retq
  %t1 = fsub <4 x float> <float 1.0, float 1.0, float 1.0, float 1.0>, %t
  %tx = fmul <4 x float> %x, %t
  %ty = fmul <4 x float> %y, %t1
  %r = fadd <4 x float> %tx, %ty
  ret <4 x float> %r
}

define <8 x float> @test_v8f32_interp(<8 x float> %x, <8 x float> %y, <8 x float> %t) {
; CHECK_FMA-LABEL: test_v8f32_interp:
; CHECK_FMA:       # BB#0:
; CHECK_FMA-NEXT:    vfnmadd213ps %ymm1, %ymm2, %ymm1
; CHECK_FMA-NEXT:    vfmadd213ps %ymm1, %ymm2, %ymm0
; CHECK_FMA-NEXT:    retq
;
; CHECK_FMA4-LABEL: test_v8f32_interp:
; CHECK_FMA4:       # BB#0:
; CHECK_FMA4-NEXT:    vfnmaddps %ymm1, %ymm1, %ymm2, %ymm1
; CHECK_FMA4-NEXT:    vfmaddps %ymm1, %ymm2, %ymm0, %ymm0
; CHECK_FMA4-NEXT:    retq
  %t1 = fsub <8 x float> <float 1.0, float 1.0, float 1.0, float 1.0, float 1.0, float 1.0, float 1.0, float 1.0>, %t
  %tx = fmul <8 x float> %x, %t
  %ty = fmul <8 x float> %y, %t1
  %r = fadd <8 x float> %tx, %ty
  ret <8 x float> %r
}

define double @test_f64_interp(double %x, double %y, double %t) {
; CHECK_FMA-LABEL: test_f64_interp:
; CHECK_FMA:       # BB#0:
; CHECK_FMA-NEXT:    vfnmadd213sd %xmm1, %xmm2, %xmm1
; CHECK_FMA-NEXT:    vfmadd213sd %xmm1, %xmm2, %xmm0
; CHECK_FMA-NEXT:    retq
;
; CHECK_FMA4-LABEL: test_f64_interp:
; CHECK_FMA4:       # BB#0:
; CHECK_FMA4-NEXT:    vfnmaddsd %xmm1, %xmm1, %xmm2, %xmm1
; CHECK_FMA4-NEXT:    vfmaddsd %xmm1, %xmm2, %xmm0, %xmm0
; CHECK_FMA4-NEXT:    retq
  %t1 = fsub double 1.0, %t
  %tx = fmul double %x, %t
  %ty = fmul double %y, %t1
  %r = fadd double %tx, %ty
  ret double %r
}

define <2 x double> @test_v2f64_interp(<2 x double> %x, <2 x double> %y, <2 x double> %t) {
; CHECK_FMA-LABEL: test_v2f64_interp:
; CHECK_FMA:       # BB#0:
; CHECK_FMA-NEXT:    vfnmadd213pd %xmm1, %xmm2, %xmm1
; CHECK_FMA-NEXT:    vfmadd213pd %xmm1, %xmm2, %xmm0
; CHECK_FMA-NEXT:    retq
;
; CHECK_FMA4-LABEL: test_v2f64_interp:
; CHECK_FMA4:       # BB#0:
; CHECK_FMA4-NEXT:    vfnmaddpd %xmm1, %xmm1, %xmm2, %xmm1
; CHECK_FMA4-NEXT:    vfmaddpd %xmm1, %xmm2, %xmm0, %xmm0
; CHECK_FMA4-NEXT:    retq
  %t1 = fsub <2 x double> <double 1.0, double 1.0>, %t
  %tx = fmul <2 x double> %x, %t
  %ty = fmul <2 x double> %y, %t1
  %r = fadd <2 x double> %tx, %ty
  ret <2 x double> %r
}

define <4 x double> @test_v4f64_interp(<4 x double> %x, <4 x double> %y, <4 x double> %t) {
; CHECK_FMA-LABEL: test_v4f64_interp:
; CHECK_FMA:       # BB#0:
; CHECK_FMA-NEXT:    vfnmadd213pd %ymm1, %ymm2, %ymm1
; CHECK_FMA-NEXT:    vfmadd213pd %ymm1, %ymm2, %ymm0
; CHECK_FMA-NEXT:    retq
;
; CHECK_FMA4-LABEL: test_v4f64_interp:
; CHECK_FMA4:       # BB#0:
; CHECK_FMA4-NEXT:    vfnmaddpd %ymm1, %ymm1, %ymm2, %ymm1
; CHECK_FMA4-NEXT:    vfmaddpd %ymm1, %ymm2, %ymm0, %ymm0
; CHECK_FMA4-NEXT:    retq
  %t1 = fsub <4 x double> <double 1.0, double 1.0, double 1.0, double 1.0>, %t
  %tx = fmul <4 x double> %x, %t
  %ty = fmul <4 x double> %y, %t1
  %r = fadd <4 x double> %tx, %ty
  ret <4 x double> %r
}

; (fneg (fma x, y, z)) -> (fma x, -y, -z)

define <4 x float> @test_v4f32_fneg_fmadd(<4 x float> %a0, <4 x float> %a1, <4 x float> %a2) #0 {
; CHECK_FMA-LABEL: test_v4f32_fneg_fmadd:
; CHECK_FMA:       # BB#0:
; CHECK_FMA-NEXT:    vfnmsub213ps %xmm2, %xmm1, %xmm0
; CHECK_FMA-NEXT:    retq
;
; CHECK_FMA4-LABEL: test_v4f32_fneg_fmadd:
; CHECK_FMA4:       # BB#0:
; CHECK_FMA4-NEXT:    vfnmsubps %xmm2, %xmm1, %xmm0, %xmm0
; CHECK_FMA4-NEXT:    retq
  %mul = fmul <4 x float> %a0, %a1
  %add = fadd <4 x float> %mul, %a2
  %neg = fsub <4 x float> <float -0.0, float -0.0, float -0.0, float -0.0>, %add
  ret <4 x float> %neg
}

define <4 x double> @test_v4f64_fneg_fmsub(<4 x double> %a0, <4 x double> %a1, <4 x double> %a2) #0 {
; CHECK_FMA-LABEL: test_v4f64_fneg_fmsub:
; CHECK_FMA:       # BB#0:
; CHECK_FMA-NEXT:    vfnmadd213pd %ymm2, %ymm1, %ymm0
; CHECK_FMA-NEXT:    retq
;
; CHECK_FMA4-LABEL: test_v4f64_fneg_fmsub:
; CHECK_FMA4:       # BB#0:
; CHECK_FMA4-NEXT:    vfnmaddpd %ymm2, %ymm1, %ymm0, %ymm0
; CHECK_FMA4-NEXT:    retq
  %mul = fmul <4 x double> %a0, %a1
  %sub = fsub <4 x double> %mul, %a2
  %neg = fsub <4 x double> <double -0.0, double -0.0, double -0.0, double -0.0>, %sub
  ret <4 x double> %neg
}

define <4 x float> @test_v4f32_fneg_fnmadd(<4 x float> %a0, <4 x float> %a1, <4 x float> %a2) #0 {
; CHECK_FMA-LABEL: test_v4f32_fneg_fnmadd:
; CHECK_FMA:       # BB#0:
; CHECK_FMA-NEXT:    vfmsub213ps %xmm2, %xmm1, %xmm0
; CHECK_FMA-NEXT:    retq
;
; CHECK_FMA4-LABEL: test_v4f32_fneg_fnmadd:
; CHECK_FMA4:       # BB#0:
; CHECK_FMA4-NEXT:    vfmsubps %xmm2, %xmm1, %xmm0, %xmm0
; CHECK_FMA4-NEXT:    retq
  %mul = fmul <4 x float> %a0, %a1
  %neg0 = fsub <4 x float> <float -0.0, float -0.0, float -0.0, float -0.0>, %mul
  %add = fadd <4 x float> %neg0, %a2
  %neg1 = fsub <4 x float> <float -0.0, float -0.0, float -0.0, float -0.0>, %add
  ret <4 x float> %neg1
}

define <4 x double> @test_v4f64_fneg_fnmsub(<4 x double> %a0, <4 x double> %a1, <4 x double> %a2) #0 {
; CHECK_FMA-LABEL: test_v4f64_fneg_fnmsub:
; CHECK_FMA:       # BB#0:
; CHECK_FMA-NEXT:    vfmadd213pd %ymm2, %ymm1, %ymm0
; CHECK_FMA-NEXT:    retq
;
; CHECK_FMA4-LABEL: test_v4f64_fneg_fnmsub:
; CHECK_FMA4:       # BB#0:
; CHECK_FMA4-NEXT:    vfmaddpd %ymm2, %ymm1, %ymm0, %ymm0
; CHECK_FMA4-NEXT:    retq
  %mul = fmul <4 x double> %a0, %a1
  %neg0 = fsub <4 x double> <double -0.0, double -0.0, double -0.0, double -0.0>, %mul
  %sub = fsub <4 x double> %neg0, %a2
  %neg1 = fsub <4 x double> <double -0.0, double -0.0, double -0.0, double -0.0>, %sub
  ret <4 x double> %neg1
}

; (fma x, c1, (fmul x, c2)) -> (fmul x, c1+c2)

define <4 x float> @test_v4f32_fma_x_c1_fmul_x_c2(<4 x float> %x) #0 {
; ALL-LABEL: test_v4f32_fma_x_c1_fmul_x_c2:
; ALL:       # BB#0:
; ALL-NEXT:    vmulps {{.*}}(%rip), %xmm0, %xmm0
; ALL-NEXT:    retq
  %m0 = fmul <4 x float> %x, <float 1.0, float 2.0, float 3.0, float 4.0>
  %m1 = fmul <4 x float> %x, <float 4.0, float 3.0, float 2.0, float 1.0>
  %a  = fadd <4 x float> %m0, %m1
  ret <4 x float> %a
}

; (fma (fmul x, c1), c2, y) -> (fma x, c1*c2, y)

define <4 x float> @test_v4f32_fma_fmul_x_c1_c2_y(<4 x float> %x, <4 x float> %y) #0 {
; CHECK_FMA-LABEL: test_v4f32_fma_fmul_x_c1_c2_y:
; CHECK_FMA:       # BB#0:
; CHECK_FMA-NEXT:    vfmadd132ps {{.*}}(%rip), %xmm1, %xmm0
; CHECK_FMA-NEXT:    retq
;
; CHECK_FMA4-LABEL: test_v4f32_fma_fmul_x_c1_c2_y:
; CHECK_FMA4:       # BB#0:
; CHECK_FMA4-NEXT:    vfmaddps %xmm1, {{.*}}(%rip), %xmm0, %xmm0
; CHECK_FMA4-NEXT:    retq
  %m0 = fmul <4 x float> %x,  <float 1.0, float 2.0, float 3.0, float 4.0>
  %m1 = fmul <4 x float> %m0, <float 4.0, float 3.0, float 2.0, float 1.0>
  %a  = fadd <4 x float> %m1, %y
  ret <4 x float> %a
}

attributes #0 = { "unsafe-fp-math"="true" }
