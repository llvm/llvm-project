; RUN: llc < %s -mtriple=x86_64-pc-win32 -mcpu=core-avx2 | FileCheck %s
; RUN: llc < %s -mtriple=x86_64-pc-win32 -mattr=+fma,+fma4 | FileCheck %s
; RUN: llc < %s -mcpu=bdver2 -mtriple=x86_64-pc-win32 -mattr=-fma4 | FileCheck %s

attributes #0 = { nounwind }

declare <4 x float> @llvm.x86.fma.vfmadd.ps(<4 x float>, <4 x float>, <4 x float>) nounwind readnone
define <4 x float> @test_x86_fmadd_baa_ps(<4 x float> %a, <4 x float> %b) #0 {
; CHECK-LABEL: test_x86_fmadd_baa_ps:
; CHECK:       # BB#0:
; CHECK-NEXT: vmovaps	(%rcx), %xmm0
; CHECK-NEXT: vfmadd132ps (%rdx), %xmm0, %xmm0
; CHECK-NEXT: retq
  %res = call <4 x float> @llvm.x86.fma.vfmadd.ps(<4 x float> %b, <4 x float> %a, <4 x float> %a) nounwind
  ret <4 x float> %res
}

define <4 x float> @test_x86_fmadd_aba_ps(<4 x float> %a, <4 x float> %b) #0 {
; CHECK-LABEL: test_x86_fmadd_aba_ps:
; CHECK:       # BB#0:
; CHECK-NEXT: vmovaps	(%rcx), %xmm0
; CHECK-NEXT: vfmadd231ps (%rdx), %xmm0, %xmm0
; CHECK-NEXT: retq
  %res = call <4 x float> @llvm.x86.fma.vfmadd.ps(<4 x float> %a, <4 x float> %b, <4 x float> %a) nounwind
  ret <4 x float> %res
}

define <4 x float> @test_x86_fmadd_bba_ps(<4 x float> %a, <4 x float> %b) #0 {
; CHECK-LABEL: test_x86_fmadd_bba_ps:
; CHECK:       # BB#0:
; CHECK-NEXT: vmovaps	(%rdx), %xmm0
; CHECK-NEXT: vfmadd213ps (%rcx), %xmm0, %xmm0
; CHECK-NEXT: retq
  %res = call <4 x float> @llvm.x86.fma.vfmadd.ps(<4 x float> %b, <4 x float> %b, <4 x float> %a) nounwind
  ret <4 x float> %res
}

declare <8 x float> @llvm.x86.fma.vfmadd.ps.256(<8 x float>, <8 x float>, <8 x float>) nounwind readnone
define <8 x float> @test_x86_fmadd_baa_ps_y(<8 x float> %a, <8 x float> %b) #0 {
; CHECK-LABEL: test_x86_fmadd_baa_ps_y:
; CHECK:       # BB#0:
; CHECK-NEXT: vmovaps	(%rcx), %ymm0
; CHECK-NEXT: vfmadd132ps (%rdx), %ymm0, %ymm0
; CHECK-NEXT: retq
  %res = call <8 x float> @llvm.x86.fma.vfmadd.ps.256(<8 x float> %b, <8 x float> %a, <8 x float> %a) nounwind
  ret <8 x float> %res
}

define <8 x float> @test_x86_fmadd_aba_ps_y(<8 x float> %a, <8 x float> %b) #0 {
; CHECK-LABEL: test_x86_fmadd_aba_ps_y:
; CHECK:       # BB#0:
; CHECK-NEXT: vmovaps	(%rcx), %ymm0
; CHECK-NEXT: vfmadd231ps (%rdx), %ymm0, %ymm0
; CHECK-NEXT: retq
  %res = call <8 x float> @llvm.x86.fma.vfmadd.ps.256(<8 x float> %a, <8 x float> %b, <8 x float> %a) nounwind
  ret <8 x float> %res
}

define <8 x float> @test_x86_fmadd_bba_ps_y(<8 x float> %a, <8 x float> %b) #0 {
; CHECK-LABEL: test_x86_fmadd_bba_ps_y:
; CHECK:       # BB#0:
; CHECK-NEXT: vmovaps	(%rdx), %ymm0
; CHECK-NEXT: vfmadd213ps (%rcx), %ymm0, %ymm0
; CHECK-NEXT: retq
  %res = call <8 x float> @llvm.x86.fma.vfmadd.ps.256(<8 x float> %b, <8 x float> %b, <8 x float> %a) nounwind
  ret <8 x float> %res
}

declare <2 x double> @llvm.x86.fma.vfmadd.pd(<2 x double>, <2 x double>, <2 x double>) nounwind readnone
define <2 x double> @test_x86_fmadd_baa_pd(<2 x double> %a, <2 x double> %b) #0 {
; CHECK-LABEL: test_x86_fmadd_baa_pd:
; CHECK:       # BB#0:
; CHECK-NEXT: vmovapd	(%rcx), %xmm0
; CHECK-NEXT: vfmadd132pd (%rdx), %xmm0, %xmm0
; CHECK-NEXT: retq
  %res = call <2 x double> @llvm.x86.fma.vfmadd.pd(<2 x double> %b, <2 x double> %a, <2 x double> %a) nounwind
  ret <2 x double> %res
}

define <2 x double> @test_x86_fmadd_aba_pd(<2 x double> %a, <2 x double> %b) #0 {
; CHECK-LABEL: test_x86_fmadd_aba_pd:
; CHECK:       # BB#0:
; CHECK-NEXT: vmovapd	(%rcx), %xmm0
; CHECK-NEXT: vfmadd231pd (%rdx), %xmm0, %xmm0
; CHECK-NEXT: retq
  %res = call <2 x double> @llvm.x86.fma.vfmadd.pd(<2 x double> %a, <2 x double> %b, <2 x double> %a) nounwind
  ret <2 x double> %res
}

define <2 x double> @test_x86_fmadd_bba_pd(<2 x double> %a, <2 x double> %b) #0 {
; CHECK-LABEL: test_x86_fmadd_bba_pd:
; CHECK:       # BB#0:
; CHECK-NEXT: vmovapd	(%rdx), %xmm0
; CHECK-NEXT: vfmadd213pd (%rcx), %xmm0, %xmm0
; CHECK-NEXT: retq
  %res = call <2 x double> @llvm.x86.fma.vfmadd.pd(<2 x double> %b, <2 x double> %b, <2 x double> %a) nounwind
  ret <2 x double> %res
}

declare <4 x double> @llvm.x86.fma.vfmadd.pd.256(<4 x double>, <4 x double>, <4 x double>) nounwind readnone
define <4 x double> @test_x86_fmadd_baa_pd_y(<4 x double> %a, <4 x double> %b) #0 {
; CHECK-LABEL: test_x86_fmadd_baa_pd_y:
; CHECK:       # BB#0:
; CHECK-NEXT: vmovapd	(%rcx), %ymm0
; CHECK-NEXT: vfmadd132pd (%rdx), %ymm0, %ymm0
; CHECK-NEXT: retq
  %res = call <4 x double> @llvm.x86.fma.vfmadd.pd.256(<4 x double> %b, <4 x double> %a, <4 x double> %a) nounwind
  ret <4 x double> %res
}

define <4 x double> @test_x86_fmadd_aba_pd_y(<4 x double> %a, <4 x double> %b) #0 {
; CHECK-LABEL: test_x86_fmadd_aba_pd_y:
; CHECK:       # BB#0:
; CHECK-NEXT: vmovapd	(%rcx), %ymm0
; CHECK-NEXT: vfmadd231pd (%rdx), %ymm0, %ymm0
; CHECK-NEXT: retq
  %res = call <4 x double> @llvm.x86.fma.vfmadd.pd.256(<4 x double> %a, <4 x double> %b, <4 x double> %a) nounwind
  ret <4 x double> %res
}

define <4 x double> @test_x86_fmadd_bba_pd_y(<4 x double> %a, <4 x double> %b) #0 {
; CHECK-LABEL: test_x86_fmadd_bba_pd_y:
; CHECK:       # BB#0:
; CHECK-NEXT: vmovapd	(%rdx), %ymm0
; CHECK-NEXT: vfmadd213pd (%rcx), %ymm0, %ymm0
; CHECK-NEXT: retq
  %res = call <4 x double> @llvm.x86.fma.vfmadd.pd.256(<4 x double> %b, <4 x double> %b, <4 x double> %a) nounwind
  ret <4 x double> %res
}



declare <4 x float> @llvm.x86.fma.vfnmadd.ps(<4 x float>, <4 x float>, <4 x float>) nounwind readnone
define <4 x float> @test_x86_fnmadd_baa_ps(<4 x float> %a, <4 x float> %b) #0 {
; CHECK-LABEL: test_x86_fnmadd_baa_ps:
; CHECK:       # BB#0:
; CHECK-NEXT: vmovaps	(%rcx), %xmm0
; CHECK-NEXT: vfnmadd132ps (%rdx), %xmm0, %xmm0
; CHECK-NEXT: retq
  %res = call <4 x float> @llvm.x86.fma.vfnmadd.ps(<4 x float> %b, <4 x float> %a, <4 x float> %a) nounwind
  ret <4 x float> %res
}

define <4 x float> @test_x86_fnmadd_aba_ps(<4 x float> %a, <4 x float> %b) #0 {
; CHECK-LABEL: test_x86_fnmadd_aba_ps:
; CHECK:       # BB#0:
; CHECK-NEXT: vmovaps	(%rcx), %xmm0
; CHECK-NEXT: vfnmadd231ps (%rdx), %xmm0, %xmm0
; CHECK-NEXT: retq
  %res = call <4 x float> @llvm.x86.fma.vfnmadd.ps(<4 x float> %a, <4 x float> %b, <4 x float> %a) nounwind
  ret <4 x float> %res
}

define <4 x float> @test_x86_fnmadd_bba_ps(<4 x float> %a, <4 x float> %b) #0 {
; CHECK-LABEL: test_x86_fnmadd_bba_ps:
; CHECK:       # BB#0:
; CHECK-NEXT: vmovaps	(%rdx), %xmm0
; CHECK-NEXT: vfnmadd213ps (%rcx), %xmm0, %xmm0
; CHECK-NEXT: retq
  %res = call <4 x float> @llvm.x86.fma.vfnmadd.ps(<4 x float> %b, <4 x float> %b, <4 x float> %a) nounwind
  ret <4 x float> %res
}

declare <8 x float> @llvm.x86.fma.vfnmadd.ps.256(<8 x float>, <8 x float>, <8 x float>) nounwind readnone
define <8 x float> @test_x86_fnmadd_baa_ps_y(<8 x float> %a, <8 x float> %b) #0 {
; CHECK-LABEL: test_x86_fnmadd_baa_ps_y:
; CHECK:       # BB#0:
; CHECK-NEXT: vmovaps	(%rcx), %ymm0
; CHECK-NEXT: vfnmadd132ps (%rdx), %ymm0, %ymm0
; CHECK-NEXT: retq
  %res = call <8 x float> @llvm.x86.fma.vfnmadd.ps.256(<8 x float> %b, <8 x float> %a, <8 x float> %a) nounwind
  ret <8 x float> %res
}

define <8 x float> @test_x86_fnmadd_aba_ps_y(<8 x float> %a, <8 x float> %b) #0 {
; CHECK-LABEL: test_x86_fnmadd_aba_ps_y:
; CHECK:       # BB#0:
; CHECK-NEXT: vmovaps	(%rcx), %ymm0
; CHECK-NEXT: vfnmadd231ps (%rdx), %ymm0, %ymm0
; CHECK-NEXT: retq
  %res = call <8 x float> @llvm.x86.fma.vfnmadd.ps.256(<8 x float> %a, <8 x float> %b, <8 x float> %a) nounwind
  ret <8 x float> %res
}

define <8 x float> @test_x86_fnmadd_bba_ps_y(<8 x float> %a, <8 x float> %b) #0 {
; CHECK-LABEL: test_x86_fnmadd_bba_ps_y:
; CHECK:       # BB#0:
; CHECK-NEXT: vmovaps	(%rdx), %ymm0
; CHECK-NEXT: vfnmadd213ps (%rcx), %ymm0, %ymm0
; CHECK-NEXT: retq
  %res = call <8 x float> @llvm.x86.fma.vfnmadd.ps.256(<8 x float> %b, <8 x float> %b, <8 x float> %a) nounwind
  ret <8 x float> %res
}

declare <2 x double> @llvm.x86.fma.vfnmadd.pd(<2 x double>, <2 x double>, <2 x double>) nounwind readnone
define <2 x double> @test_x86_fnmadd_baa_pd(<2 x double> %a, <2 x double> %b) #0 {
; CHECK-LABEL: test_x86_fnmadd_baa_pd:
; CHECK:       # BB#0:
; CHECK-NEXT: vmovapd	(%rcx), %xmm0
; CHECK-NEXT: vfnmadd132pd (%rdx), %xmm0, %xmm0
; CHECK-NEXT: retq
  %res = call <2 x double> @llvm.x86.fma.vfnmadd.pd(<2 x double> %b, <2 x double> %a, <2 x double> %a) nounwind
  ret <2 x double> %res
}

define <2 x double> @test_x86_fnmadd_aba_pd(<2 x double> %a, <2 x double> %b) #0 {
; CHECK-LABEL: test_x86_fnmadd_aba_pd:
; CHECK:       # BB#0:
; CHECK-NEXT: vmovapd	(%rcx), %xmm0
; CHECK-NEXT: vfnmadd231pd (%rdx), %xmm0, %xmm0
; CHECK-NEXT: retq
  %res = call <2 x double> @llvm.x86.fma.vfnmadd.pd(<2 x double> %a, <2 x double> %b, <2 x double> %a) nounwind
  ret <2 x double> %res
}

define <2 x double> @test_x86_fnmadd_bba_pd(<2 x double> %a, <2 x double> %b) #0 {
; CHECK-LABEL: test_x86_fnmadd_bba_pd:
; CHECK:       # BB#0:
; CHECK-NEXT: vmovapd	(%rdx), %xmm0
; CHECK-NEXT: vfnmadd213pd (%rcx), %xmm0, %xmm0
; CHECK-NEXT: retq
  %res = call <2 x double> @llvm.x86.fma.vfnmadd.pd(<2 x double> %b, <2 x double> %b, <2 x double> %a) nounwind
  ret <2 x double> %res
}

declare <4 x double> @llvm.x86.fma.vfnmadd.pd.256(<4 x double>, <4 x double>, <4 x double>) nounwind readnone
define <4 x double> @test_x86_fnmadd_baa_pd_y(<4 x double> %a, <4 x double> %b) #0 {
; CHECK-LABEL: test_x86_fnmadd_baa_pd_y:
; CHECK:       # BB#0:
; CHECK-NEXT: vmovapd	(%rcx), %ymm0
; CHECK-NEXT: vfnmadd132pd (%rdx), %ymm0, %ymm0
; CHECK-NEXT: retq
  %res = call <4 x double> @llvm.x86.fma.vfnmadd.pd.256(<4 x double> %b, <4 x double> %a, <4 x double> %a) nounwind
  ret <4 x double> %res
}

define <4 x double> @test_x86_fnmadd_aba_pd_y(<4 x double> %a, <4 x double> %b) #0 {
; CHECK-LABEL: test_x86_fnmadd_aba_pd_y:
; CHECK:       # BB#0:
; CHECK-NEXT: vmovapd	(%rcx), %ymm0
; CHECK-NEXT: vfnmadd231pd (%rdx), %ymm0, %ymm0
; CHECK-NEXT: retq
  %res = call <4 x double> @llvm.x86.fma.vfnmadd.pd.256(<4 x double> %a, <4 x double> %b, <4 x double> %a) nounwind
  ret <4 x double> %res
}

define <4 x double> @test_x86_fnmadd_bba_pd_y(<4 x double> %a, <4 x double> %b) #0 {
; CHECK-LABEL: test_x86_fnmadd_bba_pd_y:
; CHECK:       # BB#0:
; CHECK-NEXT: vmovapd	(%rdx), %ymm0
; CHECK-NEXT: vfnmadd213pd (%rcx), %ymm0, %ymm0
; CHECK-NEXT: retq
  %res = call <4 x double> @llvm.x86.fma.vfnmadd.pd.256(<4 x double> %b, <4 x double> %b, <4 x double> %a) nounwind
  ret <4 x double> %res
}


declare <4 x float> @llvm.x86.fma.vfmsub.ps(<4 x float>, <4 x float>, <4 x float>) nounwind readnone
define <4 x float> @test_x86_fmsub_baa_ps(<4 x float> %a, <4 x float> %b) #0 {
; CHECK-LABEL: test_x86_fmsub_baa_ps:
; CHECK:       # BB#0:
; CHECK-NEXT: vmovaps	(%rcx), %xmm0
; CHECK-NEXT: vfmsub132ps (%rdx), %xmm0, %xmm0
; CHECK-NEXT: retq
  %res = call <4 x float> @llvm.x86.fma.vfmsub.ps(<4 x float> %b, <4 x float> %a, <4 x float> %a) nounwind
  ret <4 x float> %res
}

define <4 x float> @test_x86_fmsub_aba_ps(<4 x float> %a, <4 x float> %b) #0 {
; CHECK-LABEL: test_x86_fmsub_aba_ps:
; CHECK:       # BB#0:
; CHECK-NEXT: vmovaps	(%rcx), %xmm0
; CHECK-NEXT: vfmsub231ps (%rdx), %xmm0, %xmm0
; CHECK-NEXT: retq
  %res = call <4 x float> @llvm.x86.fma.vfmsub.ps(<4 x float> %a, <4 x float> %b, <4 x float> %a) nounwind
  ret <4 x float> %res
}

define <4 x float> @test_x86_fmsub_bba_ps(<4 x float> %a, <4 x float> %b) #0 {
; CHECK-LABEL: test_x86_fmsub_bba_ps:
; CHECK:       # BB#0:
; CHECK-NEXT: vmovaps	(%rdx), %xmm0
; CHECK-NEXT: vfmsub213ps (%rcx), %xmm0, %xmm0
; CHECK-NEXT: retq
  %res = call <4 x float> @llvm.x86.fma.vfmsub.ps(<4 x float> %b, <4 x float> %b, <4 x float> %a) nounwind
  ret <4 x float> %res
}

declare <8 x float> @llvm.x86.fma.vfmsub.ps.256(<8 x float>, <8 x float>, <8 x float>) nounwind readnone
define <8 x float> @test_x86_fmsub_baa_ps_y(<8 x float> %a, <8 x float> %b) #0 {
; CHECK-LABEL: test_x86_fmsub_baa_ps_y:
; CHECK:       # BB#0:
; CHECK-NEXT: vmovaps	(%rcx), %ymm0
; CHECK-NEXT: vfmsub132ps (%rdx), %ymm0, %ymm0
; CHECK-NEXT: retq
  %res = call <8 x float> @llvm.x86.fma.vfmsub.ps.256(<8 x float> %b, <8 x float> %a, <8 x float> %a) nounwind
  ret <8 x float> %res
}

define <8 x float> @test_x86_fmsub_aba_ps_y(<8 x float> %a, <8 x float> %b) #0 {
; CHECK-LABEL: test_x86_fmsub_aba_ps_y:
; CHECK:       # BB#0:
; CHECK-NEXT: vmovaps	(%rcx), %ymm0
; CHECK-NEXT: vfmsub231ps (%rdx), %ymm0, %ymm0
; CHECK-NEXT: retq
  %res = call <8 x float> @llvm.x86.fma.vfmsub.ps.256(<8 x float> %a, <8 x float> %b, <8 x float> %a) nounwind
  ret <8 x float> %res
}

define <8 x float> @test_x86_fmsub_bba_ps_y(<8 x float> %a, <8 x float> %b) #0 {
; CHECK-LABEL: test_x86_fmsub_bba_ps_y:
; CHECK:       # BB#0:
; CHECK-NEXT: vmovaps	(%rdx), %ymm0
; CHECK-NEXT: vfmsub213ps (%rcx), %ymm0, %ymm0
; CHECK-NEXT: retq
  %res = call <8 x float> @llvm.x86.fma.vfmsub.ps.256(<8 x float> %b, <8 x float> %b, <8 x float> %a) nounwind
  ret <8 x float> %res
}

declare <2 x double> @llvm.x86.fma.vfmsub.pd(<2 x double>, <2 x double>, <2 x double>) nounwind readnone
define <2 x double> @test_x86_fmsub_baa_pd(<2 x double> %a, <2 x double> %b) #0 {
; CHECK-LABEL: test_x86_fmsub_baa_pd:
; CHECK:       # BB#0:
; CHECK-NEXT: vmovapd	(%rcx), %xmm0
; CHECK-NEXT: vfmsub132pd (%rdx), %xmm0, %xmm0
; CHECK-NEXT: retq
  %res = call <2 x double> @llvm.x86.fma.vfmsub.pd(<2 x double> %b, <2 x double> %a, <2 x double> %a) nounwind
  ret <2 x double> %res
}

define <2 x double> @test_x86_fmsub_aba_pd(<2 x double> %a, <2 x double> %b) #0 {
; CHECK-LABEL: test_x86_fmsub_aba_pd:
; CHECK:       # BB#0:
; CHECK-NEXT: vmovapd	(%rcx), %xmm0
; CHECK-NEXT: vfmsub231pd (%rdx), %xmm0, %xmm0
; CHECK-NEXT: retq
  %res = call <2 x double> @llvm.x86.fma.vfmsub.pd(<2 x double> %a, <2 x double> %b, <2 x double> %a) nounwind
  ret <2 x double> %res
}

define <2 x double> @test_x86_fmsub_bba_pd(<2 x double> %a, <2 x double> %b) #0 {
; CHECK-LABEL: test_x86_fmsub_bba_pd:
; CHECK:       # BB#0:
; CHECK-NEXT: vmovapd	(%rdx), %xmm0
; CHECK-NEXT: vfmsub213pd (%rcx), %xmm0, %xmm0
; CHECK-NEXT: retq
  %res = call <2 x double> @llvm.x86.fma.vfmsub.pd(<2 x double> %b, <2 x double> %b, <2 x double> %a) nounwind
  ret <2 x double> %res
}

declare <4 x double> @llvm.x86.fma.vfmsub.pd.256(<4 x double>, <4 x double>, <4 x double>) nounwind readnone
define <4 x double> @test_x86_fmsub_baa_pd_y(<4 x double> %a, <4 x double> %b) #0 {
; CHECK-LABEL: test_x86_fmsub_baa_pd_y:
; CHECK:       # BB#0:
; CHECK-NEXT: vmovapd	(%rcx), %ymm0
; CHECK-NEXT: vfmsub132pd (%rdx), %ymm0, %ymm0
; CHECK-NEXT: retq
  %res = call <4 x double> @llvm.x86.fma.vfmsub.pd.256(<4 x double> %b, <4 x double> %a, <4 x double> %a) nounwind
  ret <4 x double> %res
}

define <4 x double> @test_x86_fmsub_aba_pd_y(<4 x double> %a, <4 x double> %b) #0 {
; CHECK-LABEL: test_x86_fmsub_aba_pd_y:
; CHECK:       # BB#0:
; CHECK-NEXT: vmovapd	(%rcx), %ymm0
; CHECK-NEXT: vfmsub231pd (%rdx), %ymm0, %ymm0
; CHECK-NEXT: retq
  %res = call <4 x double> @llvm.x86.fma.vfmsub.pd.256(<4 x double> %a, <4 x double> %b, <4 x double> %a) nounwind
  ret <4 x double> %res
}

define <4 x double> @test_x86_fmsub_bba_pd_y(<4 x double> %a, <4 x double> %b) #0 {
; CHECK-LABEL: test_x86_fmsub_bba_pd_y:
; CHECK:       # BB#0:
; CHECK-NEXT: vmovapd	(%rdx), %ymm0
; CHECK-NEXT: vfmsub213pd (%rcx), %ymm0, %ymm0
; CHECK-NEXT: retq
  %res = call <4 x double> @llvm.x86.fma.vfmsub.pd.256(<4 x double> %b, <4 x double> %b, <4 x double> %a) nounwind
  ret <4 x double> %res
}


declare <4 x float> @llvm.x86.fma.vfnmsub.ps(<4 x float>, <4 x float>, <4 x float>) nounwind readnone
define <4 x float> @test_x86_fnmsub_baa_ps(<4 x float> %a, <4 x float> %b) #0 {
; CHECK-LABEL: test_x86_fnmsub_baa_ps:
; CHECK:       # BB#0:
; CHECK-NEXT: vmovaps	(%rcx), %xmm0
; CHECK-NEXT: vfnmsub132ps (%rdx), %xmm0, %xmm0
; CHECK-NEXT: retq
  %res = call <4 x float> @llvm.x86.fma.vfnmsub.ps(<4 x float> %b, <4 x float> %a, <4 x float> %a) nounwind
  ret <4 x float> %res
}

define <4 x float> @test_x86_fnmsub_aba_ps(<4 x float> %a, <4 x float> %b) #0 {
; CHECK-LABEL: test_x86_fnmsub_aba_ps:
; CHECK:       # BB#0:
; CHECK-NEXT: vmovaps	(%rcx), %xmm0
; CHECK-NEXT: vfnmsub231ps (%rdx), %xmm0, %xmm0
; CHECK-NEXT: retq
  %res = call <4 x float> @llvm.x86.fma.vfnmsub.ps(<4 x float> %a, <4 x float> %b, <4 x float> %a) nounwind
  ret <4 x float> %res
}

define <4 x float> @test_x86_fnmsub_bba_ps(<4 x float> %a, <4 x float> %b) #0 {
; CHECK-LABEL: test_x86_fnmsub_bba_ps:
; CHECK:       # BB#0:
; CHECK-NEXT: vmovaps	(%rdx), %xmm0
; CHECK-NEXT: vfnmsub213ps (%rcx), %xmm0, %xmm0
; CHECK-NEXT: retq
  %res = call <4 x float> @llvm.x86.fma.vfnmsub.ps(<4 x float> %b, <4 x float> %b, <4 x float> %a) nounwind
  ret <4 x float> %res
}

declare <8 x float> @llvm.x86.fma.vfnmsub.ps.256(<8 x float>, <8 x float>, <8 x float>) nounwind readnone
define <8 x float> @test_x86_fnmsub_baa_ps_y(<8 x float> %a, <8 x float> %b) #0 {
; CHECK-LABEL: test_x86_fnmsub_baa_ps_y:
; CHECK:       # BB#0:
; CHECK-NEXT: vmovaps	(%rcx), %ymm0
; CHECK-NEXT: vfnmsub132ps (%rdx), %ymm0, %ymm0
; CHECK-NEXT: retq
  %res = call <8 x float> @llvm.x86.fma.vfnmsub.ps.256(<8 x float> %b, <8 x float> %a, <8 x float> %a) nounwind
  ret <8 x float> %res
}

define <8 x float> @test_x86_fnmsub_aba_ps_y(<8 x float> %a, <8 x float> %b) #0 {
; CHECK-LABEL: test_x86_fnmsub_aba_ps_y:
; CHECK:       # BB#0:
; CHECK-NEXT: vmovaps	(%rcx), %ymm0
; CHECK-NEXT: vfnmsub231ps (%rdx), %ymm0, %ymm0
; CHECK-NEXT: retq
  %res = call <8 x float> @llvm.x86.fma.vfnmsub.ps.256(<8 x float> %a, <8 x float> %b, <8 x float> %a) nounwind
  ret <8 x float> %res
}

define <8 x float> @test_x86_fnmsub_bba_ps_y(<8 x float> %a, <8 x float> %b) #0 {
; CHECK-LABEL: test_x86_fnmsub_bba_ps_y:
; CHECK:       # BB#0:
; CHECK-NEXT: vmovaps	(%rdx), %ymm0
; CHECK-NEXT: vfnmsub213ps (%rcx), %ymm0, %ymm0
; CHECK-NEXT: retq
  %res = call <8 x float> @llvm.x86.fma.vfnmsub.ps.256(<8 x float> %b, <8 x float> %b, <8 x float> %a) nounwind
  ret <8 x float> %res
}

declare <2 x double> @llvm.x86.fma.vfnmsub.pd(<2 x double>, <2 x double>, <2 x double>) nounwind readnone
define <2 x double> @test_x86_fnmsub_baa_pd(<2 x double> %a, <2 x double> %b) #0 {
; CHECK-LABEL: test_x86_fnmsub_baa_pd:
; CHECK:       # BB#0:
; CHECK-NEXT: vmovapd	(%rcx), %xmm0
; CHECK-NEXT: vfnmsub132pd (%rdx), %xmm0, %xmm0
; CHECK-NEXT: retq
  %res = call <2 x double> @llvm.x86.fma.vfnmsub.pd(<2 x double> %b, <2 x double> %a, <2 x double> %a) nounwind
  ret <2 x double> %res
}

define <2 x double> @test_x86_fnmsub_aba_pd(<2 x double> %a, <2 x double> %b) #0 {
; CHECK-LABEL: test_x86_fnmsub_aba_pd:
; CHECK:       # BB#0:
; CHECK-NEXT: vmovapd	(%rcx), %xmm0
; CHECK-NEXT: vfnmsub231pd (%rdx), %xmm0, %xmm0
; CHECK-NEXT: retq
  %res = call <2 x double> @llvm.x86.fma.vfnmsub.pd(<2 x double> %a, <2 x double> %b, <2 x double> %a) nounwind
  ret <2 x double> %res
}

define <2 x double> @test_x86_fnmsub_bba_pd(<2 x double> %a, <2 x double> %b) #0 {
; CHECK-LABEL: test_x86_fnmsub_bba_pd:
; CHECK:       # BB#0:
; CHECK-NEXT: vmovapd	(%rdx), %xmm0
; CHECK-NEXT: vfnmsub213pd (%rcx), %xmm0, %xmm0
; CHECK-NEXT: retq
  %res = call <2 x double> @llvm.x86.fma.vfnmsub.pd(<2 x double> %b, <2 x double> %b, <2 x double> %a) nounwind
  ret <2 x double> %res
}

declare <4 x double> @llvm.x86.fma.vfnmsub.pd.256(<4 x double>, <4 x double>, <4 x double>) nounwind readnone
define <4 x double> @test_x86_fnmsub_baa_pd_y(<4 x double> %a, <4 x double> %b) #0 {
; CHECK-LABEL: test_x86_fnmsub_baa_pd_y:
; CHECK:       # BB#0:
; CHECK-NEXT: vmovapd	(%rcx), %ymm0
; CHECK-NEXT: vfnmsub132pd (%rdx), %ymm0, %ymm0
; CHECK-NEXT: retq
  %res = call <4 x double> @llvm.x86.fma.vfnmsub.pd.256(<4 x double> %b, <4 x double> %a, <4 x double> %a) nounwind
  ret <4 x double> %res
}

define <4 x double> @test_x86_fnmsub_aba_pd_y(<4 x double> %a, <4 x double> %b) #0 {
; CHECK-LABEL: test_x86_fnmsub_aba_pd_y:
; CHECK:       # BB#0:
; CHECK-NEXT: vmovapd	(%rcx), %ymm0
; CHECK-NEXT: vfnmsub231pd (%rdx), %ymm0, %ymm0
; CHECK-NEXT: retq
  %res = call <4 x double> @llvm.x86.fma.vfnmsub.pd.256(<4 x double> %a, <4 x double> %b, <4 x double> %a) nounwind
  ret <4 x double> %res
}

define <4 x double> @test_x86_fnmsub_bba_pd_y(<4 x double> %a, <4 x double> %b) #0 {
; CHECK-LABEL: test_x86_fnmsub_bba_pd_y:
; CHECK:       # BB#0:
; CHECK-NEXT: vmovapd	(%rdx), %ymm0
; CHECK-NEXT: vfnmsub213pd (%rcx), %ymm0, %ymm0
; CHECK-NEXT: retq
  %res = call <4 x double> @llvm.x86.fma.vfnmsub.pd.256(<4 x double> %b, <4 x double> %b, <4 x double> %a) nounwind
  ret <4 x double> %res
}

