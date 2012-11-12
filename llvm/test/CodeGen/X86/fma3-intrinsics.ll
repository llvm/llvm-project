; RUN: llc < %s -mtriple=x86_64-pc-win32 -mcpu=core-avx2 | FileCheck %s
; RUN: llc < %s -mtriple=x86_64-pc-win32 -mattr=+fma,+fma4 | FileCheck %s
; RUN: llc < %s -mcpu=bdver2 -mtriple=x86_64-pc-win32 -mattr=-fma4 | FileCheck %s

define <4 x float> @test_x86_fmadd_ss(<4 x float> %a0, <4 x float> %a1, <4 x float> %a2) {
  ; CHECK: fmadd213ss %xmm
  %res = call <4 x float> @llvm.x86.fma.vfmadd.ss(<4 x float> %a0, <4 x float> %a1, <4 x float> %a2) nounwind
  ret <4 x float> %res
}
declare <4 x float> @llvm.x86.fma.vfmadd.ss(<4 x float>, <4 x float>, <4 x float>) nounwind readnone

define <4 x float> @test_x86_fmadd_ps(<4 x float> %a0, <4 x float> %a1, <4 x float> %a2) {
  ; CHECK: fmadd213ps
  %res = call <4 x float> @llvm.x86.fma.vfmadd.ps(<4 x float> %a0, <4 x float> %a1, <4 x float> %a2) nounwind
  ret <4 x float> %res
}
declare <4 x float> @llvm.x86.fma.vfmadd.ps(<4 x float>, <4 x float>, <4 x float>) nounwind readnone

define <8 x float> @test_x86_fmadd_ps_y(<8 x float> %a0, <8 x float> %a1, <8 x float> %a2) {
  ; CHECK: fmadd213ps {{.*\(%r.*}}, %ymm
  %res = call <8 x float> @llvm.x86.fma.vfmadd.ps.256(<8 x float> %a0, <8 x float> %a1, <8 x float> %a2) nounwind
  ret <8 x float> %res
}
declare <8 x float> @llvm.x86.fma.vfmadd.ps.256(<8 x float>, <8 x float>, <8 x float>) nounwind readnone

define <4 x float> @test_x86_fnmadd_ss(<4 x float> %a0, <4 x float> %a1, <4 x float> %a2) {
  ; CHECK: fnmadd213ss %xmm
  %res = call <4 x float> @llvm.x86.fma.vfnmadd.ss(<4 x float> %a0, <4 x float> %a1, <4 x float> %a2) nounwind
  ret <4 x float> %res
}
declare <4 x float> @llvm.x86.fma.vfnmadd.ss(<4 x float>, <4 x float>, <4 x float>) nounwind readnone

define <4 x float> @test_x86_fnmadd_ps(<4 x float> %a0, <4 x float> %a1, <4 x float> %a2) {
  ; CHECK: fnmadd213ps
  %res = call <4 x float> @llvm.x86.fma.vfnmadd.ps(<4 x float> %a0, <4 x float> %a1, <4 x float> %a2) nounwind
  ret <4 x float> %res
}
declare <4 x float> @llvm.x86.fma.vfnmadd.ps(<4 x float>, <4 x float>, <4 x float>) nounwind readnone

define <8 x float> @test_x86_fnmadd_ps_y(<8 x float> %a0, <8 x float> %a1, <8 x float> %a2) {
  ; CHECK: fnmadd213ps {{.*\(%r.*}}, %ymm
  %res = call <8 x float> @llvm.x86.fma.vfnmadd.ps.256(<8 x float> %a0, <8 x float> %a1, <8 x float> %a2) nounwind
  ret <8 x float> %res
}
declare <8 x float> @llvm.x86.fma.vfnmadd.ps.256(<8 x float>, <8 x float>, <8 x float>) nounwind readnone


define <4 x float> @test_x86_fmsub_ss(<4 x float> %a0, <4 x float> %a1, <4 x float> %a2) {
  ; CHECK: fmsub213ss
  %res = call <4 x float> @llvm.x86.fma.vfmsub.ss(<4 x float> %a0, <4 x float> %a1, <4 x float> %a2) nounwind
  ret <4 x float> %res
}
declare <4 x float> @llvm.x86.fma.vfmsub.ss(<4 x float>, <4 x float>, <4 x float>) nounwind readnone

define <4 x float> @test_x86_fmsub_ps(<4 x float> %a0, <4 x float> %a1, <4 x float> %a2) {
  ; CHECK: fmsub213ps
  %res = call <4 x float> @llvm.x86.fma.vfmsub.ps(<4 x float> %a0, <4 x float> %a1, <4 x float> %a2) nounwind
  ret <4 x float> %res
}
declare <4 x float> @llvm.x86.fma.vfmsub.ps(<4 x float>, <4 x float>, <4 x float>) nounwind readnone

define <4 x float> @test_x86_fnmsub_ss(<4 x float> %a0, <4 x float> %a1, <4 x float> %a2) {
  ; CHECK: fnmsub213ss
  %res = call <4 x float> @llvm.x86.fma.vfnmsub.ss(<4 x float> %a0, <4 x float> %a1, <4 x float> %a2) nounwind
  ret <4 x float> %res
}
declare <4 x float> @llvm.x86.fma.vfnmsub.ss(<4 x float>, <4 x float>, <4 x float>) nounwind readnone

define <4 x float> @test_x86_fnmsub_ps(<4 x float> %a0, <4 x float> %a1, <4 x float> %a2) {
  ; CHECK: fnmsub213ps
  %res = call <4 x float> @llvm.x86.fma.vfnmsub.ps(<4 x float> %a0, <4 x float> %a1, <4 x float> %a2) nounwind
  ret <4 x float> %res
}
declare <4 x float> @llvm.x86.fma.vfnmsub.ps(<4 x float>, <4 x float>, <4 x float>) nounwind readnone

;;;;

define <2 x double> @test_x86_fmadd_sd(<2 x double> %a0, <2 x double> %a1, <2 x double> %a2) {
  ; CHECK: fmadd213sd
  %res = call <2 x double> @llvm.x86.fma.vfmadd.sd(<2 x double> %a0, <2 x double> %a1, <2 x double> %a2) nounwind
  ret <2 x double> %res
}
declare <2 x double> @llvm.x86.fma.vfmadd.sd(<2 x double>, <2 x double>, <2 x double>) nounwind readnone

define <2 x double> @test_x86_fmadd_pd(<2 x double> %a0, <2 x double> %a1, <2 x double> %a2) {
  ; CHECK: fmadd213pd
  %res = call <2 x double> @llvm.x86.fma.vfmadd.pd(<2 x double> %a0, <2 x double> %a1, <2 x double> %a2) nounwind
  ret <2 x double> %res
}
declare <2 x double> @llvm.x86.fma.vfmadd.pd(<2 x double>, <2 x double>, <2 x double>) nounwind readnone

define <2 x double> @test_x86_fnmadd_sd(<2 x double> %a0, <2 x double> %a1, <2 x double> %a2) {
  ; CHECK: fnmadd213sd
  %res = call <2 x double> @llvm.x86.fma.vfnmadd.sd(<2 x double> %a0, <2 x double> %a1, <2 x double> %a2) nounwind
  ret <2 x double> %res
}
declare <2 x double> @llvm.x86.fma.vfnmadd.sd(<2 x double>, <2 x double>, <2 x double>) nounwind readnone

define <2 x double> @test_x86_fnmadd_pd(<2 x double> %a0, <2 x double> %a1, <2 x double> %a2) {
  ; CHECK: fnmadd213pd
  %res = call <2 x double> @llvm.x86.fma.vfnmadd.pd(<2 x double> %a0, <2 x double> %a1, <2 x double> %a2) nounwind
  ret <2 x double> %res
}
declare <2 x double> @llvm.x86.fma.vfnmadd.pd(<2 x double>, <2 x double>, <2 x double>) nounwind readnone



define <2 x double> @test_x86_fmsub_sd(<2 x double> %a0, <2 x double> %a1, <2 x double> %a2) {
  ; CHECK: fmsub213sd
  %res = call <2 x double> @llvm.x86.fma.vfmsub.sd(<2 x double> %a0, <2 x double> %a1, <2 x double> %a2) nounwind
  ret <2 x double> %res
}
declare <2 x double> @llvm.x86.fma.vfmsub.sd(<2 x double>, <2 x double>, <2 x double>) nounwind readnone

define <2 x double> @test_x86_fmsub_pd(<2 x double> %a0, <2 x double> %a1, <2 x double> %a2) {
  ; CHECK: fmsub213pd
  %res = call <2 x double> @llvm.x86.fma.vfmsub.pd(<2 x double> %a0, <2 x double> %a1, <2 x double> %a2) nounwind
  ret <2 x double> %res
}
declare <2 x double> @llvm.x86.fma.vfmsub.pd(<2 x double>, <2 x double>, <2 x double>) nounwind readnone

define <2 x double> @test_x86_fnmsub_sd(<2 x double> %a0, <2 x double> %a1, <2 x double> %a2) {
  ; CHECK: fnmsub213sd
  %res = call <2 x double> @llvm.x86.fma.vfnmsub.sd(<2 x double> %a0, <2 x double> %a1, <2 x double> %a2) nounwind
  ret <2 x double> %res
}
declare <2 x double> @llvm.x86.fma.vfnmsub.sd(<2 x double>, <2 x double>, <2 x double>) nounwind readnone

define <2 x double> @test_x86_fnmsub_pd(<2 x double> %a0, <2 x double> %a1, <2 x double> %a2) {
  ; CHECK: fnmsub213pd
  %res = call <2 x double> @llvm.x86.fma.vfnmsub.pd(<2 x double> %a0, <2 x double> %a1, <2 x double> %a2) nounwind
  ret <2 x double> %res
}
declare <2 x double> @llvm.x86.fma.vfnmsub.pd(<2 x double>, <2 x double>, <2 x double>) nounwind readnone
