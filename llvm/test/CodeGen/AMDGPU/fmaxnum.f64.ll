; RUN: llc -mtriple=amdgcn < %s | FileCheck -check-prefix=SI -check-prefix=FUNC %s
; RUN: llc -mtriple=amdgcn -mcpu=tonga -mattr=-flat-for-global < %s | FileCheck -check-prefix=SI -check-prefix=FUNC %s

declare double @llvm.maxnum.f64(double, double) #0
declare <2 x double> @llvm.maxnum.v2f64(<2 x double>, <2 x double>) #0
declare <4 x double> @llvm.maxnum.v4f64(<4 x double>, <4 x double>) #0
declare <8 x double> @llvm.maxnum.v8f64(<8 x double>, <8 x double>) #0
declare <16 x double> @llvm.maxnum.v16f64(<16 x double>, <16 x double>) #0

; FUNC-LABEL: @test_fmax_f64
; SI: v_max_f64
define amdgpu_kernel void @test_fmax_f64(ptr addrspace(1) %out, double %a, double %b) nounwind {
  %val = call double @llvm.maxnum.f64(double %a, double %b) #0
  store double %val, ptr addrspace(1) %out, align 8
  ret void
}

; FUNC-LABEL: @test_fmax_v2f64
; SI: v_max_f64
; SI: v_max_f64
define amdgpu_kernel void @test_fmax_v2f64(ptr addrspace(1) %out, <2 x double> %a, <2 x double> %b) nounwind {
  %val = call <2 x double> @llvm.maxnum.v2f64(<2 x double> %a, <2 x double> %b) #0
  store <2 x double> %val, ptr addrspace(1) %out, align 16
  ret void
}

; FUNC-LABEL: @test_fmax_v4f64
; SI: v_max_f64
; SI: v_max_f64
; SI: v_max_f64
; SI: v_max_f64
define amdgpu_kernel void @test_fmax_v4f64(ptr addrspace(1) %out, <4 x double> %a, <4 x double> %b) nounwind {
  %val = call <4 x double> @llvm.maxnum.v4f64(<4 x double> %a, <4 x double> %b) #0
  store <4 x double> %val, ptr addrspace(1) %out, align 32
  ret void
}

; FUNC-LABEL: @test_fmax_v8f64
; SI: v_max_f64
; SI: v_max_f64
; SI: v_max_f64
; SI: v_max_f64
; SI: v_max_f64
; SI: v_max_f64
; SI: v_max_f64
; SI: v_max_f64
define amdgpu_kernel void @test_fmax_v8f64(ptr addrspace(1) %out, <8 x double> %a, <8 x double> %b) nounwind {
  %val = call <8 x double> @llvm.maxnum.v8f64(<8 x double> %a, <8 x double> %b) #0
  store <8 x double> %val, ptr addrspace(1) %out, align 64
  ret void
}

; FUNC-LABEL: @test_fmax_v16f64
; SI: v_max_f64
; SI: v_max_f64
; SI: v_max_f64
; SI: v_max_f64
; SI: v_max_f64
; SI: v_max_f64
; SI: v_max_f64
; SI: v_max_f64
; SI: v_max_f64
; SI: v_max_f64
; SI: v_max_f64
; SI: v_max_f64
; SI: v_max_f64
; SI: v_max_f64
; SI: v_max_f64
; SI: v_max_f64
define amdgpu_kernel void @test_fmax_v16f64(ptr addrspace(1) %out, <16 x double> %a, <16 x double> %b) nounwind {
  %val = call <16 x double> @llvm.maxnum.v16f64(<16 x double> %a, <16 x double> %b) #0
  store <16 x double> %val, ptr addrspace(1) %out, align 128
  ret void
}

attributes #0 = { nounwind readnone }
