; RUN: llc -march=amdgcn -mcpu=tahiti < %s | FileCheck -check-prefix=SI -check-prefix=FUNC %s
; RUN: llc -march=amdgcn -mcpu=tonga < %s | FileCheck -check-prefix=SI -check-prefix=FUNC %s

declare double @llvm.minnum.f64(double, double) #0
declare <2 x double> @llvm.minnum.v2f64(<2 x double>, <2 x double>) #0
declare <4 x double> @llvm.minnum.v4f64(<4 x double>, <4 x double>) #0
declare <8 x double> @llvm.minnum.v8f64(<8 x double>, <8 x double>) #0
declare <16 x double> @llvm.minnum.v16f64(<16 x double>, <16 x double>) #0

; FUNC-LABEL: @test_fmin_f64
; SI: v_min_f64
define amdgpu_kernel void @test_fmin_f64(double addrspace(1)* %out, double %a, double %b) nounwind {
  %val = call double @llvm.minnum.f64(double %a, double %b) #0
  store double %val, double addrspace(1)* %out, align 8
  ret void
}

; FUNC-LABEL: @test_fmin_v2f64
; SI: v_min_f64
; SI: v_min_f64
define amdgpu_kernel void @test_fmin_v2f64(<2 x double> addrspace(1)* %out, <2 x double> %a, <2 x double> %b) nounwind {
  %val = call <2 x double> @llvm.minnum.v2f64(<2 x double> %a, <2 x double> %b) #0
  store <2 x double> %val, <2 x double> addrspace(1)* %out, align 16
  ret void
}

; FUNC-LABEL: @test_fmin_v4f64
; SI: v_min_f64
; SI: v_min_f64
; SI: v_min_f64
; SI: v_min_f64
define amdgpu_kernel void @test_fmin_v4f64(<4 x double> addrspace(1)* %out, <4 x double> %a, <4 x double> %b) nounwind {
  %val = call <4 x double> @llvm.minnum.v4f64(<4 x double> %a, <4 x double> %b) #0
  store <4 x double> %val, <4 x double> addrspace(1)* %out, align 32
  ret void
}

; FUNC-LABEL: @test_fmin_v8f64
; SI: v_min_f64
; SI: v_min_f64
; SI: v_min_f64
; SI: v_min_f64
; SI: v_min_f64
; SI: v_min_f64
; SI: v_min_f64
; SI: v_min_f64
define amdgpu_kernel void @test_fmin_v8f64(<8 x double> addrspace(1)* %out, <8 x double> %a, <8 x double> %b) nounwind {
  %val = call <8 x double> @llvm.minnum.v8f64(<8 x double> %a, <8 x double> %b) #0
  store <8 x double> %val, <8 x double> addrspace(1)* %out, align 64
  ret void
}

; FUNC-LABEL: @test_fmin_v16f64
; SI: v_min_f64
; SI: v_min_f64
; SI: v_min_f64
; SI: v_min_f64
; SI: v_min_f64
; SI: v_min_f64
; SI: v_min_f64
; SI: v_min_f64
; SI: v_min_f64
; SI: v_min_f64
; SI: v_min_f64
; SI: v_min_f64
; SI: v_min_f64
; SI: v_min_f64
; SI: v_min_f64
; SI: v_min_f64
define amdgpu_kernel void @test_fmin_v16f64(<16 x double> addrspace(1)* %out, <16 x double> %a, <16 x double> %b) nounwind {
  %val = call <16 x double> @llvm.minnum.v16f64(<16 x double> %a, <16 x double> %b) #0
  store <16 x double> %val, <16 x double> addrspace(1)* %out, align 128
  ret void
}

attributes #0 = { nounwind readnone }
