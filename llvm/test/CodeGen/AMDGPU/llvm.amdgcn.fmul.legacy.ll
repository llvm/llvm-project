; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s
; RUN: llc -march=amdgcn -mcpu=tonga -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s


; GCN-LABEL: {{^}}test_mul_legacy_f32:
; GCN: v_mul_legacy_f32_e32 v{{[0-9]+}}, s{{[0-9]+}}, v{{[0-9]+}}
define amdgpu_kernel void @test_mul_legacy_f32(float addrspace(1)* %out, float %a, float %b) #0 {
  %result = call float @llvm.amdgcn.fmul.legacy(float %a, float %b)
  store float %result, float addrspace(1)* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}test_mul_legacy_undef0_f32:
; GCN: v_mul_legacy_f32_e32
define amdgpu_kernel void @test_mul_legacy_undef0_f32(float addrspace(1)* %out, float %a) #0 {
  %result = call float @llvm.amdgcn.fmul.legacy(float undef, float %a)
  store float %result, float addrspace(1)* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}test_mul_legacy_undef1_f32:
; GCN: v_mul_legacy_f32_e32
define amdgpu_kernel void @test_mul_legacy_undef1_f32(float addrspace(1)* %out, float %a) #0 {
  %result = call float @llvm.amdgcn.fmul.legacy(float %a, float undef)
  store float %result, float addrspace(1)* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}test_mul_legacy_fabs_f32:
; GCN: v_mul_legacy_f32_e64 v{{[0-9]+}}, |v{{[0-9]+}}|, |s{{[0-9]+}}|
define amdgpu_kernel void @test_mul_legacy_fabs_f32(float addrspace(1)* %out, float %a, float %b) #0 {
  %a.fabs = call float @llvm.fabs.f32(float %a)
  %b.fabs = call float @llvm.fabs.f32(float %b)
  %result = call float @llvm.amdgcn.fmul.legacy(float %a.fabs, float %b.fabs)
  store float %result, float addrspace(1)* %out, align 4
  ret void
}

; TODO: Should match mac_legacy/mad_legacy
; GCN-LABEL: {{^}}test_mad_legacy_f32:
; GCN: v_mul_legacy_f32_e32 v{{[0-9]+}}, s{{[0-9]+}}, v{{[0-9]+}}
; GCN: v_add_f32_e32
define amdgpu_kernel void @test_mad_legacy_f32(float addrspace(1)* %out, float %a, float %b, float %c) #0 {
  %mul = call float @llvm.amdgcn.fmul.legacy(float %a, float %b)
  %add = fadd float %mul, %c
  store float %add, float addrspace(1)* %out, align 4
  ret void
}

declare float @llvm.fabs.f32(float) #1
declare float @llvm.amdgcn.fmul.legacy(float, float) #1

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }
