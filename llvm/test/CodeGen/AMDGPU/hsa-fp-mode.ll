; RUN: llc -mtriple=amdgcn--amdhsa -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s

; GCN-LABEL: {{^}}test_default_ci:
; GCN: float_mode = 192
; GCN: enable_dx10_clamp = 1
; GCN: enable_ieee_mode = 1
define amdgpu_kernel void @test_default_ci(float addrspace(1)* %out0, double addrspace(1)* %out1) #0 {
  store float 0.0, float addrspace(1)* %out0
  store double 0.0, double addrspace(1)* %out1
  ret void
}

; GCN-LABEL: {{^}}test_default_vi:
; GCN: float_mode = 192
; GCN: enable_dx10_clamp = 1
; GCN: enable_ieee_mode = 1
define amdgpu_kernel void @test_default_vi(float addrspace(1)* %out0, double addrspace(1)* %out1) #1 {
  store float 0.0, float addrspace(1)* %out0
  store double 0.0, double addrspace(1)* %out1
  ret void
}

; GCN-LABEL: {{^}}test_f64_denormals:
; GCN: float_mode = 192
; GCN: enable_dx10_clamp = 1
; GCN: enable_ieee_mode = 1
define amdgpu_kernel void @test_f64_denormals(float addrspace(1)* %out0, double addrspace(1)* %out1) #2 {
  store float 0.0, float addrspace(1)* %out0
  store double 0.0, double addrspace(1)* %out1
  ret void
}

; GCN-LABEL: {{^}}test_f32_denormals:
; GCN: float_mode = 48
; GCN: enable_dx10_clamp = 1
; GCN: enable_ieee_mode = 1
define amdgpu_kernel void @test_f32_denormals(float addrspace(1)* %out0, double addrspace(1)* %out1) #3 {
  store float 0.0, float addrspace(1)* %out0
  store double 0.0, double addrspace(1)* %out1
  ret void
}

; GCN-LABEL: {{^}}test_f32_f64_denormals:
; GCN: float_mode = 240
; GCN: enable_dx10_clamp = 1
; GCN: enable_ieee_mode = 1
define amdgpu_kernel void @test_f32_f64_denormals(float addrspace(1)* %out0, double addrspace(1)* %out1) #4 {
  store float 0.0, float addrspace(1)* %out0
  store double 0.0, double addrspace(1)* %out1
  ret void
}

; GCN-LABEL: {{^}}test_no_denormals:
; GCN: float_mode = 0
; GCN: enable_dx10_clamp = 1
; GCN: enable_ieee_mode = 1
define amdgpu_kernel void @test_no_denormals(float addrspace(1)* %out0, double addrspace(1)* %out1) #5 {
  store float 0.0, float addrspace(1)* %out0
  store double 0.0, double addrspace(1)* %out1
  ret void
}

; GCN-LABEL: {{^}}test_no_dx10_clamp_vi:
; GCN: float_mode = 192
; GCN: enable_dx10_clamp = 0
; GCN: enable_ieee_mode = 1
define amdgpu_kernel void @test_no_dx10_clamp_vi(float addrspace(1)* %out0, double addrspace(1)* %out1) #6 {
  store float 0.0, float addrspace(1)* %out0
  store double 0.0, double addrspace(1)* %out1
  ret void
}

; GCN-LABEL: {{^}}test_no_ieee_mode_vi:
; GCN: float_mode = 192
; GCN: enable_dx10_clamp = 1
; GCN: enable_ieee_mode = 0
define amdgpu_kernel void @test_no_ieee_mode_vi(float addrspace(1)* %out0, double addrspace(1)* %out1) #7 {
  store float 0.0, float addrspace(1)* %out0
  store double 0.0, double addrspace(1)* %out1
  ret void
}

; GCN-LABEL: {{^}}test_no_ieee_mode_no_dx10_clamp_vi:
; GCN: float_mode = 192
; GCN: enable_dx10_clamp = 0
; GCN: enable_ieee_mode = 0
define amdgpu_kernel void @test_no_ieee_mode_no_dx10_clamp_vi(float addrspace(1)* %out0, double addrspace(1)* %out1) #8 {
  store float 0.0, float addrspace(1)* %out0
  store double 0.0, double addrspace(1)* %out1
  ret void
}

attributes #0 = { nounwind "target-cpu"="kaveri" "target-features"="-code-object-v3" }
attributes #1 = { nounwind "target-cpu"="fiji" "target-features"="-code-object-v3" }
attributes #2 = { nounwind "target-features"="-code-object-v3,-fp32-denormals,+fp64-fp16-denormals" }
attributes #3 = { nounwind "target-features"="-code-object-v3,+fp32-denormals,-fp64-fp16-denormals" }
attributes #4 = { nounwind "target-features"="-code-object-v3,+fp32-denormals,+fp64-fp16-denormals" }
attributes #5 = { nounwind "target-features"="-code-object-v3,-fp32-denormals,-fp64-fp16-denormals" }
attributes #6 = { nounwind "amdgpu-dx10-clamp"="false" "target-cpu"="fiji" "target-features"="-code-object-v3" }
attributes #7 = { nounwind "amdgpu-ieee"="false" "target-cpu"="fiji" "target-features"="-code-object-v3" }
attributes #8 = { nounwind "amdgpu-dx10-clamp"="false" "amdgpu-ieee"="false" "target-cpu"="fiji" "target-features"="-code-object-v3" }
