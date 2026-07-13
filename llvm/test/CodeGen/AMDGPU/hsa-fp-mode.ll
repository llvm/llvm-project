; RUN: llc -mtriple=amdgpu7.00-amd-amdhsa < %s | FileCheck -check-prefix=GCN %s
; RUN: llc -mtriple=amdgpu8.03-amd-amdhsa < %s | FileCheck -check-prefix=GCN %s

; GCN-LABEL: {{^}}test_default:
; GCN: .amdhsa_dx10_clamp 1
; GCN: .amdhsa_ieee_mode 1
; GCN: FloatMode: 240
define amdgpu_kernel void @test_default(ptr addrspace(1) %out0, ptr addrspace(1) %out1) {
  store float 0.0, ptr addrspace(1) %out0
  store double 0.0, ptr addrspace(1) %out1
  ret void
}

; GCN-LABEL: {{^}}test_f64_denormals:
; GCN: .amdhsa_dx10_clamp 1
; GCN: .amdhsa_ieee_mode 1
; GCN: FloatMode: 192
define amdgpu_kernel void @test_f64_denormals(ptr addrspace(1) %out0, ptr addrspace(1) %out1) #0 {
  store float 0.0, ptr addrspace(1) %out0
  store double 0.0, ptr addrspace(1) %out1
  ret void
}

; GCN-LABEL: {{^}}test_f32_denormals:
; GCN: .amdhsa_dx10_clamp 1
; GCN: .amdhsa_ieee_mode 1
; GCN: FloatMode: 48
define amdgpu_kernel void @test_f32_denormals(ptr addrspace(1) %out0, ptr addrspace(1) %out1) #1 {
  store float 0.0, ptr addrspace(1) %out0
  store double 0.0, ptr addrspace(1) %out1
  ret void
}

; GCN-LABEL: {{^}}test_f32_f64_denormals:
; GCN: .amdhsa_dx10_clamp 1
; GCN: .amdhsa_ieee_mode 1
; GCN: FloatMode: 240
define amdgpu_kernel void @test_f32_f64_denormals(ptr addrspace(1) %out0, ptr addrspace(1) %out1) #2 {
  store float 0.0, ptr addrspace(1) %out0
  store double 0.0, ptr addrspace(1) %out1
  ret void
}

; GCN-LABEL: {{^}}test_no_denormals:
; GCN: .amdhsa_dx10_clamp 1
; GCN: .amdhsa_ieee_mode 1
; GCN: FloatMode: 0
define amdgpu_kernel void @test_no_denormals(ptr addrspace(1) %out0, ptr addrspace(1) %out1) #3 {
  store float 0.0, ptr addrspace(1) %out0
  store double 0.0, ptr addrspace(1) %out1
  ret void
}

; GCN-LABEL: {{^}}test_no_dx10_clamp_vi:
; GCN: .amdhsa_dx10_clamp 0
; GCN: .amdhsa_ieee_mode 1
; GCN: FloatMode: 240
define amdgpu_kernel void @test_no_dx10_clamp_vi(ptr addrspace(1) %out0, ptr addrspace(1) %out1) #4 {
  store float 0.0, ptr addrspace(1) %out0
  store double 0.0, ptr addrspace(1) %out1
  ret void
}

; GCN-LABEL: {{^}}test_no_ieee_mode_vi:
; GCN: .amdhsa_dx10_clamp 1
; GCN: .amdhsa_ieee_mode 0
; GCN: FloatMode: 240
define amdgpu_kernel void @test_no_ieee_mode_vi(ptr addrspace(1) %out0, ptr addrspace(1) %out1) #5 {
  store float 0.0, ptr addrspace(1) %out0
  store double 0.0, ptr addrspace(1) %out1
  ret void
}

; GCN-LABEL: {{^}}test_no_ieee_mode_no_dx10_clamp_vi:
; GCN: .amdhsa_dx10_clamp 0
; GCN: .amdhsa_ieee_mode 0
; GCN: FloatMode: 240
define amdgpu_kernel void @test_no_ieee_mode_no_dx10_clamp_vi(ptr addrspace(1) %out0, ptr addrspace(1) %out1) #6 {
  store float 0.0, ptr addrspace(1) %out0
  store double 0.0, ptr addrspace(1) %out1
  ret void
}

attributes #0 = { nounwind denormal_fpenv(float: preservesign) }
attributes #1 = { nounwind denormal_fpenv(preservesign, float: ieee) }
attributes #2 = { nounwind denormal_fpenv(ieee) }
attributes #3 = { nounwind denormal_fpenv(preservesign) }
attributes #4 = { nounwind "amdgpu-dx10-clamp"="false" }
attributes #5 = { nounwind "amdgpu-ieee"="false" }
attributes #6 = { nounwind "amdgpu-dx10-clamp"="false" "amdgpu-ieee"="false" }

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"amdhsa_code_object_version", i32 400}
