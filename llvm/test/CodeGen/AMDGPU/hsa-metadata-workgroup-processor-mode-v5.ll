; RUN: llc -mtriple=amdgpu10.10-amd-amdhsa -mattr=+cumode < %s | FileCheck -check-prefix=GFX10 %s
; RUN: llc -mtriple=amdgpu10.10-amd-amdhsa < %s | FileCheck -check-prefix=GFX10-CU %s
; RUN: llc -mtriple=amdgpu11.00-amd-amdhsa -mattr=+cumode < %s | FileCheck -check-prefix=GFX10 %s
; RUN: llc -mtriple=amdgpu11.00-amd-amdhsa < %s | FileCheck -check-prefix=GFX10-CU %s

; GFX10:    .amdhsa_workgroup_processor_mode 0
; GFX10:    .workgroup_processor_mode: 0
; GFX10-CU: .amdhsa_workgroup_processor_mode 1
; GFX10-CU: .workgroup_processor_mode: 1

define amdgpu_kernel void @wavefrontsize() {
entry:
  ret void
}

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"amdhsa_code_object_version", i32 500}
