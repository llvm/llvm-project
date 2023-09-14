; RUN: llc -mtriple=amdgcn -mtriple=amdgcn-unknown-amdhsa < %s | FileCheck %s

; Make sure that with an HSA triple, we don't default to an
; unsupported device.

; CHECK: .hsa_code_object_isa 7,0,0,"AMD","AMDGPU"
define amdgpu_kernel void @test_kernel(ptr addrspace(1) %out0, ptr addrspace(1) %out1) nounwind {
  store float 0.0, ptr addrspace(1) %out0
  ret void
}

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"amdgpu_code_object_version", i32 200}
