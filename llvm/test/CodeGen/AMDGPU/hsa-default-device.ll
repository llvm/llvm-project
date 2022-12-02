; RUN: llc -march=amdgcn -mtriple=amdgcn-unknown-amdhsa --amdhsa-code-object-version=2 < %s | FileCheck %s

; Make sure that with an HSA triple, we don't default to an
; unsupported device.

; CHECK: .hsa_code_object_isa 7,0,0,"AMD","AMDGPU"
define amdgpu_kernel void @test_kernel(ptr addrspace(1) %out0, ptr addrspace(1) %out1) nounwind {
  store float 0.0, ptr addrspace(1) %out0
  ret void
}

