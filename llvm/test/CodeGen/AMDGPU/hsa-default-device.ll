; RUN: llc -march=amdgcn -mtriple=amdgcn-unknown-amdhsa < %s | FileCheck %s

; Make sure that with an HSA triple, we don't default to an
; unsupported device.

; CHECK: .hsa_code_object_isa 7,0,0,"AMD","AMDGPU"
define void @test_kernel(float addrspace(1)* %out0, double addrspace(1)* %out1) nounwind {
  store float 0.0, float addrspace(1)* %out0
  ret void
}

