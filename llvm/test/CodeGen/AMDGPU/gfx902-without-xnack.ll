; RUN: llc -march=amdgcn -mtriple=amdgcn-amd-amdhsa -mcpu=gfx902 -mattr=-code-object-v3,-xnack < %s | FileCheck %s

; CHECK: .hsa_code_object_isa 9,0,2,"AMD","AMDGPU"
define amdgpu_kernel void @test_kernel(float addrspace(1)* %out0, double addrspace(1)* %out1) nounwind {
  store float 0.0, float addrspace(1)* %out0
  ret void
}

