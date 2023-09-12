; RUN: llc -mtriple=amdgcn -mtriple=amdgcn-amd-amdhsa -mcpu=gfx902 -mattr=-xnack < %s | FileCheck %s

; CHECK: .hsa_code_object_isa 9,0,2,"AMD","AMDGPU"
define amdgpu_kernel void @test_kernel(ptr addrspace(1) %out0, ptr addrspace(1) %out1) nounwind {
  store float 0.0, ptr addrspace(1) %out0
  ret void
}

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"amdgpu_code_object_version", i32 200}

