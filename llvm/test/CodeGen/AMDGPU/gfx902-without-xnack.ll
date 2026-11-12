; RUN: llc -mtriple=amdgpu9.02-amd-amdhsa < %s | FileCheck %s

; CHECK: .amdgcn_target "amdgpu9.02-amd-amdhsa-unknown-gfx902:xnack-"
define amdgpu_kernel void @test_kernel(ptr addrspace(1) %out0, ptr addrspace(1) %out1) nounwind {
  store float 0.0, ptr addrspace(1) %out0
  ret void
}

!llvm.module.flags = !{!0, !1}
!0 = !{i32 1, !"amdhsa_code_object_version", i32 400}
!1 = !{i32 1, !"amdgpu.xnack", i32 0}
