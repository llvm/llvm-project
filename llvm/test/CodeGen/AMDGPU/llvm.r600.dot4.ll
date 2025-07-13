; RUN: llc -mtriple=r600 -mcpu=redwood -verify-machineinstrs < %s

declare float @llvm.r600.dot4(<4 x float>, <4 x float>) nounwind readnone

define amdgpu_kernel void @test_dp4(ptr addrspace(1) %out, ptr addrspace(1) %a, ptr addrspace(1) %b) nounwind {
  %src0 = load <4 x float>, ptr addrspace(1) %a, align 16
  %src1 = load <4 x float>, ptr addrspace(1) %b, align 16
  %dp4 = call float @llvm.r600.dot4(<4 x float> %src0, <4 x float> %src1) nounwind readnone
  store float %dp4, ptr addrspace(1) %out, align 4
  ret void
}
