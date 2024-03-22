; RUN: llc -mtriple=amdgcn -mcpu=tahiti -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s
; RUN: llc -mtriple=amdgcn -mcpu=fiji -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s

declare float @llvm.amdgcn.cubesc(float, float, float) nounwind readnone

; GCN-LABEL: {{^}}test_cubesc:
; GCN: v_cubesc_f32 v{{[0-9]+}}, s{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
define amdgpu_kernel void @test_cubesc(ptr addrspace(1) %out, float %a, float %b, float %c) nounwind {
  %result = call float @llvm.amdgcn.cubesc(float %a, float %b, float %c)
  store float %result, ptr addrspace(1) %out
  ret void
}
